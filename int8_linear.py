"""INT8 Linear layer using Triton tensor core matmul on sm_121.

No dequantization — INT8 weights stay as int8, matmul via tl.dot(int8, int8) -> int32,
then multiply by per-channel scale and convert to bfloat16.

Usage:
    from int8_linear import patch_int8_linear
    patch_int8_linear()  # monkey-patches nn.Linear.forward
"""

import torch
import triton
import triton.language as tl

_original_linear_forward = torch.nn.Linear.forward


@triton.jit
def _int8_matmul_kernel(
    a_ptr, b_ptr, scale_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sm,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute out[M,N] = (A_int8[M,K] @ B_int8[N,K]^T) * scale[N,1]

    A is input quantized to int8, B is weight int8 [N,K], scale is [N,1] bfloat16.
    Output is bfloat16.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is [N, K], we want B^T, so we load B[n, k] and dot with A[m, k]
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = offs_k + k_start
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (k_offs[None, :] < K), other=0)
        # A[M, K] @ B[N, K]^T = A @ B^T -> dot(a, trans(b))
        acc += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Multiply by per-channel scale: scale[N, 1]
    scale = tl.load(scale_ptr + offs_n * stride_sm, mask=offs_n < N, other=1.0)
    out = acc.to(tl.float32) * scale[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def int8_linear_forward(x_bf16: torch.Tensor, weight_int8: torch.Tensor,
                        weight_scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """INT8 matmul: x[..., K] @ w[N, K]^T * scale[N] -> [..., N] bfloat16."""
    orig_shape = x_bf16.shape
    x_2d = x_bf16.reshape(-1, orig_shape[-1])  # [M, K]
    M, K = x_2d.shape
    N = weight_int8.shape[0]

    # Quantize input to int8 (per-token symmetric)
    x_abs_max = x_2d.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    x_scale = (127.0 / x_abs_max)
    x_int8 = (x_2d.float() * x_scale).round().clamp(-128, 127).to(torch.int8)
    # Combined scale: input_scale * weight_scale -> per [M, N] but weight_scale is per-channel [N, 1]
    # out = (x_int8 / x_scale) @ (w_int8 * w_scale)^T = (x_int8 @ w_int8^T) * (w_scale / x_scale)
    # Actually: real_out = (x / x_scale_factor) means x_real = x_int8 / x_scale
    # real_out[m,n] = sum_k (x_int8[m,k] / x_scale[m]) * (w_int8[n,k] * w_scale[n])
    #              = (sum_k x_int8[m,k] * w_int8[n,k]) * w_scale[n] / x_scale[m]
    # So: multiply int32 result by (w_scale[n] / x_scale[m])

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x_bf16.device)

    # Flatten weight_scale to [N]
    w_scale_flat = weight_scale.squeeze().float()  # [N]
    # Combined scale per [m, n]: w_scale[n] / x_scale[m] -- but kernel does per-N scale only
    # We need to handle per-M scale separately
    # Simpler: just do the int8 matmul, get int32, then scale in Python
    int32_out = torch.empty((M, N), dtype=torch.int32, device=x_bf16.device)

    BLOCK_M = min(16, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(128, triton.next_power_of_2(K))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _int8_matmul_raw[grid](
        x_int8, weight_int8, int32_out,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        weight_int8.stride(0), weight_int8.stride(1),
        int32_out.stride(0), int32_out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Apply scales: result = int32_out * w_scale[n] / x_scale[m]
    result = int32_out.float() * (w_scale_flat[None, :] / x_scale)
    result = result.to(torch.bfloat16)

    if bias is not None:
        result = result + bias

    return result.reshape(*orig_shape[:-1], N)


@triton.jit
def _int8_matmul_raw(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Raw INT8 matmul: out[M,N] = A_int8[M,K] @ B_int8[N,K]^T -> int32."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = offs_k + k_start
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (k_offs[None, :] < K), other=0)
        acc += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _patched_linear_forward(self, input):
    if self.weight.dtype == torch.int8 and hasattr(self, "weight_scale"):
        return int8_linear_forward(input, self.weight, self.weight_scale, self.bias)
    return _original_linear_forward(self, input)


def patch_int8_linear():
    torch.nn.Linear.forward = _patched_linear_forward
    print("[int8] Patched nn.Linear with Triton INT8 tensor core matmul ✅", flush=True)
