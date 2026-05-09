"""Test: can Triton tl.dot do INT8 x INT8 matmul on sm_121?"""
import torch
import triton
import triton.language as tl


@triton.jit
def int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((offs_k + k)[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=((offs_k + k)[:, None] < K) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def test():
    M, N, K = 16, 64, 128
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda")
    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    int8_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=128,
    )

    ref = a.float() @ b.float()
    diff = (c - ref).abs().max().item()
    print(f"Max diff: {diff}")
    if diff == 0:
        print("Triton INT8 tensor core matmul on sm_121: PERFECT ✅")
    else:
        print(f"Triton INT8 matmul: diff={diff} (may still be correct if small)")


if __name__ == "__main__":
    test()
