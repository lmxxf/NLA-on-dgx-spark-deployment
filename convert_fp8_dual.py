"""双机 FP8 量化：28 个 shard 按文件分配到两台 Spark，各自 GPU 量化，rank 0 合并元数据。

不做 tensor 切分——每个 shard 完整加载到一台机器的 GPU 上量化。
28 个 shard 各 ~5GB，单台峰值 ~5GB 显存，128GB 绰绰有余。

用法：通过 run_convert_fp8.sh 启动双机。
"""

import gc
import json
import os
import shutil

import safetensors
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F

FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_MAX, FP8_MIN = FP8_INFO.max, FP8_INFO.min

INPUT_DIR = "/work/models/Llama-3.3-70B-NLA-L53-av"
OUTPUT_DIR = "/work/models/Llama-3.3-70B-NLA-L53-av-fp8"


def ceildiv(a, b):
    return -(-a // b)


def block_fp8(weight, block_size=(128, 128)):
    block_n, block_k = block_size
    shape_0, shape_1 = weight.shape
    n_tiles = ceildiv(shape_0, block_n)
    k_tiles = ceildiv(shape_1, block_k)

    q_weight = F.pad(
        weight,
        (0, k_tiles * block_k - shape_1, 0, n_tiles * block_n - shape_0),
        mode="constant", value=0.0,
    )
    qweight = q_weight.reshape(n_tiles, block_n, k_tiles, block_k)
    block_max = torch.max(torch.abs(qweight), dim=1, keepdim=True)[0]
    block_max = torch.max(block_max, dim=3, keepdim=True)[0]

    scale = block_max.to(torch.float32) / FP8_MAX
    qweight = (
        (qweight / scale)
        .clamp(min=FP8_MIN, max=FP8_MAX)
        .reshape((n_tiles * block_n, k_tiles * block_k))
        .to(torch.float8_e4m3fn)
    )
    qweight = qweight[:shape_0, :shape_1].clone().detach()
    scale = scale.reshape(n_tiles, k_tiles)
    return qweight, scale


def should_quantize(key):
    skip = ("layernorm", "embed", "router", "mlp.gate.", "norm", "lm_head", "eh_proj", "weights_proj")
    return "weight" in key and not any(s in key for s in skip)


def process_shard(filename):
    print(f"[rank {dist.get_rank()}] Processing {filename}")
    q_weights = {}
    modules_to_not_convert = []

    with safetensors.safe_open(os.path.join(INPUT_DIR, filename), framework="pt", device="cuda") as f:
        for k in f.keys():
            w = f.get_tensor(k)
            if should_quantize(k):
                qw, s = block_fp8(w)
                q_weights[k] = qw.cpu()
                q_weights[k.replace(".weight", ".weight_scale_inv")] = s.cpu()
            else:
                modules_to_not_convert.append(k.replace(".weight", ""))
                q_weights[k] = w.cpu()

    safetensors.torch.save_file(q_weights, os.path.join(OUTPUT_DIR, filename), metadata={"format": "pt"})
    torch.cuda.empty_cache()

    weight_map = {k: filename for k in q_weights}
    param_count = sum(v.numel() for v in q_weights.values())
    print(f"[rank {dist.get_rank()}] Done {filename}: {len(q_weights)} tensors, {param_count} params")
    return weight_map, modules_to_not_convert, param_count


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(0)

    all_shards = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".safetensors"))
    print(f"[rank {rank}] Total shards: {len(all_shards)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # rank 0 复制非 safetensors 文件
    if rank == 0:
        for f in os.listdir(INPUT_DIR):
            if not f.endswith(".safetensors") and not os.path.isdir(os.path.join(INPUT_DIR, f)):
                shutil.copyfile(os.path.join(INPUT_DIR, f), os.path.join(OUTPUT_DIR, f))

    dist.barrier()

    # 按 shard 编号分配：rank 0 拿偶数，rank 1 拿奇数
    my_shards = [s for i, s in enumerate(all_shards) if i % world_size == rank]
    print(f"[rank {rank}] My shards ({len(my_shards)}): {my_shards}")

    all_weight_map = {}
    all_modules_to_not_convert = []
    total_params = 0

    for shard in my_shards:
        wm, mtnc, pc = process_shard(shard)
        all_weight_map.update(wm)
        all_modules_to_not_convert.extend(mtnc)
        total_params += pc
        gc.collect()

    dist.barrier()
    print(f"[rank {rank}] Quantization complete. Processed {len(my_shards)} shards.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
