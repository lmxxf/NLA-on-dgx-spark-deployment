"""Dual-node INT8 quantization for NLA AV model (compressed-tensors format).

Each node processes half the shards (even/odd split).
Quantizes Linear weights to per-channel symmetric INT8, matching the format
of neuralmagic/Llama-3.3-70B-Instruct-INT8:
  - weight: int8 [out, in]
  - weight_scale: bfloat16 [out, 1]
  - LayerNorm, embed_tokens, lm_head: kept as BF16

Environment variables (set by run_convert_int8.sh):
  RANK, WORLD_SIZE (0/1 of 2)
"""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_DIR = Path("/work/models/Llama-3.3-70B-NLA-L53-av")
DST_DIR = Path("/work/models/Llama-3.3-70B-NLA-L53-av-int8")

SKIP_QUANTIZE_KEYS = {"model.embed_tokens.weight", "lm_head.weight"}


def should_quantize(key: str) -> bool:
    if key in SKIP_QUANTIZE_KEYS:
        return False
    if "layernorm" in key.lower() or "norm" in key.lower():
        return False
    if key.endswith(".weight") and not key.endswith("_scale"):
        return True
    return False


def quantize_per_channel(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT8: scale = absmax / 127, weight_int8 = round(weight / scale)."""
    weight_f32 = weight_bf16.float()
    absmax = weight_f32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = absmax / 127.0
    weight_int8 = (weight_f32 / scale).round().clamp(-128, 127).to(torch.int8)
    scale_bf16 = scale.to(torch.bfloat16)
    return weight_int8, scale_bf16


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    index_path = SRC_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Collect unique shards in order
    shards = sorted(set(weight_map.values()))
    my_shards = [s for i, s in enumerate(shards) if i % world_size == rank]

    print(f"[rank {rank}] Total shards: {len(shards)}, mine: {len(my_shards)}", flush=True)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    new_weight_map = {}

    for shard_name in my_shards:
        src_path = str(SRC_DIR / shard_name)
        shard_keys = [k for k, v in weight_map.items() if v == shard_name]

        out_tensors = {}
        with safe_open(src_path, framework="pt", device="cuda") as f:
            for key in shard_keys:
                tensor = f.get_tensor(key)
                if should_quantize(key):
                    w_int8, w_scale = quantize_per_channel(tensor)
                    out_tensors[key] = w_int8.cpu()
                    scale_key = key.replace(".weight", ".weight_scale")
                    out_tensors[scale_key] = w_scale.cpu()
                    new_weight_map[key] = shard_name
                    new_weight_map[scale_key] = shard_name
                else:
                    out_tensors[key] = tensor.cpu()
                    new_weight_map[key] = shard_name

        dst_path = str(DST_DIR / shard_name)
        save_file(out_tensors, dst_path)
        print(f"[rank {rank}] {shard_name}: {len(shard_keys)} keys -> {len(out_tensors)} keys", flush=True)

    # Save weight map fragment for this rank
    frag_path = DST_DIR / f"weight_map_rank{rank}.json"
    with open(frag_path, "w") as f:
        json.dump(new_weight_map, f)
    print(f"[rank {rank}] Done ✅", flush=True)


if __name__ == "__main__":
    main()
