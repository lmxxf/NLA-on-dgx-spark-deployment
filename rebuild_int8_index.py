"""Rebuild model.safetensors.index.json from actual shard files + write config."""

import json
import shutil
from pathlib import Path
from safetensors import safe_open

SRC_DIR = Path("/work/models/Llama-3.3-70B-NLA-L53-av")
DST_DIR = Path("/work/models/Llama-3.3-70B-NLA-L53-av-int8")

QUANTIZATION_CONFIG = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "actorder": None, "block_structure": None, "dynamic": True,
                "group_size": None, "num_bits": 8, "observer": None,
                "observer_kwargs": {}, "strategy": "token", "symmetric": True, "type": "int"
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None, "block_structure": None, "dynamic": False,
                "group_size": None, "num_bits": 8, "observer": "mse",
                "observer_kwargs": {}, "strategy": "channel", "symmetric": True, "type": "int"
            }
        }
    },
    "format": "int-quantized",
    "ignore": ["lm_head"],
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed"
}


def main():
    weight_map = {}
    shards = sorted(DST_DIR.glob("model-*.safetensors"))
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard.name
        print(f"{shard.name}: {len([k for k,v in weight_map.items() if v == shard.name])} keys", flush=True)

    index = {"metadata": {"total_size": 0}, "weight_map": dict(sorted(weight_map.items()))}
    with open(DST_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index: {len(weight_map)} keys total", flush=True)

    with open(SRC_DIR / "config.json") as f:
        config = json.load(f)
    config["quantization_config"] = QUANTIZATION_CONFIG
    with open(DST_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "nla_meta.yaml"]:
        src = SRC_DIR / name
        if src.exists():
            shutil.copy2(str(src), str(DST_DIR / name))

    print("Done ✅", flush=True)


if __name__ == "__main__":
    main()
