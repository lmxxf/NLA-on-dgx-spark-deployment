"""rsync 合并完所有 shard 后，在 host 上生成 config.json 和 index.json。"""

import json
import os

import safetensors

INPUT_DIR = "/work/models/Llama-3.3-70B-NLA-L53-av"
OUTPUT_DIR = "/work/models/Llama-3.3-70B-NLA-L53-av-fp8"

weight_map = {}
modules_to_not_convert = []
total_params = 0

skip_keywords = ("layernorm", "embed", "router", "mlp.gate.", "norm", "lm_head", "eh_proj", "weights_proj")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    if not filename.endswith(".safetensors"):
        continue
    path = os.path.join(OUTPUT_DIR, filename)
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            weight_map[k] = filename
            total_params += f.get_tensor(k).numel()
            if any(s in k for s in skip_keywords):
                modules_to_not_convert.append(k.replace(".weight", ""))

# config.json
cfg = json.load(open(os.path.join(INPUT_DIR, "config.json")))
cfg["quantization_config"] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": list(set(modules_to_not_convert)),
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

# index.json
index_dict = {
    "weight_map": weight_map,
    "metadata": {"total_size": total_params},
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index_dict, f, indent=2)

print(f"Done! {len(weight_map)} keys, {total_params} params total")
