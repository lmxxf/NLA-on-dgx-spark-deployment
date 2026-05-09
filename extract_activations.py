"""Extract layer-53 residual stream activations from Llama-3.3-70B-Instruct.

Uses the same loading method as previous experiments (paper15, wechat121):
device_map={"": 0} + low_cpu_mem_usage=True. Truncated to 54 layers via
symlinked checkpoint to reduce memory.

Run on slave inside Docker container:
    python3 /work/NLA-on-dgx-spark-deployment/extract_activations.py
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

EXTRACTION_LAYER = 53
KEEP_LAYERS = 54


def make_truncated_checkpoint(model_path: str, keep_layers: int) -> str:
    """Symlink model dir with truncated config + filtered weight index."""
    model_path = Path(model_path)
    tmp_dir = tempfile.mkdtemp(prefix="nla_trunc_")

    for f in model_path.iterdir():
        if f.name == "config.json" or f.name == "model.safetensors.index.json":
            continue
        os.symlink(str(f), str(Path(tmp_dir) / f.name))

    # Truncated config
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    original_layers = config.num_hidden_layers
    config.num_hidden_layers = keep_layers
    config.save_pretrained(tmp_dir)
    print(f"[load] Truncated config: {original_layers} -> {keep_layers} layers", flush=True)

    # Filtered weight index
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    skip_prefixes = tuple(f"model.layers.{i}." for i in range(keep_layers, original_layers))
    original_count = len(index["weight_map"])
    index["weight_map"] = {
        k: v for k, v in index["weight_map"].items()
        if not k.startswith(skip_prefixes) and k != "lm_head.weight"
    }
    with open(Path(tmp_dir) / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"[load] Weight index: {original_count} -> {len(index['weight_map'])} keys", flush=True)

    return tmp_dir


def extract(model_path: str, text: str, output_path: str):
    print(f"[extract] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    print(f"[extract] Text: {text!r}", flush=True)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    print(f"[extract] {input_ids.shape[1]} tokens: {tokens}", flush=True)

    tmp_dir = make_truncated_checkpoint(model_path, KEEP_LAYERS)
    try:
        print(f"[extract] Loading model...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            tmp_dir,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        print(f"[extract] Model loaded ✅", flush=True)
    finally:
        shutil.rmtree(tmp_dir)

    activations = {}

    def hook_fn(module, input, output):
        activations["hidden_states"] = output[0].detach().cpu().float()

    handle = model.model.layers[EXTRACTION_LAYER].register_forward_hook(hook_fn)

    print(f"[extract] Forward pass...", flush=True)
    with torch.no_grad():
        model(input_ids.to(model.device))
    handle.remove()

    h = activations["hidden_states"]
    print(f"[extract] Activations shape: {h.shape}", flush=True)

    torch.save({
        "hidden_states": h,
        "input_ids": input_ids.cpu(),
        "tokens": tokens,
        "text": text,
        "layer": EXTRACTION_LAYER,
    }, output_path)
    print(f"[extract] Saved to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/work/models/Llama-3.3-70B-Instruct-INT8")
    parser.add_argument("--output", default="/work/NLA-on-dgx-spark-deployment/activations.pt")
    parser.add_argument("--text", default="A rhyming couplet about a carrot:\nHe saw a carrot and had to grab it,\nSo he dressed up like a rabbit.")
    args = parser.parse_args()
    extract(args.model, args.text, args.output)


if __name__ == "__main__":
    main()
