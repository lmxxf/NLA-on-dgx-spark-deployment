"""AV inference: activation vector → natural language explanation.

Stream-loads INT8 AV model directly to GPU (no from_pretrained, no OOM).
Uses Triton INT8 tensor core matmul — no dequantization needed.

Run on host inside Docker container:
    python3 /work/NLA-on-dgx-spark-deployment/run_av_inference.py \
        --av-model /work/models/Llama-3.3-70B-NLA-L53-av-int8
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Add project dir to path for int8_linear import
sys.path.insert(0, str(Path(__file__).parent))
from int8_linear import patch_int8_linear

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
MIN_POSITION = 10


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm / target_scale).to(v.dtype)


def load_nla_meta(av_model_path: str) -> dict:
    with open(f"{av_model_path}/nla_meta.yaml") as f:
        return yaml.safe_load(f)


def stream_load_model(model_path: str):
    """Stream-load INT8 model: meta skeleton + safe_open directly to GPU."""
    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

    # Remove quantization_config so from_config creates plain Linear
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    print(f"[load] Creating model skeleton on meta device...", flush=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.eval()
    print(f"[load] Skeleton created ✅", flush=True)

    # Load weight index
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_to_keys = defaultdict(list)
    for key, shard in weight_map.items():
        shard_to_keys[shard].append(key)

    # Stream load
    loaded = 0
    for shard_name in sorted(shard_to_keys):
        shard_path = str(model_path / shard_name)
        keys = shard_to_keys[shard_name]
        with safe_open(shard_path, framework="pt", device="cuda") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                parts = key.split(".")
                module = model
                for part in parts[:-1]:
                    module = getattr(module, part)
                param_name = parts[-1]

                old = getattr(module, param_name, None)
                if old is not None and isinstance(old, torch.nn.Parameter):
                    setattr(module, param_name, torch.nn.Parameter(tensor, requires_grad=False))
                else:
                    setattr(module, param_name, tensor)
                loaded += 1
        print(f"[load] {shard_name}: {len(keys)} tensors ({loaded}/{len(weight_map)})", flush=True)

    # Fix rotary embedding inv_freq (buffer, not in safetensors)
    rope_theta = getattr(config, "rope_theta", 500000.0)
    head_dim = config.hidden_size // config.num_attention_heads
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and module.inv_freq is not None and module.inv_freq.device.type == "meta":
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device="cuda") / head_dim))
            module.inv_freq = inv_freq
        for buf_name, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type == "meta":
                module._buffers[buf_name] = torch.zeros(buf.shape, dtype=buf.dtype, device="cuda")

    # Fix any remaining meta parameters
    for name, param in list(model.named_parameters()):
        if param.device.type == "meta":
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, parts[-1], torch.nn.Parameter(
                torch.zeros(param.shape, dtype=param.dtype, device="cuda"), requires_grad=False))
            print(f"[load] Fixed meta param: {name}", flush=True)

    print(f"[load] Model loaded ✅", flush=True)
    return model


def run_av(av_model_path: str, activations_path: str, positions: list[int] | None):
    meta = load_nla_meta(av_model_path)
    injection_scale = meta["extraction"]["injection_scale"]
    injection_char = meta["tokens"]["injection_char"]
    injection_token_id = meta["tokens"]["injection_token_id"]
    left_neighbor_id = meta["tokens"]["injection_left_neighbor_id"]
    right_neighbor_id = meta["tokens"]["injection_right_neighbor_id"]
    prompt_template = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
    d_model = meta["d_model"]

    print(f"[av] injection_scale={injection_scale}, d_model={d_model}", flush=True)

    print(f"[av] Loading activations from {activations_path}", flush=True)
    data = torch.load(activations_path, weights_only=False)
    hidden_states = data["hidden_states"]
    if hidden_states.ndim == 2:
        hidden_states = hidden_states.unsqueeze(0)
    tokens = data["tokens"]
    text = data["text"]
    layer = data["layer"]
    print(f"[av] Layer: {layer}, seq_len: {hidden_states.shape[1]}", flush=True)
    print(f"[av] Tokens: {tokens}", flush=True)

    print(f"[av] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(av_model_path, trust_remote_code=True)

    content = prompt_template.format(injection_char=injection_char)
    chat_out = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(chat_out, "input_ids"):
        input_ids = chat_out.input_ids[0].tolist()
    elif isinstance(chat_out, torch.Tensor):
        input_ids = chat_out[0].tolist()
    else:
        input_ids = list(chat_out)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)

    inj_positions = [
        i for i, tid in enumerate(input_ids)
        if tid == injection_token_id
        and 0 < i < len(input_ids) - 1
        and input_ids[i - 1] == left_neighbor_id
        and input_ids[i + 1] == right_neighbor_id
    ]
    assert len(inj_positions) == 1, f"Expected 1 injection site, found {len(inj_positions)}"
    inj_pos = inj_positions[0]
    print(f"[av] Injection position: {inj_pos}", flush=True)

    # Patch nn.Linear for INT8 Triton matmul BEFORE loading model
    patch_int8_linear()

    # Stream load model
    model = stream_load_model(av_model_path)
    embed_layer = model.model.embed_tokens

    seq_len = hidden_states.shape[1]
    if positions is None:
        positions = list(range(MIN_POSITION, seq_len))
    positions = [p for p in positions if MIN_POSITION <= p < seq_len]
    print(f"[av] Decoding {len(positions)} positions: {positions}", flush=True)

    for pos in positions:
        token_str = tokens[pos] if pos < len(tokens) else "?"
        activation = hidden_states[0, pos]
        act_norm = activation.norm().item()

        v_scaled = normalize_activation(activation.unsqueeze(0), injection_scale)

        ids_t = input_ids_t.unsqueeze(0).to("cuda")
        with torch.no_grad():
            embeds = embed_layer(ids_t).float()
        embeds[0, inj_pos] = v_scaled[0].to(embeds.device)

        with torch.no_grad():
            gen_ids = model.generate(
                inputs_embeds=embeds.to(torch.bfloat16),
                max_new_tokens=200,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else tokenizer.eos_token_id[0],
            )

        new_tokens = gen_ids[0][len(input_ids):]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=False)

        m = EXPLANATION_RE.search(raw_output)
        explanation = m.group(1).strip() if m else raw_output.strip()

        print(f"\n{'='*60}", flush=True)
        print(f"Position {pos}: token={token_str!r}  ||activation||={act_norm:.1f}", flush=True)
        print(f"{'='*60}", flush=True)
        print(explanation, flush=True)
        print(flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-model", default="/work/models/Llama-3.3-70B-NLA-L53-av-int8")
    parser.add_argument("--activations", default="/work/NLA-on-dgx-spark-deployment/activations.pt")
    parser.add_argument("--positions", type=str, default=None,
                        help="Comma-separated positions, e.g. '10,20'. Default: all from 10+")
    args = parser.parse_args()

    positions = None
    if args.positions:
        positions = [int(x.strip()) for x in args.positions.split(",")]

    run_av(args.av_model, args.activations, positions)


if __name__ == "__main__":
    main()
