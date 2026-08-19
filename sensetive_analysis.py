import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl_gpu(model, tokenizer, text, max_seq_len=4096):
    model.eval()

    device = next(model.parameters()).device

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    total_loss = 0.0
    count = 0
    n_tokens = input_ids.shape[1]
    print(f">>> Input tokens = {n_tokens}")

    for i in range(0, n_tokens, max_seq_len):
        ids = input_ids[:, i:i+max_seq_len]
        labels = ids.clone()

        with torch.no_grad():
            out = model(ids, labels=labels)
            loss = out.loss

        total_loss += loss.item()
        count += 1

    return math.exp(total_loss / count)


def get_llama32_linear_modules(model):
    linear_modules = []

    n_layers = model.config.num_hidden_layers
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]

        sub_layers = [
            ("self_attn.q_proj",  layer.self_attn.q_proj),
            ("self_attn.k_proj",  layer.self_attn.k_proj),
            ("self_attn.v_proj",  layer.self_attn.v_proj),
            ("self_attn.o_proj",  layer.self_attn.o_proj),
            ("mlp.gate_proj",     layer.mlp.gate_proj),
            ("mlp.up_proj",       layer.mlp.up_proj),
            ("mlp.down_proj",     layer.mlp.down_proj),
        ]

        for sub_name, mod in sub_layers:
            name = f"layer{layer_idx:02d}.{sub_name}"
            linear_modules.append((name, mod))

    print(f">>> Found {len(linear_modules)} linear modules (expected {n_layers * 7})")
    return linear_modules


def evaluate_layer_sensitivity(
    model,
    tokenizer,
    text,
    noise_std=0.5,
    max_seq_len=4096,
):
    print(">>> Computing baseline PPL (no noise)...")
    base_ppl = compute_ppl_gpu(model, tokenizer, text, max_seq_len=max_seq_len)
    print(f">>> Baseline PPL = {base_ppl:.6f}")

    linear_modules = get_llama32_linear_modules(model)

    results = []

    for idx, (name, module) in enumerate(linear_modules):
        print(f"\n[{idx+1}/{len(linear_modules)}] Testing noise on {name} ...")

        with torch.no_grad():
            orig_weight = module.weight.data.clone()

            noise = torch.randn_like(module.weight) * noise_std
            module.weight.mul_(1.0 + noise)

        ppl = compute_ppl_gpu(model, tokenizer, text, max_seq_len=max_seq_len)
        delta = ppl - base_ppl
        print(f"    PPL = {ppl:.6f}, ΔPPL = {delta:.6f}")

        results.append({
            "name": name,
            "ppl": ppl,
            "delta": delta,
        })

        with torch.no_grad():
            module.weight.data.copy_(orig_weight)

    return base_ppl, results


def main():
    model_id = "meta-llama/Llama-3.2-3B"
    text_file = "ppl_corpus.txt"
    device = "cuda"

    torch.manual_seed(0)

    print(">>> Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> Loading FP16 model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    model.to(device)

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    NOISE_STD = 0.5

    base_ppl, results = evaluate_layer_sensitivity(
        model=model,
        tokenizer=tokenizer,
        text=text,
        noise_std=NOISE_STD,
        max_seq_len=4096,
    )

    print("\n====================")
    print(f"Baseline PPL (no noise): {base_ppl:.6f}")
    print(f"Noise std: {NOISE_STD}")
    print("Per-layer sensitivity (sorted by ΔPPL desc):")

    results_sorted = sorted(results, key=lambda x: x["delta"], reverse=True)
    for r in results_sorted:
        print(f"{r['name']:30s}  PPL={r['ppl']:.6f}  Δ={r['delta']:.6f}")
    print("====================")


if __name__ == "__main__":
    main()
