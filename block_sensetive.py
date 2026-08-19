import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl_gpu(model, tokenizer, text, max_seq_len=4096, trunc_len=None):
    model.eval()
    device = next(model.parameters()).device

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]

    max_model_len = getattr(model.config, "max_position_embeddings", 131072)

    if trunc_len is not None:
        effective_len = min(trunc_len, max_model_len, input_ids.shape[1])
    else:
        effective_len = min(max_model_len, input_ids.shape[1])

    if input_ids.shape[1] > effective_len:
        print(f">>> Truncating tokens from {input_ids.shape[1]} to {effective_len}")
        input_ids = input_ids[:, :effective_len]

    input_ids = input_ids.to(device)

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


def get_llama32_block_linear_modules(model):
    blocks = []

    n_layers = model.config.num_hidden_layers
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]

        modules = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ]

        block_name = f"layer{layer_idx:02d}"
        blocks.append((block_name, modules))

    print(f">>> Found {len(blocks)} blocks (each with 7 linear layers)")
    return blocks


def evaluate_block_sensitivity(
    model,
    tokenizer,
    text,
    noise_std=0.01,
    max_seq_len=4096,
    trunc_len=None,
):
    print(">>> Computing baseline PPL (no noise)...")
    base_ppl = compute_ppl_gpu(
        model, tokenizer, text,
        max_seq_len=max_seq_len,
        trunc_len=trunc_len,
    )
    print(f">>> Baseline PPL = {base_ppl:.6f}")

    blocks = get_llama32_block_linear_modules(model)

    results = []

    for idx, (block_name, modules) in enumerate(blocks):
        print(f"\n[{idx+1}/{len(blocks)}] Testing noise on {block_name} (7 linear layers)...")

        backups = []
        with torch.no_grad():
            for m in modules:
                w = m.weight
                backups.append(w.data.clone())

                noise = torch.randn_like(w) * noise_std
                w.mul_(1.0 + noise)

        ppl = compute_ppl_gpu(
            model, tokenizer, text,
            max_seq_len=max_seq_len,
            trunc_len=trunc_len,
        )
        delta = ppl - base_ppl
        print(f"    PPL = {ppl:.6f}, ΔPPL = {delta:.6f}")

        results.append({
            "block": block_name,
            "ppl": ppl,
            "delta": delta,
        })

        with torch.no_grad():
            for m, backup in zip(modules, backups):
                m.weight.data.copy_(backup)

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
        dtype=torch.float16,
        device_map="cpu"
    )

    model.to(device)

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    NOISE_STD = 0.5
    MAX_SEQ_LEN = 4096
    TRUNC_LEN = None

    base_ppl, results = evaluate_block_sensitivity(
        model=model,
        tokenizer=tokenizer,
        text=text,
        noise_std=NOISE_STD,
        max_seq_len=MAX_SEQ_LEN,
        trunc_len=TRUNC_LEN,
    )

    print("\n====================")
    print(f"Baseline PPL (no noise): {base_ppl:.6f}")
    print(f"Noise std: {NOISE_STD}")
    print(f"Trunc len: {TRUNC_LEN}")
    print("Per-block sensitivity (sorted by ΔPPL desc):")

    results_sorted = sorted(results, key=lambda x: x["delta"], reverse=True)
    for r in results_sorted:
        print(f"{r['block']:10s}  PPL={r['ppl']:.6f}  Δ={r['delta']:.6f}")
    print("====================")


if __name__ == "__main__":
    main()
