import time
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


MODEL_ID = "Qwen/Qwen3-4B"
DEVICE = "cuda"
DTYPE = torch.float16

SEQ_LEN = 512
NUM_SAMPLES = 128
MAX_BATCHES = 64
BATCH_SIZE = 1

TARGET_AVG_BIT = 3.4

EXCLUDE_PREFIX = (
    "lm_head",
    "model.embed_tokens",
    "model.norm",
)


def build_calib_loader(tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    texts = []
    for x in ds:
        t = x["text"].strip()
        if len(t) > 0:
            texts.append(t)
        if len(texts) >= NUM_SAMPLES:
            break

    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=SEQ_LEN,
        return_tensors="pt",
    )

    labels = enc["input_ids"].clone()
    data = list(zip(enc["input_ids"], enc["attention_mask"], labels))

    def collate(batch):
        return {
            "input_ids": torch.stack([b[0] for b in batch]),
            "attention_mask": torch.stack([b[1] for b in batch]),
            "labels": torch.stack([b[2] for b in batch]),
        }

    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)


def is_excluded(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in EXCLUDE_PREFIX)


def compute_hessian_trace_proxy(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, float], float]:
    model.train()

    sens = {}
    linear_weights = {
        n: m.weight
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not is_excluded(n)
    }

    for n in linear_weights:
        sens[n] = 0.0

    t0 = time.time()

    for i, batch in enumerate(loader):
        if i >= MAX_BATCHES:
            break

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()

        for n, w in linear_weights.items():
            if w.grad is not None:
                g = w.grad.detach()
                sens[n] += float(g.float().pow(2).sum().item())

    steps = max(1, min(MAX_BATCHES, i + 1))
    for k in sens:
        sens[k] /= steps

    t1 = time.time()
    model.eval()
    return sens, (t1 - t0)


def assign_bits_hawq(
    sens: Dict[str, float],
    target_avg_bit: float,
) -> Dict[str, int]:
    items = sorted(sens.items(), key=lambda x: x[1], reverse=True)
    n = len(items)

    num_4bit = int(round((target_avg_bit - 3.0) * n))
    num_4bit = max(0, min(n, num_4bit))

    bits = {}
    for i, (name, _) in enumerate(items):
        bits[name] = 4 if i < num_4bit else 3

    return bits


def main():
    assert torch.cuda.is_available(), "需要 CUDA"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
    )

    loader = build_calib_loader(tokenizer)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("==> Running HAWQ (grad^2 Hessian proxy)...")
    sens, hawq_time = compute_hessian_trace_proxy(model, loader)

    torch.cuda.synchronize()
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / 1024**3

    bits = assign_bits_hawq(sens, TARGET_AVG_BIT)

    bit3 = [n for n, b in bits.items() if b == 3]
    bit4 = [n for n, b in bits.items() if b == 4]

    print("\n================ HAWQ BIT ASSIGNMENT ================\n")
    print(f"Target average bit: {TARGET_AVG_BIT:.2f}")
    print(f"4-bit layers: {len(bit4)}")
    print(f"3-bit layers: {len(bit3)}\n")

    print("---- 4-bit layers (most sensitive) ----")
    for n in bit4:
        print(n)

    print("\n---- 3-bit layers ----")
    for n in bit3:
        print(n)

    print("\n================ RUNTIME PROFILE ================\n")
    print(f"Peak GPU memory during HAWQ: {peak_mem_gb:.2f} GB")
    print(f"HAWQ total time: {hawq_time:.1f} seconds")


if __name__ == "__main__":
    main()
