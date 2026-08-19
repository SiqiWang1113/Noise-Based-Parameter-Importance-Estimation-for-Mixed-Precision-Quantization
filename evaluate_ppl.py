import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_round import load_quantized_model
from datasets import load_dataset
import math

model_path = "./AutoRound/Llama-3.2-3B-autoround-int2-gs128-asym"

print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

print("Loading AutoRound quantized model…")
model = load_quantized_model(model_path)
model.eval()

print("Loading WikiText-2…")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def compute_ppl(model, tokenizer, dataset, max_length=2048):
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    input_ids = encodings["input_ids"]

    nlls = []
    for i in range(0, input_ids.size(1), max_length):
        inp = input_ids[:, i:i+max_length]
        target_ids = inp.clone()

        with torch.no_grad():
            outputs = model(inp, labels=target_ids)
            nlls.append(outputs.loss)

    return math.exp(torch.stack(nlls).mean().item())

print("Computing perplexity…")
ppl = compute_ppl(model, tokenizer, dataset)
print("WikiText-2 PPL:", ppl)
