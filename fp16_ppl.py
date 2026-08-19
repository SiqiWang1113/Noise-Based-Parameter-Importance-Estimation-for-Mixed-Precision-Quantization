import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

device = "cuda"
model_name = "meta-llama/Llama-3.2-3B"

print(">>> Loading FP16 model & tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

print(">>> Loading WikiText-2 validation split...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
text = "\n\n".join(ds["text"])

print(">>> Tokenizing text...")
enc = tokenizer(text, return_tensors="pt")
input_ids = enc.input_ids.to(device)

max_pos = model.config.max_position_embeddings
max_eval_tokens = 120000
eval_max_len = 2048

if input_ids.shape[1] > max_pos:
    print(f"Truncating input from {input_ids.shape[1]} to max_position_embeddings={max_pos}")
    input_ids = input_ids[:, :max_pos]

if input_ids.shape[1] > max_eval_tokens:
    print(f"Further truncating input from {input_ids.shape[1]} to max_eval_tokens={max_eval_tokens}")
    input_ids = input_ids[:, :max_eval_tokens]

print(f"Final evaluation tokens: {input_ids.shape[1]}")

print(">>> Computing FP16 perplexity...")
nlls = []
for i in tqdm(range(0, input_ids.shape[1] - 1, eval_max_len)):
    inp = input_ids[:, i:i+eval_max_len]
    with torch.no_grad():
        out = model(inp, labels=inp)
        nlls.append(out.loss.detach())

ppl = torch.exp(torch.stack(nlls).mean())
print("WikiText-2 PPL (FP16):", ppl.item())
