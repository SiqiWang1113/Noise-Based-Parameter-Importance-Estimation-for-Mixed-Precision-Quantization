import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-3B"

print(">>> Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print(">>> Inspecting quantizer structure...")

for name, module in model.named_modules():
    if "model.layers.0" in name and isinstance(module, torch.nn.Linear):
        print("\n=== Found Linear Layer:", name, "===")
        print("Attributes:", dir(module))
        for attr in dir(module):
            val = getattr(module, attr)
            if "quant" in attr.lower() or "bit" in attr.lower():
                print(f"  -> {attr}: {val}")
        break

print("Done.")
