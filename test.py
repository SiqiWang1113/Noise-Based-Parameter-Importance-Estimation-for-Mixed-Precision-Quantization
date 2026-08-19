import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA NOT AVAILABLE!")
    exit()

print(">>> Loading tiny GPT2 for test…")
model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

print(">>> Model loaded OK!")

print(">>> Testing AutoRound GPU kernel…")
try:
    from auto_round_extension.cuda import qlinear_exllamav2
    print("AutoRound GPU kernel OK!")
except Exception as e:
    print("AutoRound GPU kernel ERROR:", e)

print("\nAll tests finished.")
