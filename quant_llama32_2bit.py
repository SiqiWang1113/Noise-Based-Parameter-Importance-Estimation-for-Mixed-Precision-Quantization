import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "meta-llama/Llama-3.2-3B"

bits = 2
group_size = 128
sym = False
device = "cuda"
amp = False
nsamples = 128
iters = 200
seqlen = 512
batch_size = 4

print(">>> Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(">>> 配置 AutoRound...")
autoround = AutoRound(
    model,
    tokenizer,
    nsamples=nsamples,
    iters=iters,
    seqlen=seqlen,
    batch_size=batch_size,
    bits=bits,
    group_size=group_size,
    sym=sym,
    device=device,
    amp=amp,
)

print(">>> Start quantization (2-bit)...")
autoround.quantize()

output_dir = "./AutoRound/Llama-3.2-3B-autoround-int2-gs128-asym"
autoround.save_quantized(output_dir, format="auto_round", inplace=True)

print(">>> DONE! Saved at:", output_dir)
