import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "meta-llama/Llama-3.2-3B"
device = "cuda"

bits = 3
group_size = 128
sym = False
amp = False
nsamples = 128
iters = 200
seqlen = 512
batch_size = 4

eval_max_len = 2048
max_eval_tokens = 120000


def main():
    print(">>> Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(">>> Setting up AutoRound...")
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

    q_model = getattr(autoround, "model", model)
    q_model.to(device)
    q_model.eval()

    print(">>> Loading WikiText-2 validation split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])

    print(">>> Tokenizing text...")
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    max_pos = q_model.config.max_position_embeddings
    if input_ids.shape[1] > max_pos:
        print(f"Truncating input from {input_ids.shape[1]} to max_position_embeddings={max_pos}")
        input_ids = input_ids[:, :max_pos]

    if input_ids.shape[1] > max_eval_tokens:
        print(f"Further truncating input from {input_ids.shape[1]} to max_eval_tokens={max_eval_tokens}")
        input_ids = input_ids[:, :max_eval_tokens]

    print(f"Final evaluation tokens: {input_ids.shape[1]}")

    print(">>> Computing perplexity...")
    nlls = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[1] - 1, eval_max_len):
            end = start + eval_max_len
            inp = input_ids[:, start:end]
            outputs = q_model(inp, labels=inp)
            loss = outputs.loss
            nlls.append(loss)

    mean_nll = torch.stack(nlls).mean()
    ppl = torch.exp(mean_nll)
    print(f"WikiText-2 PPL (quantized 4-bit): {ppl.item():.4f}")


if __name__ == "__main__":
    main()
