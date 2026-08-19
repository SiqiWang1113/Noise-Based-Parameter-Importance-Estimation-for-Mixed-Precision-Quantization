import torch
import os
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_round import AutoRound


def get_layer_config(model, bits_low=3, bits_high=4):
    n_layers = model.config.num_hidden_layers
    layer_config = {}
    
    for i in range(n_layers):
        if i < 3 or i >= n_layers - 2:
            bits = bits_high
        else:
            bits = bits_low
        
        layer_prefix = f"model.layers.{i}"
        for module_name in ["self_attn.q_proj", "self_attn.k_proj", 
                           "self_attn.v_proj", "self_attn.o_proj",
                           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            full_name = f"{layer_prefix}.{module_name}"
            layer_config[full_name] = {
                "bits": bits,
                "group_size": 128,
                "sym": False
            }
            print(f"[Config] {full_name} → {bits}bit")
    
    return layer_config


def compute_ppl_gpu(model, tokenizer, text, max_seq_len=2048, device="cuda"):
    model.to(device)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    MAX_TOKENS = 200_000

    input_ids = enc["input_ids"][:, :MAX_TOKENS]
    losses = []
    n_tokens = input_ids.shape[1]
    print(f">>> 输入 tokens 数量 = {n_tokens}")

    for i in range(0, n_tokens, max_seq_len):
        ids = input_ids[:, i:i+max_seq_len].to(device)
        
        if ids.shape[1] < 2:
            continue
            
        labels = ids.clone()

        with torch.no_grad():
            try:
                out = model(ids, labels=labels)
                loss = out.loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 第 {i} 块出现异常 loss")
                    continue
                    
                losses.append(loss.item())
            except Exception as e:
                print(f"错误: 第 {i} 块计算失败: {e}")
                continue

    if len(losses) == 0:
        return float('inf'), 0.0
    
    mean_loss = np.mean(losses)
    std_loss = np.std(losses, ddof=1) if len(losses) > 1 else 0.0
    
    ppl_mean = math.exp(mean_loss)
    ppl_std = ppl_mean * std_loss
    
    print(f"\n>>> Loss统计: mean={mean_loss:.4f}, std={std_loss:.4f} (n={len(losses)} chunks)")
    
    return ppl_mean, ppl_std


def main():
    model_id = "meta-llama/Llama-3.2-3B"
    text_file = "ppl_corpus.txt"
    out_dir = "quantized_model_mixed/"

    print(">>> 加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> 加载 FP16 模型")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )

    print(">>> 生成混合精度配置...")
    layer_config = get_layer_config(model, bits_low=3, bits_high=4)

    print(">>> 初始化 AutoRound (混合精度)...")
    ar = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=3,
        group_size=128,
        sym=False,
        layer_config=layer_config,
        n_samples=512,
        seqlen=512,
        batch_size=1,
        enable_torch_compile=False,
        dtype=torch.float16,
    )

    print(">>> 运行量化 (这可能需要几分钟)...")
    ar.quantize()

    os.makedirs(out_dir, exist_ok=True)
    print(f">>> 保存量化模型到 {out_dir}")
    
    ar.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(">>> 重新加载量化模型")
    q_model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        device_map="cpu",
        trust_remote_code=True
    )

    if not os.path.exists(text_file):
        print(f"警告: {text_file} 不存在,使用示例文本")
        text = "The quick brown fox jumps over the lazy dog. " * 100
    else:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

    print(">>> 计算困惑度...")
    ppl, ppl_std = compute_ppl_gpu(q_model, tokenizer, text, max_seq_len=2048, device="cuda")

    print("\n" + "="*50)
    print(f"量化模型 PPL: {ppl:.2f} ± {ppl_std:.2f}")
    print("="*50)


if __name__ == "__main__":
    main()