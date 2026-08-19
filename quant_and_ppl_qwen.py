import torch
import os
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_round import AutoRound


def inspect_model_structure(model):
    print("\n" + "="*60)
    print(">>> 模型结构检查:")
    print("="*60)
    print(f"模型类型: {model.config.model_type}")
    print(f"总层数: {model.config.num_hidden_layers}")
    print(f"隐藏层大小: {model.config.hidden_size}")
    
    print("\n>>> 第0层的 Linear 模块:")
    for name, module in model.named_modules():
        if "model.layers.0." in name and isinstance(module, torch.nn.Linear):
            print(f"  - {name}: {module.in_features} -> {module.out_features}")
    print("="*60 + "\n")


def get_layer_config(model, bits_low=3, bits_high=4):
    n_layers = model.config.num_hidden_layers
    layer_config = {}
    
    print(f">>> 生成混合精度配置 (低精度={bits_low}bit, 高精度={bits_high}bit)")
    print(f">>> 策略: 前3层和后3层用 {bits_high}bit, 中间层用 {bits_low}bit\n")
    
    high_precision_count = 0
    low_precision_count = 0
    
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        
        if "model.layers." not in name:
            continue
        
        try:
            layer_idx = int(name.split("model.layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        
        if layer_idx < 3 or layer_idx >= n_layers - 3:
            bits = bits_high
            high_precision_count += 1
        else:
            bits = bits_low
            low_precision_count += 1
        
        layer_config[name] = {
            "bits": bits,
            "group_size": 128,
            "sym": False
        }
        
        if ".0." in name or "q_proj" in name or "gate_proj" in name:
            print(f"[Layer {layer_idx:2d}] {bits}bit - {name}")
    
    print(f"\n>>> 配置统计:")
    print(f"  - 高精度 ({bits_high}bit) 模块数: {high_precision_count}")
    print(f"  - 低精度 ({bits_low}bit) 模块数: {low_precision_count}")
    print(f"  - 总计: {len(layer_config)} 个 Linear 层\n")
    
    return layer_config


def compute_ppl_gpu(model, tokenizer, text, max_seq_len=2048, device="cuda"):
    model.to(device)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]

    MAX_TOKENS = 200_000
    input_ids = enc["input_ids"][:, :MAX_TOKENS]

    total_loss = 0
    count = 0
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
                    
                total_loss += loss.item()
                count += 1
            except Exception as e:
                print(f"错误: 第 {i} 块计算失败: {e}")
                continue

    if count == 0:
        return float('inf')
    
    return math.exp(total_loss / count)


def main():
    model_id = "Qwen/Qwen3-1.7B"
    text_file = "ppl_corpus.txt"
    out_dir = "quantized_model_mixed/"
    
    bits_low = 3
    bits_high = 4
    
    print("="*60)
    print("AutoRound 混合精度量化工具")
    print("="*60)
    print(f"模型: {model_id}")
    print(f"输出目录: {out_dir}")
    print(f"混合精度配置: {bits_low}bit / {bits_high}bit")
    print("="*60 + "\n")

    print(">>> 步骤 1/6: 加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer 加载完成\n")

    print(">>> 步骤 2/6: 加载 FP16 原始模型")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    print("✓ 模型加载完成\n")

    inspect_model_structure(model)

    print(">>> 步骤 3/6: 生成混合精度配置")
    layer_config = get_layer_config(model, bits_low=bits_low, bits_high=bits_high)
    
    if len(layer_config) == 0:
        print("错误: 未找到任何可量化的层！")
        return
    
    print("✓ 配置生成完成\n")

    print(">>> 步骤 4/6: 初始化 AutoRound")
    ar = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=bits_low,
        group_size=128,
        sym=False,
        layer_config=layer_config,
        n_samples=512,
        seqlen=512,
        batch_size=1,
        enable_torch_compile=False,
        dtype=torch.float16,
    )
    print("✓ AutoRound 初始化完成\n")

    print(">>> 步骤 5/6: 运行量化 (这可能需要几分钟到几十分钟)...")
    print("提示: 可以通过 Ctrl+C 中断\n")
    
    try:
        ar.quantize()
        print("\n✓ 量化完成\n")
    except KeyboardInterrupt:
        print("\n用户中断量化过程")
        return
    except Exception as e:
        print(f"\n错误: 量化过程失败: {e}")
        return

    print(f">>> 步骤 6/6: 保存量化模型到 {out_dir}")
    
    if os.path.exists(out_dir):
        import shutil
        print(f"提示: 删除旧的输出目录 {out_dir}")
        shutil.rmtree(out_dir)
    
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        ar.save_quantized(
            output_dir=out_dir,
            format="auto_round",
            inplace=True
        )
        tokenizer.save_pretrained(out_dir)
        print("✓ 模型保存完成\n")
    except Exception as e:
        print(f"错误: 保存模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(">>> 重新加载量化模型进行验证...")
    try:
        q_model = AutoModelForCausalLM.from_pretrained(
            out_dir,
            device_map="cpu",
            trust_remote_code=True
        )
        print("✓ 量化模型加载成功\n")
    except Exception as e:
        print(f"警告: 重新加载失败: {e}")
        print("\n提示: 模型已保存到 {out_dir}")
        print("你可以稍后单独测试加载:")
        print(f"  from transformers import AutoModelForCausalLM")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{out_dir}')")
        return

    print(">>> 计算困惑度 (PPL)...")
    
    if not os.path.exists(text_file):
        print(f"警告: {text_file} 不存在,使用示例文本")
        text = "The quick brown fox jumps over the lazy dog. " * 100
    else:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if len(text) < 100:
            print("警告: 文本太短，添加重复内容")
            text = text * 10

    try:
        ppl = compute_ppl_gpu(q_model, tokenizer, text, max_seq_len=2048, device="cuda")
        
        print("\n" + "="*60)
        print("量化结果")
        print("="*60)
        print(f"量化模型 PPL: {ppl:.2f}")
        print(f"输出目录: {out_dir}")
        print("="*60)
    except Exception as e:
        print(f"错误: PPL 计算失败: {e}")
        print("提示: 如果是 CUDA OOM 错误，可以尝试:")
        print("  1. 减小 max_seq_len")
        print("  2. 使用更短的测试文本")


if __name__ == "__main__":
    main()