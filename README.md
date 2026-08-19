# LLM Sensitivity Analysis and Mixed-Precision Quantization

This repository provides an end-to-end framework for **Sensitivity Analysis**, **Hessian-based Bit-width Allocation (HAWQ)**, and **Adaptive Mixed-Precision Quantization** for Large Language Models (e.g., LLaMA-3.2, Qwen).

It leverages [AutoRound](https://github.com/intel/auto-round) for low-bit weight quantization (2-bit, 3-bit, 4-bit) and provides analytical tools to measure layer-wise and operator-wise sensitivity via multiplicative perturbation and Hessian gradient proxies.

---

## 📋 Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Quick Sanity Check](#2-quick-sanity-check)
- [3. Workflow & Usage](#3-workflow--usage)
  - [Step 1: FP16 Baseline Evaluation](#step-1-fp16-baseline-evaluation)
  - [Step 2: Sensitivity Analysis](#step-2-sensitivity-analysis)
  - [Step 3: HAWQ Hessian Profiling & Bit-width Allocation](#step-3-hawq-hessian-profiling--bit-width-allocation)
  - [Step 4: Mixed-Precision & Uniform Quantization](#step-4-mixed-precision--uniform-quantization)
  - [Step 5: Model Evaluation](#step-5-model-evaluation)
- [4. Project Structure](#4-project-structure)
- [5. Methodology Overview](#5-methodology-overview)

---

## 1. Environment Setup

Create a Conda environment and install the required dependencies:

```bash
# 1. Create and activate conda environment
conda create -n llm-quant python=3.10 -y
conda activate llm-quant

# 2. Install PyTorch with CUDA support (adjust CUDA version if necessary)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install core dependencies
pip install transformers datasets accelerate auto-round tqdm numpy
```

---

## 2. Quick Sanity Check

Verify CUDA availability and the AutoRound GPU kernel extension:

```bash
python test.py
```

---

## 3. Workflow & Usage

### Step 1: FP16 Baseline Evaluation
Compute the baseline WikiText-2 perplexity (PPL) on the unquantized FP16 model:

```bash
python fp16_ppl.py
```

---

### Step 2: Sensitivity Analysis
Identify which layers or modules are most critical to model perplexity under perturbations ($W \leftarrow W \cdot (1 + \mathcal{N}(0, \sigma^2))$):

- **Operator-Level Sensitivity** (evaluates each `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`):
  ```bash
  python sensetive_analysis.py
  ```

- **Block/Layer-Level Sensitivity** (evaluates each Transformer layer block):
  ```bash
  python block_sensetive.py
  ```

- **Group-Level Sensitivity** (evaluates `Attention` modules vs. `FFN` modules separately per layer):
  ```bash
  python group_sensetive.py
  ```

---

### Step 3: HAWQ Hessian Profiling & Bit-width Allocation
Use Hessian trace proxy estimation ($\text{Tr}(H) \approx \mathbb{E}[\|\nabla_W L\|^2]$) to profile layer importance and automatically assign 3-bit / 4-bit configurations under a target average bit-width constraint:

```bash
python memory.py
```

---

### Step 4: Mixed-Precision & Uniform Quantization

- **Uniform 2-bit Quantization (AutoRound)**:
  ```bash
  python quant_llama32_2bit.py
  # or evaluate directly during quantization
  python quant_and_ppl.py
  ```

- **Adaptive Mixed-Precision Quantization (LLaMA / Qwen)**:
  Applies higher precision (4-bit) to the shallow and deep layers while compressing middle layers to 3-bit:
  ```bash
  # For LLaMA models
  python quant_and_ppl_v2.py

  # For Qwen models
  python quant_and_ppl_qwen.py
  ```

- **Module-Specific Mixed-Precision (FFN 4-bit + Attention 3-bit)**:
  ```bash
  python quant_and_ppl_v3.py
  ```

- **Mixed-Precision with PPL Statistics (Mean & Std)**:
  ```bash
  python quant_and_ppl_mix.py
  ```

---

### Step 5: Model Evaluation
Evaluate a saved quantized model on the WikiText-2 test set:

```bash
python evaluate_ppl.py
```

*(Optional) Prepare offline evaluation corpus from C4:*
```bash
python save_wiki2.py
```

---

## 4. Project Structure

| File | Description |
|---|---|
| `test.py` | Environment and AutoRound CUDA kernel verification script. |
| `fp16_ppl.py` | Computes baseline FP16 model perplexity on WikiText-2. |
| `sensetive_analysis.py` | Layer-by-layer and operator-level multiplicative noise sensitivity analysis. |
| `block_sensetive.py` | Block-level (Transformer layer) sensitivity evaluation. |
| `group_sensetive.py` | Attention vs. FFN group-level sensitivity evaluation across layers. |
| `memory.py` | HAWQ gradient-based Hessian proxy computation, bit-width allocation, and peak memory profiling. |
| `quant_llama32_2bit.py` | Uniform 2-bit AutoRound quantization for LLaMA-3.2-3B. |
| `quant_and_ppl.py` | Quantization pipeline with WikiText-2 PPL evaluation. |
| `quant_and_ppl_mix.py` | Mixed-precision quantization with PPL standard deviation metrics. |
| `quant_and_ppl_v2.py` | Adaptive layer-wise mixed-precision quantization (LLaMA architecture). |
| `quant_and_ppl_qwen.py` | Adaptive layer-wise mixed-precision quantization (Qwen architecture). |
| `quant_and_ppl_v3.py` | Targeted mixed-precision quantization (e.g., higher bit-width on boundary FFN layers). |
| `inspect_quantizer.py` | Inspects linear layer parameters and quantizer configurations. |
| `save_wiki2.py` | Prepares and caches the `ppl_corpus.txt` evaluation dataset. |
| `evaluate_ppl.py` | Evaluates perplexity of saved quantized checkpoints. |

---

## 5. Methodology Overview

1. **Multiplicative Perturbation**: Inject zero-mean Gaussian noise $w_{ij} \leftarrow w_{ij} \times (1 + \epsilon)$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$, to measure output perplexity degradation per operator/layer.
2. **Hessian Trace Proxy (HAWQ)**: Approximates the diagonal of the Hessian via the squared expectation of backpropagated gradients:
   $$\text{Sens}(W) \approx \frac{1}{N} \sum_{i=1}^N \|\nabla_W \mathcal{L}_i\|_F^2$$
3. **AutoRound Optimization**: Optimizes rounding values and scale parameters through gradient-based sign-rounding, minimizing block-wise reconstruction errors.
