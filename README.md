# SVD-LLM 自主复现

> 从零复现论文 **"SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression"** (ICLR 2025)

## 论文信息

- **论文**: [arXiv:2403.07378](https://arxiv.org/abs/2403.07378)
- **作者**: Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang
- **机构**: The Ohio State University / Michigan State University
- **官方代码**: [AIoT-MLSys-Lab/SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) (不参考官方实现，完全从论文独立复现)

---

## 方法概述

SVD-LLM 是一种基于 SVD 的 LLM 后训练压缩方法，解决了现有方法的两个关键问题：

### 问题 1: 截断最小奇异值 ≠ 最小压缩损失

传统 SVD 直接对权重矩阵 W 做分解，但截断最小奇异值并不一定对应最小的输出损失，因为输入激活的分布不均匀。

### 解决: Truncation-Aware Data Whitening

1. 收集校准数据的激活矩阵 X
2. 通过 Cholesky 分解 XX^T = SS^T，得到白化矩阵 S
3. 对 WS 做 SVD（而非对 W），使得：
   - 白化后激活 S⁻¹X 各通道正交独立
   - 截断第 i 个奇异值的损失 L_i = σ_i（直接等于奇异值本身）
   - 因此截断最小奇异值保证最低压缩损失

### 问题 2: 截断后无权重更新 → 高压缩率下精度严重下降

### 解决: Sequential Low-Rank Approximation (层级参数更新)

- 截断 SVD 后，逐层用新激活更新压缩后的权重
- 补偿截断造成的误差累积
- 高压缩率 (≥40%) 时尤为关键

### 数学公式

```
压缩比定义: R_w = 1 - (d + n) * r / (d * n)
  其中 W ∈ R^{d×n}, r 为保留的秩

白化: SS^T = XX^T  (Cholesky 分解)
SVD:  WS = UΣV^T
截断: 保留前 r 个最大奇异值
恢复: W_compressed = U_r Σ_r V_r^T S^{-1}
```

---

## 复现目标：主实验 (Table 1)

### 目标模型

| 模型 | 原始大小 | 本地路径 |
|------|---------|---------|
| LLaMA-7B | 13.5GB | `/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/` |

### 压缩比

| 压缩比 | 压缩后大小 | 含义 |
|--------|-----------|------|
| 20% | 10.2GB | 移除 20% 参数 |
| 40% | 7.76GB | 移除 40% 参数 |
| 60% | 5.35GB | 移除 60% 参数 |
| 80% | 2.58GB | 移除 80% 参数 |

### 对比方法 (Baselines)

| 方法 | 说明 |
|------|------|
| **SVD** (Vanilla) | 直接对 W 做 SVD 截断，无任何处理 |
| **FWSVD** | Fisher-Weighted SVD，用 Fisher 信息加权 |
| **ASVD** | Activation-aware SVD，用激活缩放 |
| **SVD-LLM (W)** | 仅用 truncation-aware data whitening |
| **SVD-LLM** | whitening + sequential low-rank update |

### 目标结果 (Table 1: LLaMA-7B)

#### 语言建模 (Perplexity ↓)

| 压缩比 | 数据集 | SVD | FWSVD | ASVD | SVD-LLM(W) | SVD-LLM |
|--------|--------|-----|-------|------|-----------|---------|
| 0% | WikiText-2 | 5.68 | 5.68 | 5.68 | 5.68 | 5.68 |
| 0% | C4 | 7.34 | 7.34 | 7.34 | 7.34 | 7.34 |
| 20% | WikiText-2 | 20061 | 1727 | 11.14 | 7.94 | **7.73** |
| 20% | C4 | 18800 | 1511 | 15.93 | 15.84 | **12.23** |
| 40% | WikiText-2 | 52489 | 18156 | 1407 | 13.73 | **9.27** |
| 40% | C4 | 47774 | 12847 | 1109 | 75.42 | **15.63** |
| 60% | WikiText-2 | 105474 | 32194 | 57057 | 66.62 | **15.00** |
| 60% | C4 | 106976 | 29292 | 43036 | 471.83 | **26.26** |
| 80% | WikiText-2 | 687291 | 96872 | 80425 | 1349 | **31.79** |
| 80% | C4 | 708243 | 89243 | 67927 | 6224 | **43.71** |

#### 下游任务 (Accuracy ↑, 20% 压缩比)

| 方法 | OpenbookQA | ARC_e | WinoGrande | HellaSwag | PIQA | MathQA | Avg | TruthfulQA | GSM8K |
|------|-----------|-------|-----------|----------|------|--------|-----|-----------|-------|
| Original | 0.34 | 0.75 | 0.70 | 0.57 | 0.79 | 0.27 | 0.57 | 0.30 | 0.09 |
| SVD | 0.05 | 0.04 | 0.01 | 0.03 | 0.02 | 0.03 | 0.03 | 0.00 | 0.00 |
| FWSVD | 0.09 | 0.11 | 0.05 | 0.08 | 0.10 | 0.05 | 0.08 | 0.00 | 0.00 |
| ASVD | 0.29 | 0.53 | 0.64 | 0.41 | 0.68 | 0.17 | 0.45 | 0.21 | 0.04 |
| SVD-LLM(W) | 0.31 | 0.62 | 0.61 | 0.45 | 0.71 | 0.21 | 0.49 | 0.26 | 0.05 |
| SVD-LLM | **0.33** | **0.67** | **0.69** | **0.55** | **0.79** | **0.26** | **0.55** | **0.28** | **0.08** |

---

## 实验设置

### 校准数据
- **来源**: WikiText-2 训练集
- **样本数**: 256 条
- **用途**: 生成激活矩阵用于白化和参数更新

### 微调数据 (用于 Sequential Update)
- **来源**: Alpaca 数据集 (yahma/alpaca-cleaned)
- **样本数**: 50K

### 评估数据集 (10 个)

| 类别 | 数据集 | 指标 |
|------|--------|------|
| 语言建模 | WikiText-2 | Perplexity ↓ |
| 语言建模 | C4 | Perplexity ↓ |
| 分类 | OpenbookQA | Accuracy ↑ |
| 分类 | ARC-Easy | Accuracy ↑ |
| 分类 | WinoGrande | Accuracy ↑ |
| 分类 | HellaSwag | Accuracy ↑ |
| 分类 | PIQA | Accuracy ↑ |
| 分类 | MathQA | Accuracy ↑ |
| 生成 | TruthfulQA | BLEU ↑ |
| 生成 | GSM8K | Exact Match ↑ |

### 评估框架
- **lm-evaluation-harness** (EleutherAI)

### 运行环境
- **GPU**: 6x NVIDIA A800 80GB PCIe
- **Conda 环境**: `compactifai`
- **Python**: 3.12.10
- **PyTorch**: 2.6.0+cu124
- **Transformers**: 4.57.1
- **lm-eval**: 0.4.9
- **PEFT**: 0.6.0

### 压缩的权重矩阵
- Transformer 中所有线性层的权重矩阵（Q, K, V, O, Gate, Up, Down projections）

---

## 实验流程

```
┌─────────────────────────────────────────────────────────────┐
│                    实验总流程                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: 环境与数据准备                                      │
│  ├── 安装依赖 (torch, transformers, datasets, lm-eval)       │
│  ├── 验证 LLaMA-7B 模型加载                                   │
│  └── 准备校准数据 (256 samples from WikiText-2)               │
│                                                             │
│  Phase 2: 实现 Baseline 方法                                  │
│  ├── Vanilla SVD 压缩                                        │
│  ├── FWSVD (Fisher-Weighted SVD)                             │
│  └── ASVD (Activation-aware SVD)                             │
│                                                             │
│  Phase 3: 实现 SVD-LLM 核心算法                               │
│  ├── Step 1: Truncation-Aware Data Whitening                 │
│  │   ├── 收集激活 X                                          │
│  │   ├── 计算 XX^T                                           │
│  │   ├── Cholesky 分解得到 S                                  │
│  │   └── 对 WS 做 SVD 并截断                                  │
│  ├── Step 2: Sequential Low-Rank Approximation               │
│  │   ├── 逐层更新压缩后权重                                    │
│  │   └── 用新激活补偿截断误差                                   │
│  └── (可选) Step 3: LoRA 微调                                 │
│                                                             │
│  Phase 4: 评估                                               │
│  ├── WikiText-2 / C4 Perplexity                              │
│  ├── 6 个分类任务 Accuracy                                    │
│  ├── TruthfulQA / GSM8K                                      │
│  └── 推理速度测试                                              │
│                                                             │
│  Phase 5: 结果对比                                            │
│  └── 对比 Table 1 数据，分析复现差距                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
SVD_LLM/
├── README.md                    # 本文件
├── requirements.txt             # 依赖
├── docs/
│   └── plans/                   # 实现计划
├── src/
│   ├── data/
│   │   ├── calibration.py       # 校准数据加载与激活收集
│   │   └── evaluation.py        # 评估数据加载
│   ├── compress/
│   │   ├── svd_vanilla.py       # Vanilla SVD baseline
│   │   ├── fwsvd.py             # FWSVD baseline
│   │   ├── asvd.py              # ASVD baseline
│   │   ├── whitening.py         # Truncation-aware data whitening
│   │   ├── svd_truncate.py      # SVD 截断核心逻辑
│   │   └── sequential_update.py # Sequential low-rank approximation
│   ├── model/
│   │   ├── loader.py            # 模型加载与保存
│   │   └── replace.py           # 替换原始层为压缩层
│   └── eval/
│       ├── perplexity.py        # Perplexity 评估
│       └── downstream.py        # 下游任务评估 (lm-eval-harness)
├── scripts/
│   ├── compress.py              # 主压缩脚本
│   └── evaluate.py              # 主评估脚本
└── tests/
    ├── test_whitening.py        # 白化正确性测试
    ├── test_svd.py              # SVD 截断测试
    └── test_compress.py         # 端到端压缩测试
```

---

## 快速开始

```bash
# 0. 激活环境
source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai

# 1. 压缩模型 (SVD-LLM, 20% 压缩比)
python scripts/compress.py \
    --model_path /home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/ \
    --method svd_llm \
    --ratio 0.2 \
    --calib_dataset wikitext2 \
    --calib_nsamples 256 \
    --save_path outputs/llama7b_svdllm_20

# 2. 评估 Perplexity
python scripts/evaluate.py \
    --model_path outputs/llama7b_svdllm_20 \
    --eval perplexity \
    --datasets wikitext2 c4

# 3. 评估下游任务
python scripts/evaluate.py \
    --model_path outputs/llama7b_svdllm_20 \
    --eval downstream \
    --tasks openbookqa arc_easy winogrande hellaswag piqa mathqa truthfulqa gsm8k
```

---

## 参考文献

```bibtex
@inproceedings{wang2025svdllm,
  title={SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression},
  author={Wang, Xin and Zheng, Yu and Wan, Zhongwei and Zhang, Mi},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
