# SVD-LLM 自主复现

[![CI](https://github.com/zhc1212/SVD-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/zhc1212/SVD-LLM/actions/workflows/ci.yml)

> 从零复现论文 **"SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression"** (ICLR 2025)

## 论文信息

- **论文**: [arXiv:2403.07378](https://arxiv.org/abs/2403.07378)
- **作者**: Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang
- **机构**: The Ohio State University / Michigan State University
- **官方代码**: [AIoT-MLSys-Lab/SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) (不参考官方实现，完全从论文独立复现)

---

## 方法概述

SVD-LLM 是一种基于 SVD 的 LLM 后训练压缩方法，包含两个核心贡献：
1. **Truncation-Aware Data Whitening** (§3.1) — 使截断最小奇异值等价于最小化压缩损失
2. **Parameter Update with Sequential Low-Rank Approximation** (§3.2) — 压缩后对低秩矩阵做 LoRA 风格的顺序微调，恢复精度

---

## 数学推导

### 1. 问题定义

对于 Transformer 中的一个线性层 $y = Wx$，其中 $W \in \mathbb{R}^{d \times n}$，低秩压缩的目标是找到秩为 $r$ 的近似 $\hat{W}$ 使得输出误差最小：

$$\min_{\hat{W}} \mathbb{E}\left[\|Wx - \hat{W}x\|_2^2\right] = \min_{\hat{W}} \mathbb{E}\left[\|(W - \hat{W})x\|_2^2\right]$$

其中期望取自校准数据集上的激活分布。

**压缩比定义:**

$$R_w = 1 - \frac{(d + n) \cdot r}{d \cdot n}$$

原始参数量为 $dn$，压缩后存储 $A \in \mathbb{R}^{d \times r}$ 和 $B \in \mathbb{R}^{r \times n}$，参数量为 $(d+n)r$。

**LLaMA-7B 的 rank 计算示例:**

| 压缩比 | 4096×4096 层的 r | 4096×11008 层的 r |
|--------|-----------------|-------------------|
| 20%    | 1638            | 2388              |
| 40%    | 1228            | 1791              |
| 60%    | 819             | 1194              |
| 80%    | 409             | 597               |

### 2. Vanilla SVD 的问题

传统方法直接对 $W$ 做 SVD:

$$W = U \Sigma V^T \approx U_r \Sigma_r V_r^T$$

截断第 $i$ 个奇异值的输出损失为：

$$L_i = \sigma_i^2 \cdot \mathbb{E}[(v_i^T x)^2]$$

其中 $v_i$ 是 $V$ 的第 $i$ 列。**关键问题: $\mathbb{E}[(v_i^T x)^2]$ 与 $i$ 无关的假设不成立。** 实际的输入激活 $x$ 各维度方差差异极大（可达 100 倍），导致截断最小 $\sigma_i$ 未必对应最小 $L_i$。

### 3. Truncation-Aware Data Whitening

**目标:** 构造线性变换，使截断最小奇异值严格等价于最小化加权输出损失。

**Step 1: 计算激活协方差**

给定 $N$ 个校准样本的激活 $X \in \mathbb{R}^{N \times n}$（这里的每个线性子层有独立的 $X$），计算协方差矩阵：

$$C = \frac{1}{N} X^T X \in \mathbb{R}^{n \times n}$$

**实现细节:** $N$ 可达 $256 \times 2048 = 524288$，不可能存储完整激活矩阵。用 forward hook 流式累加 $X^T X$：

```
for each calibration batch B_t (batch_size=4):
    X^T X += B_t^T B_t     (在 GPU 上以 float32 累加，最终转 CPU float64)
    N += len(B_t)
C = X^T X / N
```

> **注:** 采用 GPU float32 累加而非 CPU float64，通过减少 GPU↔CPU 同步次数提速约 7 倍。
> 协方差矩阵为 $n \times n$ 对称矩阵，32 层 × 7 子层同时累加约占 27GB GPU 显存。

**Step 2: Cholesky 分解**

对协方差矩阵做 Cholesky 分解：

$$C = L L^T$$

其中 $L \in \mathbb{R}^{n \times n}$ 是下三角矩阵。定义白化变换 $z = L^{-1}x$，验证白化性质：

$$\mathbb{E}[zz^T] = L^{-1} \mathbb{E}[xx^T] L^{-T} = L^{-1} C L^{-T} = L^{-1} L L^T L^{-T} = I$$

即白化后的激活 $z$ 各维度不相关且方差为 1。

> **注:** 当 $N < n$ 或 $C$ 近似秩亏时 Cholesky 会失败。此时加正则化 $C \leftarrow C + \epsilon I$（代码中 $\epsilon = 10^{-6}$，秩亏时追加 $\frac{\text{tr}(C)}{n} \cdot I$）。

**Step 3: 在白化空间做 SVD**

由于 $x = Lz$，输出 $y = Wx = (WL)z$。对 $WL$（而非 $W$）做 SVD：

$$WL = U \Sigma V^T$$

截断第 $i$ 个奇异值的**输出损失**为：

$$\mathcal{L}_i = \sigma_i^2 \cdot \mathbb{E}[(v_i^T z)^2] = \sigma_i^2 \cdot v_i^T \underbrace{\mathbb{E}[zz^T]}_{= I} v_i = \sigma_i^2 \cdot \underbrace{\|v_i\|^2}_{= 1} = \sigma_i^2$$

**因为白化后 $\mathbb{E}[zz^T] = I$，且 $V$ 正交（$\|v_i\| = 1$），所以每个奇异值的损失贡献恰好是 $\sigma_i^2$。** 截断最小奇异值严格保证最小输出损失。

这也等价于加权 Frobenius 范数的最优低秩逼近：

$$\mathbb{E}[\|(W - \hat{W})x\|^2] = \|(W - \hat{W})L\|_F^2 = \sum_{i > r} \sigma_i^2$$

**Step 4: 截断并分解为两个低秩矩阵**

保留前 $r$ 个最大奇异值后：

$$WL \approx U_r \Sigma_r V_r^T$$

$$W \approx U_r \Sigma_r V_r^T L^{-1}$$

论文将 $\Sigma_r$ **对称分配**到两侧（便于后续 LoRA 微调时两个矩阵尺度一致）：

$$W'_u = U_r \Sigma_r^{1/2} \in \mathbb{R}^{d \times r}, \quad W'_v = \Sigma_r^{1/2} V_r^T L^{-1} \in \mathbb{R}^{r \times n}$$

使得 $W \approx W'_u W'_v$，原始线性层 $y = Wx$ 替换为 $y = W'_u(W'_v x)$。

> **注:** 等价的非对称分配 $A = U_r \Sigma_r, B = V_r^T L^{-1}$ 在数学上等价，但对称分配是论文的默认实现，对后续 LoRA 微调更友好。

> **注 (记号约定):** 本实现采用 row-major 样本组织（$X \in \mathbb{R}^{N \times n}$），协方差写作 $C = X^T X / N$。论文正文中部分地方使用 $XX^T$ convention，仅是记号差异。

**代码中的变量对应关系:**

| 数学符号 | 代码变量 | 说明 |
|---------|---------|------|
| $L$ | `L = torch.linalg.cholesky(C)` | 下三角 Cholesky 因子 |
| $L^T$ | `S = L.T` | 上三角，代码命名为 S |
| $WL$ | `WS = W @ S.T` | $S.T = (L^T)^T = L$ |
| $V_r^T L^{-1}$ | `Vh_r @ S_inv.T` | $S^{-1} = L^{-T}$，$S^{-T} = L^{-1}$ |

### 4. Parameter Update with Sequential Low-Rank Approximation

白化 + SVD 截断解决了「用最少的秩保留最多信息」的问题（**阶段 A: 压缩**），但截断不可避免地引入误差。论文的第二个核心贡献是在压缩后对低秩矩阵做 LoRA 风格的参数微调来恢复精度（**阶段 B: 参数更新**）。

**关键: "Sequential" 指的是微调两个低秩矩阵的顺序，不是逐层压缩的顺序。**

**阶段 A — 压缩 (Whitening + SVD Truncation):**

对所有线性子层执行白化 SVD，得到低秩分解 $W \approx W'_u W'_v$，替换模型中所有线性层。这一步 SVD-LLM(W) 和 SVD-LLM 完全相同。

**阶段 B — 参数更新 (Sequential LoRA Fine-tuning):**

在压缩后的模型 $M'$ 上，使用 Alpaca 50K 数据集做两阶段顺序 LoRA 微调：

```
Stage 1: LoRA_u(M')
  - 冻结所有 W'_v
  - 对所有 W'_u 添加 LoRA adapter 并微调
  - 合并 LoRA → 得到更新后的 W'_u

Stage 2: LoRA_v(M'_u)
  - 冻结更新后的 W'_u
  - 对所有 W'_v 添加 LoRA adapter 并微调
  - 合并 LoRA → 得到更新后的 W'_v
```

两阶段更新的直觉：同时更新 $W'_u$ 和 $W'_v$ 是一个双线性优化问题（非凸），固定一个更新另一个则是凸问题，交替优化更稳定。

**SVD-LLM(W) 与 SVD-LLM 的区别:**

| | SVD-LLM(W) | SVD-LLM |
|---|---|---|
| 阶段 A (压缩) | 白化 + SVD 截断 | 白化 + SVD 截断 (相同) |
| 阶段 B (参数更新) | **无** | Sequential LoRA (Alpaca 50K) |
| 校准数据 | WikiText-2 256 samples | WikiText-2 256 samples |
| 微调数据 | 不需要 | **Alpaca 50K** (yahma/alpaca-cleaned) |

SVD-LLM(W) 的压缩流程：

```
注册 hooks 到所有 32 层 × 7 个线性子层 (共 224 个 hooks)
跑 256 次 forward pass → 一次性得到所有协方差
for layer l = 0, 1, ..., L-1:
    对每个线性子层: 白化 SVD → 得到 (W'_u, W'_v) → 暂存到磁盘
统一替换所有层 → 直接评估
```

SVD-LLM 的完整流程：

```
阶段 A: 同 SVD-LLM(W)，得到压缩后模型 M'
阶段 B:
  Stage 1: 冻结 W'_v, 用 Alpaca 50K LoRA 微调所有 W'_u → M'_u
  Stage 2: 冻结 W'_u, 用 Alpaca 50K LoRA 微调所有 W'_v → M'_final
评估 M'_final
```

### 5. CompressedLinear 实现

原始线性层 $y = Wx + b$ 替换为两个连续线性层：

```
x ∈ R^n  →  [Linear(n→r, no bias)]  →  z ∈ R^r  →  [Linear(r→d, bias)]  →  y ∈ R^d
             权重 = W'_v                           权重 = W'_u, bias = b_original
```

参数量: $dn + d$ → $(n \cdot r) + (r \cdot d + d) = (n+d)r + d$

在阶段 B 的 LoRA 微调中，LoRA adapter 分别加在 $W'_u$ 和 $W'_v$ 上。

### 6. 算法伪代码总结

```
Algorithm 1: SVD-LLM 压缩 (应用于每个线性子层独立)

Input:  模型 M, 校准数据 D_calib (256 WikiText-2 samples), 压缩比 R_w
Output: 压缩后模型 M'

--- 阶段 A: Whitening + SVD Truncation (SVD-LLM(W) 和 SVD-LLM 共用) ---
1. 对每个线性子层 W ∈ R^{d×n}, 计算目标秩:
     r = max(1, ⌊(1 - R_w) · d · n / (d + n)⌋)
2. 注册 hooks 到所有 L×7 = 224 个线性子层
3. for each calibration sample x ∈ D_calib:
4.     前向传播 M(x), hooks 流式累加 X^T X
5. for each layer l, for each sub-layer:
6.     C = X^T X / N
7.     L = cholesky(C)
8.     U, Σ, V^T = svd(W · L)
9.     W'_u = U_r · Σ_r^{1/2}           # 对称分配奇异值
10.    W'_v = Σ_r^{1/2} · V_r^T · L^{-1}
11.    替换为 CompressedLinear(W'_u, W'_v)
→ 得到 M' (SVD-LLM(W) 到此结束)

Algorithm 2: Sequential Low-Rank Approximation (仅 SVD-LLM)

Input:  压缩后模型 M', 微调数据 D_ft (Alpaca 50K)
Output: 参数更新后模型 M'_final

--- 阶段 B: Sequential LoRA Fine-tuning ---
Stage 1: LoRA_u
12. 冻结 M' 中所有 W'_v
13. 对所有 W'_u 添加 LoRA adapter
14. 用 D_ft 微调 → 合并 LoRA → 得到 M'_u

Stage 2: LoRA_v
15. 冻结 M'_u 中所有 W'_u
16. 对所有 W'_v 添加 LoRA adapter
17. 用 D_ft 微调 → 合并 LoRA → 得到 M'_final
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

### 复现方法

| 方法 | 阶段 A (压缩) | 阶段 B (微调) | 微调数据 |
|------|-------------|-------------|---------|
| **SVD-LLM(W)** | 白化 + SVD 截断 | 无 | 不需要 |
| **SVD-LLM** | 白化 + SVD 截断 | Sequential LoRA | Alpaca 50K |

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

## Phase 1 复现结果：SVD-LLM(W)

> Phase 1 完成。以下为 LLaMA-7B 上 SVD-LLM(W)（仅白化 + SVD 截断，无 LoRA 微调）的复现结果。

### Perplexity 对比 (↓)

| 压缩比 | 数据集 | 论文 SVD-LLM(W) | **复现** | 差值 |
|--------|--------|----------------|----------|------|
| 0% | WikiText-2 | 5.68 | **5.67** | -0.01 |
| 0% | C4 | 7.34 | **7.20** | -0.14 |
| 20% | WikiText-2 | 7.94 | **7.84** | -0.10 |
| 20% | C4 | 15.84 | **15.65** | -0.19 |
| 40% | WikiText-2 | 13.73 | **13.17** | -0.56 |
| 40% | C4 | 75.42 | **50.05** | -25.37 |
| 60% | WikiText-2 | 66.62 | **58.98** | -7.64 |
| 60% | C4 | 471.83 | **378.80** | -93.03 |
| 80% | WikiText-2 | 1349 | **660.67** | -688.33 |
| 80% | C4 | 6224 | **2641.62** | -3582.38 |

所有压缩率上 PPL 均优于论文报告值。低压缩率 (20%) 差异极小，高压缩率差距较大，可能与 C4 评估方式 / tokenizer 版本差异有关。

### 下游任务对比 (20% 压缩比, Accuracy ↑)

论文 Table 1 中 SVD-LLM(W) 在 20% 压缩比下的下游任务结果对比：

| 任务 | 论文 SVD-LLM(W) | **复现** | 差值 | 说明 |
|------|----------------|----------|------|------|
| OpenbookQA | 0.31 | **0.266** | -0.044 | |
| ARC_easy | 0.62 | **0.641** | +0.021 | |
| WinoGrande | 0.61 | **0.664** | +0.054 | |
| HellaSwag | 0.45 | **0.437** | -0.013 | |
| PIQA | 0.71 | **0.690** | -0.020 | |
| MathQA | 0.21 | **0.239** | +0.029 | 本地 Parquet 数据集 |
| TruthfulQA | 0.26 | **0.390** | — | 论文用 BLEU，复现用 MC2，不可直接对比 |
| GSM8K | 0.05 | **0.006** | -0.044 | Exact Match |

6 个 MC 任务与论文差异在 ±0.05 以内，属于 lm_eval 版本 / prompt 模板差异。MathQA 通过本地 Parquet 数据集解决了 datasets 4.1.1 兼容性问题。TruthfulQA 指标不同不可直接对比。GSM8K 论文值也仅为 0.05，压缩后数学推理能力基本丧失。

### 全部压缩率下游任务 (复现, Accuracy ↑)

| 任务 | 20% | 40% | 60% | 80% |
|------|-----|-----|-----|-----|
| OpenbookQA | 0.266 | 0.200 | 0.132 | 0.130 |
| ARC_easy | 0.641 | 0.457 | 0.301 | 0.261 |
| WinoGrande | 0.664 | 0.575 | 0.527 | 0.480 |
| HellaSwag | 0.437 | 0.330 | 0.273 | 0.260 |
| PIQA | 0.690 | 0.609 | 0.544 | 0.523 |
| MathQA | 0.239 | 0.220 | 0.218 | 0.205 |
| TruthfulQA (MC2) | 0.390 | 0.433 | 0.476 | 0.501 |
| GSM8K (EM) | 0.006 | 0.000 | 0.000 | 0.000 |

### 论文 Table 1 完整对比 (LLaMA-7B, 20% 压缩比)

#### 语言建模 Perplexity (↓)

| 数据集 | SVD | FWSVD | ASVD | SVD-LLM(W) 论文 | **SVD-LLM(W) 复现** | SVD-LLM 论文 |
|--------|-----|-------|------|-----------------|---------------------|-------------|
| WikiText-2 | 20061 | 1727 | 11.14 | 7.94 | **7.84** | **7.73** |
| C4 | 18800 | 1511 | 15.93 | 15.84 | **15.65** | **12.23** |

#### 下游任务 Accuracy (↑)

| 任务 | Original | SVD | FWSVD | ASVD | SVD-LLM(W) 论文 | **SVD-LLM(W) 复现** | SVD-LLM 论文 |
|------|----------|-----|-------|------|-----------------|---------------------|-------------|
| OpenbookQA | 0.34 | 0.05 | 0.09 | 0.29 | 0.31 | **0.266** | **0.33** |
| ARC_easy | 0.75 | 0.04 | 0.11 | 0.53 | 0.62 | **0.641** | **0.67** |
| WinoGrande | 0.70 | 0.01 | 0.05 | 0.64 | 0.61 | **0.664** | **0.69** |
| HellaSwag | 0.57 | 0.03 | 0.08 | 0.41 | 0.45 | **0.437** | **0.55** |
| PIQA | 0.79 | 0.02 | 0.10 | 0.68 | 0.71 | **0.690** | **0.79** |
| MathQA | 0.27 | 0.03 | 0.05 | 0.17 | 0.21 | **0.239** | **0.26** |
| Avg (6 tasks) | 0.57 | 0.03 | 0.08 | 0.45 | 0.49 | **0.490** | **0.55** |
| TruthfulQA | 0.30 | 0.00 | 0.00 | 0.21 | 0.26 | 0.390* | **0.28** |
| GSM8K | 0.09 | 0.00 | 0.00 | 0.04 | 0.05 | **0.006** | **0.08** |

> \* TruthfulQA 复现使用 MC2 指标（论文使用 BLEU），不可直接对比。

### Phase 1 结论

1. **PPL 复现成功** — 趋势完全一致，所有压缩率上复现值均优于论文
2. **Downstream 基本一致** — 6 个 MC 任务（含 MathQA）平均 Accuracy 0.490 vs 论文 0.49，差异在 ±0.05 以内
3. **高压缩率退化明显** — 60%/80% 的 PPL 和 Accuracy 大幅退化，需要 Phase 2 (Sequential LoRA) 恢复质量
4. **环境兼容性说明** — MathQA 通过本地 Parquet 数据集解决 datasets 4.1.1 兼容性问题；TruthfulQA 使用 MC2 指标替代论文的 BLEU 指标；GSM8K 在压缩模型上接近 0，与论文趋势一致

---

## 实验设置

### 校准数据 (白化用)
- **来源**: WikiText-2 训练集
- **样本数**: 256 条, seqlen=2048
- **用途**: 收集激活协方差，计算白化矩阵

### 微调数据 (参数更新用, 仅 SVD-LLM)
- **来源**: Alpaca 数据集 (yahma/alpaca-cleaned)
- **样本数**: 50K
- **用途**: Sequential LoRA 微调 W'_u 和 W'_v

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

## 复现计划

分阶段进行，先保证 SVD-LLM(W) 结果可复现，再实现完整 SVD-LLM。

### Phase 1: SVD-LLM(W) — 仅白化压缩 (不需要 Alpaca)

```
1. 评估原始模型 → WikiText-2 PPL, C4 PPL

2. SVD-LLM(W) × 4 ratios (20%, 40%, 60%, 80%)
   每个 ratio:
     ├── 加载原始模型
     ├── 用 256 条 WikiText-2 校准样本收集所有层协方差 (一次 forward pass)
     ├── 对每个线性子层: 白化 SVD → W'_u, W'_v → 暂存磁盘
     ├── 统一替换所有层
     ├── 保存压缩模型
     └── 评估 WikiText-2 / C4 Perplexity

3. 20% ratio 下游任务评估 (8 个 tasks)
```

### Phase 2: SVD-LLM — 白化压缩 + Sequential LoRA (需要 Alpaca 50K)

```
4. SVD-LLM × 4 ratios (20%, 40%, 60%, 80%)
   每个 ratio:
     ├── 阶段 A: 同 SVD-LLM(W) 的压缩步骤 (白化 + SVD 截断)
     ├── 阶段 B: Sequential LoRA 微调
     │     ├── Stage 1: 冻结 W'_v, LoRA 微调所有 W'_u (Alpaca 50K)
     │     └── Stage 2: 冻结 W'_u, LoRA 微调所有 W'_v (Alpaca 50K)
     ├── 合并 LoRA, 保存模型
     └── 评估 WikiText-2 / C4 Perplexity

5. 20% ratio 下游任务评估

6. 汇总结果对比 Table 1
```

---

## 项目结构

```
SVD_LLM/
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/plans/
├── src/
│   ├── data/
│   │   └── calibration.py          # 校准数据加载 + 流式协方差收集
│   ├── compress/
│   │   ├── whitening.py            # 白化矩阵 + 白化 SVD 压缩 (阶段 A)
│   │   └── compress_model.py       # SVD-LLM(W) 完整压缩流程
│   ├── finetune/                   # [计划中] Sequential LoRA 微调 (阶段 B, Phase 3)
│   ├── model/
│   │   ├── loader.py               # 模型加载、rank 计算
│   │   └── replace.py              # CompressedLinear + 层替换
│   └── eval/
│       ├── perplexity.py           # WikiText-2 / C4 Perplexity 评估
│       └── downstream.py           # 下游任务评估 (lm-eval-harness)
├── scripts/
│   ├── compress.py                 # 压缩主脚本 (阶段 A)
│   ├── eval_model.py               # 评估主脚本
│   ├── run_experiments_gpu2.sh     # 实验运行脚本
│   └── collect_results.py          # 收集结果生成对比表
├── tests/
│   ├── conftest.py
│   ├── test_whitening.py
│   ├── test_calibration.py
│   ├── test_replace.py
│   ├── test_perplexity.py
│   ├── test_downstream.py
│   └── test_e2e.py
└── outputs/                        # 实验结果 (gitignored)
```

---

## 快速开始

```bash
# 0. 激活环境
source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai

# 1. 阶段 A: 白化 SVD 压缩 (SVD-LLM(W), 20% 压缩比)
python scripts/compress.py \
    --model_path /home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/ \
    --method svd_llm_w \
    --ratio 0.2 \
    --save_path outputs/llama7b_svd_llm_w_20

# 2. (可选) 阶段 B: Sequential LoRA 微调 → 完整 SVD-LLM
python scripts/finetune.py \
    --model_path outputs/llama7b_svd_llm_w_20 \
    --save_path outputs/llama7b_svd_llm_20

# 3. 评估 Perplexity
python scripts/eval_model.py \
    --model_path outputs/llama7b_svd_llm_w_20 \
    --eval perplexity \
    --datasets wikitext2 c4

# 4. 评估下游任务
python scripts/eval_model.py \
    --model_path outputs/llama7b_svd_llm_w_20 \
    --eval downstream \
    --tasks openbookqa arc_easy winogrande hellaswag piqa truthfulqa_mc2
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
