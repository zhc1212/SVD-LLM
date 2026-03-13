# SVD-LLM 技术报告

## Truncation-aware Singular Value Decomposition for Large Language Model Compression

> **论文:** arXiv:2403.07378 (ICLR 2025)
> **作者:** Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang
> **机构:** The Ohio State University / Michigan State University

---

## 目录

1. [引言与动机](#1-引言与动机)
2. [问题形式化](#2-问题形式化)
3. [Vanilla SVD 压缩及其缺陷](#3-vanilla-svd-压缩及其缺陷)
4. [SVD-LLM 核心方法](#4-svd-llm-核心方法)
   - 4.1 [Truncation-Aware Data Whitening](#41-truncation-aware-data-whitening)
   - 4.2 [Sequential Low-Rank Approximation](#42-sequential-low-rank-approximation)
5. [CompressedLinear 架构](#5-compressedlinear-架构)
6. [完整算法流程](#6-完整算法流程)
7. [实现细节与工程优化](#7-实现细节与工程优化)
8. [实验设置与复现结果](#8-实验设置与复现结果)
9. [附录：代码-数学符号对照表](#9-附录代码-数学符号对照表)

---

## 1. 引言与动机

大语言模型 (LLM) 的参数量通常在数十亿级别（如 LLaMA-7B 有 68 亿参数），部署时面临显存、延迟和能耗的巨大挑战。**后训练压缩 (post-training compression)** 旨在不重新预训练的前提下减少模型参数量，主要有三类方法：

| 压缩方法 | 代表工作 | 核心思路 |
|---------|---------|---------|
| 量化 (Quantization) | GPTQ, AWQ | 降低每个参数的位宽 (FP16→INT4) |
| 剪枝 (Pruning) | SparseGPT, Wanda | 移除不重要的权重/结构 |
| **低秩分解 (Low-Rank)** | **SVD-LLM**, ASVD, FWSVD | 将权重矩阵分解为两个小矩阵的乘积 |

SVD-LLM 属于低秩分解类方法。它的核心问题是：**传统 SVD 直接在权重矩阵上截断最小奇异值，但这并不等价于最小化模型输出误差。** SVD-LLM 通过两个创新解决这一问题：

1. **Truncation-Aware Data Whitening** — 利用校准数据的激活统计信息构造白化变换，使截断最小奇异值严格等价于最小化输出误差
2. **Sequential Low-Rank Approximation** — 压缩后对分解得到的两个低秩矩阵做交替 LoRA 微调，恢复截断造成的精度损失

---

## 2. 问题形式化

### 2.1 线性层压缩目标

Transformer 中每个线性层执行 $y = Wx + b$，其中 $W \in \mathbb{R}^{d \times n}$。低秩压缩的目标是找到秩为 $r$ 的近似 $\hat{W}$ 使得**输出误差最小**：

$$
\min_{\hat{W}} \; \mathbb{E}_{x \sim \mathcal{D}} \left[\| Wx - \hat{W}x \|_2^2 \right] = \min_{\hat{W}} \; \mathbb{E}_{x \sim \mathcal{D}} \left[\| (W - \hat{W})x \|_2^2 \right]
$$

其中 $x$ 是该线性层的输入激活，期望取自校准数据集上的激活分布 $\mathcal{D}$。

> **注意:** 优化目标是输出空间的 $\ell_2$ 误差，不是权重空间的 Frobenius 范数 $\|W - \hat{W}\|_F^2$。这个区别是 SVD-LLM 方法论的出发点。

### 2.2 压缩比与秩的关系

原始线性层参数量为 $d \times n$。低秩分解 $\hat{W} = AB$（$A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times n}$）的参数量为 $(d + n) \times r$。**压缩比**定义为移除的参数比例：

$$
R_w = 1 - \frac{(d + n) \cdot r}{d \cdot n}
$$

给定压缩比 $R_w$，目标秩的计算公式：

$$
r = \left\lfloor (1 - R_w) \cdot \frac{d \cdot n}{d + n} \right\rfloor, \quad r \geq 1
$$

**LLaMA-7B 的具体 rank 值（32 层 × 7 个线性子层/层 = 224 个矩阵）：**

| 压缩比 $R_w$ | $W \in \mathbb{R}^{4096 \times 4096}$ 的 $r$ | $W \in \mathbb{R}^{4096 \times 11008}$ 的 $r$ |
|:---:|:---:|:---:|
| 20% | 1638 | 2388 |
| 40% | 1228 | 1791 |
| 60% | 819 | 1194 |
| 80% | 409 | 597 |

其中 $4096 \times 4096$ 对应 Q/K/V/O projections，$4096 \times 11008$ 对应 Gate/Up projections，$11008 \times 4096$ 对应 Down projection。

### 2.3 需要压缩的权重矩阵

对每个 Transformer 层，需压缩以下 7 个线性子层：

| 子模块 | 权重维度 | 功能 |
|--------|---------|------|
| `self_attn.q_proj` | $4096 \times 4096$ | Query 投影 |
| `self_attn.k_proj` | $4096 \times 4096$ | Key 投影 |
| `self_attn.v_proj` | $4096 \times 4096$ | Value 投影 |
| `self_attn.o_proj` | $4096 \times 4096$ | Output 投影 |
| `mlp.gate_proj` | $11008 \times 4096$ | SwiGLU 门控投影 |
| `mlp.up_proj` | $11008 \times 4096$ | SwiGLU 上投影 |
| `mlp.down_proj` | $4096 \times 11008$ | SwiGLU 下投影 |

LLaMA-7B 共 32 层，因此总计 $32 \times 7 = 224$ 个线性层需要独立压缩。

---

## 3. Vanilla SVD 压缩及其缺陷

### 3.1 标准 SVD 低秩逼近

对权重矩阵 $W$ 做 SVD：

$$
W = U \Sigma V^T = \sum_{i=1}^{\min(d,n)} \sigma_i \, u_i v_i^T
$$

保留前 $r$ 个最大奇异值：

$$
\hat{W}_{\text{SVD}} = U_r \Sigma_r V_r^T = \sum_{i=1}^{r} \sigma_i \, u_i v_i^T
$$

由 Eckart–Young–Mirsky 定理，这是**权重空间** Frobenius 范数误差的最优 rank-$r$ 逼近：

$$
\hat{W}_{\text{SVD}} = \arg\min_{\text{rank}(\hat{W}) \leq r} \|W - \hat{W}\|_F^2 = \arg\min_{\text{rank}(\hat{W}) \leq r} \sum_{i > r} \sigma_i^2
$$

### 3.2 为什么权重空间最优 ≠ 输出空间最优

截断第 $i$ 个奇异值带来的**输出**误差为：

$$
\mathcal{L}_i = \mathbb{E}\left[\|\sigma_i u_i (v_i^T x)\|_2^2\right] = \sigma_i^2 \cdot \mathbb{E}\left[(v_i^T x)^2\right]
$$

展开期望项：

$$
\mathbb{E}\left[(v_i^T x)^2\right] = v_i^T \, \mathbb{E}[xx^T] \, v_i = v_i^T C \, v_i
$$

其中 $C = \mathbb{E}[xx^T]$ 是激活协方差矩阵。**如果 $C = I$**（即输入各维度独立同方差），则 $v_i^T C v_i = \|v_i\|^2 = 1$，此时 $\mathcal{L}_i = \sigma_i^2$，截断最小 $\sigma_i$ 恰好最小化输出误差。

**但实际 LLM 中 $C \neq I$。** 激活 $x$ 各维度的方差差异极大（可达 $10^2$ 倍量级），这意味着：
- 某些方向上 $\mathbb{E}[(v_i^T x)^2]$ 很大，即使 $\sigma_i$ 小，截断代价也高
- 某些方向上 $\mathbb{E}[(v_i^T x)^2]$ 很小，即使 $\sigma_i$ 大，截断代价也低

**结论:** 在非白化空间中，截断最小奇异值不保证最小化输出误差。SVD、FWSVD（加权 SVD）、ASVD（激活感知 SVD）都不同程度地存在这个问题。

---

## 4. SVD-LLM 核心方法

### 4.1 Truncation-Aware Data Whitening

**核心思想:** 在做 SVD 之前，先对输入激活做白化变换，使 $C_{\text{white}} = I$，从而使截断最小奇异值严格等价于最小化输出误差。

#### Step 1: 计算激活协方差矩阵

给定 $N$ 个校准样本通过该线性层的输入激活 $x_1, x_2, \ldots, x_N$（每个 $x_k \in \mathbb{R}^n$，实际 $N = 256 \times 2048 = 524288$，因为每个 token 位置都是独立样本），协方差矩阵为：

$$
C = \frac{1}{N} \sum_{k=1}^{N} x_k x_k^T = \frac{1}{N} X^T X \in \mathbb{R}^{n \times n}
$$

其中 $X \in \mathbb{R}^{N \times n}$ 是所有激活样本按行堆叠的矩阵。

**实现细节:** $N$ 可达 52 万，不可能存储完整 $X$ 矩阵。代码使用 forward hook **流式累加** $X^TX$：

```python
# 每个 batch (batch_size=4, seqlen=2048):
#   x shape: (4×2048, n) = (8192, n)
#   x^T x: (n, n)  — 在 GPU float32 上累加
accumulators[key]["XtX"] += x.T @ x   # O(n²) GPU 内存
accumulators[key]["N"] += x.shape[0]
# 最终: C = XtX / N
```

GPU 内存占用分析（LLaMA-7B, float32）：
- 6 个 $4096 \times 4096$ 矩阵 × 32 层 = $6 \times 32 \times 4096^2 \times 4\text{B} \approx 12\text{GB}$
- 1 个 $11008 \times 11008$ 矩阵 × 32 层 = $32 \times 11008^2 \times 4\text{B} \approx 15\text{GB}$
- **总计约 27GB GPU 显存**

#### Step 2: Cholesky 分解

对协方差矩阵做 Cholesky 分解：

$$
C = LL^T
$$

其中 $L \in \mathbb{R}^{n \times n}$ 是下三角矩阵。代码中令 $S = L^T$（上三角），则 $C = S^T S$。

**白化变换:** 定义 $z = L^{-1} x$，验证白化性质：

$$
\mathbb{E}[zz^T] = L^{-1} \mathbb{E}[xx^T] (L^{-1})^T = L^{-1} C \, L^{-T} = L^{-1} LL^T L^{-T} = I
$$

白化后的激活 $z$ 各维度不相关且方差均为 1。

**数值稳定性处理:**

1. **基础正则化:** $C \leftarrow C + \epsilon I$，$\epsilon = 10^{-6}$
2. **秩亏时追加正则化:** 若 Cholesky 失败（$C$ 近奇异），追加 $C \leftarrow C + \frac{\text{tr}(C)}{n} \cdot I$

```python
try:
    L = torch.linalg.cholesky(C)
except torch.linalg.LinAlgError:
    diag_mean = C.diagonal().mean().item()
    C = C + diag_mean * torch.eye(n)
    L = torch.linalg.cholesky(C)
```

#### Step 3: 在白化空间做 SVD

由 $x = Lz$，原始输出 $y = Wx = W(Lz) = (WL)z$。对 $WL$（而非 $W$）做 SVD：

$$
WL = U \Sigma V^T
$$

截断第 $i$ 个奇异值的输出损失：

$$
\mathcal{L}_i = \sigma_i^2 \cdot \mathbb{E}\left[(v_i^T z)^2\right] = \sigma_i^2 \cdot v_i^T \underbrace{\mathbb{E}[zz^T]}_{=\,I} v_i = \sigma_i^2 \cdot \underbrace{\|v_i\|^2}_{=\,1} = \sigma_i^2
$$

**关键结论:** 在白化空间中，每个奇异值的输出损失贡献恰好是 $\sigma_i^2$。截断最小的奇异值严格保证最小输出损失。

总输出误差为：

$$
\mathbb{E}\left[\|(W - \hat{W})x\|_2^2\right] = \|(W - \hat{W})L\|_F^2 = \sum_{i > r} \sigma_i^2
$$

#### Step 4: 截断并分解为两个低秩矩阵

保留前 $r$ 个最大奇异值：

$$
WL \approx U_r \Sigma_r V_r^T
$$

还原到原始空间：

$$
W \approx U_r \Sigma_r V_r^T L^{-1}
$$

论文将 $\Sigma_r$ **对称分配 (symmetric split)** 到两侧：

$$
\boxed{
W'_u = U_r \, \Sigma_r^{1/2} \in \mathbb{R}^{d \times r}, \qquad
W'_v = \Sigma_r^{1/2} \, V_r^T \, L^{-1} \in \mathbb{R}^{r \times n}
}
$$

验证：$W'_u \, W'_v = U_r \Sigma_r^{1/2} \cdot \Sigma_r^{1/2} V_r^T L^{-1} = U_r \Sigma_r V_r^T L^{-1} \approx W$

```python
sqrt_sigma = torch.sqrt(Sigma_r)
A = U_r * sqrt_sigma.unsqueeze(0)                  # W'_u: (d, r)
B = sqrt_sigma.unsqueeze(1) * (Vh_r @ S_inv.T)     # W'_v: (r, n)
# 其中 S_inv.T = (L^T)^{-T} = L^{-1}
```

**为什么选择对称分配？**
- 非对称分配（如 $A = U_r\Sigma_r, B = V_r^TL^{-1}$）在数学上等价
- 对称分配使 $W'_u$ 和 $W'_v$ 的数值尺度更接近，对后续 LoRA 微调更友好（梯度更均衡）

#### 完整推导链（一步一步）

$$
\begin{aligned}
& \text{目标:} \quad \min_{\hat{W}} \; \mathbb{E}\left[\|(W - \hat{W})x\|_2^2\right] \\[6pt]
& \text{激活协方差:} \quad C = \mathbb{E}[xx^T] = \frac{1}{N}X^TX \\[6pt]
& \text{Cholesky:} \quad C = LL^T \\[6pt]
& \text{白化:} \quad z = L^{-1}x, \quad \mathbb{E}[zz^T] = I \\[6pt]
& \text{变量代换:} \quad \|(W - \hat{W})x\|^2 = \|(W - \hat{W})Lz\|^2 \\[6pt]
& \text{在白化空间 SVD:} \quad WL = U\Sigma V^T \\[6pt]
& \text{输出误差:} \quad \mathbb{E}\left[\|(W - \hat{W})x\|^2\right] = \|(W - \hat{W})L\|_F^2 = \sum_{i>r} \sigma_i^2 \\[6pt]
& \text{截断:} \quad WL \approx U_r\Sigma_r V_r^T \\[6pt]
& \text{还原:} \quad W \approx U_r\Sigma_r V_r^T L^{-1} \\[6pt]
& \text{对称分配:} \quad W'_u = U_r\Sigma_r^{1/2}, \quad W'_v = \Sigma_r^{1/2} V_r^T L^{-1}
\end{aligned}
$$

### 4.2 Sequential Low-Rank Approximation

白化 SVD 截断解决了「用最少的秩保留最多信息」的问题（**阶段 A: 压缩**），但截断不可避免地引入误差——尤其在高压缩比（60%/80%）下 perplexity 急剧退化。论文的第二个核心贡献是在压缩后做 **Sequential LoRA 微调** 恢复精度（**阶段 B: 参数更新**）。

#### 为什么不直接全参数微调？

压缩后的模型有两个低秩矩阵 $W'_u \in \mathbb{R}^{d \times r}$ 和 $W'_v \in \mathbb{R}^{r \times n}$，同时优化两者是**双线性优化问题**：

$$
\min_{W'_u, W'_v} \mathcal{L}(W'_u W'_v)
$$

这是非凸的（$W'_u$ 和 $W'_v$ 之间有乘法耦合），梯度下降容易陷入不好的局部最优。

#### 交替优化策略

**Sequential** 的含义：固定一个矩阵，只优化另一个。这将双线性问题转化为两个线性子问题，每个子问题是凸的。

**Stage 1 — LoRA_u:** 冻结所有 $W'_v$，对所有 $W'_u$ 添加 LoRA adapter 并微调

$$
W'_u \leftarrow W'_u + \Delta W'_u, \quad \Delta W'_u = B_{\text{lora}} A_{\text{lora}}
$$

其中 $A_{\text{lora}} \in \mathbb{R}^{r_{\text{lora}} \times r}, B_{\text{lora}} \in \mathbb{R}^{d \times r_{\text{lora}}}$，$r_{\text{lora}} = 32 \ll r$。

**Stage 2 — LoRA_v:** 冻结更新后的 $W'_u$，对所有 $W'_v$ 添加 LoRA adapter 并微调

$$
W'_v \leftarrow W'_v + \Delta W'_v, \quad \Delta W'_v = B'_{\text{lora}} A'_{\text{lora}}
$$

每个 stage 微调完毕后将 LoRA 权重合并回主权重（`merge_and_unload()`）。

#### 微调数据与超参数

| 项目 | 配置 |
|------|------|
| 微调数据 | Alpaca-cleaned 50K (`yahma/alpaca-cleaned`) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| 学习率 | 2e-4 |
| 学习率调度 | Cosine with warmup (warmup_ratio=0.03) |
| Epochs | 1（每个 stage） |
| Batch size | 4 × 4 (grad_accum) = 16 |
| 序列长度 | 256 |
| 精度 | FP16 混合精度 |
| Gradient Checkpointing | 开启 |
| 目标模块数 | 224 个 CompressedLinear（Stage 1 优化 `.second`，Stage 2 优化 `.first`） |

**Alpaca 数据处理:**

训练时仅对 response 部分计算 loss，prompt 部分用 `labels=-100` 遮蔽：

```
Prompt: "Below is an instruction... ### Instruction:\n{instruction}\n### Response:\n"
Labels: [-100, -100, ..., -100, response_token_1, response_token_2, ..., eos]
```

#### SVD-LLM(W) 与 SVD-LLM 的区别

| | SVD-LLM(W) | SVD-LLM |
|---|---|---|
| 阶段 A (压缩) | 白化 + SVD 截断 | 白化 + SVD 截断（相同） |
| 阶段 B (参数更新) | **无** | Sequential LoRA (Alpaca 50K) |
| 校准数据 | WikiText-2 × 256 samples | WikiText-2 × 256 samples |
| 微调数据 | 不需要 | Alpaca 50K |
| 推理结构 | nn.Linear (合并后) | nn.Linear (合并后) |
| 压缩耗时 | ~10 分钟 | ~10 分钟 + ~1 小时微调 |

---

## 5. CompressedLinear 架构

原始线性层 $y = Wx + b$ 被替换为两个串联的小线性层：

```
输入 x ∈ R^n
  │
  ├─→ first: Linear(n → r, bias=False)    权重 = W'_v ∈ R^{r×n}
  │      z = W'_v x
  │
  └─→ second: Linear(r → d, bias=b)       权重 = W'_u ∈ R^{d×r}
         y = W'_u z + b
```

**参数量对比:**

$$
\text{原始: } d \cdot n + d \quad \longrightarrow \quad \text{压缩: } r \cdot n + r \cdot d + d = (n + d) \cdot r + d
$$

**CompressedLinear 模块:**

```python
class CompressedLinear(nn.Module):
    def __init__(self, A, B, bias=None):
        self.first = nn.Linear(n, r, bias=False)    # W'_v
        self.second = nn.Linear(r, d, bias=True)     # W'_u + bias
        self.first.weight = nn.Parameter(B)           # (r, n)
        self.second.weight = nn.Parameter(A)          # (d, r)

    def forward(self, x):
        return self.second(self.first(x))
```

**保存与加载:** 部署前通过 `merge_compressed_model()` 将所有 CompressedLinear 合并回标准 `nn.Linear`：

$$
W_{\text{merged}} = W'_u \cdot W'_v \in \mathbb{R}^{d \times n}
$$

合并后的模型与原始 HuggingFace 格式完全兼容（`save_pretrained` / `from_pretrained`）。

---

## 6. 完整算法流程

### Algorithm 1: SVD-LLM(W) — 仅白化压缩

```
Input:  模型 M, 校准数据 D_calib (256 × WikiText-2, seqlen=2048), 压缩比 R_w
Output: 压缩后模型 M'

1.  对每个线性子层 W ∈ R^{d×n}, 计算目标秩:
      r = max(1, ⌊(1 - R_w) · d · n / (d + n)⌋)

    // 阶段 A-1: 收集协方差 (一次 forward pass 覆盖所有 32×7=224 个线性层)
2.  注册 forward hooks 到所有 224 个线性子层
3.  for batch ∈ batch(D_calib, batch_size=4):        // 共 64 次 forward pass
4.      M(batch)  → hooks 流式累加 X^T X 和 N
5.  移除所有 hooks

    // 阶段 A-2: 逐层白化 SVD 压缩
6.  for layer l = 0, 1, ..., 31:
7.      for linear_name ∈ {q_proj, k_proj, v_proj, o_proj, gate, up, down}:
8.          C ← XtX / N                              // 协方差矩阵
9.          C ← C + εI                               // 正则化
10.         L ← cholesky(C)                           // 下三角
11.         WL ← W · L                                // 变换到白化空间
12.         U, Σ, V^T ← SVD(WL)                       // 截断 SVD
13.         W'_u ← U_r · Σ_r^{1/2}                    // 对称分配
14.         W'_v ← Σ_r^{1/2} · V_r^T · L^{-1}
15.         保存 (W'_u, W'_v) 到临时文件

    // 阶段 A-3: 统一替换
16. for layer l = 0, 1, ..., 31:
17.     for linear_name:
18.         加载 (W'_u, W'_v) → 替换为 CompressedLinear

19. return M' (SVD-LLM(W) 完成)
```

### Algorithm 2: SVD-LLM — 白化压缩 + Sequential LoRA

```
Input:  模型 M (原始), D_calib, R_w, D_ft (Alpaca 50K)
Output: 压缩+微调后的模型 M'_final

    // 阶段 A: 同 Algorithm 1 (白化 SVD 截断)
1.  M' ← Algorithm 1(M, D_calib, R_w)

    // 阶段 B: Sequential LoRA Fine-tuning
    // Stage 1: LoRA_u
2.  冻结 M' 中所有 CompressedLinear 的 .first (W'_v)
3.  对所有 CompressedLinear 的 .second (W'_u) 添加 LoRA(r=32, α=64)
4.  用 D_ft 训练 1 epoch (Causal LM loss, prompt masked)
5.  合并 LoRA → 得到 M'_u

    // Stage 2: LoRA_v
6.  冻结 M'_u 中所有 CompressedLinear 的 .second (W'_u)
7.  对所有 CompressedLinear 的 .first (W'_v) 添加 LoRA(r=32, α=64)
8.  用 D_ft 训练 1 epoch
9.  合并 LoRA → 得到 M'_final

10. return M'_final
```

---

## 7. 实现细节与工程优化

### 7.1 流式协方差收集

**挑战:** 256 样本 × 2048 seqlen = 524,288 个 $n$ 维向量。对 $n = 11008$（MLP Down 层），完整存储 $X \in \mathbb{R}^{524288 \times 11008}$ 需要 ~22GB，且每层独立。

**方案:** 利用 $X^TX$ 的可加性，通过 forward hook 流式累加：

$$
X^TX = \sum_{t} B_t^T B_t
$$

其中 $B_t$ 是第 $t$ 个 mini-batch 的激活。只需存储 $n \times n$ 的累加器（每层约 $11008^2 \times 4\text{B} = 484\text{MB}$）。

**GPU vs CPU 累加:** 代码在 GPU 上以 float32 累加 $X^TX$，最终一次性搬到 CPU 转 float64。相比每步搬 CPU 累加 float64，减少了 GPU↔CPU 同步次数。

### 7.2 一次 forward pass 收集所有层

**朴素方法:** 逐层收集，每层需要完整 forward pass → 32 层 × 64 batches = 2048 次 forward pass。

**SVD-LLM(W) 优化:** 由于白化压缩不修改模型权重，所有层的激活可以**一次 forward pass 同时收集**：

- 在所有 224 个线性层注册 hook
- 跑 64 次 forward pass（256 samples / batch_size 4）
- 一次性得到所有协方差

这将 forward pass 次数从 2048 降低到 64（32 倍加速）。

### 7.3 Early Exit 优化（逐层模式）

当以逐层模式收集协方差时（用于调试或 sequential update），处理第 $l$ 层时，在第 $l+1$ 层注册 pre-hook 抛出 `_EarlyExitException`：

```python
# 处理第 l 层时只跑 layers 0..l，跳过 l+1..31
def _early_exit_pre_hook(module, args):
    raise _EarlyExitException()
# 平均省 50% forward 计算
```

### 7.4 Alpaca 数据处理

- **Prompt 遮蔽:** 对 prompt 部分的 token 设置 `labels[j] = -100`（CrossEntropyLoss 忽略），仅在 response 部分计算损失
- **Padding 遮蔽:** padding token 同样设为 -100
- **过滤:** 当 prompt 长度 ≥ max_length 时 response 全被截断，过滤掉这些全 -100 样本
- **格式模板:** 区分有/无 input 字段的 Alpaca 样本，使用不同模板

### 7.5 Tokenizer 兼容性修复

`jeffwan/llama-7b-hf` 的 tokenizer 存在 special token ID 错误（bos/eos/unk 全映射为 0）。下游任务评估时使用 `model.config` 中的正确值修复：

```python
if tokenizer.eos_token_id == 0:
    tokenizer.bos_token_id = model.config.bos_token_id or 1
    tokenizer.eos_token_id = model.config.eos_token_id or 2
```

### 7.6 模型保存与加载

1. 压缩/微调过程中模型包含 `CompressedLinear` 模块
2. 保存前调用 `merge_compressed_model()` 将 $W'_u \cdot W'_v$ 烘焙为单个 $W_{\text{merged}}$
3. 替换为标准 `nn.Linear`，使 state_dict 与 `from_pretrained()` 兼容
4. 保存压缩配置到 `compression_config.json`

---

## 8. 实验设置与复现结果

### 8.1 运行环境

| 项目 | 配置 |
|------|------|
| GPU | 6× NVIDIA A800 80GB PCIe |
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.1 |
| lm-eval | 0.4.9 |
| PEFT | 0.6.0 |

### 8.2 校准与评估数据

| 用途 | 数据集 | 样本数/设置 |
|------|--------|------------|
| 白化校准 | WikiText-2 (train) | 256 samples, seqlen=2048 |
| LoRA 微调 | Alpaca-cleaned | 50K samples, max_length=256 |
| PPL 评估 | WikiText-2 (test) | 全部，滑动窗口 seqlen=2048 |
| PPL 评估 | C4 (validation) | 前 1100 条 |
| 下游任务 | 8 个任务 | lm-evaluation-harness |

### 8.3 评估任务

| 类别 | 数据集 | 指标 | 评估方式 |
|------|--------|------|---------|
| 语言建模 | WikiText-2 | Perplexity ↓ | 滑动窗口 |
| 语言建模 | C4 | Perplexity ↓ | 滑动窗口 |
| 多选分类 | OpenbookQA | Accuracy ↑ | lm-eval |
| 多选分类 | ARC-Easy | Accuracy ↑ | lm-eval |
| 多选分类 | WinoGrande | Accuracy ↑ | lm-eval |
| 多选分类 | HellaSwag | Accuracy (norm) ↑ | lm-eval |
| 多选分类 | PIQA | Accuracy ↑ | lm-eval |
| 数学推理 | MathQA | Accuracy ↑ | lm-eval (本地 Parquet) |
| 真实性 | TruthfulQA | MC2 ↑ | lm-eval |
| 数学生成 | GSM8K | Exact Match ↑ | lm-eval |

### 8.4 Perplexity 复现结果 (LLaMA-7B)

| 压缩比 | 数据集 | SVD | FWSVD | ASVD | SVD-LLM(W) 论文 | **SVD-LLM(W) 复现** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0% | WikiText-2 | 5.68 | 5.68 | 5.68 | 5.68 | **5.67** |
| 0% | C4 | 7.34 | 7.34 | 7.34 | 7.34 | **7.20** |
| 20% | WikiText-2 | 20061 | 1727 | 11.14 | 7.94 | **7.84** |
| 20% | C4 | 18800 | 1511 | 15.93 | 15.84 | **15.65** |
| 40% | WikiText-2 | 52489 | 18156 | 1407 | 13.73 | **13.17** |
| 40% | C4 | 47774 | 12847 | 1109 | 75.42 | **50.05** |
| 60% | WikiText-2 | 105474 | 32194 | 57057 | 66.62 | **58.98** |
| 60% | C4 | 106976 | 29292 | 43036 | 471.83 | **378.80** |
| 80% | WikiText-2 | 687291 | 96872 | 80425 | 1349 | **660.67** |
| 80% | C4 | 708243 | 89243 | 67927 | 6224 | **2641.62** |

**关键观察:**
- 所有压缩比上复现 PPL 均**优于论文报告值**
- SVD-LLM(W) 在 20% 压缩比下 PPL 仅从 5.67 升至 7.84（WikiText-2），退化可接受
- 80% 压缩比下 PPL 严重退化，需要阶段 B 的 LoRA 微调恢复

### 8.5 下游任务复现结果（全部压缩比）

| 压缩比 | 任务 | SVD-LLM(W) 论文 | **SVD-LLM(W) 复现** | SVD-LLM 论文 | **SVD-LLM 复现** |
|:---:|:---|:---:|:---:|:---:|:---:|
| 20% | OpenbookQA | 0.31 | **0.266** | 0.33 | — |
| 20% | ARC_easy | 0.62 | **0.641** | 0.67 | — |
| 20% | WinoGrande | 0.61 | **0.664** | 0.69 | — |
| 20% | HellaSwag | 0.45 | **0.437** | 0.55 | — |
| 20% | PIQA | 0.71 | **0.690** | 0.79 | — |
| 20% | MathQA | 0.21 | **0.239** | 0.26 | — |
| 20% | **Avg (6)** | **0.49** | **0.490** | **0.55** | — |
| 20% | TruthfulQA† | 0.26 | **0.390** | 0.28 | — |
| 20% | GSM8K | 0.05 | **0.006** | 0.08 | — |
| 40% | OpenbookQA | 0.25 | **0.200** | 0.29 | — |
| 40% | ARC_easy | 0.33 | **0.457** | 0.59 | — |
| 40% | WinoGrande | 0.55 | **0.575** | 0.68 | — |
| 40% | HellaSwag | 0.40 | **0.330** | 0.52 | — |
| 40% | PIQA | 0.63 | **0.609** | 0.69 | — |
| 40% | MathQA | 0.12 | **0.220** | 0.20 | — |
| 40% | **Avg (6)** | **0.38** | **0.399** | **0.50** | — |
| 40% | TruthfulQA† | 0.17 | **0.433** | 0.24 | — |
| 40% | GSM8K | 0.02 | **0.000** | 0.07 | — |
| 60% | OpenbookQA | 0.10 | **0.132** | 0.18 | — |
| 60% | ARC_easy | 0.05 | **0.301** | 0.42 | — |
| 60% | WinoGrande | 0.17 | **0.527** | 0.44 | — |
| 60% | HellaSwag | 0.10 | **0.273** | 0.31 | — |
| 60% | PIQA | 0.21 | **0.544** | 0.35 | — |
| 60% | MathQA | 0.04 | **0.218** | 0.12 | — |
| 60% | **Avg (6)** | **0.11** | **0.333** | **0.30** | — |
| 60% | TruthfulQA† | 0.01 | **0.476** | 0.14 | — |
| 60% | GSM8K | 0.00 | **0.000** | 0.04 | — |
| 80% | OpenbookQA | 0.07 | **0.130** | 0.11 | — |
| 80% | ARC_easy | 0.03 | **0.261** | 0.23 | — |
| 80% | WinoGrande | 0.04 | **0.480** | 0.21 | — |
| 80% | HellaSwag | 0.02 | **0.260** | 0.14 | — |
| 80% | PIQA | 0.07 | **0.523** | 0.17 | — |
| 80% | MathQA | 0.01 | **0.205** | 0.08 | — |
| 80% | **Avg (6)** | **0.04** | **0.310** | **0.16** | — |
| 80% | TruthfulQA† | 0.00 | **0.501** | 0.04 | — |
| 80% | GSM8K | 0.00 | **0.000** | 0.02 | — |

> † TruthfulQA: 论文使用 BLEU 指标，复现使用 lm-evaluation-harness 的 `truthfulqa_mc2` (MC2) 指标，两者不可直接对比。
>
> **注:** 下游任务准确率与论文存在差异，主要原因是 lm-evaluation-harness 版本不同（复现使用 v0.4.9，论文未指定版本）。不同版本的任务实现、prompt 模板和评分逻辑可能存在变化。

### 8.6 方法间对比分析

| 方法 | 核心策略 | 20% PPL (Wiki) | 80% PPL (Wiki) |
|------|---------|:-----------:|:-----------:|
| SVD | 直接截断 W 的最小奇异值 | 20061 | 687291 |
| FWSVD | 按 Fisher 信息加权 SVD | 1727 | 96872 |
| ASVD | 激活感知的奇异值截断 | 11.14 | 80425 |
| **SVD-LLM(W)** | 白化空间 SVD（本文） | **7.94** | **1349** |
| **SVD-LLM** | 白化 + Sequential LoRA | **7.73** | **31.79** |

SVD-LLM(W) 通过白化将 20% PPL 从 ASVD 的 11.14 降至 7.94；SVD-LLM 的 Sequential LoRA 在 80% 压缩比下将 PPL 从 1349 恢复到 31.79（42 倍改善）。

---

## 9. 附录：代码-数学符号对照表

| 数学符号 | 代码变量/函数 | 文件位置 | 说明 |
|---------|-------------|---------|------|
| $W$ | `target.weight.data` | `compress_model.py:70` | 原始权重矩阵 |
| $C = X^TX/N$ | `XtX / N` | `calibration.py:61` | 激活协方差矩阵 |
| $L$ (Cholesky 下三角) | `L = torch.linalg.cholesky(C)` | `whitening.py:25` | Cholesky 因子 |
| $S = L^T$ (上三角) | `S = L.T` | `whitening.py:31` | 代码命名习惯 |
| $WL$ | `WS = W @ S.T` | `whitening.py:83` | 白化空间中的权重 |
| $U_r, \Sigma_r, V_r^T$ | `U_r, Sigma_r, Vh_r` | `whitening.py:86-88` | 截断 SVD 结果 |
| $\Sigma_r^{1/2}$ | `sqrt_sigma` | `whitening.py:91` | 对称分配用 |
| $W'_u = U_r\Sigma_r^{1/2}$ | `A` | `whitening.py:92` | 左因子 (d×r) |
| $W'_v = \Sigma_r^{1/2}V_r^TL^{-1}$ | `B` | `whitening.py:93` | 右因子 (r×n) |
| $L^{-1}$ | `S_inv.T` | `whitening.py:93` | $S^{-T} = (L^T)^{-T} = L^{-1}$ |
| $r$ | `compute_rank(d, n, ratio)` | `loader.py:45` | 目标秩 |
| CompressedLinear.first | `W'_v` (权重 B) | `replace.py:17` | 第一个线性层 |
| CompressedLinear.second | `W'_u` (权重 A) | `replace.py:18` | 第二个线性层 |

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
