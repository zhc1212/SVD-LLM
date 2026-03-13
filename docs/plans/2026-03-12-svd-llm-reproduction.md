# SVD-LLM 自主复现 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 从论文独立复现 SVD-LLM，在 LLaMA-7B 上实现 SVD-LLM(W) 和 SVD-LLM 两种方法，重现 Table 1 对应结果（2种方法 × 4种压缩比 × 10个评估指标）

**Architecture:** 模块化设计。核心流程：加载模型 → 收集校准激活 → 白化+SVD压缩权重 → 替换层 → 评估。不实现 Baseline 方法（Vanilla SVD / FWSVD / ASVD）。

**NOTE:** Task 3 (Vanilla SVD), Task 4 (FWSVD), Task 5 (ASVD) 已跳过，不实现 Baseline 方法。

**Tech Stack:** Python 3.12, PyTorch 2.6.0+cu124, Transformers 4.57.1, lm-eval 0.4.9, SciPy 1.16.1, 6×A800 80GB

---

## 关键技术参数

### LLaMA-7B 架构
- 32 层 Transformer, hidden_size=4096, intermediate_size=11008, 32 heads
- 需压缩的线性层（每层 7 个）:
  - `self_attn.q_proj`: 4096 × 4096
  - `self_attn.k_proj`: 4096 × 4096
  - `self_attn.v_proj`: 4096 × 4096
  - `self_attn.o_proj`: 4096 × 4096
  - `mlp.gate_proj`: 4096 × 11008
  - `mlp.up_proj`: 4096 × 11008
  - `mlp.down_proj`: 11008 × 4096

### 压缩比与保留秩 r
压缩比公式: `R_w = 1 - (d + n) * r / (d * n)`

| 压缩比 | Q/K/V/O (4096×4096) r= | Gate/Up (4096×11008) r= | Down (11008×4096) r= |
|--------|------------------------|------------------------|---------------------|
| 20%    | 1638                   | 2388                   | 2388                |
| 40%    | 1228                   | 1791                   | 1791                |
| 60%    | 819                    | 1194                   | 1194                |
| 80%    | 409                    | 597                    | 597                 |

### 压缩后层的表示
原始 `nn.Linear(in_features=n, out_features=d)` 替换为两个串联线性层:
- `first`: `nn.Linear(n, r, bias=False)` — 存储 V_r^T (或含缩放)
- `second`: `nn.Linear(r, d, bias=original_bias)` — 存储 U_r Σ_r (或含缩放)

### 环境激活
```bash
source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai
```

### 模型路径
```
/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/
```

---

## Task 1: 项目骨架与模型加载

**Files:**
- Create: `src/__init__.py`
- Create: `src/model/__init__.py`
- Create: `src/model/loader.py`
- Create: `src/data/__init__.py`
- Create: `src/compress/__init__.py`
- Create: `src/eval/__init__.py`
- Test: `tests/test_model_loader.py`

**Step 1: 创建项目目录结构**

```bash
mkdir -p src/model src/data src/compress src/eval scripts tests
touch src/__init__.py src/model/__init__.py src/data/__init__.py src/compress/__init__.py src/eval/__init__.py
```

**Step 2: 写 model loader**

```python
# src/model/loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_model(model_path: str, device_map: str = "auto", dtype=torch.float16):
    """加载 HuggingFace 模型，返回 (model, tokenizer)"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def get_model_config(model_path: str):
    """获取模型配置（不加载权重）"""
    return AutoConfig.from_pretrained(model_path)


def get_linear_layers(model):
    """返回所有需要压缩的线性层及其名称

    Returns: list of (name, module) tuples
    跳过 lm_head 和 embed_tokens
    """
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            linear_layers.append((name, module))
    return linear_layers


def compute_rank(d: int, n: int, ratio: float) -> int:
    """根据压缩比计算保留的秩 r

    R_w = 1 - (d + n) * r / (d * n)
    => r = (1 - R_w) * d * n / (d + n)
    """
    r = int((1 - ratio) * d * n / (d + n))
    return max(1, r)
```

**Step 3: 写测试**

```python
# tests/test_model_loader.py
import pytest
import torch
from src.model.loader import load_model, get_linear_layers, compute_rank, get_model_config

MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"


def test_compute_rank():
    """测试压缩比到秩的转换"""
    # 4096x4096, 20% compression
    r = compute_rank(4096, 4096, 0.2)
    assert r == 1638
    # 4096x11008, 40% compression
    r = compute_rank(4096, 11008, 0.4)
    assert r == 1791
    # Edge case: 最小为 1
    r = compute_rank(4096, 4096, 0.999)
    assert r >= 1


def test_get_model_config():
    """测试加载模型配置"""
    config = get_model_config(MODEL_PATH)
    assert config.hidden_size == 4096
    assert config.intermediate_size == 11008
    assert config.num_hidden_layers == 32


def test_load_model():
    """测试加载完整模型"""
    model, tokenizer = load_model(MODEL_PATH)
    assert model is not None
    assert tokenizer is not None
    # 检查基本结构
    layers = get_linear_layers(model)
    # 每层 7 个线性层 × 32 层 = 224
    assert len(layers) == 224
    # 检查名称格式
    names = [n for n, _ in layers]
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.31.mlp.down_proj" in names
    del model
    torch.cuda.empty_cache()
```

**Step 4: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_model_loader.py -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/ tests/test_model_loader.py
git commit -m "feat: add project structure and model loader"
```

---

## Task 2: 校准数据加载与激活收集

**Files:**
- Create: `src/data/calibration.py`
- Test: `tests/test_calibration.py`

**Step 1: 写校准数据加载与激活收集模块**

```python
# src/data/calibration.py
import torch
from datasets import load_dataset
from functools import partial


def get_calibration_data(tokenizer, dataset_name="wikitext2", nsamples=256, seqlen=2048, seed=42):
    """从数据集加载校准数据，返回 token ids list

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: "wikitext2"
        nsamples: 校准样本数
        seqlen: 序列长度
        seed: 随机种子

    Returns:
        list of torch.LongTensor, each shape (1, seqlen)
    """
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 将所有文本拼接成一个长字符串
    full_text = "\n\n".join([t for t in dataset[text_column] if t.strip()])

    # Tokenize
    tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]  # (total_len,)

    # 随机采样 nsamples 个长度为 seqlen 的片段
    import random
    random.seed(seed)

    samples = []
    max_start = len(tokens) - seqlen - 1
    for _ in range(nsamples):
        start = random.randint(0, max_start)
        sample = tokens[start : start + seqlen].unsqueeze(0)  # (1, seqlen)
        samples.append(sample)

    return samples


def collect_layer_activations(model, calibration_data, layer_idx, device="cuda"):
    """收集指定 transformer 层的输入激活

    对 model.layers[layer_idx] 注册 forward hook，
    将所有校准样本的输入激活拼接返回。

    Args:
        model: HuggingFace CausalLM model
        calibration_data: list of (1, seqlen) tensors
        layer_idx: transformer 层索引 (0-31)
        device: 计算设备

    Returns:
        activations: tensor of shape (seqlen * nsamples, hidden_size)
    """
    layer = model.model.layers[layer_idx]
    activations = []

    def hook_fn(module, input, output):
        # input 是 tuple, input[0] shape: (batch, seqlen, hidden_size)
        inp = input[0].detach()
        # reshape to (seqlen, hidden_size) since batch=1
        activations.append(inp.squeeze(0).to(torch.float32))

    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for sample in calibration_data:
            sample = sample.to(device)
            model(sample)

    handle.remove()

    # Concatenate: (nsamples * seqlen, hidden_size)
    return torch.cat(activations, dim=0)


def collect_linear_input_activations(model, calibration_data, layer_idx,
                                      linear_name, device="cuda"):
    """收集指定线性层的输入激活

    Args:
        model: HuggingFace CausalLM model
        calibration_data: list of (1, seqlen) tensors
        layer_idx: transformer 层索引
        linear_name: 线性层名称，如 "self_attn.q_proj", "mlp.gate_proj"
        device: 计算设备

    Returns:
        activations: tensor of shape (nsamples * seqlen, in_features)
    """
    # 定位目标线性层
    layer = model.model.layers[layer_idx]
    parts = linear_name.split(".")
    target = layer
    for p in parts:
        target = getattr(target, p)

    activations = []

    def hook_fn(module, input, output):
        inp = input[0].detach()
        # 展平 batch 和 seq 维度
        activations.append(inp.reshape(-1, inp.shape[-1]).to(torch.float32))

    handle = target.register_forward_hook(hook_fn)

    with torch.no_grad():
        for sample in calibration_data:
            sample = sample.to(device)
            model(sample)

    handle.remove()

    return torch.cat(activations, dim=0)
```

**Step 2: 写测试**

```python
# tests/test_calibration.py
import pytest
import torch
from src.model.loader import load_model
from src.data.calibration import (
    get_calibration_data,
    collect_layer_activations,
    collect_linear_input_activations,
)

MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = load_model(MODEL_PATH)
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


def test_get_calibration_data(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    samples = get_calibration_data(tokenizer, nsamples=4, seqlen=128)
    assert len(samples) == 4
    assert samples[0].shape == (1, 128)
    assert samples[0].dtype == torch.long


def test_collect_linear_input_activations(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    samples = get_calibration_data(tokenizer, nsamples=2, seqlen=64)

    # 收集第 0 层 q_proj 的输入激活
    X = collect_linear_input_activations(
        model, samples, layer_idx=0, linear_name="self_attn.q_proj"
    )
    # shape: (2 * 64, 4096)
    assert X.shape == (2 * 64, 4096)
    assert X.dtype == torch.float32
    assert not torch.isnan(X).any()
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_calibration.py -v`
Expected: 2 PASS (需要 GPU，加载模型较慢)

**Step 4: Commit**

```bash
git add src/data/calibration.py tests/test_calibration.py
git commit -m "feat: add calibration data loading and activation collection"
```

---

## Task 3: Vanilla SVD 压缩

**Files:**
- Create: `src/compress/svd_vanilla.py`
- Test: `tests/test_svd_vanilla.py`

**Step 1: 实现 Vanilla SVD**

核心算法: 直接对权重矩阵 W 做 SVD，截断到 rank r。

```python
# src/compress/svd_vanilla.py
import torch


def compress_linear_svd(weight, rank):
    """对单个权重矩阵做 Vanilla SVD 压缩

    W = U Σ V^T, 保留前 rank 个奇异值
    返回两个矩阵: A = U_r Σ_r (d, r), B = V_r^T (r, n)
    使得 W ≈ A @ B

    Args:
        weight: tensor (d, n) — 线性层的权重 (out_features, in_features)
        rank: int — 保留的秩

    Returns:
        A: tensor (d, r)
        B: tensor (r, n)
    """
    W = weight.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # 截断到 rank
    U_r = U[:, :rank]       # (d, r)
    S_r = S[:rank]           # (r,)
    Vh_r = Vh[:rank, :]      # (r, n)

    A = U_r * S_r.unsqueeze(0)  # (d, r) = U_r @ diag(S_r)
    B = Vh_r                     # (r, n)

    return A, B
```

**Step 2: 写测试**

```python
# tests/test_svd_vanilla.py
import pytest
import torch
from src.compress.svd_vanilla import compress_linear_svd
from src.model.loader import compute_rank


def test_compress_linear_svd_shape():
    """测试输出形状"""
    W = torch.randn(4096, 4096)
    r = compute_rank(4096, 4096, 0.2)  # 1638
    A, B = compress_linear_svd(W, r)
    assert A.shape == (4096, r)
    assert B.shape == (r, 4096)


def test_compress_linear_svd_reconstruction():
    """测试低秩近似误差：rank 越高，误差越小"""
    torch.manual_seed(42)
    W = torch.randn(256, 256)

    errors = []
    for ratio in [0.2, 0.5, 0.8]:
        r = compute_rank(256, 256, ratio)
        A, B = compress_linear_svd(W, r)
        W_hat = A @ B
        error = torch.norm(W - W_hat).item() / torch.norm(W).item()
        errors.append(error)

    # 压缩比越大，误差越大
    assert errors[0] < errors[1] < errors[2]


def test_compress_linear_svd_mlp_shape():
    """测试非方阵 (MLP layers)"""
    W = torch.randn(4096, 11008)
    r = compute_rank(4096, 11008, 0.4)  # 1791
    A, B = compress_linear_svd(W, r)
    assert A.shape == (4096, r)
    assert B.shape == (r, 11008)
    # 验证重建
    W_hat = A @ B
    assert W_hat.shape == W.shape
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_svd_vanilla.py -v`
Expected: 3 PASS

**Step 4: Commit**

```bash
git add src/compress/svd_vanilla.py tests/test_svd_vanilla.py
git commit -m "feat: add vanilla SVD compression"
```

---

## Task 4: FWSVD 压缩 (Fisher-Weighted SVD)

**Files:**
- Create: `src/compress/fwsvd.py`
- Test: `tests/test_fwsvd.py`

**Step 1: 实现 FWSVD**

核心思想: 用 Fisher 信息矩阵的对角近似来加权 SVD。Fisher 信息 F_ij ≈ E[(∂L/∂w_ij)²]，对 √F ⊙ W 做 SVD，然后恢复原始尺度。

```python
# src/compress/fwsvd.py
import torch


def compute_fisher_info(model, calibration_data, layer_idx, linear_name, device="cuda"):
    """计算指定线性层权重的 Fisher 信息（对角近似）

    F_ij = E[(∂L/∂w_ij)²] ≈ (1/N) Σ (∂L/∂w_ij)²

    使用语言模型的 next-token prediction loss。

    Args:
        model: CausalLM model
        calibration_data: list of (1, seqlen) tensors
        layer_idx: transformer 层索引
        linear_name: 如 "self_attn.q_proj"
        device: 计算设备

    Returns:
        fisher: tensor (d, n), 与权重同形状的 Fisher 信息
    """
    # 定位目标线性层
    layer = model.model.layers[layer_idx]
    parts = linear_name.split(".")
    target = layer
    for p in parts:
        target = getattr(target, p)

    weight = target.weight  # (d, n)
    fisher = torch.zeros_like(weight, dtype=torch.float32, device=device)

    n_samples = 0
    for sample in calibration_data:
        sample = sample.to(device)
        model.zero_grad()

        outputs = model(sample, labels=sample)
        loss = outputs.loss
        loss.backward()

        if weight.grad is not None:
            fisher += weight.grad.float() ** 2
            n_samples += 1

    fisher /= n_samples
    return fisher


def compress_linear_fwsvd(weight, fisher, rank):
    """Fisher-Weighted SVD 压缩

    1. 计算 √F
    2. 对 √F ⊙ W 做 SVD 截断
    3. 恢复: W_hat = (U_r Σ_r V_r^T) ⊘ √F

    Args:
        weight: tensor (d, n)
        fisher: tensor (d, n)
        rank: int

    Returns:
        A: tensor (d, r)
        B: tensor (r, n)
    """
    W = weight.float()
    F = fisher.float()

    # 避免除零
    sqrt_F = torch.sqrt(F + 1e-10)

    # 加权
    W_weighted = sqrt_F * W

    # SVD
    U, S, Vh = torch.linalg.svd(W_weighted, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # 恢复原始尺度
    W_hat = (U_r * S_r.unsqueeze(0)) @ Vh_r / sqrt_F

    # 再对恢复后的 W_hat 做一次 SVD 得到低秩分解形式
    U2, S2, Vh2 = torch.linalg.svd(W_hat, full_matrices=False)
    A = U2[:, :rank] * S2[:rank].unsqueeze(0)
    B = Vh2[:rank, :]

    return A, B
```

**Step 2: 写测试**

```python
# tests/test_fwsvd.py
import pytest
import torch
from src.compress.fwsvd import compress_linear_fwsvd


def test_compress_fwsvd_shape():
    """测试输出形状"""
    W = torch.randn(256, 256)
    F = torch.rand(256, 256).abs() + 0.01  # Fisher info 非负
    A, B = compress_linear_fwsvd(W, F, rank=100)
    assert A.shape == (256, 100)
    assert B.shape == (100, 256)


def test_fwsvd_vs_vanilla():
    """FWSVD 和 Vanilla SVD 应该给出不同的结果（除非 Fisher 全为常数）"""
    torch.manual_seed(42)
    W = torch.randn(256, 256)
    F = torch.rand(256, 256).abs() + 0.01

    from src.compress.svd_vanilla import compress_linear_svd

    A_v, B_v = compress_linear_svd(W, 100)
    A_f, B_f = compress_linear_fwsvd(W, F, 100)

    W_vanilla = A_v @ B_v
    W_fwsvd = A_f @ B_f

    # 两者应该不同
    assert not torch.allclose(W_vanilla, W_fwsvd, atol=1e-3)
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_fwsvd.py -v`
Expected: 2 PASS

**Step 4: Commit**

```bash
git add src/compress/fwsvd.py tests/test_fwsvd.py
git commit -m "feat: add FWSVD (Fisher-Weighted SVD) compression"
```

---

## Task 5: ASVD 压缩 (Activation-aware SVD)

**Files:**
- Create: `src/compress/asvd.py`
- Test: `tests/test_asvd.py`

**Step 1: 实现 ASVD**

核心思想: 用激活的列范数来缩放权重的列（输入维度），使得重要的输入通道在 SVD 中被优先保留。

```python
# src/compress/asvd.py
import torch


def compute_activation_scales(activations):
    """计算激活矩阵各输入通道的缩放因子

    s_j = ||X_j||_2 (第 j 列的 L2 范数)

    Args:
        activations: tensor (N, n) — N 个样本, n 个输入特征

    Returns:
        scales: tensor (n,)
    """
    # 列的 L2 范数
    scales = torch.norm(activations.float(), dim=0)  # (n,)
    # 避免除零
    scales = torch.clamp(scales, min=1e-8)
    return scales


def compress_linear_asvd(weight, activations, rank):
    """Activation-aware SVD 压缩

    1. 计算缩放: s_j = ||X_j||_2
    2. 缩放权重: W' = W @ diag(s)
    3. SVD 截断: W' = U Σ V^T → U_r Σ_r V_r^T
    4. 恢复: A = U_r Σ_r, B = V_r^T @ diag(1/s)
    使得 W ≈ A @ B

    Args:
        weight: tensor (d, n)
        activations: tensor (N, n)
        rank: int

    Returns:
        A: tensor (d, r)
        B: tensor (r, n)
    """
    W = weight.float()
    scales = compute_activation_scales(activations)

    # 缩放权重列
    W_scaled = W * scales.unsqueeze(0)  # (d, n)

    # SVD
    U, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    A = U_r * S_r.unsqueeze(0)             # (d, r)
    B = Vh_r / scales.unsqueeze(0)          # (r, n) — 恢复缩放

    return A, B
```

**Step 2: 写测试**

```python
# tests/test_asvd.py
import pytest
import torch
from src.compress.asvd import compress_linear_asvd, compute_activation_scales


def test_activation_scales():
    """测试激活缩放计算"""
    X = torch.randn(100, 64)
    scales = compute_activation_scales(X)
    assert scales.shape == (64,)
    assert (scales > 0).all()


def test_compress_asvd_shape():
    """测试输出形状"""
    W = torch.randn(256, 512)
    X = torch.randn(100, 512)
    A, B = compress_linear_asvd(W, X, rank=80)
    assert A.shape == (256, 80)
    assert B.shape == (80, 512)


def test_asvd_reconstruction():
    """ASVD 应保持 W @ X ≈ (A @ B) @ X"""
    torch.manual_seed(42)
    W = torch.randn(128, 256)
    X = torch.randn(50, 256)

    A, B = compress_linear_asvd(W, X, rank=60)

    # 原始输出 vs 压缩输出
    Y_orig = X @ W.T         # (50, 128)
    Y_comp = X @ (A @ B).T   # (50, 128)

    error = torch.norm(Y_orig - Y_comp) / torch.norm(Y_orig)
    # 合理的近似误差
    assert error < 0.5
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_asvd.py -v`
Expected: 3 PASS

**Step 4: Commit**

```bash
git add src/compress/asvd.py tests/test_asvd.py
git commit -m "feat: add ASVD (Activation-aware SVD) compression"
```

---

## Task 6: SVD-LLM 白化 (Truncation-Aware Data Whitening)

**Files:**
- Create: `src/compress/whitening.py`
- Test: `tests/test_whitening.py`

**Step 1: 实现白化算法**

核心算法:
1. 计算激活协方差: C = X^T X (注意: 论文写 XX^T，但这里 X 是 (N, n)，所以协方差是 X^T X ∈ R^{n×n})
2. Cholesky 分解: C = S^T S (S 是上三角) 或 C = LL^T (L 是下三角)
3. 对 W @ S^T 做 SVD (S^T 使得 S^{-T} X^T 各通道正交)
4. 截断后恢复: W_hat = U_r Σ_r V_r^T @ S^{-T}

**注意**: 论文的记号中 X 的排列可能不同。关键等式是:
- 白化后的激活 X_w = S^{-1} X 满足 X_w X_w^T = I
- 对 WS 做 SVD 使得截断第 i 个奇异值的输出损失 = σ_i

```python
# src/compress/whitening.py
import torch


def compute_whitening_matrix(activations, eps=1e-6):
    """计算白化矩阵 S (通过 Cholesky 分解)

    给定激活 X ∈ R^{N×n}:
    1. C = X^T X / N  (协方差矩阵, n×n)
    2. Cholesky: C = L L^T  (L 是下三角)
    3. S = L^T (上三角), 满足 C = S^T S

    白化操作: X_whitened = X @ S^{-1}
    对 W @ S^T 做 SVD 等价于在白化空间中做 SVD

    Args:
        activations: tensor (N, n)
        eps: 正则化项，防止协方差矩阵奇异

    Returns:
        S: tensor (n, n) — 上三角白化矩阵
        S_inv: tensor (n, n) — S 的逆
    """
    X = activations.float()
    N, n = X.shape

    # 协方差矩阵
    C = X.T @ X / N  # (n, n)

    # 加正则化防止奇异
    C += eps * torch.eye(n, device=C.device, dtype=C.dtype)

    # Cholesky 分解: C = L L^T
    L = torch.linalg.cholesky(C)  # 下三角

    # S = L^T (上三角), C = L L^T = S^T S
    S = L.T

    # S 的逆: S^{-1} = (L^T)^{-1} = (L^{-1})^T
    S_inv = torch.linalg.inv(S)

    return S, S_inv


def compress_linear_whitening(weight, activations, rank, eps=1e-6):
    """SVD-LLM 白化压缩 (仅白化，无 sequential update)

    1. 计算白化矩阵 S (通过 Cholesky)
    2. 计算 WS^T: 将权重投影到白化空间
    3. 对 WS^T 做 SVD 截断
    4. 恢复: W_hat = U_r Σ_r V_r^T @ S^{-T}
    5. 分解为低秩形式: A @ B

    Args:
        weight: tensor (d, n) — out_features × in_features
        activations: tensor (N, n)
        rank: int
        eps: Cholesky 正则化

    Returns:
        A: tensor (d, r)
        B: tensor (r, n)
    """
    W = weight.float()
    d, n = W.shape

    # Step 1: 白化矩阵
    S, S_inv = compute_whitening_matrix(activations, eps=eps)

    # Step 2: WS^T — 在白化空间中的权重
    # W @ S^T 使得截断奇异值的输出损失最小
    WS = W @ S.T  # (d, n)

    # Step 3: SVD 截断
    U, Sigma, Vh = torch.linalg.svd(WS, full_matrices=False)
    U_r = U[:, :rank]          # (d, r)
    Sigma_r = Sigma[:rank]     # (r,)
    Vh_r = Vh[:rank, :]        # (r, n)

    # Step 4: 恢复到原始空间
    # W_hat = U_r diag(Σ_r) V_r^T @ S^{-T}
    # 分解为: A = U_r diag(Σ_r), B = V_r^T @ S^{-T}
    A = U_r * Sigma_r.unsqueeze(0)   # (d, r)
    B = Vh_r @ S_inv.T               # (r, n) = (r, n) @ (n, n)

    return A, B
```

**Step 2: 写测试**

```python
# tests/test_whitening.py
import pytest
import torch
from src.compress.whitening import compute_whitening_matrix, compress_linear_whitening
from src.compress.svd_vanilla import compress_linear_svd
from src.model.loader import compute_rank


def test_whitening_matrix_shape():
    """测试白化矩阵的形状和性质"""
    X = torch.randn(500, 64)
    S, S_inv = compute_whitening_matrix(X)
    assert S.shape == (64, 64)
    assert S_inv.shape == (64, 64)

    # S @ S_inv ≈ I
    I_approx = S @ S_inv
    I_true = torch.eye(64)
    assert torch.allclose(I_approx, I_true, atol=1e-4)


def test_whitening_decorrelates():
    """白化后的激活应该近似不相关"""
    torch.manual_seed(42)
    # 构造相关的激活
    A_mat = torch.randn(64, 64)
    X = torch.randn(1000, 64) @ A_mat  # 相关的

    S, S_inv = compute_whitening_matrix(X)

    # 白化
    X_w = X @ S_inv

    # 白化后的协方差应近似单位阵
    C_w = X_w.T @ X_w / X_w.shape[0]
    I = torch.eye(64)
    assert torch.allclose(C_w, I, atol=0.1)


def test_compress_whitening_shape():
    """测试白化压缩的输出形状"""
    W = torch.randn(256, 512)
    X = torch.randn(200, 512)
    r = compute_rank(256, 512, 0.3)
    A, B = compress_linear_whitening(W, X, r)
    assert A.shape == (256, r)
    assert B.shape == (r, 512)


def test_whitening_better_than_vanilla():
    """白化 SVD 在考虑激活分布时应优于 Vanilla SVD

    构造一个激活高度不均匀的场景：
    某些通道激活很大，某些很小
    """
    torch.manual_seed(42)
    d, n = 128, 256
    W = torch.randn(d, n)

    # 构造不均匀激活: 前 50 个通道激活很大，后面很小
    scales = torch.ones(n)
    scales[:50] = 10.0
    scales[50:] = 0.1
    X = torch.randn(500, n) * scales.unsqueeze(0)

    rank = compute_rank(d, n, 0.5)

    # Vanilla SVD
    A_v, B_v = compress_linear_svd(W, rank)
    Y_orig = X @ W.T
    Y_vanilla = X @ (A_v @ B_v).T
    err_vanilla = torch.norm(Y_orig - Y_vanilla) / torch.norm(Y_orig)

    # Whitening SVD
    A_w, B_w = compress_linear_whitening(W, X, rank)
    Y_whiten = X @ (A_w @ B_w).T
    err_whiten = torch.norm(Y_orig - Y_whiten) / torch.norm(Y_orig)

    # 白化应该在输出空间上误差更小
    assert err_whiten < err_vanilla, f"Whitening error {err_whiten:.4f} should be < Vanilla error {err_vanilla:.4f}"
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_whitening.py -v`
Expected: 4 PASS

**Step 4: Commit**

```bash
git add src/compress/whitening.py tests/test_whitening.py
git commit -m "feat: add truncation-aware data whitening (SVD-LLM core)"
```

---

## Task 7: 压缩层替换模块

**Files:**
- Create: `src/model/replace.py`
- Test: `tests/test_replace.py`

**Step 1: 实现层替换**

将压缩后的 (A, B) 矩阵对替换为两个串联的 nn.Linear。

```python
# src/model/replace.py
import torch
import torch.nn as nn


class CompressedLinear(nn.Module):
    """压缩后的线性层: 两个串联的小线性层

    原始: y = Wx + b, W ∈ R^{d×n}
    压缩: y = A(Bx) + b, A ∈ R^{d×r}, B ∈ R^{r×n}
    """

    def __init__(self, A, B, bias=None):
        """
        Args:
            A: tensor (d, r)
            B: tensor (r, n)
            bias: tensor (d,) or None
        """
        super().__init__()
        d, r = A.shape
        _, n = B.shape

        self.first = nn.Linear(n, r, bias=False)
        self.second = nn.Linear(r, d, bias=bias is not None)

        self.first.weight = nn.Parameter(B.clone())      # (r, n)
        self.second.weight = nn.Parameter(A.clone())      # (d, r)
        if bias is not None:
            self.second.bias = nn.Parameter(bias.clone())

    def forward(self, x):
        return self.second(self.first(x))


def replace_linear_with_compressed(model, layer_idx, linear_name, A, B):
    """替换模型中指定的线性层为压缩版本

    Args:
        model: HuggingFace model
        layer_idx: transformer 层索引
        linear_name: 如 "self_attn.q_proj"
        A: tensor (d, r)
        B: tensor (r, n)
    """
    layer = model.model.layers[layer_idx]
    parts = linear_name.split(".")

    # 获取原始线性层
    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)

    original = getattr(parent, parts[-1])
    bias = original.bias.data if original.bias is not None else None

    # 创建压缩层
    compressed = CompressedLinear(
        A.to(original.weight.dtype).to(original.weight.device),
        B.to(original.weight.dtype).to(original.weight.device),
        bias=bias.to(original.weight.device) if bias is not None else None,
    )

    # 替换
    setattr(parent, parts[-1], compressed)

    # 释放原始权重
    del original
    torch.cuda.empty_cache()
```

**Step 2: 写测试**

```python
# tests/test_replace.py
import pytest
import torch
from src.model.replace import CompressedLinear, replace_linear_with_compressed


def test_compressed_linear_forward():
    """测试压缩层的前向传播"""
    A = torch.randn(64, 20)
    B = torch.randn(20, 128)
    bias = torch.randn(64)

    layer = CompressedLinear(A, B, bias)
    x = torch.randn(4, 128)
    y = layer(x)

    assert y.shape == (4, 64)

    # 验证等价于 A @ B @ x^T + bias
    y_manual = x @ (A @ B).T + bias
    assert torch.allclose(y, y_manual, atol=1e-5)


def test_compressed_linear_no_bias():
    """测试无 bias 的压缩层"""
    A = torch.randn(64, 20)
    B = torch.randn(20, 128)

    layer = CompressedLinear(A, B, bias=None)
    x = torch.randn(4, 128)
    y = layer(x)

    assert y.shape == (4, 64)
    assert layer.second.bias is None
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_replace.py -v`
Expected: 2 PASS

**Step 4: Commit**

```bash
git add src/model/replace.py tests/test_replace.py
git commit -m "feat: add compressed linear layer and model replacement"
```

---

## Task 8: Sequential Low-Rank Approximation

**Files:**
- Create: `src/compress/sequential_update.py`
- Test: `tests/test_sequential.py`

**Step 1: 实现 Sequential Update**

核心思想: 逐层处理，每层压缩后用更新后的激活来处理下一层，补偿误差累积。

SVD-LLM(W) 是一次性用原始激活白化所有层。
SVD-LLM 是逐层更新：压缩第 l 层 → 用压缩后模型重新算第 l+1 层的激活 → 白化压缩第 l+1 层。

```python
# src/compress/sequential_update.py
import torch
from src.data.calibration import collect_linear_input_activations
from src.compress.whitening import compress_linear_whitening
from src.model.replace import replace_linear_with_compressed
from src.model.loader import compute_rank


# 每个 transformer 层内需要压缩的线性层，按执行顺序排列
LINEAR_LAYERS_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def compress_model_sequential(model, tokenizer, calibration_data, ratio, device="cuda"):
    """SVD-LLM 完整压缩流程: 白化 + Sequential Update

    逐层逐线性层处理:
    1. 对每一层的每个线性层，收集当前（更新后的）输入激活
    2. 用白化 SVD 压缩该线性层
    3. 替换为压缩层
    4. 继续处理下一个线性层（此时激活会反映之前的压缩）

    Args:
        model: HuggingFace CausalLM model
        tokenizer: tokenizer
        calibration_data: list of (1, seqlen) tensors
        ratio: 压缩比 (0.0 - 1.0)
        device: 计算设备

    Returns:
        model: 压缩后的模型 (in-place 修改)
    """
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers - 1}...")

        for linear_name in LINEAR_LAYERS_ORDER:
            # 获取当前线性层的权重维度
            layer = model.model.layers[layer_idx]
            parts = linear_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)

            weight = target.weight.data  # (d, n)
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            # Step 1: 用当前模型收集该线性层的输入激活
            X = collect_linear_input_activations(
                model, calibration_data, layer_idx, linear_name, device=device
            )

            # Step 2: 白化 SVD 压缩
            A, B = compress_linear_whitening(weight.float(), X, rank)

            # Step 3: 替换为压缩层
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)

            # 清理
            del X, A, B
            torch.cuda.empty_cache()

    return model


def compress_model_whitening_only(model, tokenizer, calibration_data, ratio, device="cuda"):
    """SVD-LLM(W) 压缩: 仅白化，无 sequential update

    用原始模型的激活一次性白化压缩所有层。
    不逐层更新激活。

    与 sequential 版本的区别:
    - 这里所有激活都来自原始未压缩模型
    - sequential 版本每压缩一层后，后续层的激活会更新

    Args:
        同 compress_model_sequential

    Returns:
        model: 压缩后的模型
    """
    num_layers = model.config.num_hidden_layers

    # 预收集所有层所有线性层的激活（来自原始模型）
    all_activations = {}
    print("Collecting activations from original model...")
    for layer_idx in range(num_layers):
        for linear_name in LINEAR_LAYERS_ORDER:
            print(f"  Layer {layer_idx}, {linear_name}")
            X = collect_linear_input_activations(
                model, calibration_data, layer_idx, linear_name, device=device
            )
            all_activations[(layer_idx, linear_name)] = X

    # 用预收集的激活压缩所有层
    print("Compressing with whitening...")
    for layer_idx in range(num_layers):
        print(f"Compressing layer {layer_idx}/{num_layers - 1}...")
        for linear_name in LINEAR_LAYERS_ORDER:
            layer = model.model.layers[layer_idx]
            parts = linear_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)

            weight = target.weight.data
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            X = all_activations[(layer_idx, linear_name)]
            A, B = compress_linear_whitening(weight.float(), X, rank)
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)

            del X

    del all_activations
    torch.cuda.empty_cache()
    return model


def compress_model_vanilla(model, ratio):
    """Vanilla SVD 压缩: 直接对权重做 SVD，无激活信息"""
    from src.compress.svd_vanilla import compress_linear_svd

    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        print(f"Compressing layer {layer_idx}/{num_layers - 1}...")
        for linear_name in LINEAR_LAYERS_ORDER:
            layer = model.model.layers[layer_idx]
            parts = linear_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)

            weight = target.weight.data
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            A, B = compress_linear_svd(weight.float(), rank)
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)

    torch.cuda.empty_cache()
    return model


def compress_model_asvd(model, calibration_data, ratio, device="cuda"):
    """ASVD 压缩: 用激活缩放权重后做 SVD"""
    from src.compress.asvd import compress_linear_asvd

    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        print(f"Compressing layer {layer_idx}/{num_layers - 1}...")
        for linear_name in LINEAR_LAYERS_ORDER:
            layer = model.model.layers[layer_idx]
            parts = linear_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)

            weight = target.weight.data
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            X = collect_linear_input_activations(
                model, calibration_data, layer_idx, linear_name, device=device
            )
            A, B = compress_linear_asvd(weight.float(), X, rank)
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)
            del X

    torch.cuda.empty_cache()
    return model


def compress_model_fwsvd(model, calibration_data, ratio, device="cuda"):
    """FWSVD 压缩: Fisher 信息加权 SVD"""
    from src.compress.fwsvd import compute_fisher_info, compress_linear_fwsvd

    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        print(f"Compressing layer {layer_idx}/{num_layers - 1}...")
        for linear_name in LINEAR_LAYERS_ORDER:
            layer = model.model.layers[layer_idx]
            parts = linear_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)

            weight = target.weight.data
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            fisher = compute_fisher_info(
                model, calibration_data, layer_idx, linear_name, device=device
            )
            A, B = compress_linear_fwsvd(weight.float(), fisher, rank)
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)
            del fisher

    torch.cuda.empty_cache()
    return model
```

**Step 2: 写测试**

```python
# tests/test_sequential.py
import pytest
import torch
from src.model.loader import compute_rank


def test_linear_layers_order():
    """验证线性层顺序列表"""
    from src.compress.sequential_update import LINEAR_LAYERS_ORDER
    assert len(LINEAR_LAYERS_ORDER) == 7
    assert "self_attn.q_proj" in LINEAR_LAYERS_ORDER
    assert "mlp.down_proj" in LINEAR_LAYERS_ORDER


def test_compress_model_vanilla_smoke():
    """Vanilla SVD 端到端 smoke test (仅压缩前2层)"""
    from src.model.loader import load_model
    from src.compress.sequential_update import compress_model_vanilla
    from src.model.replace import CompressedLinear

    MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"
    model, tokenizer = load_model(MODEL_PATH)

    # 仅压缩前 2 层测试
    original_num_layers = model.config.num_hidden_layers
    model.config.num_hidden_layers = 2

    compress_model_vanilla(model, ratio=0.5)

    model.config.num_hidden_layers = original_num_layers

    # 验证第 0 层已被替换
    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, CompressedLinear)

    # 验证第 2 层未被替换 (仍是原始 nn.Linear)
    q_proj_2 = model.model.layers[2].self_attn.q_proj
    assert isinstance(q_proj_2, torch.nn.Linear)
    assert not isinstance(q_proj_2, CompressedLinear)

    del model
    torch.cuda.empty_cache()
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_sequential.py -v -x`
Expected: 2 PASS

**Step 4: Commit**

```bash
git add src/compress/sequential_update.py tests/test_sequential.py
git commit -m "feat: add sequential update and all compression method orchestrators"
```

---

## Task 9: Perplexity 评估

**Files:**
- Create: `src/eval/perplexity.py`
- Test: `tests/test_perplexity.py`

**Step 1: 实现 Perplexity 评估**

```python
# src/eval/perplexity.py
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss


def evaluate_perplexity(model, tokenizer, dataset_name="wikitext2", device="cuda"):
    """评估模型在指定数据集上的 Perplexity

    使用滑动窗口方法，序列长度 = model.config.max_position_embeddings (2048)

    Args:
        model: HuggingFace CausalLM
        tokenizer: tokenizer
        dataset_name: "wikitext2" 或 "c4"
        device: 计算设备

    Returns:
        perplexity: float
    """
    model.eval()
    seqlen = model.config.max_position_embeddings  # 2048

    # 加载测试集
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    elif dataset_name == "c4":
        dataset = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = "\n\n".join(dataset["text"][:1100])  # 与常见做法一致
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # (total_len,)

    # 滑动窗口评估
    nll_sum = 0.0
    n_tokens = 0

    loss_fn = CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for start in range(0, len(input_ids) - 1, seqlen):
            end = min(start + seqlen, len(input_ids))
            ids = input_ids[start:end].unsqueeze(0).to(device)  # (1, chunk_len)

            outputs = model(ids)
            logits = outputs.logits  # (1, chunk_len, vocab_size)

            # shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            nll_sum += loss.item()
            n_tokens += shift_labels.numel()

    avg_nll = nll_sum / n_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    return perplexity
```

**Step 2: 写测试**

```python
# tests/test_perplexity.py
import pytest
import torch


def test_perplexity_original_model():
    """测试原始模型的 WikiText-2 Perplexity 应在合理范围"""
    from src.model.loader import load_model
    from src.eval.perplexity import evaluate_perplexity

    MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"
    model, tokenizer = load_model(MODEL_PATH)

    ppl = evaluate_perplexity(model, tokenizer, "wikitext2")
    print(f"WikiText-2 Perplexity: {ppl:.2f}")

    # 论文报告 5.68, 允许合理偏差
    assert 4.0 < ppl < 8.0, f"Perplexity {ppl} out of expected range"

    del model
    torch.cuda.empty_cache()
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_perplexity.py -v -s`
Expected: PASS, WikiText-2 PPL ≈ 5.68

**Step 4: Commit**

```bash
git add src/eval/perplexity.py tests/test_perplexity.py
git commit -m "feat: add perplexity evaluation on WikiText-2 and C4"
```

---

## Task 10: 下游任务评估 (lm-eval-harness)

**Files:**
- Create: `src/eval/downstream.py`
- Test: `tests/test_downstream.py`

**Step 1: 实现下游任务评估**

```python
# src/eval/downstream.py
import lm_eval
from lm_eval.models.huggingface import HFLM


def evaluate_downstream(model, tokenizer, tasks=None, batch_size=8, device="cuda"):
    """使用 lm-evaluation-harness 评估下游任务

    Args:
        model: HuggingFace CausalLM
        tokenizer: tokenizer
        tasks: 任务列表，默认为论文中的 8 个任务
        batch_size: 评估 batch size
        device: 计算设备

    Returns:
        results: dict, 每个任务的评估结果
    """
    if tasks is None:
        tasks = [
            "openbookqa",
            "arc_easy",
            "winogrande",
            "hellaswag",
            "piqa",
            "mathqa",
            "truthfulqa_gen",
            "gsm8k",
        ]

    # 包装模型为 lm-eval 格式
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=str(device),
    )

    # 运行评估
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
    )

    return results["results"]


def format_downstream_results(results):
    """格式化下游任务结果为可读表格

    Args:
        results: simple_evaluate 返回的结果字典

    Returns:
        formatted: dict {task_name: metric_value}
    """
    formatted = {}
    for task_name, task_results in results.items():
        # 提取主要指标
        if "acc,none" in task_results:
            formatted[task_name] = task_results["acc,none"]
        elif "acc_norm,none" in task_results:
            formatted[task_name] = task_results["acc_norm,none"]
        elif "bleu_max,none" in task_results:
            formatted[task_name] = task_results["bleu_max,none"]
        elif "exact_match,strict-match" in task_results:
            formatted[task_name] = task_results["exact_match,strict-match"]
        elif "exact_match,flexible-extract" in task_results:
            formatted[task_name] = task_results["exact_match,flexible-extract"]

    return formatted
```

**Step 2: 写测试**

```python
# tests/test_downstream.py
import pytest


def test_format_downstream_results():
    """测试结果格式化"""
    from src.eval.downstream import format_downstream_results

    mock_results = {
        "openbookqa": {"acc,none": 0.34, "acc_norm,none": 0.35},
        "arc_easy": {"acc,none": 0.75, "acc_norm,none": 0.72},
    }

    formatted = format_downstream_results(mock_results)
    assert formatted["openbookqa"] == 0.34
    assert formatted["arc_easy"] == 0.75
```

**Step 3: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_downstream.py -v`
Expected: 1 PASS

**Step 4: Commit**

```bash
git add src/eval/downstream.py tests/test_downstream.py
git commit -m "feat: add downstream task evaluation via lm-eval-harness"
```

---

## Task 11: 主压缩脚本 (scripts/compress.py)

**Files:**
- Create: `scripts/compress.py`

**Step 1: 实现主压缩脚本**

```python
# scripts/compress.py
"""SVD-LLM 主压缩脚本

用法:
    python scripts/compress.py \
        --model_path <path> \
        --method svd_llm \
        --ratio 0.2 \
        --save_path outputs/llama7b_svdllm_20
"""
import argparse
import os
import sys
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.loader import load_model
from src.data.calibration import get_calibration_data
from src.compress.sequential_update import (
    compress_model_vanilla,
    compress_model_fwsvd,
    compress_model_asvd,
    compress_model_whitening_only,
    compress_model_sequential,
)


def main():
    parser = argparse.ArgumentParser(description="SVD-LLM Compression")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["svd", "fwsvd", "asvd", "svd_llm_w", "svd_llm"])
    parser.add_argument("--ratio", type=float, required=True,
                        help="Compression ratio (0.0-1.0)")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--calib_dataset", type=str, default="wikitext2")
    parser.add_argument("--calib_nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"=== SVD-LLM Compression ===")
    print(f"Method: {args.method}")
    print(f"Ratio: {args.ratio}")
    print(f"Model: {args.model_path}")

    # 加载模型
    print("Loading model...")
    model, tokenizer = load_model(args.model_path)

    # 准备校准数据（非 vanilla 方法需要）
    calibration_data = None
    if args.method != "svd":
        print(f"Loading calibration data ({args.calib_nsamples} samples)...")
        calibration_data = get_calibration_data(
            tokenizer, args.calib_dataset, args.calib_nsamples, args.seqlen, args.seed
        )

    # 压缩
    start_time = time.time()
    print(f"Compressing with {args.method}...")

    if args.method == "svd":
        model = compress_model_vanilla(model, args.ratio)
    elif args.method == "fwsvd":
        model = compress_model_fwsvd(model, calibration_data, args.ratio, args.device)
    elif args.method == "asvd":
        model = compress_model_asvd(model, calibration_data, args.ratio, args.device)
    elif args.method == "svd_llm_w":
        model = compress_model_whitening_only(
            model, tokenizer, calibration_data, args.ratio, args.device
        )
    elif args.method == "svd_llm":
        model = compress_model_sequential(
            model, tokenizer, calibration_data, args.ratio, args.device
        )

    elapsed = time.time() - start_time
    print(f"Compression done in {elapsed:.1f}s")

    # 保存
    print(f"Saving to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    # 保存压缩配置
    import json
    config = {
        "method": args.method,
        "ratio": args.ratio,
        "calib_dataset": args.calib_dataset,
        "calib_nsamples": args.calib_nsamples,
        "seqlen": args.seqlen,
        "compression_time_seconds": elapsed,
    }
    with open(os.path.join(args.save_path, "compression_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/compress.py
git commit -m "feat: add main compression script"
```

---

## Task 12: 主评估脚本 (scripts/evaluate.py)

**Files:**
- Create: `scripts/evaluate.py`

**Step 1: 实现主评估脚本**

```python
# scripts/evaluate.py
"""SVD-LLM 主评估脚本

用法:
    # Perplexity 评估
    python scripts/evaluate.py \
        --model_path outputs/llama7b_svdllm_20 \
        --eval perplexity \
        --datasets wikitext2 c4

    # 下游任务评估
    python scripts/evaluate.py \
        --model_path outputs/llama7b_svdllm_20 \
        --eval downstream \
        --tasks openbookqa arc_easy winogrande hellaswag piqa mathqa truthfulqa_gen gsm8k
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.loader import load_model
from src.eval.perplexity import evaluate_perplexity
from src.eval.downstream import evaluate_downstream, format_downstream_results


def main():
    parser = argparse.ArgumentParser(description="SVD-LLM Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval", type=str, required=True, choices=["perplexity", "downstream", "all"])
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    print(f"=== SVD-LLM Evaluation ===")
    print(f"Model: {args.model_path}")

    model, tokenizer = load_model(args.model_path, device_map="auto")

    results = {}

    if args.eval in ["perplexity", "all"]:
        print("\n--- Perplexity Evaluation ---")
        for ds in args.datasets:
            ppl = evaluate_perplexity(model, tokenizer, ds, args.device)
            results[f"ppl_{ds}"] = ppl
            print(f"  {ds}: {ppl:.2f}")

    if args.eval in ["downstream", "all"]:
        print("\n--- Downstream Task Evaluation ---")
        raw_results = evaluate_downstream(
            model, tokenizer, args.tasks, args.batch_size, args.device
        )
        formatted = format_downstream_results(raw_results)
        results["downstream"] = formatted
        for task, score in formatted.items():
            print(f"  {task}: {score:.4f}")

    # 保存结果
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(args.model_path, "eval_results.json")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add main evaluation script"
```

---

## Task 13: 端到端集成测试

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: 写端到端测试**

验证完整压缩 + 评估流程在小规模上运行正确。

```python
# tests/test_e2e.py
"""端到端集成测试

仅压缩前 2 层 + 少量校准样本，验证全流程不报错。
"""
import pytest
import torch

MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"


@pytest.fixture(scope="module")
def model_and_data():
    from src.model.loader import load_model
    from src.data.calibration import get_calibration_data

    model, tokenizer = load_model(MODEL_PATH)
    calib_data = get_calibration_data(tokenizer, nsamples=4, seqlen=128)
    yield model, tokenizer, calib_data
    del model
    torch.cuda.empty_cache()


def _compress_2_layers(model, method_fn, **kwargs):
    """辅助: 仅压缩前 2 层"""
    original = model.config.num_hidden_layers
    model.config.num_hidden_layers = 2
    method_fn(model, **kwargs)
    model.config.num_hidden_layers = original


def test_e2e_vanilla(model_and_data):
    """Vanilla SVD 端到端"""
    from src.model.loader import load_model
    from src.compress.sequential_update import compress_model_vanilla

    model, tokenizer = load_model(MODEL_PATH)
    original = model.config.num_hidden_layers
    model.config.num_hidden_layers = 2
    compress_model_vanilla(model, ratio=0.5)
    model.config.num_hidden_layers = original

    # 模型应能正常推理
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5)
    assert out.shape[1] > inputs["input_ids"].shape[1]

    del model
    torch.cuda.empty_cache()
```

**Step 2: 运行测试**

Run: `cd /home/xiyaofeng/huicheng/SVD_LLM && python -m pytest tests/test_e2e.py -v -s -x`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "feat: add end-to-end integration test"
```

---

## Task 14: 运行全部实验

**Files:**
- Create: `scripts/run_all_experiments.sh`

**Step 1: 编写实验脚本**

```bash
#!/bin/bash
# scripts/run_all_experiments.sh
# 运行 Table 1 全部实验: 5 methods × 4 ratios

source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai
cd /home/xiyaofeng/huicheng/SVD_LLM

MODEL=/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/

METHODS="svd fwsvd asvd svd_llm_w svd_llm"
RATIOS="0.2 0.4 0.6 0.8"

# Step 1: 评估原始模型 Perplexity
echo "=== Evaluating Original Model ==="
python scripts/evaluate.py \
    --model_path $MODEL \
    --eval perplexity \
    --datasets wikitext2 c4 \
    --output_file outputs/original/eval_results.json

# Step 2: 压缩 + 评估 (每个 method × ratio 组合)
for method in $METHODS; do
    for ratio in $RATIOS; do
        ratio_pct=$(echo "$ratio * 100" | bc | cut -d. -f1)
        save_dir="outputs/llama7b_${method}_${ratio_pct}"

        echo "=== ${method} @ ${ratio_pct}% compression ==="

        # 压缩
        python scripts/compress.py \
            --model_path $MODEL \
            --method $method \
            --ratio $ratio \
            --save_path $save_dir

        # 评估 Perplexity
        python scripts/evaluate.py \
            --model_path $save_dir \
            --eval perplexity \
            --datasets wikitext2 c4

        echo "Done: ${method} @ ${ratio_pct}%"
        echo ""
    done
done

# Step 3: 下游任务评估 (仅 20% 压缩比)
echo "=== Downstream Task Evaluation (20% compression) ==="
for method in $METHODS; do
    save_dir="outputs/llama7b_${method}_20"
    python scripts/evaluate.py \
        --model_path $save_dir \
        --eval downstream
done

echo "=== All experiments complete ==="
```

**Step 2: Commit**

```bash
chmod +x scripts/run_all_experiments.sh
git add scripts/run_all_experiments.sh
git commit -m "feat: add experiment runner script for Table 1 reproduction"
```

---

## Task 15: 结果汇总脚本

**Files:**
- Create: `scripts/collect_results.py`

**Step 1: 编写结果收集脚本**

```python
# scripts/collect_results.py
"""收集所有实验结果，生成 Table 1 对比表"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "outputs"
METHODS = ["svd", "fwsvd", "asvd", "svd_llm_w", "svd_llm"]
RATIOS = [20, 40, 60, 80]

# 论文 Table 1 参考值
PAPER_RESULTS = {
    ("svd", 0): {"wikitext2": 5.68, "c4": 7.34},
    ("svd", 20): {"wikitext2": 20061, "c4": 18800},
    ("svd", 40): {"wikitext2": 52489, "c4": 47774},
    ("svd", 60): {"wikitext2": 105474, "c4": 106976},
    ("svd", 80): {"wikitext2": 687291, "c4": 708243},
    ("fwsvd", 20): {"wikitext2": 1727, "c4": 1511},
    ("fwsvd", 40): {"wikitext2": 18156, "c4": 12847},
    ("fwsvd", 60): {"wikitext2": 32194, "c4": 29292},
    ("fwsvd", 80): {"wikitext2": 96872, "c4": 89243},
    ("asvd", 20): {"wikitext2": 11.14, "c4": 15.93},
    ("asvd", 40): {"wikitext2": 1407, "c4": 1109},
    ("asvd", 60): {"wikitext2": 57057, "c4": 43036},
    ("asvd", 80): {"wikitext2": 80425, "c4": 67927},
    ("svd_llm_w", 20): {"wikitext2": 7.94, "c4": 15.84},
    ("svd_llm_w", 40): {"wikitext2": 13.73, "c4": 75.42},
    ("svd_llm_w", 60): {"wikitext2": 66.62, "c4": 471.83},
    ("svd_llm_w", 80): {"wikitext2": 1349, "c4": 6224},
    ("svd_llm", 20): {"wikitext2": 7.73, "c4": 12.23},
    ("svd_llm", 40): {"wikitext2": 9.27, "c4": 15.63},
    ("svd_llm", 60): {"wikitext2": 15.00, "c4": 26.26},
    ("svd_llm", 80): {"wikitext2": 31.79, "c4": 43.71},
}


def main():
    print("=" * 80)
    print("SVD-LLM Reproduction Results vs Paper (Table 1)")
    print("=" * 80)

    print(f"\n{'Method':<12} {'Ratio':<8} {'WikiText-2':>12} {'Paper':>12} {'C4':>12} {'Paper':>12}")
    print("-" * 70)

    for method in METHODS:
        for ratio in RATIOS:
            result_path = os.path.join(OUTPUT_DIR, f"llama7b_{method}_{ratio}", "eval_results.json")

            if os.path.exists(result_path):
                with open(result_path) as f:
                    results = json.load(f)
                wt2 = results.get("ppl_wikitext2", "N/A")
                c4 = results.get("ppl_c4", "N/A")
            else:
                wt2 = "N/A"
                c4 = "N/A"

            paper = PAPER_RESULTS.get((method, ratio), {})
            paper_wt2 = paper.get("wikitext2", "N/A")
            paper_c4 = paper.get("c4", "N/A")

            wt2_str = f"{wt2:.2f}" if isinstance(wt2, float) else wt2
            c4_str = f"{c4:.2f}" if isinstance(c4, float) else c4

            print(f"{method:<12} {ratio}%{'':<4} {wt2_str:>12} {paper_wt2:>12} {c4_str:>12} {paper_c4:>12}")
        print()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/collect_results.py
git commit -m "feat: add results collection and comparison script"
```

---

## 执行顺序总结

| Task | 内容 | 依赖 | 预计时间 |
|------|------|------|---------|
| 1 | 项目骨架 + 模型加载 | 无 | 10 min |
| 2 | 校准数据 + 激活收集 | Task 1 | 15 min |
| 3 | Vanilla SVD | Task 1 | 10 min |
| 4 | FWSVD | Task 2 | 15 min |
| 5 | ASVD | Task 2 | 10 min |
| 6 | 白化算法 | Task 2 | 20 min |
| 7 | 层替换模块 | Task 3 | 10 min |
| 8 | Sequential Update + 所有方法编排 | Task 2-7 | 20 min |
| 9 | Perplexity 评估 | Task 1 | 10 min |
| 10 | 下游任务评估 | Task 1 | 10 min |
| 11 | 主压缩脚本 | Task 8 | 10 min |
| 12 | 主评估脚本 | Task 9-10 | 10 min |
| 13 | 端到端测试 | Task 8, 11 | 15 min |
| 14 | 实验运行脚本 | Task 11-12 | 5 min |
| 15 | 结果汇总 | Task 14 | 5 min |

**总编码时间**: ~3 小时
**实验运行时间**: 5 methods × 4 ratios × ~30 min/run ≈ 10 小时 (可并行)
