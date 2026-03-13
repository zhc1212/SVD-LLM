import torch


def compute_whitening_matrix_from_covariance(covariance, eps=1e-6):
    """从预计算的协方差矩阵计算白化矩阵 S

    C = L L^T  (Cholesky)
    S = L^T, C = S^T S

    Args:
        covariance: tensor (n, n) — 协方差矩阵 X^T X / N
        eps: 正则化项

    Returns:
        S: tensor (n, n) — 上三角白化矩阵
        S_inv: tensor (n, n) — S 的逆
    """
    C = covariance.float()
    n = C.shape[0]

    C = C + eps * torch.eye(n, device=C.device, dtype=C.dtype)

    # 检查是否需要额外正则化（秩亏情况）
    try:
        L = torch.linalg.cholesky(C)
    except torch.linalg.LinAlgError:
        diag_mean = C.diagonal().mean().item()
        C = C + diag_mean * torch.eye(n, device=C.device, dtype=C.dtype)
        L = torch.linalg.cholesky(C)

    S = L.T
    S_inv = torch.linalg.inv(S)

    return S, S_inv


def compute_whitening_matrix(activations, eps=1e-6):
    """从原始激活计算白化矩阵 S (兼容旧接口)

    给定激活 X ∈ R^{N×n}:
    1. C = X^T X / N  (协方差矩阵, n×n)
    2. Cholesky → S

    Args:
        activations: tensor (N, n)
        eps: 正则化项

    Returns:
        S: tensor (n, n)
        S_inv: tensor (n, n)
    """
    X = activations.float()
    N = X.shape[0]
    C = X.T @ X / N
    return compute_whitening_matrix_from_covariance(C, eps=eps)


def compress_linear_whitening_from_covariance(weight, covariance, rank, eps=1e-6):
    """SVD-LLM 白化压缩（从预计算协方差）

    1. 从协方差计算白化矩阵: C = LL^T, S = L^T
    2. WL = W @ S^T — 在白化空间中的权重
    3. SVD(WL) = UΣV^T, 截断保留前 r 个
    4. 对称分配奇异值:
       W'_u = U_r · Σ_r^{1/2}   (d × r)
       W'_v = Σ_r^{1/2} · V_r^T · L^{-1}   (r × n)
    5. W ≈ W'_u @ W'_v

    Args:
        weight: tensor (d, n)
        covariance: tensor (n, n) — X^T X / N
        rank: int
        eps: Cholesky 正则化

    Returns:
        A: tensor (d, r) — W'_u = U_r · Σ_r^{1/2}
        B: tensor (r, n) — W'_v = Σ_r^{1/2} · V_r^T · L^{-1}
    """
    W = weight.float()

    S, S_inv = compute_whitening_matrix_from_covariance(covariance.to(W.device), eps=eps)

    WS = W @ S.T  # W @ L (since S = L^T, S.T = L)

    U, Sigma, Vh = torch.linalg.svd(WS, full_matrices=False)
    U_r = U[:, :rank]
    Sigma_r = Sigma[:rank]
    Vh_r = Vh[:rank, :]

    # 对称分配: Σ^{1/2} 到两侧
    sqrt_sigma = torch.sqrt(Sigma_r)
    A = U_r * sqrt_sigma.unsqueeze(0)            # W'_u = U_r · Σ_r^{1/2}
    B = sqrt_sigma.unsqueeze(1) * (Vh_r @ S_inv.T)  # W'_v = Σ_r^{1/2} · V_r^T · L^{-1}

    return A, B


def compress_linear_whitening(weight, activations, rank, eps=1e-6):
    """SVD-LLM 白化压缩（兼容旧接口，从原始激活）

    Args:
        weight: tensor (d, n)
        activations: tensor (N, n)
        rank: int
        eps: Cholesky 正则化

    Returns:
        A: tensor (d, r)
        B: tensor (r, n)
    """
    X = activations.float()
    N = X.shape[0]
    C = X.T @ X / N
    return compress_linear_whitening_from_covariance(weight, C, rank, eps=eps)
