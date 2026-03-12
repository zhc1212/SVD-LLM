import torch


def compute_whitening_matrix(activations, eps=1e-6):
    """计算白化矩阵 S (通过 Cholesky 分解)

    给定激活 X ∈ R^{N×n}:
    1. C = X^T X / N  (协方差矩阵, n×n)
    2. Cholesky: C = L L^T  (L 是下三角)
    3. S = L^T (上三角), 满足 C = S^T S

    Args:
        activations: tensor (N, n)
        eps: 正则化项

    Returns:
        S: tensor (n, n) — 上三角白化矩阵
        S_inv: tensor (n, n) — S 的逆
    """
    X = activations.float()
    N, n = X.shape

    C = X.T @ X / N  # (n, n)
    C += eps * torch.eye(n, device=C.device, dtype=C.dtype)

    L = torch.linalg.cholesky(C)  # 下三角
    S = L.T  # 上三角

    S_inv = torch.linalg.inv(S)

    return S, S_inv


def compress_linear_whitening(weight, activations, rank, eps=1e-6):
    """SVD-LLM 白化压缩

    1. 计算白化矩阵 S (Cholesky)
    2. WS^T — 在白化空间中的权重
    3. SVD 截断
    4. 恢复: W_hat = U_r Σ_r V_r^T @ S^{-T}
    5. 分解为 A @ B

    Args:
        weight: tensor (d, n)
        activations: tensor (N, n)
        rank: int
        eps: Cholesky 正则化

    Returns:
        A: tensor (d, r)
        B: tensor (r, n)
    """
    W = weight.float()
    d, n = W.shape

    S, S_inv = compute_whitening_matrix(activations, eps=eps)

    WS = W @ S.T  # (d, n)

    U, Sigma, Vh = torch.linalg.svd(WS, full_matrices=False)
    U_r = U[:, :rank]
    Sigma_r = Sigma[:rank]
    Vh_r = Vh[:rank, :]

    A = U_r * Sigma_r.unsqueeze(0)   # (d, r)
    B = Vh_r @ S_inv.T               # (r, n)

    return A, B
