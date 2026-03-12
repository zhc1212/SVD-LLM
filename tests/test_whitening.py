import pytest
import torch
from src.compress.whitening import (
    compute_whitening_matrix,
    compute_whitening_matrix_from_covariance,
    compress_linear_whitening,
    compress_linear_whitening_from_covariance,
)
from src.model.loader import compute_rank


def test_whitening_matrix_shape():
    """测试白化矩阵的形状和性质"""
    X = torch.randn(500, 64)
    S, S_inv = compute_whitening_matrix(X)
    assert S.shape == (64, 64)
    assert S_inv.shape == (64, 64)
    I_approx = S @ S_inv
    assert torch.allclose(I_approx, torch.eye(64), atol=1e-4)


def test_whitening_from_covariance():
    """从预计算协方差和从原始激活应该给出相同结果"""
    torch.manual_seed(42)
    X = torch.randn(500, 64)
    C = X.T @ X / X.shape[0]

    S1, S1_inv = compute_whitening_matrix(X)
    S2, S2_inv = compute_whitening_matrix_from_covariance(C)

    assert torch.allclose(S1, S2, atol=1e-4)
    assert torch.allclose(S1_inv, S2_inv, atol=1e-4)


def test_whitening_decorrelates():
    """白化后的激活应该近似不相关"""
    torch.manual_seed(42)
    A_mat = torch.randn(64, 64)
    X = torch.randn(1000, 64) @ A_mat

    S, S_inv = compute_whitening_matrix(X)
    X_w = X @ S_inv

    C_w = X_w.T @ X_w / X_w.shape[0]
    assert torch.allclose(C_w, torch.eye(64), atol=0.1)


def test_compress_whitening_shape():
    """测试白化压缩的输出形状"""
    W = torch.randn(256, 512)
    X = torch.randn(200, 512)
    r = compute_rank(256, 512, 0.3)
    A, B = compress_linear_whitening(W, X, r)
    assert A.shape == (256, r)
    assert B.shape == (r, 512)


def test_compress_from_covariance_matches():
    """从协方差和从激活压缩应该给出相同结果"""
    torch.manual_seed(42)
    W = torch.randn(64, 128)
    X = torch.randn(200, 128)
    C = X.T @ X / X.shape[0]
    r = compute_rank(64, 128, 0.3)

    A1, B1 = compress_linear_whitening(W, X, r)
    A2, B2 = compress_linear_whitening_from_covariance(W, C, r)

    assert torch.allclose(A1, A2, atol=1e-4)
    assert torch.allclose(B1, B2, atol=1e-4)


def test_whitening_better_than_vanilla():
    """白化 SVD 在考虑激活分布时应优于 Vanilla SVD"""
    torch.manual_seed(42)
    d, n = 128, 256
    W = torch.randn(d, n)

    scales = torch.ones(n)
    scales[:50] = 10.0
    scales[50:] = 0.1
    X = torch.randn(500, n) * scales.unsqueeze(0)

    rank = compute_rank(d, n, 0.5)

    # Vanilla SVD
    U, S_vals, Vh = torch.linalg.svd(W, full_matrices=False)
    A_v = U[:, :rank] * S_vals[:rank].unsqueeze(0)
    B_v = Vh[:rank, :]
    Y_orig = X @ W.T
    Y_vanilla = X @ (A_v @ B_v).T
    err_vanilla = torch.norm(Y_orig - Y_vanilla) / torch.norm(Y_orig)

    # Whitening SVD
    A_w, B_w = compress_linear_whitening(W, X, rank)
    Y_whiten = X @ (A_w @ B_w).T
    err_whiten = torch.norm(Y_orig - Y_whiten) / torch.norm(Y_orig)

    assert err_whiten < err_vanilla
