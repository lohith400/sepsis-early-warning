"""Takagi-Sugeno ANFIS: Gaussian membership curves, product rules, linear consequents.

Premise parameters (centres, widths) are searched by PSO; consequents are solved by
weighted ridge least squares (Jang's hybrid learning).
"""
import numpy as np


def decode(v: np.ndarray, n_rules: int, n_inputs: int):
    """Split a particle into centres C (R x K) and widths S (R x K)."""
    R, K = n_rules, n_inputs
    return v[: R * K].reshape(R, K), np.exp(v[R * K:]).reshape(R, K)


def phi(X: np.ndarray, C: np.ndarray, S: np.ndarray):
    """Return (normalised rule strengths N x R, design matrix N x R(K+1))."""
    d = (X[:, None, :] - C[None]) / S[None]
    lw = -0.5 * (d * d).sum(2)
    lw -= lw.max(1, keepdims=True)                       # numerically stable
    w = np.exp(lw)
    w /= w.sum(1, keepdims=True) + 1e-12
    Xa = np.concatenate([np.ones((len(X), 1)), X], 1)
    return w, (w[:, :, None] * Xa[:, None, :]).reshape(len(X), -1)


def lse(P: np.ndarray, y: np.ndarray, sw: np.ndarray, lam: float = 1.0) -> np.ndarray:
    A = P.T @ (P * sw[:, None])
    A[np.diag_indices_from(A)] += lam
    return np.linalg.solve(A, P.T @ (sw * y))


def class_weights(y: np.ndarray) -> np.ndarray:
    p = y.mean()
    return np.where(y == 1, 0.5 / p, 0.5 / (1 - p))


def score(X: np.ndarray, C, S, theta, chunk: int = 40000) -> np.ndarray:
    out = []
    for s in range(0, len(X), chunk):
        _, P = phi(X[s: s + chunk], C, S)
        out.append(P @ theta)
    return np.concatenate(out)
