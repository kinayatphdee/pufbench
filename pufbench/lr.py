"""Logistic-regression attacker (paper Sections IV-C and V-A).

* Arbiter (and feed-forward, single-hyperplane model): cross-entropy with a
  ridge term lambda = 1e-8, minimized by Newton's method with a backtracking
  line search. The loss is convex, so one run from w = 0 reaches the global
  minimum; the final gradient norm is returned as a convergence check.
* k-XOR (tanh-product model) is added in step 3b.

Responses are in {-1, +1} throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit

LAMBDA = 1e-8


@dataclass
class LRResult:
    W: np.ndarray            # (d,) for linear models, (k, d) for XOR
    train_loss: float
    grad_norm: float = float("nan")
    iterations: int = 0


# ----------------------------------------------------------- linear model --

def _linear_loss(X, t, w, lam):
    z = X @ w
    return np.mean(np.logaddexp(0.0, -t * z)) + 0.5 * lam * (w @ w)


def lr_linear(Phi: np.ndarray, r: np.ndarray, lam: float = LAMBDA, w0: np.ndarray | None = None,
              max_iter: int = 100, tol: float = 1e-10) -> LRResult:
    """Ridge-regularized LR by Newton's method; r in {-1, +1}."""
    X = np.asarray(Phi, dtype=float)
    t = np.asarray(r, dtype=float)
    y = (t + 1.0) / 2.0
    N, d = X.shape
    w = np.zeros(d) if w0 is None else np.array(w0, dtype=float)
    I = np.eye(d)
    it = 0
    for it in range(1, max_iter + 1):
        s = expit(X @ w)
        g = X.T @ (s - y) / N + lam * w
        if np.linalg.norm(g) < tol:
            break
        H = (X * (s * (1.0 - s))[:, None]).T @ X / N + lam * I
        step = np.linalg.solve(H, g)
        L0, eta = _linear_loss(X, t, w, lam), 1.0
        while _linear_loss(X, t, w - eta * step, lam) > L0 - 1e-4 * eta * (g @ step) and eta > 1e-10:
            eta *= 0.5
        w = w - eta * step
    s = expit(X @ w)
    g = X.T @ (s - y) / N + lam * w
    return LRResult(w, _linear_loss(X, t, w, lam), float(np.linalg.norm(g)), it)


def predict_linear(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    out = np.sign(np.asarray(Phi, dtype=float) @ w).astype(np.int8)
    out[out == 0] = 1
    return out


# --------------------------------------------------------------- helpers --

def fidelity(pred: np.ndarray, r_clean: np.ndarray) -> float:
    """Fraction of test challenges on which the model matches the noise-free response."""
    return float(np.mean(pred == r_clean))


def angle_deg(w_hat: np.ndarray, w: np.ndarray) -> float:
    c = w_hat @ w / (np.linalg.norm(w_hat) * np.linalg.norm(w))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def attack_lr(data, model: str, N: int) -> dict:
    """Run linear LR on the first N CRPs of a CellData under noise condition ``model``."""
    Phi, r = data.Phi_train[:N], data.r_train[model][:N]
    res = lr_linear(Phi, r)
    pred = predict_linear(data.Phi_test, res.W)
    train_err = float(np.mean(predict_linear(Phi, res.W) != r))
    return {"fidelity": fidelity(pred, data.r_test), "train_error": train_err,
            "train_loss": res.train_loss, "grad_norm": res.grad_norm, "result": res}


# --------------------------------------------------------- tanh product --
# Step 3b. Model: P(r | c) = (1 + r * prod_j tanh(w_j . Phi)) / 2, as in xor_puf_attack.m.
# All restarts are stacked into one matrix so each mini-batch needs one matrix product.

def _xor_loss_grad(X, t, Wf, R, k):
    """Batch loss per restart and gradient. X: (B, d); Wf: (R*k, d)."""
    B = X.shape[0]
    U = np.tanh(X @ Wf.T).reshape(B, R, k)
    pre = np.cumprod(np.concatenate([np.ones((B, R, 1), U.dtype), U[:, :, :-1]], axis=2), axis=2)
    suf = np.cumprod(np.concatenate([np.ones((B, R, 1), U.dtype), U[:, :, :0:-1]], axis=2), axis=2)[:, :, ::-1]
    others = pre * suf                                   # product over the other chains
    p = pre[:, :, -1] * U[:, :, -1]                      # (B, R)
    q = np.maximum(1.0 + t[:, None] * p, 1e-12)
    loss = np.mean(np.log(2.0) - np.log(q), axis=0)      # (R,)
    dz = (-t[:, None] / q / B)[:, :, None] * others * (1.0 - U ** 2)
    G = dz.reshape(B, R * k).T @ X                       # (R*k, d)
    return loss, G


def lr_xor(Phi, r, k, rng, restarts=8, step=0.01, batch=1000, max_epochs=500,
           patience=50, tol=1e-4, dtype=np.float32):
    """Tanh-product LR: mini-batch Adam, all restarts trained on the same batches."""
    X = np.asarray(Phi, dtype=dtype)
    t = np.asarray(r, dtype=dtype)
    N, d = X.shape
    Wf = (rng.standard_normal((restarts * k, d)) / np.sqrt(d)).astype(dtype)
    m = np.zeros_like(Wf); v = np.zeros_like(Wf)
    b1, b2, eps, it = 0.9, 0.999, 1e-8, 0
    bs = min(batch, N)
    best = np.full(restarts, np.inf); stall = np.zeros(restarts, dtype=int)
    for _ in range(max_epochs):
        order = rng.permutation(N)
        epoch_loss = np.zeros(restarts)
        for j in range(0, N, bs):
            idx = order[j:j + bs]
            loss, G = _xor_loss_grad(X[idx], t[idx], Wf, restarts, k)
            epoch_loss += loss * len(idx) / N
            it += 1
            m = b1 * m + (1 - b1) * G
            v = b2 * v + (1 - b2) * G * G
            Wf -= step * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps)
        improved = epoch_loss < best * (1 - tol)
        best = np.where(improved, epoch_loss, best)
        stall = np.where(improved, 0, stall + 1)
        if np.all(stall >= patience):
            break
    full_loss, _ = _xor_loss_grad(X, t, Wf, restarts, k)
    b = int(np.argmin(full_loss))
    W = Wf.reshape(restarts, k, d)[b].astype(float)
    return LRResult(W, float(full_loss[b]), iterations=it)


def predict_xor(Phi, W):
    out = np.prod(np.sign(np.asarray(Phi, dtype=float) @ np.atleast_2d(W).T), axis=1).astype(np.int8)
    out[out == 0] = 1
    return out