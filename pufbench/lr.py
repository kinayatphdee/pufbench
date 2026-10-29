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