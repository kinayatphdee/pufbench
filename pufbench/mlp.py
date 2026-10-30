"""MLP attacker (paper Sections IV-C and 10), written to mirror mlp_attacks.m.

Step 4a: the network, its loss and its gradient by backpropagation.
* Layers [d, h, 2h, h, 1] with h = max(2^(k-1), 16); tanh hidden units,
  linear output (the logit).
* Initial weights N(0, 1) / sqrt(fan-in), biases zero.
* Loss: binary cross-entropy with logits, in the numerically stable form
  mean(max(z, 0) - z y + log(1 + exp(-|z|))), with labels y in {0, 1}.
* Backpropagation (Section 10.2): delta at the output is (sigma(z) - y) / N,
  and delta_l = (delta_{l+1} W_{l+1}^T) * (1 - a_l^2) through each tanh layer.
"""
from __future__ import annotations

import numpy as np


def layer_sizes(d: int, k: int = 1) -> list[int]:
    h = max(2 ** (k - 1), 16)
    return [d, h, 2 * h, h, 1]


def init_mlp(sizes: list[int], rng: np.random.Generator) -> dict:
    """Weights W[l] of shape (fan_in, fan_out), as in MATLAB's net.Wt."""
    W = [rng.standard_normal((sizes[l], sizes[l + 1])) / np.sqrt(sizes[l]) for l in range(len(sizes) - 1)]
    b = [np.zeros(sizes[l + 1]) for l in range(len(sizes) - 1)]
    return {"W": W, "b": b}


def n_params(net: dict) -> int:
    return sum(w.size for w in net["W"]) + sum(b.size for b in net["b"])


def forward(X: np.ndarray, net: dict):
    """Return the list of activations a[0..L] (a[L] is the logit, shape (N,))."""
    a = [X]
    L = len(net["W"])
    for l in range(L):
        z = a[-1] @ net["W"][l] + net["b"][l]
        a.append(np.tanh(z) if l < L - 1 else z)
    a[-1] = a[-1][:, 0]
    return a


def loss_grad(X: np.ndarray, y01: np.ndarray, net: dict):
    """Cross-entropy loss and its gradient (lists gW, gb) by backpropagation."""
    a = forward(X, net)
    z = a[-1]
    N = X.shape[0]
    loss = float(np.mean(np.maximum(z, 0) - z * y01 + np.log1p(np.exp(-np.abs(z)))))
    L = len(net["W"])
    gW, gb = [None] * L, [None] * L
    delta = ((1.0 / (1.0 + np.exp(-z)) - y01) / N)[:, None]
    for l in range(L - 1, -1, -1):
        gW[l] = a[l].T @ delta
        gb[l] = delta.sum(axis=0)
        if l > 0:
            delta = (delta @ net["W"][l].T) * (1.0 - a[l] ** 2)
    return loss, gW, gb


def predict(X: np.ndarray, net: dict) -> np.ndarray:
    out = np.sign(forward(X, net)[-1]).astype(np.int8)
    out[out == 0] = 1
    return out


# ------------------------------------------------------------- training --
# Step 4b, mirroring train_mlp in mlp_attacks.m: Adam (step 0.002, batch 1,000),
# at most 200 epochs, stop after 40 epochs without a relative improvement of 1e-4
# in the epoch loss. Like the MATLAB version, the returned loss is the best
# epoch loss and the returned network is the one after the last epoch.

def train_mlp(X, y01, net, rng, step=0.002, batch=1000, epochs=200, patience=40, tol=1e-4):
    N = X.shape[0]
    bs = min(batch, N)
    mW = [np.zeros_like(w) for w in net["W"]]; vW = [np.zeros_like(w) for w in net["W"]]
    mb = [np.zeros_like(b) for b in net["b"]]; vb = [np.zeros_like(b) for b in net["b"]]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    best, stall = np.inf, 0
    for _ in range(epochs):
        order = rng.permutation(N)
        tot = 0.0
        for j in range(0, N, bs):
            idx = order[j:j + bs]
            L, gW, gb = loss_grad(X[idx], y01[idx], net)
            t += 1
            for l in range(len(net["W"])):
                mW[l] = b1 * mW[l] + (1 - b1) * gW[l]; vW[l] = b2 * vW[l] + (1 - b2) * gW[l] ** 2
                net["W"][l] -= step * (mW[l] / (1 - b1 ** t)) / (np.sqrt(vW[l] / (1 - b2 ** t)) + eps)
                mb[l] = b1 * mb[l] + (1 - b1) * gb[l]; vb[l] = b2 * vb[l] + (1 - b2) * gb[l] ** 2
                net["b"][l] -= step * (mb[l] / (1 - b1 ** t)) / (np.sqrt(vb[l] / (1 - b2 ** t)) + eps)
            tot += L * len(idx) / N
        if tot < best * (1 - tol):
            best, stall = tot, 0
        else:
            stall += 1
        if stall >= patience:
            break
    return net, best


def fit_mlp(Phi, r, k, rng, restarts=2, **train_kw):
    """Train ``restarts`` networks on (Phi, r in +-1) and keep the lowest-loss one."""
    X = np.asarray(Phi, dtype=float)
    y01 = (np.asarray(r) > 0).astype(float)
    best_net, best_loss = None, np.inf
    for _ in range(restarts):
        net, L = train_mlp(X, y01, init_mlp(layer_sizes(X.shape[1], k), rng), rng, **train_kw)
        if L < best_loss:
            best_net, best_loss = net, L
    return best_net, best_loss


# ---------------------------------------------------------- entry point --
# Step 4c: same interface as attack_lr, so the runner treats both attackers alike.

def attack_mlp(data, model: str, N: int, rng: np.random.Generator | None = None) -> dict:
    """Run the MLP on the first N CRPs of a CellData under noise condition ``model``.

    The hidden width uses k for k-XOR PUFs and k = 1 otherwise; for every family
    in the grid this gives layers [16, 32, 16]. Restarts are seeded from
    (family, seed, instance, model, N) unless ``rng`` is given.
    """
    from .seeds import rng_for
    Phi, r = data.Phi_train[:N], data.r_train[model][:N]
    k = int(data.family[3:]) if data.family.startswith("xor") else 1
    if rng is None:
        rng = rng_for(data.family, data.seed, data.instance, "mlp-init", model, N)
    net, loss = fit_mlp(Phi, r, k, rng)
    return {"fidelity": float(np.mean(predict(np.asarray(data.Phi_test, float), net) == data.r_test)),
            "train_error": float(np.mean(predict(np.asarray(Phi, float), net) != r)),
            "train_loss": float(loss), "grad_norm": float("nan"), "result": net}