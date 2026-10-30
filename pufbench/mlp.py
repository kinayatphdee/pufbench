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