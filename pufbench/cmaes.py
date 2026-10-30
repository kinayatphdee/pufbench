"""CMA-ES attacker (paper Sections IV-C, V-D and Foundations Section 11).

Step 5a: the optimizer, a line-by-line port of cma_es in cma_vs_lr.m, the
canonical (mu/mu_w, lambda)-CMA-ES (Hansen, purecmaes). ``fitfun`` maps a
(d, P) matrix of candidates to P fitness values; lower is better.
"""
from __future__ import annotations

import numpy as np


def cma_es(fitfun, x0, sigma0, max_eval, rng, stopf=None):
    """Minimize fitfun; return (xbest, fbest, history of (evaluations, fbest))."""
    N = len(x0)
    xmean = np.asarray(x0, dtype=float).reshape(N, 1)
    sigma = float(sigma0)
    lam = 4 + int(np.floor(3 * np.log(N)))
    mu = lam / 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, int(np.floor(mu)) + 1))
    mu = int(np.floor(mu))
    w = (w / w.sum()).reshape(mu, 1)
    mueff = float(w.sum() ** 2 / (w ** 2).sum())
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
    pc = np.zeros((N, 1)); ps = np.zeros((N, 1))
    B = np.eye(N); D = np.ones((N, 1)); C = np.eye(N); invsqrtC = np.eye(N)
    eigeneval = 0
    chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
    counteval, xbest, fbest, hist = 0, xmean.ravel().copy(), np.inf, []
    while counteval < max_eval:
        Z = rng.standard_normal((N, lam))
        Xc = xmean + sigma * (B @ (D * Z))
        f = np.asarray(fitfun(Xc), dtype=float)
        counteval += lam
        idx = np.argsort(f, kind="stable")
        f = f[idx]
        if f[0] < fbest:
            fbest, xbest = float(f[0]), Xc[:, idx[0]].copy()
        hist.append((counteval, fbest))
        xold = xmean
        xmean = Xc[:, idx[:mu]] @ w
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (xmean - xold)) / sigma
        hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN
                     < 1.4 + 2 / (N + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        artmp = (Xc[:, idx[:mu]] - xold) / sigma
        C = ((1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (artmp * w.T) @ artmp.T)
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if counteval - eigeneval > lam / (c1 + cmu) / N / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            Dd, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(Dd, 0)).reshape(N, 1)
            invsqrtC = B @ np.diag(1 / D.ravel()) @ B.T
        if stopf is not None and fbest <= stopf:
            break
        if sigma > 1e14 or sigma < 1e-15:
            break
    return xbest, fbest, np.array(hist)


# --------------------------------------------------------- PUF fitness --
# Step 5b. Fitness = 0/1 training error of the hard model, for a whole
# population at once (Xc has one candidate per column), as xor_train_error in
# cma_vs_lr.m. Responses r are in {-1, +1}.

def _sign(Z):
    S = np.sign(Z)
    S[S == 0] = 1
    return S


def xor_error(Phi, r, Xc, k):
    """Arbiter (k = 1) and k-XOR. Xc: (k*d, P); candidate = k blocks of d chain weights.

    The XOR of k signs is -1 exactly when an odd number of chain outputs are
    negative, so the population is scored by one float32 matrix product and a
    parity count (step 6e speed-up; same values as the plain sign product).
    """
    Phi = np.asarray(Phi, dtype=np.float32)
    N, d = Phi.shape
    P = Xc.shape[1]
    W = np.asarray(Xc, dtype=np.float32).reshape(k, d, P).transpose(1, 0, 2).reshape(d, k * P)
    neg = (Phi @ W < 0).reshape(N, k, P)
    pred_neg = (neg.sum(axis=1) % 2).astype(bool)
    return np.mean(pred_neg != (np.asarray(r) < 0)[:, None], axis=0)


def ff_weights(P_st, Q_st, J, K):
    """Exact linear form of a one-loop feed-forward PUF from its n+1 stage delays.

    Returns (v, alpha, beta) over the challenge features Phi (length n+1):
      internal delay  Delta_J = Phi_{J+1} * (v . Phi)          (Section III-B)
      final delay     alpha . Phi if stage K is straight, beta . Phi if crossed.
    Stage K is crossed when Delta_J > 0 (FF_APUF.m convention).
    """
    from .sim import delays_to_weights
    n = len(P_st) - 1
    wf = delays_to_weights(P_st, Q_st)                        # n+2 weights over the n+1 stages
    k0 = K - 1                                                 # 0-based stage index of K
    alpha = np.empty(n + 1); beta = np.empty(n + 1)
    alpha[:k0] = wf[:k0];  beta[:k0] = -wf[:k0]
    alpha[k0] = wf[k0] + wf[k0 + 1];  beta[k0] = -wf[k0] + wf[k0 + 1]
    alpha[k0 + 1:] = wf[k0 + 2:];  beta[k0 + 1:] = wf[k0 + 2:]
    wp = delays_to_weights(P_st[:J], Q_st[:J])                # J+1 prefix weights
    v = np.zeros(n + 1)
    v[:J] = wp[:J]
    v[J] = wp[J]
    return v, alpha, beta


def ff_predict(Phi, v, alpha, beta, J):
    Phi = np.asarray(Phi, dtype=float)
    internal = Phi[:, J] * (Phi @ v)
    return _sign(np.where(internal > 0, Phi @ beta, Phi @ alpha))


def _ff_weights_population(Xc, J, K):
    """ff_weights for every column of Xc at once; returns (v, alpha, beta), each (n+1, P)."""
    m = Xc.shape[0] // 2
    Pd, Qd = Xc[:m], Xc[m:]
    a, b = (Pd - Qd) / 2.0, (Pd + Qd) / 2.0
    z = np.zeros((1, Xc.shape[1]))
    wf = np.vstack([a, z]) + np.vstack([z, b])                # (n+2, P)
    k0 = K - 1
    alpha = np.vstack([wf[:k0], wf[k0:k0 + 1] + wf[k0 + 1:k0 + 2], wf[k0 + 2:]])
    beta = np.vstack([-wf[:k0], -wf[k0:k0 + 1] + wf[k0 + 1:k0 + 2], wf[k0 + 2:]])
    ap, bp = a[:J], b[:J]
    wp = np.vstack([ap, z]) + np.vstack([z, bp])               # (J+1, P)
    v = np.zeros_like(alpha)
    v[:J + 1] = wp
    return v, alpha, beta


def ff_error(Phi, r, Xc, J, K):
    """Feed-forward PUF. Xc: (2(n+1), P); candidate = [stage delays p (n+1), q (n+1)].

    Whole population at once: three float32 matrix products (step 6e speed-up).
    """
    Phi = np.asarray(Phi, dtype=np.float32)
    v, a, b = (x.astype(np.float32) for x in _ff_weights_population(np.asarray(Xc, float), J, K))
    internal = Phi[:, J:J + 1] * (Phi @ v)
    final = np.where(internal > 0, Phi @ b, Phi @ a)
    return np.mean((final < 0) != (np.asarray(r) < 0)[:, None], axis=0)


# ---------------------------------------------------------- entry point --
# Step 5c, following cma_vs_lr.m: sigma0 = 0.3, at most 400*dim evaluations per
# restart, 3 restarts, stop as soon as the training error reaches 0. Each restart
# starts at N(0, 1) / sqrt(d). The fitness is the 0/1 training error; the restart
# with the lowest training error is kept.

def fit_cma(Phi, r, family, rng, J=None, K=None, sigma0=0.3, eval_per_dim=400,
            restarts=3, stopf=0.0):
    Phi = np.asarray(Phi, dtype=float)
    d = Phi.shape[1]
    if family == "ff":
        dim = 2 * d                                           # n+1 stage delays p and q
        fit = lambda Xc: ff_error(Phi, r, Xc, J, K)
    else:
        k = int(family[3:]) if family.startswith("xor") else 1
        dim = k * d
        fit = lambda Xc: xor_error(Phi, r, Xc, k)
    best_x, best_err, used = None, np.inf, 0
    for _ in range(restarts):
        x0 = rng.standard_normal(dim) / np.sqrt(d)
        xb, fb, hist = cma_es(fit, x0, sigma0, eval_per_dim * dim, rng, stopf)
        used += int(hist[-1, 0])
        if fb < best_err:
            best_x, best_err = xb, fb
        if best_err <= stopf:
            break
    return best_x, float(best_err), used


def _cma_predict(Phi, x, family, J=None, K=None):
    Phi = np.asarray(Phi, dtype=float)
    d = Phi.shape[1]
    if family == "ff":
        v, a, b = ff_weights(x[:d], x[d:], J, K)
        return ff_predict(Phi, v, a, b, J)
    k = int(family[3:]) if family.startswith("xor") else 1
    return np.prod(_sign(Phi @ x.reshape(k, d).T), axis=1)


def attack_cma(data, model: str, N: int, rng: np.random.Generator | None = None, **kw) -> dict:
    """Run CMA-ES on the first N CRPs of a CellData under noise condition ``model``.

    Same fields as attack_lr and attack_mlp, plus the number of fitness
    evaluations used. Restarts are seeded from the cell keys unless ``rng`` is given.
    """
    from .seeds import rng_for
    Phi, r = data.Phi_train[:N], data.r_train[model][:N]
    if rng is None:
        rng = rng_for(data.family, data.seed, data.instance, "cma-init", model, N)
    J = getattr(data.inst, "J", None); K = getattr(data.inst, "K", None)
    x, err, used = fit_cma(Phi, r, data.family, rng, J, K, **kw)
    pred = _cma_predict(data.Phi_test, x, data.family, J, K)
    return {"fidelity": float(np.mean(pred == data.r_test)), "train_error": err,
            "train_loss": err, "grad_norm": float("nan"), "result": x, "evaluations": used}