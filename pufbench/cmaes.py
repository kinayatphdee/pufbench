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
    """Arbiter (k = 1) and k-XOR. Xc: (k*d, P); candidate = k blocks of d chain weights."""
    Phi = np.asarray(Phi, dtype=float)
    N, d = Phi.shape
    P = Xc.shape[1]
    W = Xc.reshape(k, d, P)                                   # chain j = rows j*d .. (j+1)*d-1
    pred = np.prod(_sign(np.einsum("nd,kdp->nkp", Phi, W)), axis=1)
    return np.mean(pred != np.asarray(r)[:, None], axis=0)


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


def ff_error(Phi, r, Xc, J, K):
    """Feed-forward PUF. Xc: (2(n+1), P); candidate = [stage delays p (n+1), q (n+1)]."""
    Phi = np.asarray(Phi, dtype=float)
    r = np.asarray(r)
    m = Xc.shape[0] // 2
    errs = np.empty(Xc.shape[1])
    for c in range(Xc.shape[1]):
        v, a, b = ff_weights(Xc[:m, c], Xc[m:, c], J, K)
        errs[c] = np.mean(ff_predict(Phi, v, a, b, J) != r)
    return errs