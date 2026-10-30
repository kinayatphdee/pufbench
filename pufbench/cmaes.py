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