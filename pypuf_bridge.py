"""Bridge to pypuf for the cross-check of paper Section IV-E (step 2).

pypuf and this simulator implement the same additive delay model:
* pypuf challenges x are in {-1, +1}; our bits are c = (1 - x) / 2, so s = x
  and our parity features equal pypuf's ATF transform.
* With the same weights, responses are identical (tested).

Conventions that differ, and are therefore compared statistically rather
than exactly:
* pypuf draws every weight from N(0, 1) and sets the last (bias) weight to 0;
  our model derives the weights from stage delays (end weights N(0, 1/2),
  non-zero bias).
* pypuf's ``noisiness`` is set per arbiter and does not map to the output
  flip rate by our closed form, so it is calibrated numerically to the
  target output flip rate p.
"""
from __future__ import annotations

import numpy as np

from .sim import Arbiter, XORArbiter, CellData, parity_features
from .seeds import derive_seed, rng_for


def to_pypuf_challenges(C: np.ndarray) -> np.ndarray:
    return (1 - 2 * np.asarray(C, dtype=np.int8)).astype(np.int8)


def _pypuf_factory(family: str, n: int, seed: int):
    from pypuf.simulation import ArbiterPUF, XORArbiterPUF
    if family == "arbiter":
        return lambda nu: ArbiterPUF(n=n, seed=seed, noisiness=nu)
    if family.startswith("xor"):
        k = int(family[3:])
        return lambda nu: XORArbiterPUF(n=n, k=k, seed=seed, noisiness=nu)
    raise ValueError(f"pypuf cross-check supports arbiter and xorK, not {family!r}")


def as_our_instance(puf):
    """Our model with pypuf's weights (noise-free)."""
    W = np.atleast_2d(puf.weight_array)
    chains = [Arbiter.from_weights(w) for w in W]
    return chains[0] if len(chains) == 1 else XORArbiter(chains)


def calibrate_noisiness(factory, X: np.ndarray, p: float, iters: int = 20) -> float:
    """Find pypuf noisiness giving output flip rate p on challenges X (bisection)."""
    clean = factory(0.0).eval(X)
    lo, hi = 0.0, 2.0
    for _ in range(iters):
        nu = 0.5 * (lo + hi)
        rate = np.mean(factory(nu).eval(X) != clean)
        lo, hi = (nu, hi) if rate < p else (lo, nu)
    return 0.5 * (lo + hi)


def pypuf_cell_data(family: str, seed: int, instance: int, p: float = 0.05,
                    n_train: int = 100_000, n_test: int = 20_000, n: int = 64) -> CellData:
    """Paired data like draw_cell_data, but on a pypuf-generated instance."""
    key = ("pypuf", family, seed, instance)
    pseed = derive_seed(*key, "instance") % (2**31)
    factory = _pypuf_factory(family, n, pseed)
    C_tr = rng_for(*key, "train-challenges").integers(0, 2, (n_train, n), dtype=np.int8)
    C_te = rng_for(*key, "test-challenges").integers(0, 2, (n_test, n), dtype=np.int8)
    X_tr, X_te = to_pypuf_challenges(C_tr), to_pypuf_challenges(C_te)
    clean_puf = factory(0.0)
    clean = clean_puf.eval(X_tr).astype(np.int8)
    if p > 0:
        X_cal = to_pypuf_challenges(rng_for(*key, "calibration").integers(0, 2, (200_000, n)))
        nu = calibrate_noisiness(factory, X_cal, p)
        delay = factory(nu).eval(X_tr).astype(np.int8)
    else:
        delay = clean.copy()
    flips = rng_for(*key, "flip-mask").random(n_train) < p
    uniform = np.where(flips, -clean, clean).astype(np.int8)
    return CellData(family, seed, instance, p, as_our_instance(clean_puf), C_tr,
                    parity_features(C_tr), {"clean": clean, "delay": delay, "uniform": uniform},
                    C_te, parity_features(C_te), clean_puf.eval(X_te).astype(np.int8))
