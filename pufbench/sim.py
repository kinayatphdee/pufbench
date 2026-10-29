"""PUF simulator for the benchmark (paper Section III).

Conventions
-----------
* Challenges ``C`` are integer arrays of shape (N, n) with bits in {0, 1}.
* Responses are int8 in {-1, +1}; +1 means the delay difference is positive.
  (Attackers that need labels in {0, 1} use y = (r + 1) / 2.)
* Parity features ``Phi`` have shape (N, n + 1): Phi_i = prod_{j>=i} s_j with
  s_j = 1 - 2 c_j, and a final constant feature equal to 1.

Models
------
* Arbiter: Delta = w . Phi(c), with w built from N(0, 1) stage delays p_i, q_i.
* k-XOR: product of k independent arbiter chains (in +-1 form).
* Feed-forward (one loop): n challenge bits plus one extra stage K whose
  crossing is set by an internal arbiter after stage J, as in FF_APUF.m
  (crossed when the internal delay difference is positive).

Noise (Section III-C), always calibrated so the OUTPUT flips at rate p:
* "delay":   Gaussian noise added to every arbiter's delay difference.
* "uniform": each noise-free output flipped independently with prob. p.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .seeds import rng_for

N_STAGES = 64
FF_J, FF_K = 20, 34          # FF_APUF.m default for one loop with n = 64


# ---------------------------------------------------------------- features --

def parity_features(C: np.ndarray) -> np.ndarray:
    """Parity transform Phi(c), shape (N, n + 1), values in {-1, +1} (int8)."""
    S = (1 - 2 * np.asarray(C, dtype=np.int8)).astype(np.int8)
    Phi = np.cumprod(S[:, ::-1], axis=1, dtype=np.int8)[:, ::-1]
    return np.hstack([Phi, np.ones((S.shape[0], 1), dtype=np.int8)])


def delays_to_weights(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Weights of the additive delay model: w_1 = a_1, w_i = a_i + b_{i-1}, w_{n+1} = b_n."""
    a, b = (P - Q) / 2.0, (P + Q) / 2.0
    return np.r_[a, 0.0] + np.r_[0.0, b]


def race_delta(C: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Stage-by-stage race (ground truth for the linear model)."""
    C = np.asarray(C)
    delta = np.zeros(C.shape[0])
    for i in range(C.shape[1]):
        crossed = C[:, i] == 1
        delta = np.where(crossed, -delta + Q[i], delta + P[i])
    return delta


def _sign(x: np.ndarray) -> np.ndarray:
    out = np.sign(x).astype(np.int8)
    out[out == 0] = 1
    return out


# ------------------------------------------------------------------ models --

@dataclass
class Arbiter:
    """One arbiter chain. ``sigma`` is the delay-noise std set by calibrate()."""
    P: np.ndarray
    Q: np.ndarray
    sigma: float = 0.0

    def __post_init__(self):
        self.w = delays_to_weights(self.P, self.Q)
        self.n = len(self.P)

    @property
    def sigma_delta(self) -> float:
        """sqrt(sum_{i<=n} w_i^2): spread of Delta over uniform challenges."""
        return float(np.linalg.norm(self.w[:-1]))

    def delta(self, Phi: np.ndarray) -> np.ndarray:
        return Phi @ self.w

    def eval(self, Phi: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        """Responses in {-1, +1}. ``noise``: standard-normal draws, scaled by sigma."""
        d = self.delta(Phi)
        if noise is not None:
            d = d + self.sigma * noise[:, 0]
        return _sign(d)

    n_noise_draws = 1


@dataclass
class XORArbiter:
    """k independent chains; response = product of chain responses."""
    chains: list

    @property
    def k(self) -> int:
        return len(self.chains)

    @property
    def n_noise_draws(self) -> int:
        return self.k

    def eval(self, Phi: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        r = np.ones(Phi.shape[0], dtype=np.int8)
        for j, ch in enumerate(self.chains):
            d = ch.delta(Phi)
            if noise is not None:
                d = d + ch.sigma * noise[:, j]
            r = r * _sign(d)
        return r


@dataclass
class FeedForward:
    """One-loop feed-forward arbiter PUF (FF_APUF.m convention).

    ``P``, ``Q`` hold the delays of all n + 1 stages (1-indexed stage K is the
    extra stage). Challenge bits fill the other stages in order. The internal
    arbiter after stage J outputs 1 (stage K crossed) when its delay
    difference is positive. One common noise std ``sigma`` is used for both
    the internal and the final arbiter.
    """
    P: np.ndarray
    Q: np.ndarray
    J: int = FF_J
    K: int = FF_K
    sigma: float = 0.0

    n_noise_draws = 2

    @property
    def n(self) -> int:
        return len(self.P) - 1   # number of challenge bits

    def race(self, C: np.ndarray, noise: np.ndarray | None = None,
             force_internal: int | None = None):
        """Return (final delta, internal delta Delta_J, internal bit used)."""
        C = np.asarray(C)
        N = C.shape[0]
        delta = np.zeros(N)
        internal_delta = None
        internal_bit = None
        cbit = 0
        for stage in range(1, self.n + 2):          # stages 1 .. n+1
            i = stage - 1
            if stage == self.K:
                crossed = internal_bit == 1
            else:
                crossed = C[:, cbit] == 1
                cbit += 1
            delta = np.where(crossed, -delta + self.Q[i], delta + self.P[i])
            if stage == self.J:
                internal_delta = delta.copy()
                d_int = internal_delta if noise is None else internal_delta + self.sigma * noise[:, 1]
                internal_bit = (d_int > 0).astype(np.int8)
                if force_internal is not None:
                    internal_bit = np.full(N, force_internal, dtype=np.int8)
        return delta, internal_delta, internal_bit

    def eval_challenges(self, C: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        d, _, _ = self.race(C, noise)
        if noise is not None:
            d = d + self.sigma * noise[:, 0]
        return _sign(d)


def make_instance(family: str, rng: np.random.Generator, n: int = N_STAGES):
    """Draw one instance. family in {'arbiter', 'xor3', 'xor4', 'xor5', 'ff'} (or 'xorK')."""
    if family == "arbiter":
        return Arbiter(rng.standard_normal(n), rng.standard_normal(n))
    if family.startswith("xor"):
        k = int(family[3:])
        return XORArbiter([Arbiter(rng.standard_normal(n), rng.standard_normal(n))
                           for _ in range(k)])
    if family == "ff":
        return FeedForward(rng.standard_normal(n + 1), rng.standard_normal(n + 1))
    raise ValueError(f"unknown family {family!r}")


# ------------------------------------------------------------ calibration --

def _eval(inst, C, Phi, noise):
    if isinstance(inst, FeedForward):
        return inst.eval_challenges(C, noise)
    return inst.eval(Phi, noise)


def calibrate(inst, p: float, rng: np.random.Generator, n_cal: int = 200_000) -> None:
    """Set delay-noise std(s) so that the OUTPUT flips at rate p (Section III-C).

    Arbiter: sigma = tan(pi p) sigma_Delta.
    k-XOR:   per-chain rate p_c = (1 - (1 - 2p)^(1/k)) / 2, then the arbiter rule per chain.
    FF:      common sigma found by bisection on a fixed calibration set.
    """
    if p <= 0:
        if isinstance(inst, XORArbiter):
            for ch in inst.chains:
                ch.sigma = 0.0
        else:
            inst.sigma = 0.0
        return
    if isinstance(inst, Arbiter):
        inst.sigma = np.tan(np.pi * p) * inst.sigma_delta
        return
    if isinstance(inst, XORArbiter):
        pc = (1.0 - (1.0 - 2.0 * p) ** (1.0 / inst.k)) / 2.0
        for ch in inst.chains:
            ch.sigma = np.tan(np.pi * pc) * ch.sigma_delta
        return
    if isinstance(inst, FeedForward):
        C = rng.integers(0, 2, (n_cal, inst.n))
        Z = rng.standard_normal((n_cal, 2))
        inst.sigma = 0.0
        clean = inst.eval_challenges(C)
        lo, hi = 0.0, 4.0 * np.sqrt(inst.n)
        for _ in range(24):           # resolution 4 sqrt(n) / 2^24, far below need
            inst.sigma = 0.5 * (lo + hi)
            rate = np.mean(inst.eval_challenges(C, Z) != clean)
            lo, hi = (inst.sigma, hi) if rate < p else (lo, inst.sigma)
        inst.sigma = 0.5 * (lo + hi)
        return
    raise TypeError(type(inst))


# ------------------------------------------------------- paired cell data --

@dataclass
class CellData:
    """Everything one (family, seed, instance) needs, shared by all attackers.

    Training challenges are one stream; a budget N uses the first N rows.
    ``r_train[m]`` holds responses under noise condition m in
    {'clean', 'delay', 'uniform'}. The test set is noise-free.
    """
    family: str
    seed: int
    instance: int
    p: float
    inst: object
    C_train: np.ndarray
    Phi_train: np.ndarray
    r_train: dict
    C_test: np.ndarray
    Phi_test: np.ndarray
    r_test: np.ndarray


def draw_cell_data(family: str, seed: int, instance: int, p: float = 0.05,
                   n_train: int = 100_000, n_test: int = 20_000,
                   n: int = N_STAGES) -> CellData:
    """Deterministic, paired data for one instance (Section IV-B)."""
    key = (family, seed, instance)
    inst = make_instance(family, rng_for(*key, "delays"), n)
    calibrate(inst, p, rng_for(*key, "calibration", p))

    C_tr = rng_for(*key, "train-challenges").integers(0, 2, (n_train, n), dtype=np.int8)
    C_te = rng_for(*key, "test-challenges").integers(0, 2, (n_test, n), dtype=np.int8)
    Phi_tr, Phi_te = parity_features(C_tr), parity_features(C_te)

    Z = rng_for(*key, "delay-noise").standard_normal((n_train, inst.n_noise_draws))
    flips = rng_for(*key, "flip-mask").random(n_train) < p

    clean = _eval(inst, C_tr, Phi_tr, None)
    delay = _eval(inst, C_tr, Phi_tr, Z) if p > 0 else clean.copy()
    uniform = np.where(flips, -clean, clean).astype(np.int8)
    test = _eval(inst, C_te, Phi_te, None)

    return CellData(family, seed, instance, p, inst, C_tr, Phi_tr,
                    {"clean": clean, "delay": delay, "uniform": uniform},
                    C_te, Phi_te, test)
