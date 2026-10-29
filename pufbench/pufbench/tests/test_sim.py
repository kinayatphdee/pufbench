"""Tests for the simulator. Each test checks a claim made in paper Section III.

Run with:  pytest -q
"""
import numpy as np
import pytest

from pufbench import (Arbiter, FeedForward, XORArbiter, calibrate, delays_to_weights,
                      draw_cell_data, make_instance, parity_features, race_delta, rng_for)

N_STAGES = 64


def _chal(rng, N, n=N_STAGES):
    return rng.integers(0, 2, (N, n), dtype=np.int8)


# III-B: the unrolled recursion equals the dot product w . Phi(c), exactly.
def test_linear_model_equals_race():
    rng = rng_for("test", "race")
    P, Q = rng.standard_normal(N_STAGES), rng.standard_normal(N_STAGES)
    C = _chal(rng, 20_000)
    lin = parity_features(C) @ delays_to_weights(P, Q)
    assert np.allclose(lin, race_delta(C, P, Q), atol=1e-10)


# III-B: c -> Phi(c) is a bijection, s_i = Phi_i * Phi_{i+1}.
def test_parity_map_is_invertible():
    rng = rng_for("test", "bijection")
    C = _chal(rng, 1000)
    Phi = parity_features(C).astype(int)
    S = Phi[:, :-1] * Phi[:, 1:]
    assert np.array_equal((1 - S) // 2, C)


# III-B: E[Phi Phi^T] = I for uniform challenges.
def test_features_are_uncorrelated():
    rng = rng_for("test", "features")
    Phi = parity_features(_chal(rng, 200_000)).astype(float)
    M = Phi.T @ Phi / len(Phi)
    assert np.max(np.abs(M - np.eye(N_STAGES + 1))) < 0.02


# III-B: weight distributions; E[sigma_Delta^2] = 63.5 for n = 64.
def test_weight_statistics():
    rng = rng_for("test", "weights")
    W = np.array([make_instance("arbiter", rng).w for _ in range(4000)])
    assert abs(W[:, 0].var() - 0.5) < 0.05 and abs(W[:, -1].var() - 0.5) < 0.05
    assert abs(W[:, 1:-1].var() - 1.0) < 0.03
    assert abs(np.mean(np.sum(W[:, :-1] ** 2, axis=1)) - 63.5) < 0.6


# Foundations Prediction A: chip-to-chip bias spread is about 3.5%.
def test_bias_spread():
    rng = rng_for("test", "bias")
    Phi = parity_features(_chal(rng, 20_000)).astype(float)
    W = np.array([make_instance("arbiter", rng).w for _ in range(2000)]).T
    bias = np.mean(Phi @ W > 0, axis=0)
    assert 0.031 < bias.std() < 0.040


# III-B: k-XOR response is the product of chain responses.
def test_xor_is_product_of_chains():
    rng = rng_for("test", "xor")
    inst = make_instance("xor3", rng)
    Phi = parity_features(_chal(rng, 5000))
    prod = np.prod([np.sign(Phi @ ch.w) for ch in inst.chains], axis=0)
    assert np.array_equal(inst.eval(Phi), prod.astype(np.int8))


# III-B: internal delay difference factors as Delta_J = Phi_{J+1} (v . Phi).
def test_ff_internal_factorization():
    rng = rng_for("test", "ff-factor")
    inst = make_instance("ff", rng)
    C = _chal(rng, 3000)
    Phi = parity_features(C).astype(float)
    _, dJ, _ = inst.race(C)
    target = dJ * Phi[:, inst.J]          # Phi_{J+1} in 1-indexed notation
    v, *_ = np.linalg.lstsq(Phi, target, rcond=None)
    assert np.max(np.abs(Phi @ v - target)) < 1e-9


# III-B: for each value of the internal bit, the final delay difference is linear in Phi.
@pytest.mark.parametrize("bit", [0, 1])
def test_ff_selects_between_two_hyperplanes(bit):
    rng = rng_for("test", "ff-select", bit)
    inst = make_instance("ff", rng)
    C = _chal(rng, 3000)
    Phi = parity_features(C).astype(float)
    d, _, _ = inst.race(C, force_internal=bit)
    alpha, *_ = np.linalg.lstsq(Phi, d, rcond=None)
    assert np.max(np.abs(Phi @ alpha - d)) < 1e-9


# III-C: per-challenge flip probability Phi_N(-|Delta| / sigma_n) concentrates near the boundary.
def test_delay_noise_concentrates_near_boundary():
    data = draw_cell_data("arbiter", 0, 0, p=0.05, n_train=100_000, n_test=1000)
    margin = np.abs(data.inst.delta(data.Phi_train))
    flip = data.r_train["delay"] != data.r_train["clean"]
    q = np.quantile(margin, [0.25, 0.75])
    assert flip[margin < q[0]].mean() > 0.15 and flip[margin > q[1]].mean() < 0.001


# III-C: output-matched calibration hits the target flip rate for every family and model.
@pytest.mark.parametrize("family", ["arbiter", "xor3", "xor5", "ff"])
@pytest.mark.parametrize("model", ["delay", "uniform"])
def test_output_flip_rate_matches_target(family, model):
    rates = []
    for inst_id in range(4):
        d = draw_cell_data(family, 0, inst_id, p=0.05, n_train=100_000, n_test=1000)
        rates.append(np.mean(d.r_train[model] != d.r_train["clean"]))
    assert abs(np.mean(rates) - 0.05) < 0.004


# IV-B: pairing and determinism.
def test_cell_data_is_deterministic_and_paired():
    a = draw_cell_data("xor3", 1, 2, n_train=5000, n_test=500)
    b = draw_cell_data("xor3", 1, 2, n_train=5000, n_test=500)
    assert np.array_equal(a.C_train, b.C_train) and np.array_equal(a.r_train["delay"], b.r_train["delay"])
    # all noise conditions share the same challenges and the same clean responses
    assert len({len(v) for v in a.r_train.values()}) == 1
    c = draw_cell_data("xor3", 1, 3, n_train=5000, n_test=500)
    assert not np.array_equal(a.C_train, c.C_train)


# IV-B: budgets are prefixes of one stream (drawing more CRPs does not change the first ones).
def test_budgets_are_prefixes():
    small = draw_cell_data("arbiter", 3, 0, n_train=2000, n_test=100)
    large = draw_cell_data("arbiter", 3, 0, n_train=100_000, n_test=100)
    assert np.array_equal(small.C_train, large.C_train[:2000])
