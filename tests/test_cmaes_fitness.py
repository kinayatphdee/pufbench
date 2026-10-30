"""Step 5b: CMA-ES fitness functions for the PUF models."""
import numpy as np

from pufbench import draw_cell_data, parity_features, rng_for
from pufbench.cmaes import ff_error, ff_predict, ff_weights, xor_error


# The true weights score zero training error on noise-free data, for arbiter and 3-XOR.
def test_true_model_has_zero_error():
    a = draw_cell_data("arbiter", 0, 0, n_train=3000, n_test=10)
    assert xor_error(a.Phi_train, a.r_train["clean"], a.inst.w[:, None], 1)[0] == 0.0
    x = draw_cell_data("xor3", 0, 0, n_train=3000, n_test=10)
    cand = np.concatenate([ch.w for ch in x.inst.chains])[:, None]
    assert xor_error(x.Phi_train, x.r_train["clean"], cand, 3)[0] == 0.0


# A population is scored column by column, and a random model scores about 50%.
def test_population_scoring():
    a = draw_cell_data("arbiter", 0, 0, n_train=20_000, n_test=10)
    rng = np.random.default_rng(0)
    pop = np.column_stack([a.inst.w, rng.standard_normal(65)])
    e = xor_error(a.Phi_train, a.r_train["clean"], pop, 1)
    assert e.shape == (2,) and e[0] == 0.0 and 0.4 < e[1] < 0.6


# The linear feed-forward formula reproduces the stage-by-stage race exactly.
def test_ff_linear_form_equals_race():
    d = draw_cell_data("ff", 0, 0, p=0.0, n_train=10, n_test=10)
    inst = d.inst
    C = rng_for("test", "ff-linear").integers(0, 2, (20_000, 64), dtype=np.int8)
    v, a, b = ff_weights(inst.P, inst.Q, inst.J, inst.K)
    assert np.array_equal(ff_predict(parity_features(C), v, a, b, inst.J),
                          inst.eval_challenges(C).astype(float))


# The true stage delays score zero error as a CMA-ES candidate.
def test_ff_true_candidate_has_zero_error():
    d = draw_cell_data("ff", 0, 0, p=0.0, n_train=3000, n_test=10)
    cand = np.concatenate([d.inst.P, d.inst.Q])[:, None]
    assert ff_error(d.Phi_train, d.r_train["clean"], cand, d.inst.J, d.inst.K)[0] == 0.0