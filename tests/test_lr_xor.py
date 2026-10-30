"""Step 3b: tanh-product LR on k-XOR PUFs (paper Section IV-C)."""
import numpy as np

from pufbench import draw_cell_data
from pufbench.lr import fidelity, lr_xor, predict_xor


# The prediction rule reproduces the XOR PUF exactly when given the true chain weights.
def test_predict_xor_with_true_weights_is_exact():
    d = draw_cell_data("xor3", 0, 0, n_train=1000, n_test=5000)
    W = np.array([ch.w for ch in d.inst.chains])
    assert fidelity(predict_xor(d.Phi_test, W), d.r_test) == 1.0


# LR breaks a noise-free 64-stage 3-XOR PUF with 5,000 CRPs.
def test_breaks_3xor_noise_free():
    d = draw_cell_data("xor3", 0, 0, n_train=5000)
    res = lr_xor(d.Phi_train, d.r_train["clean"], 3, np.random.default_rng(0))
    assert fidelity(predict_xor(d.Phi_test, res.W), d.r_test) > 0.95


# Same data and same seed give the same model (reproducibility, Section IV-B).
def test_lr_xor_is_deterministic():
    d = draw_cell_data("xor3", 0, 1, n_train=2000, n_test=100)
    a = lr_xor(d.Phi_train, d.r_train["clean"], 3, np.random.default_rng(5), max_epochs=20)
    b = lr_xor(d.Phi_train, d.r_train["clean"], 3, np.random.default_rng(5), max_epochs=20)
    assert np.array_equal(a.W, b.W)