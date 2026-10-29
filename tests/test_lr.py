"""Step 3a: LR on the arbiter PUF. Each test checks a prediction of paper Section V-A."""
import numpy as np

from pufbench import draw_cell_data
from pufbench.lr import angle_deg, attack_lr, lr_linear


def _angles(model, N, n_inst=8):
    out = []
    for i in range(n_inst):
        d = draw_cell_data("arbiter", 0, i, p=0.05)
        out.append(angle_deg(attack_lr(d, model, N)["result"].W, d.inst.w))
    return np.array(out)


# V-A: the loss is convex, so different starting points reach the same model.
def test_convex_one_run_suffices():
    d = draw_cell_data("arbiter", 0, 0, p=0.05)
    Phi, r = d.Phi_train[:5000], d.r_train["delay"][:5000]
    w_zero = lr_linear(Phi, r).W
    w_rand = lr_linear(Phi, r, w0=np.random.default_rng(1).standard_normal(65)).W
    assert angle_deg(w_zero, w_rand) < 1e-4


# V-E prediction 1: reference constants c = theta^2 N at p = 0.05.
def test_reference_constant_delay_noise():
    th = np.radians(_angles("delay", 20_000))
    assert 12.0 < np.mean(th ** 2) * 20_000 < 18.0          # predicted 14.9


def test_reference_constant_uniform_flips():
    th = np.radians(_angles("uniform", 20_000))
    assert 31.0 < np.mean(th ** 2) * 20_000 < 46.0          # predicted 38.2


# V-A: angle falls as 1/sqrt(N), so 4x the CRPs halves it (log-log slope -1/2).
def test_slope_minus_half():
    ratio = _angles("delay", 5000).mean() / _angles("delay", 20_000).mean()
    assert 1.7 < ratio < 2.3


# III-E: measured fidelity equals 1 - theta/pi; the run converged.
def test_fidelity_equals_angle_and_converges():
    d = draw_cell_data("arbiter", 0, 0, p=0.05)
    out = attack_lr(d, "delay", 20_000)
    th = angle_deg(out["result"].W, d.inst.w)
    assert abs(out["fidelity"] - (1 - th / 180)) < 0.003
    assert out["grad_norm"] < 1e-6