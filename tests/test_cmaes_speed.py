"""Step 6e: the fast CMA-ES fitness gives the same values as the plain definitions."""
import numpy as np

from pufbench import draw_cell_data
from pufbench.cmaes import ff_error, ff_predict, ff_weights, xor_error


def _xor_error_reference(Phi, r, Xc, k):
    """Plain definition: product of chain signs, one candidate at a time."""
    Phi = np.asarray(Phi, dtype=float); d = Phi.shape[1]
    out = []
    for c in range(Xc.shape[1]):
        W = Xc[:, c].reshape(k, d)
        s = np.sign(Phi @ W.T); s[s == 0] = 1
        out.append(np.mean(np.prod(s, axis=1) != r))
    return np.array(out)


def _ff_error_reference(Phi, r, Xc, J, K):
    m = Xc.shape[0] // 2
    return np.array([np.mean(ff_predict(Phi, *ff_weights(Xc[:m, c], Xc[m:, c], J, K), J) != r)
                     for c in range(Xc.shape[1])])


def test_fast_xor_fitness_equals_reference():
    rng = np.random.default_rng(0)
    for fam, k in [("arbiter", 1), ("xor3", 3), ("xor5", 5)]:
        d = draw_cell_data(fam, 0, 0, n_train=2000, n_test=10)
        Xc = rng.standard_normal((65 * k, 20))
        Xc[:, 0] = np.concatenate([ch.w for ch in d.inst.chains]) if k > 1 else d.inst.w
        assert np.allclose(xor_error(d.Phi_train, d.r_train["delay"], Xc, k),
                           _xor_error_reference(d.Phi_train, d.r_train["delay"], Xc, k))


def test_fast_ff_fitness_equals_reference():
    rng = np.random.default_rng(1)
    d = draw_cell_data("ff", 0, 0, n_train=2000, n_test=10)
    Xc = rng.standard_normal((130, 20))
    Xc[:, 0] = np.concatenate([d.inst.P, d.inst.Q])
    assert np.allclose(ff_error(d.Phi_train, d.r_train["delay"], Xc, 20, 34),
                       _ff_error_reference(d.Phi_train, d.r_train["delay"], Xc, 20, 34))