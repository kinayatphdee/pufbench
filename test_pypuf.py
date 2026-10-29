"""Step 2: cross-check the simulator against pypuf (paper Section IV-E).

Skipped automatically if pypuf is not installed (pip install pypuf).
"""
import numpy as np
import pytest

pypuf = pytest.importorskip("pypuf")
from pypuf.simulation import ArbiterPUF, XORArbiterPUF            # noqa: E402

from pufbench import parity_features, rng_for                         # noqa: E402
from pufbench.pypuf_bridge import (as_our_instance, pypuf_cell_data,   # noqa: E402
                                   to_pypuf_challenges)


def _challenges(n=64, N=20_000):
    return rng_for("test", "pypuf").integers(0, 2, (N, n), dtype=np.int8)


# Same weights -> identical responses: our parity features equal pypuf's ATF transform.
@pytest.mark.parametrize("k", [1, 3, 5])
def test_identical_responses_with_same_weights(k):
    C = _challenges()
    puf = ArbiterPUF(n=64, seed=7) if k == 1 else XORArbiterPUF(n=64, k=k, seed=7)
    ours = as_our_instance(puf).eval(parity_features(C))
    assert np.array_equal(ours, puf.eval(to_pypuf_challenges(C)).astype(np.int8))


# Documented convention difference: pypuf weights are all N(0, 1) and the bias weight is 0.
def test_pypuf_weight_convention():
    W = np.array([ArbiterPUF(n=64, seed=s).weight_array[0] for s in range(1000)])
    assert np.all(W[:, -1] == 0)
    assert abs(W[:, :-1].std() - 1.0) < 0.03 and abs(W[:, 0].std() - 1.0) < 0.06


# Calibrated pypuf noise hits the target output flip rate, like our own calibration.
@pytest.mark.parametrize("family", ["arbiter", "xor3"])
def test_pypuf_noise_calibrated_to_target(family):
    d = pypuf_cell_data(family, 0, 0, p=0.05, n_train=100_000, n_test=1000)
    assert abs(np.mean(d.r_train["delay"] != d.r_train["clean"]) - 0.05) < 0.005
    assert abs(np.mean(d.r_train["uniform"] != d.r_train["clean"]) - 0.05) < 0.005
