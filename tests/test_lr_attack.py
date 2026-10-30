"""Step 3c: one LR entry point for every PUF family (paper Sections IV-C and V-E)."""
import numpy as np

from pufbench import draw_cell_data
from pufbench.lr import attack_lr


# attack_lr runs on every family and returns the same fields.
def test_attack_lr_runs_on_every_family():
    for fam in ["arbiter", "xor3", "ff"]:
        d = draw_cell_data(fam, 0, 0, n_train=2000, n_test=1000)
        out = attack_lr(d, "clean", 2000)
        assert set(out) >= {"fidelity", "train_error", "train_loss", "grad_norm"}
        assert 0.0 <= out["fidelity"] <= 1.0


# XOR runs are reproducible without passing a generator (seeded from the cell keys).
def test_attack_lr_xor_reproducible():
    d = draw_cell_data("xor3", 0, 0, n_train=2000, n_test=1000)
    assert attack_lr(d, "clean", 2000)["fidelity"] == attack_lr(d, "clean", 2000)["fidelity"]


# V-E prediction 7: single-hyperplane LR does not reach high fidelity on the
# feed-forward PUF, and more CRPs barely help.
def test_lr_fails_on_feed_forward():
    d = draw_cell_data("ff", 0, 0)
    f20 = attack_lr(d, "clean", 20_000)["fidelity"]
    f100 = attack_lr(d, "clean", 100_000)["fidelity"]
    assert f20 < 0.90 and f100 < 0.90
    assert f100 - f20 < 0.02