"""Step 5c: the CMA-ES entry point (same interface as attack_lr and attack_mlp)."""
from pufbench import draw_cell_data
from pufbench.cmaes import attack_cma
from pufbench.lr import attack_lr


# Same fields as attack_lr, plus the number of fitness evaluations used.
def test_fields_and_evaluation_cap():
    d = draw_cell_data("arbiter", 0, 0, n_train=1000, n_test=1000)
    out = attack_cma(d, "clean", 1000, eval_per_dim=20)
    assert set(out) == set(attack_lr(d, "clean", 1000)) | {"evaluations"}
    assert out["evaluations"] <= 3 * (20 * 65 + 16)          # cap per restart, plus one generation


# CMA-ES models a noise-free arbiter PUF from 2,000 CRPs.
def test_breaks_arbiter_noise_free():
    d = draw_cell_data("arbiter", 0, 0, n_train=2000)
    assert attack_cma(d, "clean", 2000)["fidelity"] > 0.95


# With the stage-delay model, CMA-ES models a noise-free feed-forward PUF from 2,000 CRPs.
def test_breaks_feed_forward_noise_free():
    d = draw_cell_data("ff", 0, 0, n_train=2000)
    assert attack_cma(d, "clean", 2000)["fidelity"] > 0.95


# Runs are reproducible without passing a generator (seeded from the cell keys).
def test_reproducible():
    d = draw_cell_data("xor3", 0, 0, n_train=1000, n_test=1000)
    a = attack_cma(d, "delay", 1000, eval_per_dim=10)
    b = attack_cma(d, "delay", 1000, eval_per_dim=10)
    assert a["fidelity"] == b["fidelity"] and a["evaluations"] == b["evaluations"]