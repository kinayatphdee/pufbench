"""Step 4c: the MLP entry point (same interface as attack_lr)."""
from pufbench import draw_cell_data
from pufbench.lr import attack_lr
from pufbench.mlp import attack_mlp


# attack_mlp runs on every family and returns the same fields as attack_lr.
def test_attack_mlp_runs_on_every_family():
    for fam in ["arbiter", "xor3", "ff"]:
        d = draw_cell_data(fam, 0, 0, n_train=2000, n_test=1000)
        out = attack_mlp(d, "clean", 2000)
        assert set(out) == set(attack_lr(d, "clean", 2000))
        assert 0.0 <= out["fidelity"] <= 1.0


# Runs are reproducible without passing a generator (seeded from the cell keys).
def test_attack_mlp_reproducible():
    d = draw_cell_data("arbiter", 0, 1, n_train=2000, n_test=1000)
    assert attack_mlp(d, "delay", 2000)["fidelity"] == attack_mlp(d, "delay", 2000)["fidelity"]