"""Step 8a: per-instance budgets N_A and their summary (Sections III-E, IV-D)."""
import csv

import numpy as np

from pufbench.analysis import bootstrap_median, instance_budgets, load_results, summarize
from pufbench.runner import FIELDS


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _row(inst, N, fid, noise="delay"):
    return {"family": "arbiter", "attacker": "lr", "noise": noise, "p": 0.05, "seed": 0,
            "instance": inst, "N": N, "fidelity": fid}


# N_A is the first budget reaching A; infinity if never reached.
def test_instance_budget_definition(tmp_path):
    p = tmp_path / "r.csv"
    _write(p, [_row(0, 500, 0.85), _row(0, 1000, 0.91), _row(0, 2000, 0.96),
               _row(1, 500, 0.80), _row(1, 1000, 0.89), _row(1, 2000, 0.89)])
    b90 = instance_budgets(load_results([p]), 0.90, "grid")[("arbiter", "lr", "delay")]
    assert b90[(0, 0)] == 1000 and np.isinf(b90[(0, 1)])
    b95 = instance_budgets(load_results([p]), 0.95, "grid")[("arbiter", "lr", "delay")]
    assert b95[(0, 0)] == 2000


# Merging files keeps each run once.
def test_merging_removes_duplicates(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, [_row(0, 500, 0.9), _row(0, 1000, 0.95)])
    _write(b, [_row(0, 1000, 0.95), _row(1, 500, 0.8)])
    assert len(load_results([a, b])) == 3


# The bootstrap interval contains the median, and is a single point for identical values.
def test_bootstrap_median():
    med, lo, hi = bootstrap_median([1000] * 10)
    assert med == lo == hi == 1000
    med, lo, hi = bootstrap_median([500, 1000, 1000, 2000, 5000])
    assert lo <= med <= hi and med == 1000


# The summary reports the fraction of instances never reaching A.
def test_summary_never_reached(tmp_path):
    p = tmp_path / "r.csv"
    _write(p, [_row(i, 1000, 0.92 if i < 3 else 0.85) for i in range(4)])
    s = [t for t in summarize(load_results([p]), thresholds=[0.90])][0]
    assert s["instances"] == 4 and s["never_reached"] == 0.25 and s["median_NA"] == 1000


# Step 8b: on an exact learning curve 1 - F = c / sqrt(N), interpolation recovers the true N_A.
def test_interpolation_recovers_true_budget(tmp_path):
    from pufbench.analysis import _crossing
    c = 3.0
    pts = [(N, 1 - c / np.sqrt(N)) for N in [500, 1000, 2000, 5000, 10_000]]
    assert abs(_crossing(pts, 0.90, "interp") - (c / 0.10) ** 2) < 1e-6        # 900
    assert abs(_crossing(pts, 0.95, "interp") - (c / 0.05) ** 2) < 1e-6        # 3,600
    assert _crossing(pts, 0.90, "grid") == 1000


# Reached at the first budget: no extrapolation below the grid. Never reached: infinity.
def test_interpolation_edges():
    from pufbench.analysis import _crossing
    assert _crossing([(500, 0.97), (1000, 0.99)], 0.95, "interp") == 500
    assert np.isinf(_crossing([(500, 0.80), (1000, 0.85)], 0.90, "interp"))