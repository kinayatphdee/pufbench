"""Analysis of benchmark results (paper Sections III-E and IV-D).

Step 8a: per-instance CRP budgets N_A and their summary per cell.
* N_A of one instance: by default interpolated between the two grid budgets
  that bracket A, linearly in log(1 - F) against log N (step 8b); the plain
  grid value (smallest budget with fidelity >= A) is reported alongside.
  Infinity if A is never reached ("never broken within the grid").
* Per cell (family, attacker, noise): median N_A over instances, a bootstrap
  95% interval for the median, and the fraction of instances never reaching A.

Command line:  python -m pufbench.analysis results/A1.csv [more.csv ...]
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

THRESHOLDS = (0.90, 0.95, 0.99)
KEY = ("family", "attacker", "noise", "p", "seed", "instance", "N")


def load_results(paths) -> list[dict]:
    """Read and merge results files; a run appearing in several files is kept once."""
    rows, seen = [], set()
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                k = tuple(r[c] for c in KEY)
                if k in seen:
                    continue
                seen.add(k)
                r["N"] = int(r["N"]); r["seed"] = int(r["seed"]); r["instance"] = int(r["instance"])
                r["fidelity"] = float(r["fidelity"])
                rows.append(r)
    return rows


def _crossing(pts, A: float, method: str) -> float:
    """N_A for one instance from its (N, fidelity) points.

    grid:   smallest grid budget with fidelity >= A.
    interp: between the bracketing budgets, log(1 - F) is interpolated linearly
            in log N (the learning curve is a straight line on log-log axes,
            Section V-A). If A is already reached at the smallest budget, that
            budget is returned (no extrapolation below the grid).
    Both give infinity if A is never reached.
    """
    pts = sorted(pts)
    for j, (N, f) in enumerate(pts):
        if f >= A:
            if method == "grid" or j == 0:
                return float(N)
            (N0, f0), (N1, f1) = pts[j - 1], pts[j]
            e0, e1, e = 1.0 - f0, max(1.0 - f1, 1e-12), 1.0 - A
            t = (np.log(e) - np.log(e0)) / (np.log(e1) - np.log(e0))
            return float(np.exp(np.log(N0) + t * (np.log(N1) - np.log(N0))))
    return np.inf


def instance_budgets(rows, A: float, method: str = "interp") -> dict:
    """{(family, attacker, noise): {(seed, instance): N_A}}, N_A = inf if never reached."""
    curves = defaultdict(lambda: defaultdict(list))
    for r in rows:
        curves[(r["family"], r["attacker"], r["noise"])][(r["seed"], r["instance"])].append(
            (r["N"], r["fidelity"]))
    out = {}
    for cell, insts in curves.items():
        out[cell] = {}
        for inst, pts in insts.items():
            out[cell][inst] = _crossing(pts, A, method)
    return out


def bootstrap_median(values, n_boot: int = 10_000, seed: int = 0):
    """Median and bootstrap 95% interval (percentile method); infinities allowed."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    # no interpolation: interval ends are actual values (grid budgets or infinity)
    lo = np.percentile(meds, 2.5, method="inverted_cdf")
    hi = np.percentile(meds, 97.5, method="inverted_cdf")
    return float(np.median(v)), float(lo), float(hi)


def summarize(rows, thresholds=THRESHOLDS, method: str = "interp") -> list[dict]:
    """One summary row per (cell, threshold)."""
    table = []
    for A in thresholds:
        grid = instance_budgets(rows, A, "grid")
        for cell, insts in sorted(instance_budgets(rows, A, method).items()):
            vals = list(insts.values())
            med, lo, hi = bootstrap_median(vals)
            smallest = min(r["N"] for r in rows if (r["family"], r["attacker"], r["noise"]) == cell)
            table.append({"family": cell[0], "attacker": cell[1], "noise": cell[2], "A": A,
                          "instances": len(vals), "median_NA": med, "ci_low": lo, "ci_high": hi,
                          "median_NA_grid": float(np.median(list(grid[cell].values()))),
                          "at_smallest_budget": float(np.mean([v == smallest for v in vals])),
                          "never_reached": float(np.mean(np.isinf(vals)))})
    return table


def _fmt(x):
    return "never" if np.isinf(x) else f"{int(x):,}"


def main(argv=None):
    import sys
    paths = (argv if argv is not None else sys.argv[1:])
    rows = load_results(paths)
    print(f"{len(rows)} runs from {len(paths)} file(s)\n")
    print("median N_A interpolated (grid value in brackets); '<=smallest' = share already at the first budget")
    print(f"{'family':8s} {'attacker':8s} {'noise':8s} {'A':>5s} {'n':>3s} "
          f"{'median N_A':>11s} {'(grid)':>9s} {'95% interval':>22s} {'<=smallest':>10s} {'never':>6s}")
    for t in summarize(rows):
        print(f"{t['family']:8s} {t['attacker']:8s} {t['noise']:8s} {t['A']:5.2f} {t['instances']:3d} "
              f"{_fmt(t['median_NA']):>11s} {'(' + _fmt(t['median_NA_grid']) + ')':>9s} "
              f"{_fmt(t['ci_low']) + ' - ' + _fmt(t['ci_high']):>22s} "
              f"{t['at_smallest_budget']:10.0%} {t['never_reached']:6.0%}")


if __name__ == "__main__":
    main()