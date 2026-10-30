"""Runner for the benchmark grid (paper Sections IV-A, IV-B and IV-E).

Step 6a: run one experiment and store its result as one row of a CSV file.
The file is appended to after every run and flushed immediately, so an
interrupted session loses at most the run in progress; on restart, rows that
are already in the file are skipped.
"""
from __future__ import annotations

import csv
import os
import time

from .cmaes import attack_cma
from .lr import attack_lr
from .mlp import attack_mlp

ATTACKERS = {"lr": attack_lr, "mlp": attack_mlp, "cma": attack_cma}

# Noise conditions of Section IV-A: name -> output flip rate p.
NOISE_CONDITIONS = {"clean": 0.0, "delay": 0.05, "uniform": 0.05}

KEY_FIELDS = ["family", "attacker", "noise", "p", "seed", "instance", "N"]
FIELDS = KEY_FIELDS + ["fidelity", "train_error", "train_loss", "grad_norm",
                       "evaluations", "seconds"]


def run_one(data, attacker: str, noise: str, N: int) -> dict:
    """Run one attacker on the first N CRPs of ``data`` under one noise condition."""
    t0 = time.time()
    out = ATTACKERS[attacker](data, noise, N)
    return {"family": data.family, "attacker": attacker, "noise": noise,
            "p": NOISE_CONDITIONS[noise], "seed": data.seed, "instance": data.instance, "N": N,
            "fidelity": out["fidelity"], "train_error": out["train_error"],
            "train_loss": out["train_loss"], "grad_norm": out["grad_norm"],
            "evaluations": out.get("evaluations", ""), "seconds": round(time.time() - t0, 3)}


def _key(row: dict) -> tuple:
    return (row["family"], row["attacker"], row["noise"], float(row["p"]),
            int(row["seed"]), int(row["instance"]), int(row["N"]))


class ResultsFile:
    """Append-only CSV of results, with a set of the keys already done."""

    def __init__(self, path: str):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    self.done.add(_key(row))

    def has(self, family, attacker, noise, seed, instance, N) -> bool:
        return (family, attacker, noise, NOISE_CONDITIONS[noise], seed, instance, N) in self.done

    def append(self, row: dict) -> None:
        new_file = not os.path.exists(self.path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        self.done.add(_key(row))


# -------------------------------------------------------------- the grid --
# Step 6b. The defaults are the grid of Section IV-A. Each (family, seed,
# instance) is drawn once and shared by every attacker, noise condition and
# budget (Section IV-B); finished runs are skipped.

FAMILIES = ["arbiter", "xor3", "xor4", "xor5", "ff"]
ATTACKER_NAMES = ["lr", "mlp", "cma"]
NOISES = ["clean", "delay", "uniform"]
BUDGETS = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
SEEDS = [0, 1, 2, 3, 4]
INSTANCES = [0, 1, 2, 3, 4]


def run_grid(results_path, families=FAMILIES, attackers=ATTACKER_NAMES, noises=NOISES,
             budgets=BUDGETS, seeds=SEEDS, instances=INSTANCES, log=print):
    """Run every missing combination; return the number of new runs."""
    from .sim import draw_cell_data
    res = ResultsFile(results_path)
    n_train = max(budgets)
    new = 0
    for family in families:
        for seed in seeds:
            for inst in instances:
                todo = [(a, m, N) for a in attackers for m in noises for N in budgets
                        if not res.has(family, a, m, seed, inst, N)]
                if not todo:
                    continue
                data = draw_cell_data(family, seed, inst, p=0.05, n_train=max(n_train, 100_000))
                for a, m, N in todo:
                    row = run_one(data, a, m, N)
                    res.append(row)
                    new += 1
                    log(f"{family} seed={seed} inst={inst} {a:4s} {m:7s} N={N:6d}  "
                        f"fidelity={row['fidelity']:.4f}  ({row['seconds']:.1f} s)")
    return new


def main(argv=None):
    """Command line: python -m pufbench.runner --families xor3 --attackers lr ..."""
    import argparse
    ap = argparse.ArgumentParser(description="Run (part of) the benchmark grid.")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--attackers", nargs="+", default=ATTACKER_NAMES, choices=ATTACKER_NAMES)
    ap.add_argument("--noises", nargs="+", default=NOISES, choices=NOISES)
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--instances", nargs="+", type=int, default=INSTANCES)
    a = ap.parse_args(argv)
    n = run_grid(a.out, a.families, a.attackers, a.noises, a.budgets, a.seeds, a.instances)
    print(f"done: {n} new runs written to {a.out}")


if __name__ == "__main__":
    main()