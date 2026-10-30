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