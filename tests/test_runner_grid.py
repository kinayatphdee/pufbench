"""Step 6b: the grid loop and its command line."""
import csv
import subprocess
import sys

from pufbench.runner import run_grid

TINY = dict(families=["arbiter"], attackers=["lr"], noises=["clean", "delay", "uniform"],
            budgets=[500, 1000], seeds=[0], instances=[0, 1], log=lambda s: None)


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# Every combination runs once; a second call finds nothing left to do.
def test_grid_runs_each_combination_once(tmp_path):
    out = str(tmp_path / "r.csv")
    assert run_grid(out, **TINY) == 1 * 1 * 3 * 2 * 1 * 2        # 12 runs
    assert run_grid(out, **TINY) == 0
    assert len(_rows(out)) == 12


# Extending the grid later only runs what is new.
def test_grid_extends_without_repeating(tmp_path):
    out = str(tmp_path / "r.csv")
    run_grid(out, **TINY)
    more = dict(TINY, budgets=[500, 1000, 2000])
    assert run_grid(out, **more) == 3 * 2                           # only N = 2000 is new
    assert len(_rows(out)) == 18


# The command line runs a chosen part of the grid.
def test_command_line(tmp_path):
    out = str(tmp_path / "cli.csv")
    cmd = [sys.executable, "-m", "pufbench.runner", "--out", out, "--families", "arbiter",
           "--attackers", "lr", "--noises", "delay", "--budgets", "500", "--seeds", "0",
           "--instances", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "done: 1 new runs" in r.stdout and len(_rows(out)) == 1