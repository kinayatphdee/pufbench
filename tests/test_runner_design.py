"""Step 6f: the paper's design encoded in the runner (Sections IV-A, IV-C)."""
import subprocess
import sys

from pufbench.runner import count_plan, paper_plan, run_plan


def test_paper_design_size():
    plan = list(paper_plan())
    c = count_plan(plan)
    for fam in ["arbiter", "xor3", "xor4", "xor5", "ff"]:
        assert c[(fam, "lr")] == c[(fam, "mlp")] == 25 * 3 * 8          # 600 each
    assert c[("arbiter", "cma")] == c[("ff", "cma")] == 5 * 3 * 8         # 120 each
    assert c[("xor3", "cma")] == 5 * 3 * 3                                 # 45
    assert ("xor4", "cma") not in c and ("xor5", "cma") not in c
    assert len(plan) == 6285


def test_plan_has_no_duplicates():
    plan = list(paper_plan())
    assert len(plan) == len(set(plan))


def test_run_plan_runs_once(tmp_path):
    out = str(tmp_path / "r.csv")
    plan = [p for p in paper_plan(["arbiter"], ["lr"]) if p[3] == 0 and p[4] < 2 and p[5] <= 1000]
    assert len(plan) == 2 * 3 * 2
    assert run_plan(out, plan, log=lambda s: None) == 12
    assert run_plan(out, plan, log=lambda s: None) == 0


def test_count_option(tmp_path):
    r = subprocess.run([sys.executable, "-m", "pufbench.runner", "--design", "paper", "--count",
                        "--out", str(tmp_path / "none.csv")], capture_output=True, text=True, check=True)
    assert "total: 6285 planned, 6285 still to run" in r.stdout