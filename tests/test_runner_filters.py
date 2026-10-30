"""Step 6g: selecting part of the paper design with --seeds and the other filters."""
import subprocess
import sys

from pufbench.runner import count_plan, filter_plan, paper_plan


# Phase A (seed 0) and phase B (seeds 1-4) together are exactly the full design.
def test_phases_partition_the_design():
    full = set(paper_plan())
    a = set(filter_plan(paper_plan(), seeds=[0]))
    b = set(filter_plan(paper_plan(), seeds=[1, 2, 3, 4]))
    assert a | b == full and not (a & b)
    assert len(a) == 1485 and len(b) == 4800


# Phase B contains no CMA-ES runs: its design uses seed 0 only.
def test_phase_b_has_no_cma():
    assert ("arbiter", "cma") not in count_plan(filter_plan(paper_plan(), seeds=[1, 2, 3, 4]))


# Filters never add runs outside the design (CMA-ES on 3-XOR keeps its three budgets).
def test_filters_stay_inside_the_design():
    sel = filter_plan(paper_plan(["xor3"], ["cma"]), budgets=[500, 2000])
    assert {p[5] for p in sel} == {2000}


def test_count_with_seed_filter(tmp_path):
    r = subprocess.run([sys.executable, "-m", "pufbench.runner", "--design", "paper", "--seeds", "0",
                        "--count", "--out", str(tmp_path / "none.csv")],
                       capture_output=True, text=True, check=True)
    assert "total: 1485 planned, 1485 still to run" in r.stdout