"""Step 6a: running one experiment and storing results that survive interruption."""
import csv

from pufbench import draw_cell_data
from pufbench.runner import FIELDS, ResultsFile, run_one


def test_run_one_returns_a_complete_row():
    d = draw_cell_data("arbiter", 0, 0, n_train=2000, n_test=1000)
    row = run_one(d, "lr", "delay", 2000)
    assert list(row) == FIELDS
    assert row["p"] == 0.05 and row["N"] == 2000 and 0 <= row["fidelity"] <= 1


def test_results_file_appends_and_resumes(tmp_path):
    path = tmp_path / "results.csv"
    d = draw_cell_data("arbiter", 0, 0, n_train=2000, n_test=1000)
    res = ResultsFile(str(path))
    assert not res.has("arbiter", "lr", "clean", 0, 0, 1000)
    res.append(run_one(d, "lr", "clean", 1000))
    res.append(run_one(d, "lr", "uniform", 1000))
    # a fresh object, as after a restart, sees both rows as done
    again = ResultsFile(str(path))
    assert again.has("arbiter", "lr", "clean", 0, 0, 1000)
    assert again.has("arbiter", "lr", "uniform", 0, 0, 1000)
    assert not again.has("arbiter", "lr", "delay", 0, 0, 1000)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["noise"] == "clean" and float(rows[1]["p"]) == 0.05