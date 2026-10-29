# pufbench

Benchmark code for *Modeling Attacks under a CRP Budget: A Reproducible Benchmark*.

Implements the models of paper Section III exactly: arbiter, k-XOR and one-loop
feed-forward PUFs (n = 64), delay noise and uniform label flips calibrated to the
same output flip rate, and paired, deterministic data generation (Section IV-B).

## Status

| Step | Content | State |
|---|---|---|
| 1 | Simulator, noise models, calibration, paired data + tests | done |
| 2 | pypuf cross-check of the simulator | next |
| 3 | LR attacker | |
| 4 | MLP attacker (PyTorch) | |
| 5 | CMA-ES attacker | |
| 6 | Runner and grid | |
| 7 | MATLAB cross-check | |
| 8 | Analysis and figures | |

## Setup

In a GitHub Codespace, Google Colab, or any machine with Python 3.10+:

```bash
pip install -r requirements.txt
pytest -q          # about 1 minute; all tests should pass
```

## Quick use

```python
from pufbench import draw_cell_data

d = draw_cell_data("xor3", seed=0, instance=0, p=0.05)
d.Phi_train[:5000]          # parity features of the first 5,000 training CRPs
d.r_train["delay"][:5000]   # their responses under delay noise (+-1)
d.r_train["uniform"][:5000] # same challenges, uniform label flips
d.Phi_test, d.r_test        # 20,000 noise-free test CRPs
```

Families: `arbiter`, `xor3`, `xor4`, `xor5`, `ff`. A budget N uses the first N
rows of the training stream, so all budgets, attackers and noise models are paired.

## What the tests check

| Test | Paper claim |
|---|---|
| linear model equals race | III-B: Delta = w . Phi(c) exactly |
| parity map invertible | III-B: c -> Phi(c) is a bijection |
| features uncorrelated | III-B: E[Phi Phi^T] = I |
| weight statistics | III-B: w_1, w_65 ~ N(0, 1/2), others N(0, 1); E[sigma_Delta^2] = 63.5 |
| bias spread | about 3.5% chip-to-chip |
| XOR is product of chains | III-B |
| FF internal factorization | III-B: Delta_J = Phi_{J+1} (v . Phi) |
| FF selects between two hyperplanes | III-B |
| delay noise near boundary | III-C: flips concentrate on small margins |
| output flip rate matches target | III-C: output-matched calibration, all families, both models |
| deterministic and paired; budgets are prefixes | IV-B |
