"""pufbench: modeling attacks on delay-based PUFs under a CRP budget.

Step 1 provides the simulator (sim.py) and deterministic seeding (seeds.py).
"""
from .seeds import derive_seed, rng_for
from .sim import (
    parity_features, delays_to_weights, race_delta,
    Arbiter, XORArbiter, FeedForward, make_instance,
    calibrate, CellData, draw_cell_data,
)
__all__ = [
    "derive_seed", "rng_for", "parity_features", "delays_to_weights", "race_delta",
    "Arbiter", "XORArbiter", "FeedForward", "make_instance", "calibrate",
    "CellData", "draw_cell_data",
]
