"""Deterministic seeding (paper Section IV-B).

Every random quantity is drawn from a generator whose seed is derived from a
tuple of keys, e.g. ("arbiter", seed, instance, "delays"). The same keys always
give the same stream, on any machine, so runs are reproducible and every
attacker and noise model sees identical instances and challenges.
"""
import hashlib
import numpy as np


def derive_seed(*keys) -> int:
    """Map an ordered tuple of keys to a 64-bit integer seed."""
    text = "|".join(str(k) for k in keys).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def rng_for(*keys) -> np.random.Generator:
    """A NumPy generator seeded from the keys."""
    return np.random.default_rng(derive_seed(*keys))
