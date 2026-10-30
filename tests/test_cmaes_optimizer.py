"""Step 5a: the CMA-ES optimizer on standard test functions (as for cma_vs_lr.m)."""
import numpy as np

from pufbench.cmaes import cma_es

D = 10


def sphere(X):
    return np.sum(X ** 2, axis=0)


def rosenbrock(X):
    return np.sum(100 * (X[1:] - X[:-1] ** 2) ** 2 + (1 - X[:-1]) ** 2, axis=0)


def ellipsoid(X):
    """Condition number 1e6: only solvable fast if the covariance adapts."""
    scales = 1e6 ** (np.arange(D) / (D - 1))
    return np.sum(scales[:, None] * X ** 2, axis=0)


def test_sphere():
    _, f, _ = cma_es(sphere, np.full(D, 3.0), 1.0, 20_000, np.random.default_rng(0), stopf=1e-10)
    assert f <= 1e-10


def test_rosenbrock():
    x, f, _ = cma_es(rosenbrock, np.zeros(D), 0.5, 100_000, np.random.default_rng(1), stopf=1e-10)
    assert f <= 1e-10 and np.allclose(x, 1.0, atol=1e-4)


def test_ill_conditioned_ellipsoid():
    _, f, _ = cma_es(ellipsoid, np.full(D, 1.0), 1.0, 60_000, np.random.default_rng(2), stopf=1e-10)
    assert f <= 1e-10


def test_defaults_match_section_11():
    # lambda = 4 + floor(3 ln d) = 16 and mu = 8 for d = 65.
    seen = []
    cma_es(lambda X: seen.append(X.shape[1]) or np.zeros(X.shape[1]),
           np.zeros(65), 0.3, 16, np.random.default_rng(0))
    assert seen == [16]


def test_deterministic():
    a = cma_es(sphere, np.full(D, 3.0), 1.0, 2000, np.random.default_rng(5))
    b = cma_es(sphere, np.full(D, 3.0), 1.0, 2000, np.random.default_rng(5))
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]