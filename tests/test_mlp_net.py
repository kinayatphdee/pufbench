"""Step 4a: the MLP network and backpropagation (paper Section 10.1-10.2)."""
import numpy as np

from pufbench.mlp import forward, init_mlp, layer_sizes, loss_grad, n_params, predict


def _net_and_data(seed=0, N=50):
    rng = np.random.default_rng(seed)
    net = init_mlp(layer_sizes(65, 3), rng)
    X = rng.choice([-1.0, 1.0], size=(N, 65))
    y = rng.integers(0, 2, N).astype(float)
    return net, X, y


# Section 10.1: [16, 32, 16] hidden layers for every family in the grid; 2,145 parameters.
def test_architecture_and_parameter_count():
    for k in [1, 3, 4, 5]:
        assert layer_sizes(65, k) == [65, 16, 32, 16, 1]
    net, _, _ = _net_and_data()
    assert n_params(net) == 2145


# Section 10.2: backpropagation equals finite differences in every layer.
def test_backprop_matches_finite_differences():
    net, X, y = _net_and_data()
    _, gW, gb = loss_grad(X, y, net)
    rng, eps, worst = np.random.default_rng(1), 1e-6, 0.0
    for l in range(len(net["W"])):
        for _ in range(5):
            i, j = rng.integers(net["W"][l].shape[0]), rng.integers(net["W"][l].shape[1])
            old = net["W"][l][i, j]
            net["W"][l][i, j] = old + eps; lp, _, _ = loss_grad(X, y, net)
            net["W"][l][i, j] = old - eps; lm, _, _ = loss_grad(X, y, net)
            net["W"][l][i, j] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gW[l][i, j]) / max(abs(num) + abs(gW[l][i, j]), 1e-12))
        j = rng.integers(net["b"][l].shape[0]); old = net["b"][l][j]
        net["b"][l][j] = old + eps; lp, _, _ = loss_grad(X, y, net)
        net["b"][l][j] = old - eps; lm, _, _ = loss_grad(X, y, net)
        net["b"][l][j] = old
        num = (lp - lm) / (2 * eps)
        worst = max(worst, abs(num - gb[l][j]) / max(abs(num) + abs(gb[l][j]), 1e-12))
    assert worst < 1e-6


# The stable loss form equals the plain cross-entropy -[y log s + (1-y) log(1-s)].
def test_stable_loss_equals_plain_cross_entropy():
    net, X, y = _net_and_data()
    z = forward(X, net)[-1]
    s = 1 / (1 + np.exp(-z))
    plain = np.mean(-(y * np.log(s) + (1 - y) * np.log(1 - s)))
    assert abs(loss_grad(X, y, net)[0] - plain) < 1e-12


# Predictions are +-1 and follow the sign of the logit.
def test_predict_is_sign_of_logit():
    net, X, _ = _net_and_data()
    z = forward(X, net)[-1]
    p = predict(X, net)
    assert set(np.unique(p)) <= {-1, 1} and np.all((z > 0) == (p == 1))