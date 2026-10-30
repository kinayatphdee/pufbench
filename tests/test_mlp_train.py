"""Step 4b: MLP training (paper Section IV-C; mirrors train_mlp in mlp_attacks.m)."""
import numpy as np

from pufbench import draw_cell_data
from pufbench.lr import attack_lr
from pufbench.mlp import fit_mlp, init_mlp, layer_sizes, loss_grad, predict, train_mlp


def _data(fam="xor3", n_train=2000):
    d = draw_cell_data(fam, 0, 0, n_train=n_train, n_test=5000)
    X = d.Phi_train.astype(float)
    return d, X, (d.r_train["clean"] > 0).astype(float)


# Training lowers the loss below its starting value.
def test_training_reduces_loss():
    d, X, y = _data("arbiter")
    rng = np.random.default_rng(0)
    net = init_mlp(layer_sizes(65, 1), rng)
    start = loss_grad(X, y, net)[0]
    net, best = train_mlp(X, y, net, rng, epochs=20)
    assert best < 0.5 * start


# Same data and same seed give the same network (reproducibility, Section IV-B).
def test_training_is_deterministic():
    d, X, y = _data()
    a, _ = fit_mlp(d.Phi_train, d.r_train["clean"], 3, np.random.default_rng(7), epochs=5)
    b, _ = fit_mlp(d.Phi_train, d.r_train["clean"], 3, np.random.default_rng(7), epochs=5)
    assert all(np.array_equal(x, z) for x, z in zip(a["W"], b["W"]))


# V-E prediction 7: the MLP reaches high fidelity on the feed-forward PUF, where LR does not.
def test_mlp_succeeds_on_feed_forward_where_lr_fails():
    d = draw_cell_data("ff", 0, 0, n_train=20_000)
    net, _ = fit_mlp(d.Phi_train, d.r_train["clean"], 1, np.random.default_rng(0))
    f_mlp = np.mean(predict(d.Phi_test.astype(float), net) == d.r_test)
    f_lr = attack_lr(d, "clean", 20_000)["fidelity"]
    assert f_mlp > 0.95 and f_lr < 0.90