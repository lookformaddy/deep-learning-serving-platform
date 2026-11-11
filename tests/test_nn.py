"""Tests for the from-scratch neural network (nn.py)."""
import math
import random

import pytest

from dlserve.nn import (
    MLP,
    MLPConfig,
    Adam,
    Dropout,
    LayerNorm,
    Linear,
    SGD,
    cross_entropy_loss,
    make_optimizer,
    matmul,
    relu,
    softmax,
    softmax_ce_grad,
    transpose,
)


def test_matmul_basic():
    assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]


def test_matmul_shape_mismatch():
    with pytest.raises(ValueError):
        matmul([[1, 2]], [[1, 2, 3]])


def test_transpose():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_softmax_rows_sum_to_one():
    rows = softmax([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    for row in rows:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-9)
    assert rows[0][2] > rows[0][1] > rows[0][0]


def test_cross_entropy_and_grad():
    probs = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
    loss = cross_entropy_loss(probs, [0, 1])
    assert math.isclose(loss, -(math.log(0.7) + math.log(0.8)) / 2, rel_tol=1e-9)
    grad = softmax_ce_grad(probs, [0, 1])
    # Each grad row sums to zero (softmax CE property).
    for row in grad:
        assert math.isclose(sum(row), 0.0, abs_tol=1e-9)


def test_linear_forward_backward_shapes():
    rng = random.Random(0)
    layer = Linear(4, 3, rng)
    x = [[0.1, -0.2, 0.3, 0.4], [0.5, 0.6, -0.7, 0.8]]
    out, cache = layer.forward(x)
    assert len(out) == 2 and len(out[0]) == 3
    grad_out = [[0.1, 0.2, 0.3], [-0.1, 0.0, 0.2]]
    grad_in, grads = layer.backward(grad_out, cache)
    assert len(grad_in) == 2 and len(grad_in[0]) == 4
    assert len(grads["weight"]) == 4 and len(grads["weight"][0]) == 3
    assert len(grads["bias"]) == 3


def test_layernorm_output_zero_mean_unit_var():
    ln = LayerNorm(4)
    out, _cache = ln.forward([[1.0, 2.0, 3.0, 4.0]])
    mean = sum(out[0]) / 4
    var = sum((v - mean) ** 2 for v in out[0]) / 4
    assert math.isclose(mean, 0.0, abs_tol=1e-9)
    assert math.isclose(var, 1.0, rel_tol=1e-4)


def test_dropout_eval_mode_is_identity():
    drop = Dropout(0.5, random.Random(0))
    x = [[1.0, 2.0], [3.0, 4.0]]
    out, mask = drop.forward(x, training=False)
    assert out == x and mask is None


def test_dropout_training_scales_kept_units():
    drop = Dropout(0.5, random.Random(1))
    x = [[1.0] * 200]
    out, mask = drop.forward(x, training=True)
    kept = [v for row in out for v in row]
    assert set(kept) <= {0.0, 2.0}  # inverted dropout: kept units scaled by 1/keep
    assert 0.0 in kept and 2.0 in kept


def test_mlp_forward_shapes_and_proba():
    model = MLP(MLPConfig(input_dim=6, hidden_dims=[8, 4], output_dim=3))
    x = [[0.1] * 6, [-0.2] * 6]
    logits, _ = model.forward(x)
    assert len(logits) == 2 and len(logits[0]) == 3
    probs = model.predict_proba(x)
    for row in probs:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-9)
    assert model.predict(x) == [int(p.index(max(p))) for p in probs]


def test_mlp_num_parameters():
    # (6*8+8) + LN(16) + (8*4+4) + LN(8) + (4*3+3) = 56+16+36+8+15 = 131
    model = MLP(MLPConfig(input_dim=6, hidden_dims=[8, 4], output_dim=3))
    assert model.num_parameters() == 131


def test_mlp_state_dict_roundtrip():
    model = MLP(MLPConfig(input_dim=4, hidden_dims=[5], output_dim=2), rng=random.Random(7))
    x = [[0.3, -0.1, 0.7, 0.2]]
    before = model.predict_proba(x)
    state = model.state_dict()
    other = MLP(MLPConfig(input_dim=4, hidden_dims=[5], output_dim=2), rng=random.Random(99))
    other.load_state_dict(state)
    after = other.predict_proba(x)
    assert before == after


def test_gradient_check_tiny_mlp():
    """Finite-difference check of the manual backprop on a tiny model."""
    rng = random.Random(3)
    model = MLP(
        MLPConfig(input_dim=3, hidden_dims=[4], output_dim=2,
                  dropout_rate=0.0, use_layer_norm=False),
        rng=rng,
    )
    x = [[0.5, -0.3, 0.8], [0.1, 0.9, -0.2]]
    y = [1, 0]

    def loss_fn():
        logits, _ = model.forward(x, training=False)
        return cross_entropy_loss(softmax(logits), y)

    logits, caches = model.forward(x, training=False)
    grads = model.backward(softmax_ce_grad(softmax(logits), y), caches)
    eps = 1e-5
    # Check a handful of parameters in the first and last Linear layers.
    for layer in model.layers:
        if not isinstance(layer, Linear):
            continue
        layer_grads = grads[id(layer)]
        for (i, j) in [(0, 0), (1, 0), (0, 1)]:
            if i >= len(layer.weight) or j >= len(layer.weight[0]):
                continue
            orig = layer.weight[i][j]
            layer.weight[i][j] = orig + eps
            loss_plus = loss_fn()
            layer.weight[i][j] = orig - eps
            loss_minus = loss_fn()
            layer.weight[i][j] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            assert math.isclose(numeric, layer_grads["weight"][i][j],
                                rel_tol=1e-3, abs_tol=1e-4), (
                f"grad mismatch at [{i}][{j}]: numeric {numeric} vs "
                f"analytic {layer_grads['weight'][i][j]}"
            )


def test_optimizers_reduce_loss():
    rng = random.Random(5)
    x = [[0.5, -0.3], [0.1, 0.9], [-0.7, 0.2], [0.4, 0.4]]
    y = [1, 0, 1, 0]
    for name in ("adam", "sgd"):
        model = MLP(
            MLPConfig(input_dim=2, hidden_dims=[6], output_dim=2,
                      dropout_rate=0.0, use_layer_norm=False),
            rng=random.Random(5),
        )
        opt = make_optimizer(name, lr=0.05 if name == "adam" else 0.1)
        assert isinstance(opt, (Adam, SGD))
        first = None
        for _ in range(30):
            logits, caches = model.forward(x, training=True)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y)
            if first is None:
                first = loss
            grads = model.backward(softmax_ce_grad(probs, y), caches)
            opt.step(model, grads)
        assert loss < first, f"{name}: loss did not decrease ({first} -> {loss})"


def test_make_optimizer_unknown():
    with pytest.raises(ValueError):
        make_optimizer("rmsprop", 0.01)


def test_relu():
    assert relu([[-1.0, 0.0, 2.0]]) == [[0.0, 0.0, 2.0]]
