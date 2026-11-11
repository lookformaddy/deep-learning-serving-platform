"""From-scratch neural network primitives — pure Python standard library.

No torch, no numpy, no sklearn. Matrices are ``list[list[float]]`` and
every operation is written out explicitly so the maths is auditable.
All randomness flows through a caller-supplied ``random.Random`` so runs
are reproducible from a seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

Matrix = list[list[float]]
Vector = list[float]


# ---------------------------------------------------------------------------
# Small matrix helpers (row-major, batch-first)
# ---------------------------------------------------------------------------
def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Return ``a @ b`` for 2-D lists. Raises on shape mismatch."""
    if not a or not b or len(a[0]) != len(b):
        raise ValueError(
            f"matmul shape mismatch: {len(a)}x{len(a[0]) if a else 0} "
            f"@ {len(b)}x{len(b[0]) if b else 0}"
        )
    inner = len(b)
    cols = len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(len(a))
    ]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def add_bias(x: Matrix, bias: Vector) -> Matrix:
    if len(bias) != len(x[0]):
        raise ValueError("bias length does not match matrix width")
    return [[v + bias[j] for j, v in enumerate(row)] for row in x]


def relu(x: Matrix) -> Matrix:
    return [[v if v > 0.0 else 0.0 for v in row] for row in x]


def relu_backward(grad: Matrix, pre_activation: Matrix) -> Matrix:
    return [
        [g if v > 0.0 else 0.0 for g, v in zip(g_row, x_row)]
        for g_row, x_row in zip(grad, pre_activation)
    ]


def softmax(logits: Matrix) -> Matrix:
    out: Matrix = []
    for row in logits:
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        total = sum(exps)
        out.append([e / total for e in exps])
    return out


def cross_entropy_loss(probs: Matrix, targets: Sequence[int]) -> float:
    """Mean negative log-likelihood. ``targets`` are class indices."""
    total = 0.0
    for row, t in zip(probs, targets):
        total += -math.log(max(row[t], 1e-12))
    return total / len(probs)


def softmax_ce_grad(probs: Matrix, targets: Sequence[int]) -> Matrix:
    """dL/dlogits for softmax + cross-entropy, averaged over the batch."""
    n = len(probs)
    grad: Matrix = []
    for row, t in zip(probs, targets):
        g = list(row)
        g[t] -= 1.0
        grad.append([v / n for v in g])
    return grad


def xavier_init(rows: int, cols: int, rng: random.Random) -> Matrix:
    limit = math.sqrt(6.0 / (rows + cols))
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
@dataclass
class Linear:
    in_dim: int
    out_dim: int
    rng: random.Random
    weight: Matrix = field(init=False)
    bias: Vector = field(init=False)

    def __post_init__(self) -> None:
        self.weight = xavier_init(self.in_dim, self.out_dim, self.rng)
        self.bias = [0.0] * self.out_dim

    def forward(self, x: Matrix) -> tuple[Matrix, Matrix]:
        return add_bias(matmul(x, self.weight), self.bias), x

    def backward(
        self, grad_out: Matrix, cache: Matrix
    ) -> tuple[Matrix, dict[str, Matrix | Vector]]:
        grad_in = matmul(grad_out, transpose(self.weight))
        grad_w = matmul(transpose(cache), grad_out)
        grad_b = [sum(col) for col in zip(*grad_out)]
        return grad_in, {"weight": grad_w, "bias": grad_b}

    def parameters(self) -> dict[str, Matrix | Vector]:
        return {"weight": self.weight, "bias": self.bias}

    def set_parameters(self, params: dict[str, Matrix | Vector]) -> None:
        self.weight = [list(r) for r in params["weight"]]  # type: ignore[arg-type]
        self.bias = list(params["bias"])  # type: ignore[arg-type]


@dataclass
class LayerNorm:
    dim: int
    eps: float = 1e-5
    gamma: Vector = field(init=False)
    beta: Vector = field(init=False)

    def __post_init__(self) -> None:
        self.gamma = [1.0] * self.dim
        self.beta = [0.0] * self.dim

    def forward(self, x: Matrix) -> tuple[Matrix, tuple[Matrix, Vector, Vector]]:
        means: Vector = []
        inv_stds: Vector = []
        out: Matrix = []
        for row in x:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            inv_std = 1.0 / math.sqrt(var + self.eps)
            means.append(mean)
            inv_stds.append(inv_std)
            out.append(
                [self.gamma[j] * (v - mean) * inv_std + self.beta[j]
                 for j, v in enumerate(row)]
            )
        return out, (x, means, inv_stds)

    def backward(
        self, grad_out: Matrix, cache: tuple[Matrix, Vector, Vector]
    ) -> tuple[Matrix, dict[str, Vector]]:
        x, means, inv_stds = cache
        d = self.dim
        grad_gamma = [0.0] * d
        grad_beta = [0.0] * d
        grad_in: Matrix = []
        for i, row in enumerate(x):
            x_hat = [(v - means[i]) * inv_stds[i] for v in row]
            g = grad_out[i]
            for j in range(d):
                grad_gamma[j] += g[j] * x_hat[j]
                grad_beta[j] += g[j]
            mean_g = sum(g[j] * self.gamma[j] for j in range(d)) / d
            mean_gx = sum(g[j] * self.gamma[j] * x_hat[j] for j in range(d)) / d
            grad_in.append(
                [
                    inv_stds[i] * self.gamma[j] * (g[j] - mean_g - x_hat[j] * mean_gx)
                    for j in range(d)
                ]
            )
        return grad_in, {"gamma": grad_gamma, "beta": grad_beta}

    def parameters(self) -> dict[str, Vector]:
        return {"gamma": self.gamma, "beta": self.beta}

    def set_parameters(self, params: dict[str, Vector]) -> None:
        self.gamma = list(params["gamma"])
        self.beta = list(params["beta"])


@dataclass
class Dropout:
    rate: float
    rng: random.Random

    def forward(
        self, x: Matrix, training: bool
    ) -> tuple[Matrix, Optional[Matrix]]:
        if not training or self.rate <= 0.0:
            return x, None
        keep = 1.0 - self.rate
        mask = [
            [1.0 / keep if self.rng.random() < keep else 0.0 for _ in row]
            for row in x
        ]
        return [[v * m for v, m in zip(row, mask_row)] for row, mask_row in zip(x, mask)], mask

    def backward(self, grad_out: Matrix, mask: Optional[Matrix]) -> Matrix:
        if mask is None:
            return grad_out
        return [[g * m for g, m in zip(g_row, m_row)] for g_row, m_row in zip(grad_out, mask)]


# ---------------------------------------------------------------------------
# MLP — input -> [Linear, LayerNorm, ReLU, Dropout] x N -> Linear (logits)
# ---------------------------------------------------------------------------
Layer = Linear | LayerNorm | Dropout


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    dropout_rate: float = 0.1
    use_layer_norm: bool = True


class MLP:
    """Small feed-forward classifier with manual forward/backward passes."""

    def __init__(self, config: MLPConfig, rng: Optional[random.Random] = None) -> None:
        self.config = config
        self.rng = rng or random.Random(42)
        self.layers: list[Layer] = []
        dims = [config.input_dim] + list(config.hidden_dims)
        for in_d, out_d in zip(dims, config.hidden_dims):
            self.layers.append(Linear(in_d, out_d, self.rng))
            if config.use_layer_norm:
                self.layers.append(LayerNorm(out_d))
            self.layers.append(_Relu())
            self.layers.append(Dropout(config.dropout_rate, self.rng))
        self.layers.append(Linear(dims[-1], config.output_dim, self.rng))

    # -- forward ---------------------------------------------------------
    def forward(self, x: Matrix, training: bool = False) -> tuple[Matrix, list[object]]:
        caches: list[object] = []
        cur = x
        for layer in self.layers:
            if isinstance(layer, Dropout):
                cur, cache = layer.forward(cur, training)
            elif isinstance(layer, _Relu):
                cur, cache = layer.forward(cur)
            else:
                cur, cache = layer.forward(cur)
            caches.append(cache)
        return cur, caches

    def predict_proba(self, x: Matrix) -> Matrix:
        logits, _ = self.forward(x, training=False)
        return softmax(logits)

    def first_hidden_activations(self, x: Matrix) -> Matrix:
        """Post-ReLU output of the first hidden block (for drift monitoring)."""
        cur = x
        for layer in self.layers:
            if isinstance(layer, Dropout):
                cur, _ = layer.forward(cur, False)
            elif isinstance(layer, _Relu):
                cur, _ = layer.forward(cur)
                return cur
            else:
                cur, _ = layer.forward(cur)
        raise ValueError("MLP has no hidden ReLU layer")

    def predict(self, x: Matrix) -> list[int]:
        return [max(range(len(row)), key=lambda j: row[j])
                for row in self.predict_proba(x)]

    # -- backward --------------------------------------------------------
    def backward(self, grad_logits: Matrix, caches: list[object]) -> dict[str, dict[str, object]]:
        grads: dict[str, dict[str, object]] = {}
        grad = grad_logits
        for layer, cache in zip(reversed(self.layers), reversed(caches)):
            if isinstance(layer, Dropout):
                grad = layer.backward(grad, cache)  # type: ignore[arg-type]
            elif isinstance(layer, _Relu):
                grad = relu_backward(grad, cache)  # type: ignore[arg-type]
            else:
                grad, layer_grads = layer.backward(grad, cache)  # type: ignore[arg-type]
                grads[id(layer)] = layer_grads
        return grads

    # -- parameter (de)serialisation --------------------------------------
    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Linear, LayerNorm)):
                state[f"layer_{i}"] = {
                    k: [list(r) for r in v] if isinstance(v, list) and v and isinstance(v[0], list)
                    else list(v)
                    for k, v in layer.parameters().items()
                }
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Linear, LayerNorm)):
                params = state[f"layer_{i}"]  # type: ignore[index]
                layer.set_parameters(params)  # type: ignore[arg-type]

    def num_parameters(self) -> int:
        total = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                total += layer.in_dim * layer.out_dim + layer.out_dim
            elif isinstance(layer, LayerNorm):
                total += 2 * layer.dim
        return total


class _Relu:
    """ReLU as a layer object so the MLP layer list stays uniform."""

    def forward(self, x: Matrix) -> tuple[Matrix, Matrix]:
        return relu(x), x


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict[int, dict[str, object]] = {}

    def step(self, model: MLP, grads: dict[str, dict[str, object]]) -> None:
        for layer in model.layers:
            if id(layer) not in grads:
                continue
            params = layer.parameters()  # type: ignore[attr-defined]
            layer_grads = grads[id(layer)]
            vel = self._velocity.setdefault(id(layer), {})
            for name, param in params.items():
                g = layer_grads[name]  # type: ignore[index]
                v = vel.get(name)
                if isinstance(param[0], list):  # matrix param
                    if v is None:
                        v = [[0.0] * len(r) for r in param]  # type: ignore[union-attr]
                        vel[name] = v
                    for i, row in enumerate(param):  # type: ignore[union-attr]
                        for j in range(len(row)):
                            v[i][j] = self.momentum * v[i][j] - self.lr * g[i][j]  # type: ignore[index]
                            row[j] += v[i][j]
                else:  # vector param
                    if v is None:
                        v = [0.0] * len(param)  # type: ignore[arg-type]
                        vel[name] = v
                    for j in range(len(param)):  # type: ignore[arg-type]
                        v[j] = self.momentum * v[j] - self.lr * g[j]  # type: ignore[index]
                        param[j] += v[j]  # type: ignore[index]


class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: dict[int, dict[str, object]] = {}
        self._v: dict[int, dict[str, object]] = {}

    def step(self, model: MLP, grads: dict[str, dict[str, object]]) -> None:
        self.t += 1
        for layer in model.layers:
            if id(layer) not in grads:
                continue
            params = layer.parameters()  # type: ignore[attr-defined]
            layer_grads = grads[id(layer)]
            m_state = self._m.setdefault(id(layer), {})
            v_state = self._v.setdefault(id(layer), {})
            for name, param in params.items():
                g = layer_grads[name]  # type: ignore[index]
                m = m_state.get(name)
                v = v_state.get(name)
                if isinstance(param[0], list):
                    if m is None:
                        m = [[0.0] * len(r) for r in param]  # type: ignore[union-attr]
                        v = [[0.0] * len(r) for r in param]  # type: ignore[union-attr]
                        m_state[name] = m
                        v_state[name] = v
                    for i, row in enumerate(param):  # type: ignore[union-attr]
                        for j in range(len(row)):
                            m[i][j] = self.beta1 * m[i][j] + (1 - self.beta1) * g[i][j]  # type: ignore[index]
                            v[i][j] = self.beta2 * v[i][j] + (1 - self.beta2) * g[i][j] ** 2  # type: ignore[index]
                            m_hat = m[i][j] / (1 - self.beta1 ** self.t)
                            v_hat = v[i][j] / (1 - self.beta2 ** self.t)
                            row[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                else:
                    if m is None:
                        m = [0.0] * len(param)  # type: ignore[arg-type]
                        v = [0.0] * len(param)  # type: ignore[arg-type]
                        m_state[name] = m
                        v_state[name] = v
                    for j in range(len(param)):  # type: ignore[arg-type]
                        m[j] = self.beta1 * m[j] + (1 - self.beta1) * g[j]  # type: ignore[index]
                        v[j] = self.beta2 * v[j] + (1 - self.beta2) * g[j] ** 2  # type: ignore[index]
                        m_hat = m[j] / (1 - self.beta1 ** self.t)
                        v_hat = v[j] / (1 - self.beta2 ** self.t)
                        param[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)  # type: ignore[index]

    def state_dict(self) -> dict[str, object]:
        return {"t": self.t,
                "m": _nested_to_lists(self._m),
                "v": _nested_to_lists(self._v)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.t = int(state["t"])  # type: ignore[arg-type]
        # Optimizer moments are keyed by id(layer) which is not stable
        # across processes; resume restores the step count and re-inits
        # moments (documented trade-off: a resumed run warms up Adam
        # moments again instead of restoring exact state).


def _nested_to_lists(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _nested_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nested_to_lists(v) for v in obj]
    return obj


def make_optimizer(name: str, lr: float) -> SGD | Adam:
    name = name.lower()
    if name == "adam":
        return Adam(lr=lr)
    if name == "sgd":
        return SGD(lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer: {name!r} (expected 'adam' or 'sgd')")
