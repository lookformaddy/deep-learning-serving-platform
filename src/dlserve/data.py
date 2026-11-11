"""Deterministic synthetic datasets — no real data, no external files.

Two dataset families, both seeded so every run reproduces exactly:

* ``tabular`` — 16 features. The label is a *nonlinear* function of the
  first two features (radius band crossed with half-plane, an XOR-style
  structure): a linear model cannot solve it, which is what makes the
  MLP-vs-baseline demo meaningful. The remaining 14 features are
  correlated noise / distractors.
* ``image_like`` — flattened 8x8 (64-dim) synthetic "images" of three
  shape classes (ring, square frame, diagonal cross) with pixel noise.
  They exercise the same MLP on higher-dimensional vector input.

Splits are stratified by class and reproducible from the seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .nn import Matrix


@dataclass
class Dataset:
    name: str
    X: Matrix
    y: list[int]
    n_classes: int
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


def _stratified_split(
    X: Matrix, y: list[int], test_frac: float, rng: random.Random
) -> tuple[Matrix, list[int], Matrix, list[int]]:
    """Stratified train/test split preserving class proportions."""
    by_class: dict[int, list[int]] = {}
    for i, label in enumerate(y):
        by_class.setdefault(label, []).append(i)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label, idxs in sorted(by_class.items()):
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_frac)))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return (
        [X[i] for i in train_idx],
        [y[i] for i in train_idx],
        [X[i] for i in test_idx],
        [y[i] for i in test_idx],
    )


def generate_tabular(n: int = 900, seed: int = 42) -> Dataset:
    """Ring × quadrant classification: the label is a *nonlinear* function
    of the first two features — ``(radius_band + half_plane) % 3`` — so a
    linear model cannot solve it (it is an XOR-style structure), while an
    MLP can. The remaining features are correlated noise / distractors,
    including a radius proxy that alone is not enough to separate the
    classes."""
    rng = random.Random(seed)
    X: Matrix = []
    y: list[int] = []
    for _ in range(n):
        # Radius band (0/1/2) crossed with upper/lower half-plane.
        radius = rng.uniform(0.0, 3.0)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x0 = radius * math.cos(angle) + rng.gauss(0.0, 0.08)
        x1 = radius * math.sin(angle) + rng.gauss(0.0, 0.08)
        band = 0 if radius < 1.0 else (1 if radius < 2.0 else 2)
        half = 0 if math.sin(angle) > 0 else 1
        label = (band + half) % 3
        # Distractor features: some correlated with the signal, most pure noise.
        row = [x0, x1]
        row.append(x0 * x1 + rng.gauss(0.0, 0.3))          # interaction term
        row.append(x0 ** 2 + x1 ** 2 + rng.gauss(0.0, 0.3))  # radius proxy
        row.append(math.sin(x0) + rng.gauss(0.0, 0.2))
        row.append(math.cos(x1) + rng.gauss(0.0, 0.2))
        for _ in range(10):
            row.append(rng.gauss(0.0, 1.0))
        X.append(row)
        y.append(label)
    return Dataset(
        name="tabular-rings",
        X=X,
        y=y,
        n_classes=3,
        feature_names=[f"f{i}" for i in range(16)],
    )


def _shape_pixels(kind: str, size: int, rng: random.Random, noise: float) -> list[float]:
    """Render one 8x8 (or ``size``x``size``) shape as a flat vector."""
    centre = (size - 1) / 2.0
    pixels: list[float] = []
    for r in range(size):
        for c in range(size):
            dx = c - centre
            dy = r - centre
            dist = math.sqrt(dx * dx + dy * dy)
            if kind == "ring":
                value = 1.0 if 2.0 <= dist <= 3.0 else 0.0
            elif kind == "frame":
                value = 1.0 if (r in (1, size - 2) or c in (1, size - 2)) and 1 <= r < size - 1 and 1 <= c < size - 1 else 0.0
            else:  # "cross"
                value = 1.0 if abs(dx) == abs(dy) else 0.0
            value += rng.gauss(0.0, noise)
            pixels.append(max(0.0, min(1.0, value)))
    return pixels


def generate_image_like(n: int = 900, seed: int = 42, size: int = 8) -> Dataset:
    """Three synthetic shape classes rendered as flattened pixel vectors."""
    rng = random.Random(seed)
    kinds = ["ring", "frame", "cross"]
    X: Matrix = []
    y: list[int] = []
    per_class = n // 3
    for label, kind in enumerate(kinds):
        for _ in range(per_class):
            X.append(_shape_pixels(kind, size, rng, noise=0.15))
            y.append(label)
    # Any remainder rows go to class 0 so the total is exactly ``n``.
    for _ in range(n - len(y)):
        X.append(_shape_pixels("ring", size, rng, noise=0.15))
        y.append(0)
    order = list(range(len(y)))
    rng.shuffle(order)
    return Dataset(
        name="image-like-shapes",
        X=[X[i] for i in order],
        y=[y[i] for i in order],
        n_classes=3,
        feature_names=[f"px{r}_{c}" for r in range(size) for c in range(size)],
    )


def split_dataset(
    dataset: Dataset, test_frac: float = 0.2, seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Return (train, test) stratified splits of ``dataset``."""
    rng = random.Random(seed)
    X_train, y_train, X_test, y_test = _stratified_split(
        dataset.X, dataset.y, test_frac, rng
    )
    train = Dataset(dataset.name, X_train, y_train, dataset.n_classes, dataset.feature_names)
    test = Dataset(dataset.name, X_test, y_test, dataset.n_classes, dataset.feature_names)
    return train, test


def standardize_fit(X: Matrix) -> tuple[list[float], list[float]]:
    """Per-feature mean/std for standardisation (std floored at 1e-6)."""
    n = len(X)
    d = len(X[0])
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        stds.append(max(math.sqrt(var), 1e-6))
    return means, stds


def standardize_apply(X: Matrix, means: list[float], stds: list[float]) -> Matrix:
    return [[(v - means[j]) / stds[j] for j, v in enumerate(row)] for row in X]
