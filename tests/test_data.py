"""Tests for synthetic datasets (data.py)."""
import math

from dlserve.data import (
    generate_image_like,
    generate_tabular,
    split_dataset,
    standardize_apply,
    standardize_fit,
)


def test_tabular_shape_and_classes(tabular):
    assert len(tabular) == 300
    assert len(tabular.X[0]) == 16
    assert tabular.n_classes == 3
    assert set(tabular.y) == {0, 1, 2}
    assert len(tabular.feature_names) == 16


def test_tabular_deterministic():
    a = generate_tabular(n=50, seed=7)
    b = generate_tabular(n=50, seed=7)
    assert a.X == b.X and a.y == b.y
    c = generate_tabular(n=50, seed=8)
    assert c.X != a.X


def test_image_like_shape_and_classes(image_like):
    assert len(image_like) == 300
    assert len(image_like.X[0]) == 64  # 8x8 flattened
    assert image_like.n_classes == 3
    assert set(image_like.y) == {0, 1, 2}
    # Pixel values are clipped to [0, 1].
    assert all(0.0 <= v <= 1.0 for row in image_like.X for v in row)


def test_image_like_classes_are_distinct():
    data = generate_image_like(n=90, seed=1)
    means = []
    for label in range(3):
        rows = [r for r, y in zip(data.X, data.y) if y == label]
        means.append([sum(col) / len(rows) for col in zip(*rows)])
    # Class-mean images must differ from each other.
    for i in range(3):
        for j in range(i + 1, 3):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(means[i], means[j])))
            assert dist > 1.0


def test_split_is_stratified(tabular):
    train, test = split_dataset(tabular, test_frac=0.2, seed=1)
    assert len(train) + len(test) == len(tabular)
    for label in range(3):
        train_frac = sum(1 for y in train.y if y == label) / len(train)
        orig_frac = sum(1 for y in tabular.y if y == label) / len(tabular)
        assert abs(train_frac - orig_frac) < 0.1


def test_standardize_zero_mean_unit_var(tabular):
    means, stds = standardize_fit(tabular.X)
    scaled = standardize_apply(tabular.X, means, stds)
    for j in range(len(tabular.X[0])):
        col = [row[j] for row in scaled]
        assert abs(sum(col) / len(col)) < 1e-9
        var = sum(v * v for v in col) / len(col)
        assert abs(var - 1.0) < 1e-9
