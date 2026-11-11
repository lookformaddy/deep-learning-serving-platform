"""Pytest configuration: put the src layout on the import path so plain
``pytest`` passes from the project root, with or without an install.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlserve.data import generate_image_like, generate_tabular, split_dataset  # noqa: E402
from dlserve.nn import MLP, MLPConfig  # noqa: E402
from dlserve.registry import ModelRegistry, ModelVersion  # noqa: E402
from dlserve.serving import ServingConfig, ServingLayer  # noqa: E402
from dlserve.training import TrainConfig, TrainingJob  # noqa: E402

SEED = 42


@pytest.fixture()
def tabular():
    return generate_tabular(n=300, seed=SEED)


@pytest.fixture()
def tabular_split(tabular):
    return split_dataset(tabular, test_frac=0.2, seed=SEED)


@pytest.fixture()
def image_like():
    return generate_image_like(n=300, seed=SEED)


@pytest.fixture()
def trained_mlp(tabular_split, tmp_path):
    train, val = tabular_split
    job = TrainingJob(
        "test-job",
        TrainConfig(epochs=8, hidden_dims=[16, 8], seed=SEED),
        workdir=tmp_path,
    )
    model, report = job.run(train, val)
    return model, report, job


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry.json")


@pytest.fixture()
def serving_layer(trained_mlp):
    model, _report, _job = trained_mlp
    return ServingLayer(
        champion_fn=model.predict_proba,
        champion_version="v1",
        config=ServingConfig(canary_pct=0.0, shadow_pct=0.0),
    )
