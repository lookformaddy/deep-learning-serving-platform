"""deep-learning-serving-platform: from-scratch neural network serving.

Pure-stdlib MLP (forward/backprop, Adam/SGD, dropout, layer norm),
versioned registry with gated champion promotion, canary/shadow
deployment, dynamic batching, latency-budget fallback, PSI drift
monitoring, and offline evaluation.
"""
from .data import Dataset, generate_image_like, generate_tabular, split_dataset
from .evaluation import EvalReport, evaluate
from .monitoring import DriftMonitor, psi
from .nn import MLP, MLPConfig
from .registry import ModelRegistry, ModelVersion, PromotionGate
from .serving import ServingConfig, ServingLayer
from .training import LinearBaseline, TrainConfig, TrainingJob

__all__ = [
    "Dataset",
    "DriftMonitor",
    "EvalReport",
    "LinearBaseline",
    "MLP",
    "MLPConfig",
    "ModelRegistry",
    "ModelVersion",
    "PromotionGate",
    "ServingConfig",
    "ServingLayer",
    "TrainConfig",
    "TrainingJob",
    "evaluate",
    "generate_image_like",
    "generate_tabular",
    "psi",
    "split_dataset",
]

__version__ = "0.1.0"
