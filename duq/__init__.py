"""
Decision-aware evaluation of uncertainty-based deferral in imbalanced clinical prediction.

    load_config      read a YAML experiment configuration
    run_experiment   run one dataset x UQ-method setting
    run_all          run several settings, skipping any whose data is absent
"""

from .config import ExperimentConfig, load_config
from .pipeline import ExperimentResult, run_all, run_experiment

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "load_config",
    "run_all",
    "run_experiment",
]
__version__ = "2.0.0"
