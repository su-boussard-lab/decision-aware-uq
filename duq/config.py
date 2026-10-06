"""
Typed experiment configuration.

One YAML file defines an experiment. `defaults:` names a parent YAML merged
underneath the current file, so a config contains only its differences.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

logger = logging.getLogger(__name__)


def _build(cls, payload: dict[str, Any] | None):
    """Recursively instantiate a nested dataclass tree from a plain dict."""
    payload = dict(payload or {})
    known = {f.name for f in fields(cls)}
    hints = get_type_hints(cls, globalns=globals())
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        ftype = hints[key]
        kwargs[key] = _build(ftype, value) if is_dataclass(ftype) else value
    return cls(**kwargs)


@dataclass
class DataConfig:
    """Which dataset to load and from where."""

    dataset: str = "dpr"
    root: str = "data/DPR_Data/preprocessed_data/dpr_preprocessed"
    # POU only: validation/test fractions and seed used to build the splits.
    val_size: float = 0.15
    test_size: float = 0.15
    split_seed: int = 42
    # Optional stratified cap on training rows; null keeps all rows.
    max_train_rows: int | None = None


@dataclass
class ModelConfig:
    """Base classifier: L1-penalised logistic regression."""

    kind: str = "lasso"
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "penalty": "l1",
            "C": 0.1,
            "max_iter": 5000,
            "solver": "liblinear",
        }
    )
    standardize: bool = False


@dataclass
class EnsembleConfig:
    """Bootstrap ensemble."""

    n_bootstrap: int = 100
    sample_size: int | None = None  # None -> size of the training set
    stratify: bool = False
    max_redraws: int = 10
    # Parallel fitting changes scheduling only; the member matrix is identical
    # to the serial one (`verify_parallel` checks this by refitting serially).
    parallel: bool = False
    n_jobs: int = -1
    blas_threads_per_worker: int | None = 1
    verify_parallel: bool = False


@dataclass
class BNNConfig:
    """Bayesian logistic regression fitted by mean-field variational inference."""

    enabled: bool = False  # true -> Bayesian arm; false -> bootstrap ensemble
    prior: str = "scaled_gaussian"  # "scaled_gaussian" | "scale_mixture"
    prior_scale: float | None = None  # scaled_gaussian; None -> 1 / sqrt(fan_in)
    mixture_pi: float = 0.5
    mixture_sigma1: float = 1.0
    mixture_sigma2: float = 1e-3

    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 256
    predict_batch_size: int = 4096
    kl_warmup_frac: float = 0.2
    init_rho: float = -5.0
    grad_clip: float | None = 5.0
    num_samples: int = 500
    min_val_auroc: float = 0.55
    device: str = "cpu"


@dataclass
class ThresholdConfig:
    """Operating threshold tau, estimated on validation."""

    rule: str = "f1"  # "f1" | "fixed"
    value: float | None = None


@dataclass
class OrthogonalizationConfig:
    """Natural cubic spline of SD_logit on predicted risk, used to define SD_perp."""

    n_knots: int = 5
    predictor: str = "logit"  # "logit" | "proba"
    standardize: bool = False
    fit_split: str = "val"  # "val" (main analysis) | "self" (in-sample sensitivity)


@dataclass
class UncertaintyConfig:
    """Uncertainty metrics and their percentile-rank reference."""

    metrics: list[str] = field(
        default_factory=lambda: [
            "sd_logit",
            "sd_perp",
            "sd_prob",
            "null_mi",
            "null_entropy",
            "null_dm",
            "null_dm_logit",
        ]
    )
    rank_reference: str = "val"  # "val" | "self"
    orthogonalization: OrthogonalizationConfig = field(
        default_factory=OrthogonalizationConfig
    )


@dataclass
class SweepConfig:
    """Threshold sweep for the FP-vs-FN effect size."""

    enabled: bool = True
    min: float = 0.05
    max: float = 0.50
    n: int = 46


@dataclass
class DeferralConfig:
    """Net benefit vs coverage on a fixed denominator; deferred cases are not treated."""

    enabled: bool = True
    coverages: list[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    )
    metrics: list[str] = field(
        default_factory=lambda: ["sd_logit", "sd_perp", "null_entropy", "null_dm_logit"]
    )


@dataclass
class ErrorSeparationConfig:
    """Correct-vs-incorrect separation and validation-to-test threshold transfer."""

    enabled: bool = True
    q_correct: float = 0.75
    q_incorrect: float = 0.25
    include_minmax: bool = False


@dataclass
class ResidualUncertaintyConfig:
    """Absolute residual vs uncertainty, overall and within p-bar strata."""

    enabled: bool = True
    n_bins: int = 20
    loess_frac: float = 0.5
    n_pbar_strata: int = 10
    max_scatter_points: int = 4000


@dataclass
class UncertaintyGroupConfig:
    """Reliability and net benefit of the least- and most-uncertain groups."""

    enabled: bool = True
    low_pct: float = 20.0
    high_pct: float = 20.0
    n_calibration_bins: int = 10
    calibration_max_probability: float = 0.5  # shared axis limit of calibration panels
    brier_n_boot: int = 1000  # bootstrap replicates for Brier CIs; 0 disables
    # Metrics shown in the group-calibration figure (tables cover all metrics).
    calibration_metrics: list[str] = field(
        default_factory=lambda: ["sd_logit", "sd_perp", "sd_prob", "null_mi"]
    )
    net_benefit_metrics: list[str] | None = None  # None -> all metrics
    threshold_min: float = 0.01
    threshold_max: float = 0.50
    threshold_n: int = 50
    fixed_denominator: bool = True


@dataclass
class DecisionImpactConfig:
    """
    Change in standardised net benefit at `snb_coverage` (vs full coverage and vs
    random deferral, E[sNB_random] = coverage * sNB_full) and retained sensitivity
    at `sens_coverage`, with bootstrap CIs.
    """

    enabled: bool = True
    snb_coverage: float = 0.95
    sens_coverage: float = 0.50
    n_boot: int = 2000
    group_column: str | None = None  # meta_test column of patient ids; None -> rows


@dataclass
class AnalysisConfig:
    """Toggles for the individual analyses and their shared settings."""

    threshold_sweep: SweepConfig = field(default_factory=SweepConfig)
    deferral: DeferralConfig = field(default_factory=DeferralConfig)
    residual_coverage: bool = True
    risk_coverage: bool = True
    calibration: bool = True
    error_separation: ErrorSeparationConfig = field(default_factory=ErrorSeparationConfig)
    residual_uncertainty: ResidualUncertaintyConfig = field(
        default_factory=ResidualUncertaintyConfig
    )
    uncertainty_groups: UncertaintyGroupConfig = field(default_factory=UncertaintyGroupConfig)
    decision_impact: DecisionImpactConfig = field(default_factory=DecisionImpactConfig)
    n_bootstrap_ci: int = 2000
    ci_level: float = 0.95


@dataclass
class PlotConfig:
    """Figure output settings."""

    enabled: bool = True
    format: str = "pdf"
    dpi: int = 300
    context: str = "paper"
    metric_order: str = "pure_vs_null"  # "pure_vs_null" | "epistemic_vs_aleatoric"
    group_metric_order: str = "epistemic_vs_aleatoric"


@dataclass
class ExperimentConfig:
    """Top-level configuration for one dataset x UQ-method setting."""

    name: str = "experiment"
    seed: int = 42
    output_dir: str = "results"
    project_root: str = "."
    save_artifacts: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    bnn: BNNConfig = field(default_factory=BNNConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    @property
    def figure_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def table_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def artifact_dir(self) -> Path:
        return self.run_dir / "artifacts"

    def resolve(self, path: str | Path) -> Path:
        """Resolve a relative path against `project_root`, the cwd, then the repository root."""
        path = Path(path)
        if path.is_absolute():
            return path
        package_parent = Path(__file__).resolve().parent.parent
        for root in (Path(self.project_root), Path("."), package_parent):
            candidate = root / path
            if candidate.exists():
                return candidate
        return Path(self.project_root) / path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """Fail fast on invalid configurations."""
        if self.threshold.rule == "fixed" and self.threshold.value is None:
            raise ValueError("threshold.rule='fixed' requires threshold.value")
        if self.threshold.rule not in {"f1", "fixed"}:
            raise ValueError("threshold.rule must be 'f1' or 'fixed'")
        if self.uncertainty.orthogonalization.fit_split not in {"val", "self"}:
            raise ValueError("orthogonalization.fit_split must be 'val' or 'self'")
        if self.uncertainty.rank_reference not in {"val", "self"}:
            raise ValueError("uncertainty.rank_reference must be 'val' or 'self'")
        if self.ensemble.n_bootstrap < 2:
            raise ValueError("ensemble.n_bootstrap must be >= 2 to define an SD")
        valid_orders = {"pure_vs_null", "epistemic_vs_aleatoric"}
        for field_name in ("metric_order", "group_metric_order"):
            value = getattr(self.plots, field_name)
            if value not in valid_orders:
                raise ValueError(f"plots.{field_name} must be one of {sorted(valid_orders)}")
        if self.uncertainty.orthogonalization.n_knots < 3:
            raise ValueError("orthogonalization.n_knots must be >= 3")
        if not 0.0 < self.analysis.ci_level < 1.0:
            raise ValueError("analysis.ci_level must lie in (0, 1)")
        if self.bnn.enabled:
            if self.bnn.prior not in {"scaled_gaussian", "scale_mixture"}:
                raise ValueError("bnn.prior must be 'scaled_gaussian' or 'scale_mixture'")
            if not 0.0 < self.bnn.mixture_pi < 1.0:
                raise ValueError("bnn.mixture_pi must lie in (0, 1)")
            if self.bnn.mixture_sigma2 >= self.bnn.mixture_sigma1:
                raise ValueError(
                    "bnn.mixture_sigma2 (spike) must be smaller than mixture_sigma1 (slab)"
                )
            if not 0.0 <= self.bnn.kl_warmup_frac <= 1.0:
                raise ValueError("bnn.kl_warmup_frac must lie in [0, 1]")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_with_defaults(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Read a YAML config and merge its `defaults:` ancestor chain underneath it."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular 'defaults' chain at {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    parent = payload.pop("defaults", None)
    if not parent:
        return payload
    base = _read_with_defaults(path.parent / parent, seen + (path,))
    return _deep_merge(base, payload)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config, applying its `defaults:` chain and command-line overrides."""
    path = Path(path)
    payload = _read_with_defaults(path)
    if overrides:
        payload = _deep_merge(payload, overrides)
    cfg = _build(ExperimentConfig, payload)
    cfg.validate()
    logger.info("Config loaded from %s (experiment=%s)", path, cfg.name)
    return cfg
