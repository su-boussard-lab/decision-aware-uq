"""
Experiment orchestration: `run_experiment(cfg)` runs one dataset x UQ-method setting.

    model fitting                 <- training split
    operating threshold tau       <- validation split
    percentile-rank reference     <- validation split
    SD_logit orthogonalisation    <- validation split
    reported results              <- test split
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import plots
from .config import ExperimentConfig
from .data import DatasetUnavailable, Splits, load_dataset
from .ensemble import MemberPredictions, bnn_ensemble, bootstrap_ensemble
from .evaluation import (
    best_f1_threshold,
    calibration_summary,
    calibration_table,
    decision_impact,
    confusion_masks,
    deferral_curve,
    group_comparison,
    group_summary,
    performance_summary,
    residual_coverage,
    risk_coverage,
    threshold_sweep,
)
from .models import build_model
from .reliability import run_reliability_suite
from .uncertainty import (
    MetricSet,
    PredictionBundle,
    correlation_with_pbar,
    fit_orthogonalizer,
)
from .utils import ensure_dir, log_elapsed, save_json, save_table, set_global_seed

logger = logging.getLogger(__name__)

COVERAGES = np.round(np.linspace(0.2, 1.0, 17), 4)


@dataclass
class ExperimentResult:
    """Artifacts of one completed experiment, also written to disk."""

    config: ExperimentConfig
    tau: float
    bundles: dict[str, PredictionBundle]
    metric_set: MetricSet
    tables: dict[str, pd.DataFrame]
    summary: dict


def _arm_name(cfg: ExperimentConfig) -> str:
    """UQ arm of this run: 'blr' when `bnn.enabled`, otherwise 'ensemble'."""
    return "blr" if cfg.bnn.enabled else "ensemble"


def _test_groups(splits, column: str | None) -> np.ndarray | None:
    """Resampling unit for the decision-impact bootstrap: a patient id, or rows."""
    if column is None:
        return None
    if splits.meta_test is None or column not in splits.meta_test.columns:
        logger.warning(
            "decision_impact.group_column='%s' is not in meta_test; "
            "bootstrapping rows instead of patients", column,
        )
        return None
    return splits.meta_test[column].to_numpy()


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run the full uncertainty-evaluation study defined by `cfg`."""
    set_global_seed(cfg.seed)
    for directory in (cfg.run_dir, cfg.figure_dir, cfg.table_dir):
        ensure_dir(directory)
    plots.set_style(cfg.plots.context)
    writer = plots.FigureWriter(cfg.figure_dir, cfg.plots.format, cfg.plots.dpi, cfg.plots.enabled)
    tables: dict[str, pd.DataFrame] = {}

    with log_elapsed(f"EXPERIMENT {cfg.name}"):
        splits = load_dataset(cfg)
        tables["split_summary"] = splits.summary()

        bundles = _build_bundles(cfg, splits)
        val, test = bundles["val"], bundles["test"]

        tau = _select_threshold(cfg, val)
        reference = val if cfg.uncertainty.rank_reference == "val" else None

        orth = fit_orthogonalizer(cfg.uncertainty, val, test)
        metric_set = MetricSet(cfg.uncertainty, orth)

        scores = metric_set.scores(test, reference, tau)
        raws = metric_set.raws(test, tau)
        masks = confusion_masks(test.y_true, test.mean_proba, tau)

        tables["performance"] = pd.DataFrame(
            [
                {"split": "validation", **performance_summary(val.y_true, val.mean_proba, tau)},
                {"split": "test", **performance_summary(test.y_true, test.mean_proba, tau)},
            ]
        )
        tables["confound_check"] = _confound_table(raws, scores, test)
        tables["group_summary"] = group_summary(scores, masks)
        tables["group_comparison"] = group_comparison(
            scores,
            masks,
            n_boot=cfg.analysis.n_bootstrap_ci,
            ci_level=cfg.analysis.ci_level,
            seed=cfg.seed,
        )

        _figures(cfg, writer, metric_set, scores, masks, test, orth, tables)

        if cfg.analysis.threshold_sweep.enabled:
            sweep_cfg = cfg.analysis.threshold_sweep
            taus = np.linspace(sweep_cfg.min, sweep_cfg.max, sweep_cfg.n)
            tables["threshold_sweep"] = threshold_sweep(test, metric_set, reference, taus)
            writer.render(
                plots.plot_threshold_sweep, "threshold_sweep",
                tables["threshold_sweep"], tau, metric_set.labels(),
                title=f"Threshold dependence of FP-vs-FN separation ({cfg.name})",
            )

        if cfg.analysis.deferral.enabled:
            dcfg = cfg.analysis.deferral
            subset = {k: v for k, v in scores.items() if k in dcfg.metrics} or scores
            tables["deferral"] = deferral_curve(
                test.y_true, test.mean_proba, subset, tau,
                dcfg.coverages, seed=cfg.seed,
            )
            writer.render(
                plots.plot_deferral_net_benefit, "deferral",
                tables["deferral"], metric_set.labels(),
                title=f"Deferral rules ({cfg.name})",
            )

        if cfg.analysis.decision_impact.enabled:
            icfg = cfg.analysis.decision_impact
            dcfg = cfg.analysis.deferral
            subset = {k: v for k, v in scores.items() if k in dcfg.metrics} or scores
            tables["decision_impact"] = decision_impact(
                test.y_true, test.mean_proba, subset, tau,
                dataset=cfg.name, method=_arm_name(cfg),
                snb_coverage=icfg.snb_coverage, sens_coverage=icfg.sens_coverage,
                n_boot=icfg.n_boot, ci_level=cfg.analysis.ci_level, seed=cfg.seed,
                groups=_test_groups(splits, icfg.group_column),
            )

        if cfg.analysis.residual_coverage:
            tables["residual_coverage"] = residual_coverage(
                test.y_true, test.mean_proba, scores, COVERAGES
            )
            writer.render(
                plots.plot_residual_coverage, "residual_coverage",
                tables["residual_coverage"], metric_set.labels(),
                title=f"Residual vs coverage ({cfg.name})",
            )

        if cfg.analysis.risk_coverage:
            tables["risk_coverage"] = risk_coverage(
                test.y_true, test.mean_proba, scores, tau, COVERAGES
            )
            writer.render(
                plots.plot_risk_coverage, "risk_coverage",
                tables["risk_coverage"], "risk_accuracy", metric_set.labels(),
                title=f"Risk-coverage ({cfg.name})",
            )

        if cfg.analysis.calibration:
            cal = {
                split: calibration_table(b.y_true, b.mean_proba)
                for split, b in bundles.items()
                if split in ("val", "test")
            }
            summaries = {
                split: calibration_summary(b.y_true, b.mean_proba)
                for split, b in bundles.items()
                if split in cal
            }
            tables["calibration_test"] = cal["test"]
            tables["calibration_summary"] = pd.DataFrame(summaries).T.rename_axis("split").reset_index()
            writer.render(
                plots.plot_calibration, "calibration", cal, f"Calibration ({cfg.name})",
                summaries=summaries,
            )

        _reliability(cfg, writer, tables, _arm_name(cfg), val, test, metric_set, reference, tau)

        summary = _write_outputs(cfg, tables, bundles, tau, orth, scores)

    return ExperimentResult(cfg, tau, bundles, metric_set, tables, summary)


# -------------------------------------------------------- reliability suite
def _reliability(
    cfg, writer, tables, method: str, val, test, metric_set, reference, tau: float
) -> None:
    """Run the reliability analyses (error separation, residuals, uncertainty groups)."""
    acfg = cfg.analysis
    if not any(
        (acfg.error_separation.enabled, acfg.residual_uncertainty.enabled,
         acfg.uncertainty_groups.enabled)
    ):
        return

    labels = metric_set.labels()
    val_scores = metric_set.scores(val, reference, tau)
    test_scores = metric_set.scores(test, reference, tau)

    with log_elapsed(f"Reliability suite ({method})"):
        results = run_reliability_suite(
            val_scores, test_scores, val, test, tau, acfg, cfg.seed
        )
    tables.update(results)

    title_suffix = f"{cfg.name}, {method}"

    if acfg.error_separation.enabled and "correctness_distribution" in results:
        writer.render(
            plots.plot_error_separation, "correctness_distribution",
            results["correctness_distribution"], results["uncertainty_bounds"], labels,
            results["error_separation"],
            title=f"Correct vs incorrect, validation and test ({title_suffix})",
            order=cfg.plots.metric_order,
        )
        writer.render(
            plots.plot_error_separation_effects, "error_separation",
            results["error_separation"], labels,
            title=f"Error separation: composition vs signal ({title_suffix})",
        )

    if acfg.residual_uncertainty.enabled and "residual_uncertainty_binned" in results:
        writer.render(
            plots.plot_residual_uncertainty, "residual_uncertainty",
            results["residual_uncertainty_binned"],
            results.get("residual_uncertainty_loess"),
            _scatter_frame(test, test_scores, acfg.residual_uncertainty.max_scatter_points, cfg.seed),
            results.get("residual_partial_association"), labels,
            title=f"Residual vs uncertainty ({title_suffix})",
            order=cfg.plots.metric_order,
        )

    if acfg.uncertainty_groups.enabled and "group_calibration" in results:
        if len(results["group_calibration"]):
            writer.render(
                plots.plot_uncertainty_group_calibration, "group_calibration",
                results["group_calibration"], results.get("group_overlap"), labels,
                title=f"Calibration by uncertainty group ({title_suffix})",
                brier=results.get("group_brier"),
                max_probability=acfg.uncertainty_groups.calibration_max_probability,
                metrics_include=acfg.uncertainty_groups.calibration_metrics,
                order=cfg.plots.group_metric_order,
            )
        if acfg.uncertainty_groups.fixed_denominator:
            writer.render(
                plots.plot_uncertainty_group_net_benefit, "group_net_benefit_fixed",
                results["group_net_benefit"], tau, labels, "snb_fixed",
                title=f"Uncertainty groups on the full-cohort denominator ({title_suffix})",
                metrics_include=acfg.uncertainty_groups.net_benefit_metrics,
                order=cfg.plots.group_metric_order,
            )


def _scatter_frame(bundle, scores: dict, max_points: int, seed: int) -> pd.DataFrame:
    """Subsampled residual/uncertainty pairs for the scatter layer."""
    residual = np.abs(bundle.y_true - bundle.mean_proba)
    n = len(residual)
    idx = (
        np.random.default_rng(seed).choice(n, max_points, replace=False)
        if 0 < max_points < n
        else np.arange(n)
    )
    return pd.concat(
        [
            pd.DataFrame(
                {"metric": name, "uncertainty": np.asarray(v)[idx], "residual": residual[idx]}
            )
            for name, v in scores.items()
        ],
        ignore_index=True,
    )


# --------------------------------------------------------------------- steps
def _build_bundles(cfg: ExperimentConfig, splits: Splits) -> dict[str, PredictionBundle]:
    """Fit the bootstrap ensemble or, if `bnn.enabled`, the Bayesian model, and wrap predictions."""
    model = build_model(cfg)
    eval_sets = {"val": splits.X_val, "test": splits.X_test}
    if not cfg.bnn.enabled:
        members = bootstrap_ensemble(model, splits.X_train, splits.y_train, eval_sets, cfg)

        bundles = {
            "val": PredictionBundle.from_members(members["val"], splits.y_val),
            "test": PredictionBundle.from_members(members["test"], splits.y_test),
        }

    else:
        try:
            bnn_members = bnn_ensemble(splits.X_train, splits.y_train, eval_sets, cfg)
            bundles = {
                "val": PredictionBundle.from_members(bnn_members["val"], splits.y_val),
                "test": PredictionBundle.from_members(bnn_members["test"], splits.y_test)
            }

            from .bnn_vi import check_convergence

            check_convergence(bnn_members["val"].probs, splits.y_val, cfg.bnn, "validation")
        except ImportError as exc:
            raise RuntimeError(
                f"bnn.enabled=true but the BNN backend is unavailable ({exc}); "
                "install torch, or set bnn.enabled=false to run the ensemble arm"
            ) from exc
    return bundles


def _select_threshold(cfg: ExperimentConfig, val: PredictionBundle) -> float:
    """Choose the operating threshold on validation only."""
    if cfg.threshold.rule == "fixed":
        tau = float(cfg.threshold.value)
        logger.info("Operating threshold (fixed): tau=%.4f", tau)
        return tau
    tau, f1 = best_f1_threshold(val.y_true, val.mean_proba)
    logger.info("Operating threshold from validation F1: tau=%.4f (F1=%.4f)", tau, f1)
    return tau


def _confound_table(
    raws: dict[str, np.ndarray], scores: dict[str, np.ndarray], test: PredictionBundle
) -> pd.DataFrame:
    """Correlation of each metric (raw and ranked) with p-bar on test."""
    rows = []
    for name in raws:
        rows.append(
            {
                "metric": name,
                **{f"raw_{k}": v for k, v in correlation_with_pbar(raws[name], test.mean_proba).items()},
                **correlation_with_pbar(scores[name], test.mean_proba),
                "spearman_with_logit_pbar": correlation_with_pbar(scores[name], test.mean_logit)[
                    "spearman_rho"
                ],
            }
        )
    df = pd.DataFrame(rows)
    for row in df.itertuples():
        logger.info("  %-14s Spearman(metric, p-bar) = %+.3f", row.metric, row.spearman_rho)
    return df


def _figures(cfg, writer, metric_set, scores, masks, test, orth, tables) -> None:
    """Orthogonalisation, confusion-group and confound-check figures."""
    if not writer.enabled:
        return

    writer.render(
        plots.plot_sd_vs_pbar, "sd_orthogonalization",
        test.mean_proba, test.sd_logit, orth.expected(test.mean_proba),
        orth.transform(test.mean_proba, test.sd_logit),
        title=f"Probability geometry in ensemble disagreement ({cfg.name})",
    )

    panel = [m for m in plots.order_metrics(scores, cfg.plots.metric_order) if m in scores]
    writer.render(
        plots.plot_group_distributions, "confusion_groups",
        {m: scores[m] for m in panel}, masks, metric_set.labels(),
        tables["group_comparison"],
        title=f"Confusion groups: metrics against geometry-only nulls ({cfg.name})",
        order=cfg.plots.metric_order,
    )

    writer.render(
        plots.plot_metric_correlation, "confound_check",
        tables["confound_check"], f"Dependence on predicted probability ({cfg.name})",
    )


def _write_outputs(cfg, tables, bundles, tau, orth, scores) -> dict:
    """Persist tables, per-sample scores, a run summary, and the raw bundles."""
    for name, df in tables.items():
        save_table(df, cfg.table_dir / f"{name}.csv")

    test = bundles["test"]
    per_sample = pd.DataFrame(
        {
            "y_true": test.y_true,
            "mean_proba": test.mean_proba,
            "mean_logit": test.mean_logit,
            "sd_prob": test.sd_prob,
            "sd_logit": test.sd_logit,
            "entropy": test.entropy,
            **scores,
        }
    )
    save_table(per_sample, cfg.table_dir / "test_per_sample.csv")

    summary = {
        "experiment": cfg.name,
        "dataset": cfg.data.dataset,
        "model": cfg.model.kind,
        "n_bootstrap": cfg.ensemble.n_bootstrap,
        "tau": tau,
        "test_prevalence": test.prevalence,
        "orthogonalization": {
            "method": "spline",
            "fit_split": cfg.uncertainty.orthogonalization.fit_split,
            "n_fit": orth.fit_n,
            "r2_pbar_on_sd_logit": orth.fit_r2,
        },
        "config": cfg.to_dict(),
    }
    save_json(summary, cfg.run_dir / "summary.json")

    if cfg.save_artifacts:
        path = ensure_dir(cfg.artifact_dir) / "bundles.pkl"
        with open(path, "wb") as fh:
            pickle.dump(
                {k: {"y_true": b.y_true, "member_probs": b.member_probs} for k, b in bundles.items()},
                fh,
            )
        logger.info("Ensemble predictions saved -> %s", path)
    return summary


def run_all(configs: list[ExperimentConfig]) -> dict[str, ExperimentResult | str]:
    """Run several experiments, skipping (with a warning) any whose data is absent."""
    results: dict[str, ExperimentResult | str] = {}
    for cfg in configs:
        try:
            results[cfg.name] = run_experiment(cfg)
        except DatasetUnavailable as exc:
            logger.warning("SKIPPED %s - %s", cfg.name, exc)
            results[cfg.name] = f"skipped: {exc}"
    return results

