"""
Figures (file name: content; manuscript figure).

    confound_check            Spearman rho of each metric with p-bar (Fig. 1a-d)
    sd_orthogonalization      SD_logit vs p-bar with fitted spline, and SD_perp (Fig. 1e-h)
    confusion_groups          uncertainty by TP/FP/FN/TN (Fig. 2)
    error_separation          pooled and within-predicted-label effects (Fig. 3)
    deferral                  standardised net benefit vs coverage (Fig. 4a-d)
    risk_coverage             selective risk and retained prevalence (Fig. 4e-h, Fig. S6)
    group_calibration         reliability of least/most uncertain groups (Fig. 5)
    residual_uncertainty      |y - p-bar| vs uncertainty (Fig. S1)
    threshold_sweep           FP-vs-FN effect size vs tau (Fig. S2)
    residual_coverage         |y - p-bar| vs coverage by true class (Fig. S3)
    calibration               whole-cohort reliability diagram (Fig. S4)
    group_net_benefit_fixed   group decision curves, fixed denominator (Fig. S5)
    correctness_distribution  correct vs incorrect, validation and test, with
                              validation-derived bounds
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .utils import ensure_dir  # noqa: E402

logger = logging.getLogger(__name__)

GROUP_ORDER = ["TP", "FP", "FN", "TN"]
GROUP_PALETTE = {"TP": "#2c7fb8", "FP": "#d95f02", "FN": "#c51b7d", "TN": "#7fbc41"}

#: Metric display orders (`plots.metric_order`, `plots.group_metric_order`).
METRIC_ORDERS: dict[str, list[str]] = {
    "pure_vs_null": [
        "sd_logit", "sd_perp", "null_dm_logit",
        "sd_prob", "null_mi", "null_entropy", "null_dm",
    ],
    "epistemic_vs_aleatoric": [
        "sd_logit", "sd_perp", "sd_prob", "null_mi",
        "null_dm_logit", "null_dm", "null_entropy",
    ],
}


def order_metrics(metrics, order: str | list[str] | None = "pure_vs_null") -> list[str]:
    """
    Order metric names for display, keeping any not covered by the ordering at the end.

    `order` may be a key of METRIC_ORDERS or an explicit list of names.
    """
    present = list(dict.fromkeys(metrics))
    if order is None:
        return present
    ranking = METRIC_ORDERS.get(order, order) if isinstance(order, str) else list(order)
    if isinstance(order, str) and order not in METRIC_ORDERS:
        raise KeyError(f"unknown metric order '{order}'; available: {sorted(METRIC_ORDERS)}")
    index = {name: i for i, name in enumerate(ranking)}
    return sorted(present, key=lambda m: (index.get(m, len(index)), present.index(m)))


def select_metrics(metrics, include=None, order="pure_vs_null") -> list[str]:
    """Filter to `include` (if given) and then apply the display ordering."""
    present = list(dict.fromkeys(metrics))
    if include is not None:
        wanted = [m for m in include if m in present]
        missing = [m for m in include if m not in present]
        if missing:
            logger.warning("Requested metrics not available and skipped: %s", ", ".join(missing))
        present = wanted or present
    return order_metrics(present, order)


def grid_shape(n_metrics: int, ncols: int) -> tuple[int, int]:
    """Rows and columns for a panel grid, preferring a square layout."""
    ncols = max(1, min(ncols, n_metrics))
    for candidate in range(ncols, 1, -1):
        if n_metrics % candidate == 0 and n_metrics // candidate <= candidate:
            ncols = candidate
            break
    return int(np.ceil(n_metrics / ncols)), ncols


def _signed_limits(values, pad: float = 0.08, floor: float = 0.15) -> tuple[float, float]:
    """Y-limits with each side scaled to the data it contains."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return -1.05, 1.05
    lo, hi = float(v.min()), float(v.max())
    lo = min(0.0, lo) - max(pad * abs(lo), pad * floor)
    hi = max(0.0, hi) + max(pad * abs(hi), pad * floor)
    return max(-1.05, lo), min(1.05, hi)


def set_style(context: str = "paper") -> None:
    """Apply a consistent publication style to all figures."""
    sns.set_theme(context=context, style="whitegrid")
    plt.rcParams.update(
        {"axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10, "figure.autolayout": True}
    )


class FigureWriter:
    """Saves figures to a run directory with a consistent name, format and dpi."""

    def __init__(self, directory: str | Path, fmt: str = "pdf", dpi: int = 300, enabled: bool = True):
        self.directory = ensure_dir(directory)
        self.fmt, self.dpi, self.enabled = fmt, dpi, enabled

    def save(self, fig: plt.Figure, name: str) -> Path | None:
        if not self.enabled:
            plt.close(fig)
            return None
        path = self.directory / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved -> %s", path)
        return path

    def render(self, builder, name: str, *args, **kwargs) -> Path | None:
        """Build and save a figure, skipping construction entirely when plotting is off."""
        if not self.enabled:
            return None
        return self.save(builder(*args, **kwargs), name)


def plot_sd_vs_pbar(
    mean_proba: np.ndarray,
    sd_logit: np.ndarray,
    fitted: np.ndarray,
    sd_perp: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """Left: SD_logit vs p-bar with the fitted spline. Right: SD_perp vs p-bar."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = np.argsort(mean_proba)

    axes[0].scatter(mean_proba, sd_logit, s=6, alpha=0.25, color="#4c72b0", edgecolors="none")
    axes[0].plot(mean_proba[order], fitted[order], color="#c44e52", lw=2.2, label=r"fitted $s(\bar{p})$")
    axes[0].set_ylabel(r"SD$_{logit}$")
    axes[0].set_title("Ensemble disagreement vs predicted probability")
    axes[0].legend()

    axes[1].scatter(mean_proba, sd_perp, s=6, alpha=0.25, color="#55a868", edgecolors="none")
    axes[1].axhline(0, color="0.3", lw=1, ls="--")
    axes[1].set_ylabel(r"SD$_\perp$")
    axes[1].set_title("After removing the fitted trend")

    for ax in axes:
        ax.set_xlabel(r"$\bar{p}$")
        ax.set_xscale("log")
    if title:
        fig.suptitle(title, y=1.03)
    return fig


def _long_frame(scores: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> pd.DataFrame:
    records = []
    for metric, values in scores.items():
        for group in GROUP_ORDER:
            for v in values[masks[group]]:
                records.append({"metric": metric, "group": group, "value": v})
    return pd.DataFrame(records)


def plot_group_distributions(
    scores: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    labels: dict[str, str] | None = None,
    annotations: pd.DataFrame | None = None,
    title: str = "",
    ncols: int = 2,
    order: str | list[str] | None = "pure_vs_null",
) -> plt.Figure:
    """Metric distributions by confusion group, one panel per metric, with effect sizes."""
    labels = labels or {}
    df = _long_frame(scores, masks)
    metrics = order_metrics(scores, order)
    nrows, ncols = grid_shape(len(metrics), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)

    for ax, metric in zip(axes.ravel(), metrics):
        sub = df[df["metric"] == metric]
        sns.boxplot(
            data=sub, x="group", y="value", order=GROUP_ORDER, hue="group",
            palette=GROUP_PALETTE, legend=False, showfliers=False, width=0.6, ax=ax,
        )
        counts = sub.groupby("group")["value"].size().reindex(GROUP_ORDER).fillna(0).astype(int)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([f"{g}\nn={counts[g]}" for g in GROUP_ORDER])
        ax.set_title(labels.get(metric, metric))
        ax.set_xlabel("")
        ax.set_ylabel("Percentile rank")
        if annotations is not None:
            note = annotations[annotations["metric"] == metric]
            text = "  ".join(
                f"{row.contrast}: r={row.r:+.2f}" + ("*" if row.p_value < 0.05 else "")
                for row in note.itertuples()
            )
            if text:
                ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=8.5,
                        bbox=dict(fc="white", ec="0.8", alpha=0.85))

    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, y=1.02)
    return fig


def plot_threshold_sweep(
    sweep: pd.DataFrame,
    tau_operating: float | None = None,
    labels: dict[str, str] | None = None,
    title: str = "",
) -> plt.Figure:
    """Rank-biserial correlation for FP vs FN across the threshold range."""
    labels = labels or {}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for metric, sub in sweep.groupby("metric", sort=False):
        sub = sub.sort_values("tau")
        ax.plot(sub["tau"], sub["r"], marker="o", ms=3, lw=1.8, label=labels.get(metric, metric))
    ax.axhline(0, color="0.3", lw=1, ls="--")
    if tau_operating is not None:
        ax.axvline(tau_operating, color="0.45", lw=1.2, ls=":")
        ax.text(tau_operating, ax.get_ylim()[1], r"  operating $\tau$", va="top", fontsize=9, color="0.35")
    ax.set_xlabel(r"Decision threshold $\tau$")
    ax.set_ylabel("Rank-biserial $r$ (FP vs FN)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(ncol=2, fontsize=9)
    ax.set_title(title or "FP-vs-FN separation as a function of the operating threshold")
    return fig


def plot_residual_coverage(
    residuals: pd.DataFrame, labels: dict[str, str] | None = None, title: str = ""
) -> plt.Figure:
    """Mean absolute residual versus coverage, separately for y = 0 and y = 1."""
    labels = labels or {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, cls, panel in zip(axes, (0, 1), ("y = 0", "y = 1")):
        sub_cls = residuals[residuals["y_true"] == cls]
        for metric, sub in sub_cls.groupby("metric", sort=False):
            sub = sub.sort_values("coverage")
            ax.plot(sub["coverage"], sub["mean_abs_residual"], marker="o", ms=3.5,
                    label=labels.get(metric, metric))
        ax.set_title(panel)
        ax.set_xlabel("Coverage (least uncertain first)")
        ax.set_ylabel(r"Mean $|y-\bar{p}|$")
    axes[0].legend(fontsize=9)
    if title:
        fig.suptitle(title, y=1.03)
    return fig


def plot_deferral_net_benefit(
    curve: pd.DataFrame, labels: dict[str, str] | None = None, title: str = ""
) -> plt.Figure:
    """Standardised net benefit and retained FNs vs coverage, with treat-all/none references."""
    labels = labels or {}
    ref = curve.attrs.get("reference", {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for rule, sub in curve.groupby("rule", sort=False):
        sub = sub.sort_values("coverage")
        style = dict(marker="o", ms=3.5, lw=1.8)
        if rule == "random":
            style.update(color="0.55", ls="--", marker=None)
        axes[0].plot(sub["coverage"], sub["standardized_net_benefit"],
                     label=labels.get(rule, rule), **style)
        axes[1].plot(sub["coverage"], sub["FN_retained"], label=labels.get(rule, rule), **style)

    if "treat_all_snb" in ref:
        axes[0].axhline(ref["treat_all_snb"], color="#8c564b", lw=1.2, ls=":", label="treat all")
        axes[0].axhline(0.0, color="#1f77b4", lw=1.2, ls=":", label="treat none")

    axes[0].set_ylabel("Standardized net benefit")
    axes[0].set_title(
        rf"Net benefit at $\tau$ = {ref.get('tau', float('nan')):.3f}"
        f" (deferred: {ref.get('deferred_policy', 'n/a')})"
    )
    axes[1].set_ylabel("False negatives among retained")
    axes[1].set_title("Missed positives kept in the automated arm")
    for ax in axes:
        ax.set_xlabel("Coverage (fraction retained)")
    axes[0].legend(fontsize=9, ncol=2)
    if title:
        fig.suptitle(title, y=1.03)
    return fig


# ------------------------------------------------ risk-coverage and calibration
def plot_risk_coverage(
    risk: pd.DataFrame, metric_col: str = "risk_accuracy", labels: dict[str, str] | None = None,
    title: str = "",
) -> plt.Figure:
    """Panels: 1 - AUC, `metric_col` (default 1 - accuracy) and retained prevalence vs coverage."""
    labels = labels or {}
    panels = ["risk_auc", metric_col, "retained_prevalence"]
    ylabels = {
        "risk_auc": "Risk (1 - AUC)",
        "risk_accuracy": "Risk (1 - accuracy)",
        "fnr": "False-negative rate",
        "retained_prevalence": "Prevalence among retained",
    }
    titles = {
        "risk_auc": "Ranking quality of the retained set",
        "retained_prevalence": "Composition shift induced by selection",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    for metric, sub in risk.groupby("metric", sort=False):
        sub = sub.sort_values("coverage")
        for ax, col in zip(axes, panels):
            ax.plot(sub["coverage"], sub[col], marker="o", ms=3.5,
                    label=labels.get(metric, metric))
    for ax, col in zip(axes, panels):
        ax.set_ylabel(ylabels.get(col, col))
        ax.set_xlabel("Coverage")
        if col in titles:
            ax.set_title(titles[col])
    axes[0].legend(fontsize=9)
    if title:
        fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig


def plot_calibration(
    tables: dict[str, pd.DataFrame],
    title: str = "",
    summaries: dict[str, dict] | None = None,
) -> plt.Figure:
    """Reliability diagram per split; `summaries` adds prevalence, p-bar and Brier statistics."""
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    lim = 0.0
    for name, tab in tables.items():
        ax.plot(tab["mean_predicted"], tab["observed_rate"], marker="o", ms=4, label=name)
        lim = max(lim, tab["mean_predicted"].max(), tab["observed_rate"].max())
    lim = min(1.0, lim * 1.1)
    ax.plot([0, lim], [0, lim], ls="--", color="0.4", lw=1, label="perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title(title or "Calibration")
    ax.legend(fontsize=9, loc="upper left")

    if summaries:
        ax.text(
            0.98, 0.02, _calibration_note(summaries), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, family="monospace",
            bbox=dict(fc="white", ec="0.8", alpha=0.9),
        )
    return fig


def _calibration_note(summaries: dict[str, dict]) -> str:
    """Aligned per-split table of prevalence, p-bar location and Brier score."""
    rows = [f"{'':<11}" + "".join(f"{s:>10}" for s in summaries)]
    fields = [
        ("n", "n", "{:.0f}"),
        ("prevalence", "prevalence", "{:.4f}"),
        ("median_pbar", "median p", "{:.4f}"),
        ("mean_pbar", "mean p", "{:.4f}"),
        ("brier_score", "Brier", "{:.4f}"),
        ("climatology", "  pi(1-pi)", "{:.4f}"),
        ("brier_skill_score", "  BSS", "{:+.4f}"),
    ]
    for key, label, fmt in fields:
        cells = "".join(f"{fmt.format(s.get(key, float('nan'))):>10}" for s in summaries.values())
        rows.append(f"{label:<11}" + cells)
    return "\n".join(rows)


def plot_metric_correlation(corr: pd.DataFrame, title: str = "") -> plt.Figure:
    """Spearman correlation of each metric with p-bar."""
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(corr) + 1.6))
    order = corr.reindex(corr["spearman_rho"].abs().sort_values().index)
    colors = ["#c44e52" if abs(v) > 0.5 else "#4c72b0" for v in order["spearman_rho"]]
    ax.barh(order["metric"], order["spearman_rho"], color=colors)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(r"Spearman $\rho$ with $\bar{p}$")
    ax.set_title(title or "Dependence of each uncertainty metric on predicted probability")
    return fig


# ----------------------------------------------------- correct vs incorrect
CORRECTNESS_PALETTE = {"correct": "#4c72b0", "incorrect": "#c44e52"}


def plot_error_separation(
    long_df: pd.DataFrame,
    bounds: pd.DataFrame | None = None,
    labels: dict[str, str] | None = None,
    effects: pd.DataFrame | None = None,
    title: str = "",
    ncols: int = 3,
    order: str | list[str] | None = "pure_vs_null",
) -> plt.Figure:
    """Correct vs incorrect uncertainty on validation and test, with validation-derived bounds."""
    labels = labels or {}
    metrics = order_metrics(long_df["metric"], order)
    nrows, ncols = grid_shape(len(metrics), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)
    bound_rows = {} if bounds is None else {r.metric: r for r in bounds.itertuples()}

    for ax, metric in zip(axes.ravel(), metrics):
        sub = long_df[long_df["metric"] == metric]
        sns.boxplot(
            data=sub, x="split", y="value", hue="correctness",
            order=["validation", "test"], hue_order=list(CORRECTNESS_PALETTE),
            palette=CORRECTNESS_PALETTE, showfliers=False, width=0.65, ax=ax,
        )
        row = bound_rows.get(metric)
        if row is not None:
            ax.axhline(row.t_correct, ls="--", lw=1.3, color="#4c72b0")
            ax.axhline(row.t_incorrect, ls="--", lw=1.3, color="#c44e52")
            if row.band_inverted:
                ax.text(0.5, 0.02, "band inverted: distributions overlap",
                        transform=ax.transAxes, ha="center", fontsize=8.5, color="#8b0000")
        if effects is not None:
            e = effects[(effects["metric"] == metric)].set_index("contrast")["r"]
            note = (f"pooled r={e.get('pooled', float('nan')):+.2f}\n"
                    f"within-ŷ r={e.get('stratum_weighted', float('nan')):+.2f}")
            ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(fc="white", ec="0.8", alpha=0.85))
        ax.set_title(labels.get(metric, metric))
        ax.set_xlabel("")
        ax.set_ylabel("Percentile rank")
        ax.legend_.remove() if ax.legend_ else None

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in CORRECTNESS_PALETTE.values()]
    fig.legend(handles, list(CORRECTNESS_PALETTE), loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02))
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, y=1.02)
    return fig


def plot_error_separation_effects(
    effects: pd.DataFrame, labels: dict[str, str] | None = None, title: str = ""
) -> plt.Figure:
    """Pooled, within-predicted-label and stratum-weighted error-separation effects."""
    labels = labels or {}
    order = ["pooled", "within_pred_pos", "within_pred_neg", "stratum_weighted"]
    pretty = {
        "pooled": "pooled (all cases)",
        "within_pred_pos": r"within $\hat{y}=1$ (FP vs TP)",
        "within_pred_neg": r"within $\hat{y}=0$ (FN vs TN)",
        "stratum_weighted": "stratum-weighted",
    }
    sub = effects[effects["contrast"].isin(order)].copy()
    sub["contrast"] = pd.Categorical(sub["contrast"], order, ordered=True)
    sub["metric_label"] = sub["metric"].map(lambda m: labels.get(m, m))

    fig, ax = plt.subplots(figsize=(1.6 * sub["metric"].nunique() + 3.0, 5.8))
    sns.barplot(data=sub, x="metric_label", y="r", hue="contrast", ax=ax,
                palette="Blues_r", edgecolor="0.3")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_ylim(*_signed_limits(sub["r"]))
    ax.set_xlabel("")
    ax.set_ylabel("Rank-biserial $r$ (incorrect vs correct)", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, [pretty.get(x, x) for x in lbls], fontsize=12, ncol=2)
    ax.set_title(title or "Pooled separation decomposed into composition and signal",
                 fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------- residual vs uncertainty
def plot_residual_uncertainty(
    binned: pd.DataFrame,
    loess: pd.DataFrame | None = None,
    scatter: pd.DataFrame | None = None,
    association: pd.DataFrame | None = None,
    labels: dict[str, str] | None = None,
    title: str = "",
    ncols: int = 3,
    order: str | list[str] | None = "pure_vs_null",
) -> plt.Figure:
    """Mean |y - p-bar| vs uncertainty, with the calibration-implied 2 p-bar(1 - p-bar) (dashed)."""
    labels = labels or {}
    metrics = order_metrics(binned["metric"], order)
    nrows, ncols = grid_shape(len(metrics), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)

    for ax, metric in zip(axes.ravel(), metrics):
        if scatter is not None:
            s = scatter[scatter["metric"] == metric]
            ax.scatter(s["uncertainty"], s["residual"], s=4, alpha=0.12,
                       color="0.55", edgecolors="none", zorder=1)
        b = binned[binned["metric"] == metric].sort_values("mean_uncertainty")
        ax.plot(b["mean_uncertainty"], b["mean_abs_residual"], marker="o", ms=4.5,
                lw=1.8, color="#c44e52", label="observed", zorder=3)
        ax.plot(b["mean_uncertainty"], b["implied_residual"], ls="--", lw=1.8,
                color="#4c72b0", label=r"calibration-implied $2\bar{p}(1-\bar{p})$", zorder=3)
        if loess is not None:
            l = loess[loess["metric"] == metric]
            if len(l):
                ax.plot(l["uncertainty"], l["loess_residual"], lw=1.2,
                        color="#c44e52", alpha=0.5, zorder=2)
        if association is not None:
            a = association[association["metric"] == metric]
            if len(a):
                r = a.iloc[0]
                ax.text(0.02, 0.98,
                        f"$\\rho$ overall={r.rho_overall:+.2f}\n"
                        f"$\\rho$ within-$\\bar{{p}}$={r.rho_within_pooled:+.2f}",
                        transform=ax.transAxes, va="top", fontsize=8.5,
                        bbox=dict(fc="white", ec="0.8", alpha=0.85))
        ax.set_title(labels.get(metric, metric))
        ax.set_xlabel("Uncertainty (percentile rank)")
        ax.set_ylabel(r"Mean $|y-\bar{p}|$")

    axes.ravel()[0].legend(fontsize=8.5, loc="lower right")
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, y=1.02)
    return fig


# --------------------------- uncertainty-stratified calibration and net benefit
GROUP_STYLE = {
    "low_uncertainty": dict(color="#4c72b0", marker="o", label="20% least uncertain"),
    "high_uncertainty": dict(color="#c44e52", marker="s", label="20% most uncertain"),
}


def plot_uncertainty_group_calibration(
    calibration: pd.DataFrame,
    overlap: pd.DataFrame | None = None,
    labels: dict[str, str] | None = None,
    title: str = "",
    ncols: int = 3,
    brier: pd.DataFrame | None = None,
    max_probability: float = 0.5,
    metrics_include: list[str] | None = None,
    order: str | list[str] | None = "epistemic_vs_aleatoric",
) -> plt.Figure:
    """
    Reliability of the least- and most-uncertain groups per metric, annotated with
    median p-bar, p-bar overlap, prevalence, Brier score and BSS_within.
    `max_probability` fixes both axes on every panel.
    """
    labels = labels or {}
    metrics = select_metrics(calibration["metric"], metrics_include, order)
    nrows, ncols = grid_shape(len(metrics), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.2 * nrows), squeeze=False)
    ov = {} if overlap is None else {r.metric: r for r in overlap.itertuples()}

    bs_rows = {} if brier is None else {
        (r.metric, r.group): r for r in brier.itertuples()
    }
    cohort = None if brier is None else brier[brier["metric"] == "all"]

    lim = float(max_probability)
    for ax, metric in zip(axes.ravel(), metrics):
        sub = calibration[calibration["metric"] == metric]
        pred_max = obs_max = 0.0
        for group, style in GROUP_STYLE.items():
            g = sub[sub["group"] == group].sort_values("mean_predicted")
            if not len(g):
                continue
            ax.plot(g["mean_predicted"], g["observed_rate"], ms=4.5, lw=1.8, **style)
            pred_max = max(pred_max, g["mean_predicted"].max())
            obs_max = max(obs_max, g["observed_rate"].max())
        if pred_max > lim:
            logger.warning(
                "%s: predicted probabilities reach %.3f but max_probability=%.2f; "
                "raise it to show the full range",
                metric, pred_max, lim,
            )
        elif obs_max > lim:
            logger.debug(
                "%s: reliability curve clipped at the axis limit (observed rate %.2f)",
                metric, obs_max,
            )
        ax.plot([0, lim], [0, lim], ls="--", lw=1, color="0.4")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        row = ov.get(metric)
        note = []
        if row is not None:
            note.append(
                f"median $\\bar{{p}}$: {row.low_pbar_median:.3f} / {row.high_pbar_median:.3f}\n"
                f"$\\bar{{p}}$ overlap: {row.overlap_fraction:.2f}"
            )
        stats = [bs_rows.get((metric, g)) for g in GROUP_STYLE]
        if all(s is not None for s in stats):
            lo, hi = stats
            line = f"BSS: {lo.bss_within:+.3f} / {hi.bss_within:+.3f}"
            if np.isfinite(getattr(lo, "bss_within_lo", np.nan)):
                line += (f"\n  CI [{lo.bss_within_lo:+.2f},{lo.bss_within_hi:+.2f}]"
                         f" / [{hi.bss_within_lo:+.2f},{hi.bss_within_hi:+.2f}]")
            note.append(
                f"prevalence: {lo.prevalence:.3f} / {hi.prevalence:.3f}\n"
                f"BS: {lo.brier_score:.4f} / {hi.brier_score:.4f}\n" + line
            )
        if note:
            ax.text(0.02, 0.98, "\n".join(note), transform=ax.transAxes, va="top",
                    fontsize=8.5, bbox=dict(fc="white", ec="0.8", alpha=0.85))
        ax.set_title(labels.get(metric, metric))
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed rate")

    axes.ravel()[0].legend(fontsize=8.5, loc="lower right")
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, y=1.02)
    if cohort is not None and len(cohort):
        c = cohort.iloc[0]
        fig.text(
            0.5, -0.01,
            f"Full cohort: BS={c.brier_score:.4f}  prevalence={c.prevalence:.4f}.  "
            r"BSS$_{within}$ is referenced to each group's own base rate $\pi_g(1-\pi_g)$, "
            "so it measures skill net of prevalence.",
            ha="center", fontsize=8.5, color="0.35",
        )
    return fig


def plot_uncertainty_group_net_benefit(
    curve: pd.DataFrame,
    tau: float | None = None,
    labels: dict[str, str] | None = None,
    value_col: str = "snb_conditional",
    title: str = "",
    ncols: int = 3,
    metrics_include: list[str] | None = None,
    order: str | list[str] | None = "epistemic_vs_aleatoric",
) -> plt.Figure:
    """Decision curves for the least- and most-uncertain groups (`value_col`: snb_fixed or snb_conditional)."""
    labels = labels or {}
    conditional = value_col == "snb_conditional"
    all_col = "treat_all_conditional" if conditional else "treat_all_fixed"
    metrics = select_metrics(curve["metric"], metrics_include, order)
    nrows, ncols = grid_shape(len(metrics), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)

    for ax, metric in zip(axes.ravel(), metrics):
        sub = curve[curve["metric"] == metric]
        note = []
        for group, style in GROUP_STYLE.items():
            g = sub[sub["group"] == group].sort_values("threshold")
            if not len(g):
                continue
            ax.plot(g["threshold"], g[value_col], lw=2,
                    color=style["color"], label=style["label"])
            if conditional and all_col in g:
                ax.plot(g["threshold"], g[all_col], lw=1.1, ls=":",
                        color=style["color"], alpha=0.8)
            note.append(f"{style['label'].split('%')[0]}%: prev={g['group_prevalence'].iloc[0]:.3f}")
        if not conditional and all_col in sub:
            ref = sub.sort_values("threshold")
            ax.plot(ref["threshold"], ref[all_col], lw=1.2, ls=":",
                    color="#8c564b", label="treat all (cohort)")
        ax.axhline(0.0, lw=1.2, ls=":", color="0.45")
        if tau is not None:
            ax.axvline(tau, lw=1.1, ls="-.", color="0.35")
        ax.set_ylim(-0.35, 1.05)
        ax.set_title(labels.get(metric, metric))
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Standardized net benefit"
                      + (" (within group)" if conditional else " (full cohort)"))
        if note:
            ax.text(0.98, 0.98, "\n".join(note), transform=ax.transAxes, va="top",
                    ha="right", fontsize=8, bbox=dict(fc="white", ec="0.8", alpha=0.85))

    axes.ravel()[0].legend(fontsize=8.5, loc="lower left")
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, y=1.02)
    return fig
