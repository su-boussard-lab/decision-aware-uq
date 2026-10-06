# Deferring on uncertainty can mean deferring on risk in imbalanced clinical prediction

Code for the decision-aware evaluation suite described in the manuscript. It
evaluates uncertainty-based deferral in two clinical prediction tasks (prolonged
opioid use, POU; 30-day diabetic readmission, DPR) with two uncertainty
quantification methods (bootstrap ensemble of L1-penalised logistic regressions;
Bayesian logistic regression by variational inference), giving four settings.
See the manuscript for methods and interpretation.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt
```

## Reproducing the results

Run from the repository root:

```bash
./run_experiments.sh          # DPR, both methods (extracts the DPR data on first run)
./run_experiments.sh pou      # POU, both methods (requires the restricted data)
```

A single setting can be run with

```bash
python run_experiments.py --config configs/ensemble_dpr.yaml
```

and any configuration value can be overridden from the command line, e.g.
`--set ensemble.n_bootstrap=200` or `--set bnn.prior=scaled_gaussian`.

| Config | Setting | Output |
|---|---|---|
| `configs/ensemble_pou.yaml` | POU, bootstrap ensemble | `results/ensemble/pou/` |
| `configs/ensemble_dpr.yaml` | DPR, bootstrap ensemble | `results/ensemble/dpr/` |
| `configs/blr_pou.yaml` | POU, Bayesian logistic regression | `results/blr/pou/` |
| `configs/blr_dpr.yaml` | DPR, Bayesian logistic regression | `results/blr/dpr/` |

Shared settings (model, ensemble size, variational inference, analyses) are in
`configs/base.yaml`. Operating threshold, percentile-rank reference and the
orthogonalisation spline are estimated on the validation split; all reported
results are computed on the test split.

## Outputs

Each run writes `figures/`, `tables/` (one CSV per analysis, plus
`test_per_sample.csv`), `summary.json` (effective config, threshold and spline
R²) and `artifacts/bundles.pkl` (per-member predicted probabilities).

| Manuscript | Figure (`figures/`) | Table (`tables/`) |
|---|---|---|
| Table 1 | — | `split_summary`, `performance`, `calibration_summary` |
| Fig. 1, Table 2 | `confound_check`, `sd_orthogonalization` | `confound_check`; spline R² in `summary.json` |
| Fig. 2, Tables S1–S2 | `confusion_groups` | `performance`, `group_comparison` |
| Fig. 3, Table S3 | `error_separation` | `error_separation` |
| Fig. 4, Table 3 | `deferral`, `risk_coverage` | `deferral`, `risk_coverage`, `decision_impact` |
| Fig. 5, Table S5 | `group_calibration` | `group_overlap`, `group_brier` |
| Fig. S1, Table S4 | `residual_uncertainty` | `residual_uncertainty_binned`, `residual_partial_association` |
| Fig. S2 | `threshold_sweep` | `threshold_sweep` |
| Fig. S3 | `residual_coverage` | `residual_coverage` |
| Fig. S4 | `calibration` | `calibration_test` |
| Fig. S5 | `group_net_benefit_fixed` | `group_net_benefit` |
| Fig. S6 | `risk_coverage` | `risk_coverage` |

## Data

- **DPR** (public): raw, intermediate and preprocessed train/validation/test
  files are in `data/DPR_Data/` (see its README).
- **POU** (restricted): only a synthetic example with the same file structure
  is provided in `data/POU_Data_Fake/`. With approved access, place the data in
  `data/POU_Data/`; otherwise the POU settings are skipped.

## Repository structure

```
configs/       base.yaml and one config per setting
data/          DPR data and the synthetic POU example
duq/
  data.py          dataset loaders
  models.py        L1-penalised logistic regression
  ensemble.py      bootstrap ensemble; Bayesian LR wrapper
  bnn_vi.py        Bayesian logistic regression (mean-field VI)
  uncertainty.py   uncertainty metrics and percentile ranking
  orthogonalize.py geometry adjustment (SD_perp)
  evaluation.py    effect sizes, threshold sweep, net benefit, deferral
  reliability.py   error separation, residual analyses, uncertainty groups
  plots.py         figures
  pipeline.py      orchestration
  config.py, cli.py, utils.py
run_experiments.py
run_experiments.sh
```

## Citation

If you use this code, please cite the manuscript (citation will be added soon).

## License

MIT; see `LICENSE`.
