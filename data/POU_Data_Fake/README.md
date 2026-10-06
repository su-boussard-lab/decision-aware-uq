# Prolonged Opioid Use (POU) — synthetic example

The POU cohort (STARR OMOP; 10,636 covariates; 26,979 / 6,227 / 8,301
train / validation / test patients) cannot be shared publicly (see Data
availability in the manuscript). This folder contains 10 **synthetic** patients
that show the file format only; they cannot be used to reproduce results.

| File | Contents |
|---|---|
| `cohort_subset.csv` | `rowId`, `outcome` (1 = prolonged opioid use) |
| `data_subset.csv` | `rowId` plus one column per covariate (by `covariateId`) |
| `covariate_subset.csv` | covariate dictionary: `covariateId`, `covariateName` |

To run POU with approved access, place `cohort.csv` (`rowId`, `outcome`,
`evaluationType` = `Train`/`Test`) and `dataset.csv` in `data/POU_Data/`.
