# Diabetic Patient Readmission (DPR)

101,766 encounters from 130 US hospitals (1999–2008; Strack et al., 2014), from
the UCI *Diabetes 130-US Hospitals* release
([doi:10.24432/C5230J](https://doi.org/10.24432/C5230J)) via a Kaggle mirror
(<https://www.kaggle.com/datasets/saurabhtayal/diabetic-patients-readmission-prediction>).

Outcome `readmit_30d`: 1 if `readmitted == "<30"`, else 0.

| Folder | Contents |
|---|---|
| `raw_data/` | source file (`diabetic_data.csv`), ID mappings and data dictionary |
| `intermediate_data/` | cohort and feature tables, and train/validation/test splits before encoding |
| `preprocessed_data/dpr_preprocessed.zip` | model-ready `train.csv`, `validation.csv`, `test.csv` used in the manuscript (71,236 / 15,265 / 15,265 encounters; 2,332 features) |

`run_experiments.sh` extracts the zip to `preprocessed_data/dpr_preprocessed/`.
