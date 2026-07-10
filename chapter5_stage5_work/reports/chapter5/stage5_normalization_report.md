# Stage 5. Normalization of prior features

## Status

Stage 5 implementation is complete. The stage is not methodologically closed until the user runs and confirms the closing checks in the working Windows environment.

## Implemented changes

- Implemented `src/manual_coding_sim/prediction/prior_feature_normalizer.py`.
- Added min-max normalization for numeric `prior_*` features.
- Added inverse normalization for `lower_is_better` features.
- Added direct normalization for `higher_is_better` and `neutral` numeric features.
- Added handling of constant numeric features.
- Added skipping and reporting of non-numeric prior features.
- Added output paths to `configs/chapter5.yaml` and `Chapter5OutputPaths`.
- Added CLI flag `--normalize-inputs`.
- Aligned `data/schema/prior_feature_dictionary.yaml` with the 150-row extended corpus.
- Aligned `configs/chapter5_feature_weights.yaml` with actually available `prior_*` columns.
- Added tests for the normalizer and CLI normalization path.

## Generated artifacts

- `reports/chapter5/normalized_prior_features.csv`
- `reports/chapter5/normalization_report.json`

## Local validation results

- `python -m pytest tests/prediction -q`: 36 passed.
- `python -m pytest -q`: 284 passed.
- CLI normalization completed successfully.

## Normalization summary

- Input rows: 150.
- Input columns: 36.
- Normalized numeric prior features: 30.
- Skipped non-numeric prior features: 2.
- Non-numeric skipped features:
  - `prior_coding_tool_type`
  - `prior_condition_profile`
- Unknown input features: 0.
- Missing dictionary features: 0.
- Normalized value range: `[0, 1]`.
- Missing values in normalized columns: 0.

## Methodological note

This stage only prepares normalized `X_prior` for subsequent construction of partial criteria. It does not compute `q_acc_pred`, `q_time_pred`, `q_effort_pred`, `q_res_pred`, `q_rep_pred`, `q_fit_pred`, or `Q_pred`.
