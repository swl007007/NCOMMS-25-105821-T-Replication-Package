# Missingness Sensitivity Design

**Date:** 2026-09-02
**Status:** Approved and implemented; main-model and baseline results remain frozen
**Scope:** Test whether predictor missingness patterns materially change Forecasting and Nowcasting comparisons among XGBoost, Ensemble OLS, and Ordered Probit.

## 1. Isolation boundary

This is a sensitivity analysis. Existing main-model and baseline results are frozen and must not be overwritten or regenerated in place.

All new outputs go under:

`2.Source Code/produced_graph/missingness_sensitivity/`

The analysis retains the existing temporal split, targets, model definitions, preprocessing policies, and evaluation metric definitions. Hyperparameters are never retuned.

## 2. Feature-removal sensitivity

Use thresholds `0%`, `5%`, `10%`, `30%`, and `50%`.

For each source-role feature set:

1. calculate each feature's missing-cell proportion using pre-2022 training rows only;
2. rank features from most to least missing, breaking ties by source-column order; and
3. remove the first `ceil(threshold * feature_count)` features.

The feature sets are:

- Forecasting: one 106-feature ranking shared by XGBoost, Ensemble OLS, and Ordered Probit;
- Nowcasting XGBoost and Ensemble OLS: separate rankings for the 106 Layer-1 Forecasting-source features and 69 Layer-2 Nowcasting-source features; and
- Nowcasting Ordered Probit: one ranking over its direct 173-feature source table.

Refit all six task-model combinations at every threshold:

- Forecasting and Nowcasting; multiplied by
- XGBoost, Ensemble OLS, and Ordered Probit.

Because the Nowcasting architectures do not use identical feature roles, this experiment is interpreted as within-method availability sensitivity, not as a same-feature algorithm-only comparison.

## 3. Country-removal sensitivity

Use the same five thresholds. Rank all 29 countries using pre-2022 feature-cell missingness:

- Forecasting score: the 106 Layer-1 features; and
- Nowcasting score: the 106 Layer-1 plus 69 Layer-2 features.

For each task and threshold, remove the first `ceil(threshold * 29)` countries from both training and test data. All three models within a task use the same retained countries and observation keys.

Refit the same six task-model combinations at every threshold. If a selected country has no 2022 test rows, its removal changes training only; the metrics output must report the removed ISO3 codes together with `n_train` and `n_test`.

## 4. Missing-indicator sensitivity

Do not rerun XGBoost. Use the two frozen main XGBoost results as references.

Refit only:

- Forecasting Ensemble OLS;
- Nowcasting Ensemble OLS;
- Forecasting Ordered Probit; and
- Nowcasting Ordered Probit.

Before imputation, add one binary missing indicator for each source-role feature, with `NaN`, positive infinity, and negative infinity coded as missing. Apply the existing train-only imputation, scaling, and deterministic rank-pruning policies to the augmented baseline matrices. Do not change any other model policy.

## 5. Fixed modeling and failure rules

- Reuse the existing XGBoost hyperparameters without retuning.
- Reuse the existing Ensemble OLS estimator and preprocessing policy.
- Reuse the existing Ordered Probit configuration, including BFGS and the 1,000-iteration limit.
- Never use 2022 test rows to rank features or countries, fit preprocessing, select columns, or tune models.
- Never adapt a deletion set in response to model failure.
- If a condition cannot be estimated, record `status=not_estimable` and a concise reason, then continue the remaining conditions.

Examples of `not_estimable` include a missing Ordered Probit outcome class, no valid predictor columns, or a required key mismatch.

## 6. Outputs

Write only these required artifacts inside the sensitivity directory:

- one metrics CSV; and
- one sensitivity-curves figure published as vector PDF and 300 dpi PNG.

The metrics CSV contains one row per task, model, experiment, and threshold or indicator condition. It records:

- experiment and threshold;
- task and model;
- removed feature or country count;
- removed ISO3 codes for country-removal conditions;
- raw accuracy;
- Phase-3+ precision and recall;
- R-squared;
- `n_train` and `n_test`;
- status and reason; and
- metric deltas relative to the relevant XGBoost result: same-threshold XGBoost for feature/country removal and frozen main XGBoost for the indicator experiment.

The figure plots accuracy, Phase-3+ precision, Phase-3+ recall, and Phase-3+ R-squared for the feature- and country-removal experiments in a 4-by-4 panel grid. The first three metric columns use fixed `[0, 1]` limits. All four R-squared panels share data-derived limits that include zero and display a zero-reference line. Country-removal panels show the actual removed-country count and `n_test`. Missing-indicator results remain in the metrics CSV because they are a single condition rather than a curve.

## 7. Non-goals

This work will not:

- alter or overwrite existing main or baseline artifacts;
- retune any model;
- replace XGBoost's native missing-value routing;
- redefine the temporal evaluation population except for the explicitly selected country removals;
- treat feature-removal results as strict same-input algorithm comparisons; or
- modify manuscript text or adopt a new main result.

Implementation requires a separate approved plan.
