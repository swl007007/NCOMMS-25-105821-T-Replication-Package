# Missingness Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one isolated missingness-sensitivity metrics table and one sensitivity-curve figure in PDF and PNG for Forecasting and Nowcasting without changing any frozen main-model or baseline artifact.

**Architecture:** Add one standalone generator and one focused unittest module. The generator loads the existing prepared data and model helpers, constructs deterministic feature-removal, country-removal, and missing-indicator conditions, runs the existing XGBoost architecture plus the existing OLS/Ordered Probit numerical routines, and publishes only a metrics CSV plus matched PDF and PNG renderings of one figure. Existing generators remain read-only because their recorded source hashes belong to frozen artifacts.

**Tech Stack:** Python 3.11.3, pandas 2.2.3, NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.5.2, XGBoost 2.0.3, statsmodels 0.14.6, matplotlib 3.10.1, pathlib, argparse, tempfile, unittest.

**Spec:** `docs/superpowers/specs/2026-09-02-missingness-sensitivity-design.md`

**Status:** Approved and executed; commit, merge, and push remain unauthorized.

## Global Constraints

- Existing main-model, baseline, manuscript, notebook, data, and `produced_graph` artifacts are read-only.
- Formal outputs are restricted to `2.Source Code/produced_graph/missingness_sensitivity/`.
- The temporal split remains training before `2022-01-01` and testing on or after `2022-01-01`.
- Formal full-data counts remain 4,405 training rows and 1,170 test rows before an explicitly selected country removal.
- Feature and country thresholds are exactly `0`, `5`, `10`, `30`, and `50` percent.
- Missingness means `NaN`, positive infinity, or negative infinity.
- Feature ranking uses pre-2022 training rows only; ties retain source-column order.
- Country ranking uses pre-2022 feature cells only; ties use ascending ISO3 code.
- Forecasting uses the 106 source predictors for every method.
- Cascading Nowcasting XGBoost and Ensemble OLS use 106 Layer-1 Forecasting-source predictors and 69 Layer-2 Nowcasting-source predictors.
- Direct Nowcasting Ordered Probit uses its 173 source predictors.
- Feature removal is implemented by setting the selected feature columns to missing in condition-local copies. This makes the columns information-free while preserving the existing model helpers and fixed feature-name contracts.
- Country removal excludes the selected ISO3 codes from both training and test rows for all three methods within the task.
- Missing indicators are added only to Ensemble OLS and Ordered Probit; XGBoost is not refitted for that experiment.
- XGBoost uses the existing parameter JSON files with `random_state=None` and `estimator_n_jobs=None`, matching the selected Figure 1 lineage. No parameter is retuned.
- Ordered Probit remains `distr="probit"`, optimizer `bfgs`, and `maxiter=1000`.
- Ensemble OLS retains train-only median imputation, scaling, deterministic constant/duplicate/pivoted-QR pruning, exactly one intercept, unconstrained cumulative predictions, two-decimal phase conversion, and in-sample Layer-1 residual fitting.
- A non-estimable model-condition produces one `status=not_estimable` metrics row and does not change the deletion set or stop other conditions.
- The metrics CSV contains no selection-audit table, run-audit table, key/selection hashes, confusion counts, or indicator-count fields.
- The figure is produced only with Python/matplotlib and saved as one vector PDF plus one matching 300 dpi PNG. No JPG, SVG, TIFF, or interactive output is published.
- No new dependency, generic experiment framework, configuration registry, concurrency layer, or model cache is added.
- The existing uncommitted variable-missingness files remain untouched: `2.Source Code/generate_variable_missingness_balance.py`, `tests/test_generate_variable_missingness_balance.py`, and `2.Source Code/produced_graph/variable_missingness_balance.csv`.
- Do not commit, merge, or push unless separately authorized by the user. Each task ends with a verification checkpoint instead of a commit.

## Figure Contract

- **Core conclusion:** Show whether progressively removing the most missing features or countries changes absolute performance and the performance gap between XGBoost, Ensemble OLS, and Ordered Probit.
- **Evidence chain:** Feature-removal rows test predictor-availability sensitivity; country-removal rows test geographic concentration of missingness; columns show five-class accuracy, Phase-3+ precision, Phase-3+ recall, and Phase-3+ R-squared.
- **Archetype:** Quantitative grid.
- **Backend:** Python/matplotlib only.
- **Final size:** One landscape 4-by-4 figure published as vector PDF and matching 300 dpi PNG.
- **Panel map:** Rows are Forecasting feature removal, Nowcasting feature removal, Forecasting country removal, and Nowcasting country removal. Columns are accuracy, Phase-3+ precision, Phase-3+ recall, and Phase-3+ R-squared.
- **Evidence hierarchy:** Accuracy is the broad classification check; precision and recall are the reviewer-facing Phase-3+ evidence; R-squared shows continuous Phase-3+ fit; missing-indicator results remain in the metrics CSV because they are a single condition rather than a curve.
- **Statistics:** Plot point estimates only; this deterministic fixed-sample sensitivity design does not create uncertainty intervals.
- **Source data:** `missingness_sensitivity_metrics.csv`.
- **Integrity:** Curves use saved metric values without smoothing or interpolation; non-estimable conditions appear as broken lines rather than fabricated values.
- **Metric scales:** Accuracy, precision, and recall use fixed `[0, 1]` limits. All four R-squared panels use one data-derived y-range that includes zero and a zero-reference line.
- **Reviewer risk:** Country removal changes `n_test`; every country-panel tick must show both the actual removed-country count and retained test count.

## File Map

- Create: `2.Source Code/generate_missingness_sensitivity.py` — condition construction, model orchestration, metrics, deltas, figure, CLI, and isolated publication.
- Create: `tests/test_generate_missingness_sensitivity.py` — deterministic ranking, no-leakage, model routing, failure continuation, metrics, figure, output, and live source-contract tests.
- Generate after implementation authorization: `2.Source Code/produced_graph/missingness_sensitivity/missingness_sensitivity_metrics.csv`.
- Generate after implementation authorization: `2.Source Code/produced_graph/missingness_sensitivity/missingness_sensitivity_curves.pdf`.
- Generate after implementation authorization: `2.Source Code/produced_graph/missingness_sensitivity/missingness_sensitivity_curves.png`.
- Read only: `2.Source Code/generate_leave_one_country_out_robustness.py` — XGBoost architecture, country mapping, feature authority, cumulative targets, and phase conversion.
- Read only: `2.Source Code/generate_simple_baseline_comparison.py` — prepared inputs, numeric preprocessing, OLS arrays, Ordered Probit arrays, metric calculation, and plotting style.
- Read only: `2.Source Code/generate_multinomial_baseline_comparison.py` — direct Ordered Probit source-feature selection and evaluation-phase reconstruction.
- Read only: `2.Source Code/generate_filtered_main_result_metrics.py` — selected-lineage XGBoost parameter loading and frozen-input validation pattern.
- Read only: `2.Source Code/main_result_figure1_v1.py` — frozen main-result environment contract.
- Read only: `2.Source Code/produced_graph/simple_baseline_comparison_metrics.csv` — the two frozen XGBoost references for the missing-indicator comparison and 0% baseline regression checks.

---

### Task 1: Deterministic Missingness Selections and Transformations

**Files:**

- Create: `2.Source Code/generate_missingness_sensitivity.py`
- Create: `tests/test_generate_missingness_sensitivity.py`

**Interfaces:**

- Produces: `THRESHOLDS: tuple[int, ...]`.
- Produces: `missing_cells(data: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame`.
- Produces: `rank_features(data: pd.DataFrame, features: Sequence[str], train_mask: pd.Series) -> tuple[str, ...]`.
- Produces: `rank_countries(layers: Sequence[tuple[pd.DataFrame, Sequence[str], pd.Series]]) -> tuple[str, ...]`.
- Produces: `selection_count(percent: int, population_size: int) -> int`.
- Produces: `suppress_features(data: pd.DataFrame, selected: Sequence[str]) -> pd.DataFrame`.
- Produces: `add_missing_indicators(data: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, tuple[str, ...]]`.

- [ ] **Step 1: Create the test module and failing feature-ranking tests**

Create the import shell and tests:

```python
from __future__ import annotations

import math
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE = REPO_ROOT / "2.Source Code"
if str(SOURCE_CODE) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE))

import generate_missingness_sensitivity as sensitivity


class SelectionTests(unittest.TestCase):
    def test_feature_ranking_uses_training_only_and_source_order_for_ties(self):
        data = pd.DataFrame(
            {
                "a": [np.nan, 1.0, np.nan, np.nan],
                "b": [np.inf, 2.0, 3.0, np.nan],
                "c": [1.0, np.nan, 4.0, np.nan],
            }
        )
        train_mask = pd.Series([True, True, False, False])
        self.assertEqual(
            sensitivity.rank_features(data, ["a", "b", "c"], train_mask),
            ("a", "b", "c"),
        )

    def test_selection_count_uses_ceiling(self):
        self.assertEqual(sensitivity.selection_count(5, 29), 2)
        self.assertEqual(sensitivity.selection_count(10, 29), 3)
        self.assertEqual(sensitivity.selection_count(30, 29), 9)
        self.assertEqual(sensitivity.selection_count(50, 29), 15)
        self.assertEqual(sensitivity.selection_count(0, 29), 0)
```

- [ ] **Step 2: Add failing country-ranking and transformation tests**

```python
class TransformationTests(unittest.TestCase):
    def test_country_ranking_combines_layer_feature_cells(self):
        layer1 = pd.DataFrame(
            {
                "country_code_3": ["AAA", "AAA", "BBB", "BBB"],
                "x": [np.nan, 1.0, np.nan, np.nan],
                "y": [1.0, 1.0, np.nan, 1.0],
            }
        )
        layer2 = pd.DataFrame(
            {
                "country_code_3": ["AAA", "AAA", "BBB", "BBB"],
                "z": [1.0, np.inf, np.nan, np.nan],
            }
        )
        train = pd.Series([True, True, True, True])
        ranked = sensitivity.rank_countries(
            [(layer1, ["x", "y"], train), (layer2, ["z"], train)]
        )
        self.assertEqual(ranked, ("BBB", "AAA"))

    def test_suppress_features_does_not_mutate_source(self):
        source = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = sensitivity.suppress_features(source, ["x"])
        self.assertTrue(result["x"].isna().all())
        self.assertEqual(source["x"].tolist(), [1.0, 2.0])
        self.assertEqual(result["y"].tolist(), [3.0, 4.0])

    def test_indicators_capture_nan_and_infinity_before_imputation(self):
        source = pd.DataFrame({"x": [1.0, np.nan, np.inf, -np.inf]})
        result, indicators = sensitivity.add_missing_indicators(source, ["x"])
        self.assertEqual(indicators, ("x__missing",))
        self.assertEqual(result["x__missing"].tolist(), [0, 1, 1, 1])
        self.assertTrue(np.isinf(result.loc[[2, 3], "x"]).all())
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
python -B -m unittest discover -s tests -p "test_generate_missingness_sensitivity.py" -v
```

Expected: import failure because `generate_missingness_sensitivity.py` does not exist.

- [ ] **Step 4: Add the generator imports, constants, and selection helpers**

Start the generator with the existing authorities rather than copied feature lists:

```python
"""Run missingness sensitivity analyses without modifying frozen results."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache-missingness-sensitivity")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

import generate_filtered_main_result_metrics as selected_main
import generate_leave_one_country_out_robustness as loco
import generate_multinomial_baseline_comparison as multinomial
import generate_simple_baseline_comparison as simple
import main_result_figure1_v1 as frozen_main_result


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE_DIR = REPO_ROOT / "2.Source Code"
DEFAULT_OUTPUT_DIR = SOURCE_CODE_DIR / "produced_graph" / "missingness_sensitivity"
DEFAULT_BASE_METRICS = SOURCE_CODE_DIR / "produced_graph" / "simple_baseline_comparison_metrics.csv"
DEFAULT_GENERAL_PARAMS = SOURCE_CODE_DIR / "forecasting_hyperparameters.json"
DEFAULT_PHASE3_PARAMS = SOURCE_CODE_DIR / "forecasting_hyperparameters_p3.json"
METRICS_FILENAME = "missingness_sensitivity_metrics.csv"
FIGURE_FILENAME = "missingness_sensitivity_curves.pdf"
FIGURE_PNG_FILENAME = "missingness_sensitivity_curves.png"
THRESHOLDS = (0, 5, 10, 30, 50)
TASKS = ("Forecasting", "Nowcasting")
MODELS = ("XGBoost", "Ensemble OLS", "Ordered Probit")


def missing_cells(data: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    columns = tuple(features)
    if not columns:
        raise ValueError("Missingness calculation requires at least one feature.")
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return data.loc[:, columns].replace([np.inf, -np.inf], np.nan).isna()


def rank_features(
    data: pd.DataFrame,
    features: Sequence[str],
    train_mask: pd.Series,
) -> tuple[str, ...]:
    columns = tuple(features)
    rates = missing_cells(data.loc[train_mask], columns).mean(axis=0)
    source_order = {column: index for index, column in enumerate(columns)}
    return tuple(
        sorted(columns, key=lambda column: (-float(rates[column]), source_order[column]))
    )


def rank_countries(
    layers: Sequence[tuple[pd.DataFrame, Sequence[str], pd.Series]],
) -> tuple[str, ...]:
    summaries = []
    for data, features, train_mask in layers:
        selected = data.loc[train_mask]
        missing = missing_cells(selected, features)
        summaries.append(
            pd.DataFrame(
                {
                    "country_code_3": selected["country_code_3"].astype(str).to_numpy(),
                    "missing_cells": missing.sum(axis=1).to_numpy(dtype=int),
                    "feature_cells": len(tuple(features)),
                }
            )
        )
    totals = (
        pd.concat(summaries, ignore_index=True)
        .groupby("country_code_3", observed=True)[["missing_cells", "feature_cells"]]
        .sum()
    )
    if totals.empty or totals["feature_cells"].le(0).any():
        raise ValueError("Country missingness ranking has no eligible feature cells.")
    rates = totals["missing_cells"] / totals["feature_cells"]
    return tuple(sorted(rates.index, key=lambda iso3: (-float(rates[iso3]), iso3)))


def selection_count(percent: int, population_size: int) -> int:
    if percent not in THRESHOLDS or population_size < 1:
        raise ValueError("Selection percent or population size is outside the contract.")
    return int(math.ceil(percent / 100 * population_size))


def suppress_features(data: pd.DataFrame, selected: Sequence[str]) -> pd.DataFrame:
    result = data.copy()
    columns = tuple(selected)
    missing = set(columns).difference(result.columns)
    if missing:
        raise ValueError(f"Cannot suppress missing columns: {sorted(missing)}")
    result.loc[:, columns] = np.nan
    return result


def add_missing_indicators(
    data: pd.DataFrame,
    features: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    result = data.copy()
    indicators = tuple(f"{column}__missing" for column in features)
    collisions = set(indicators).intersection(result.columns)
    if collisions:
        raise ValueError(f"Missing-indicator columns already exist: {sorted(collisions)}")
    missing = missing_cells(result, features)
    for source, indicator in zip(features, indicators, strict=True):
        result[indicator] = missing[source].astype(np.int8)
    return result, indicators
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the same unittest command. Expected: all selection and transformation tests pass.

- [ ] **Step 6: Checkpoint the scenario contract**

Run:

```bash
python -B -m py_compile "2.Source Code/generate_missingness_sensitivity.py" tests/test_generate_missingness_sensitivity.py
```

Expected: exit status 0. Do not commit.

---

### Task 2: Condition-Local XGBoost, Ordered Probit, and Ensemble OLS Fits

**Files:**

- Modify: `2.Source Code/generate_missingness_sensitivity.py`
- Modify: `tests/test_generate_missingness_sensitivity.py`

**Interfaces:**

- Produces: `prediction_frame(actual: Sequence[int], predicted: Sequence[int]) -> pd.DataFrame`.
- Produces: `fit_xgboost_task(*, task: str, forecasting: pd.DataFrame, nowcasting: pd.DataFrame, forecasting_train_mask: pd.Series, forecasting_test_mask: pd.Series, nowcasting_train_mask: pd.Series, nowcasting_test_mask: pd.Series, general_params: Mapping[str, object], phase3_params: Mapping[str, object]) -> tuple[pd.DataFrame, int, int]`.
- Produces: `fit_ordered_probit_task(*, data: pd.DataFrame, features: Sequence[str], train_mask: pd.Series, test_mask: pd.Series, task: str) -> tuple[pd.DataFrame, int, int]`.
- Produces: `fit_ensemble_ols_task(*, task: str, forecasting: pd.DataFrame, nowcasting: pd.DataFrame, layer1_features: Sequence[str], layer2_features: Sequence[str], forecasting_train_mask: pd.Series, forecasting_test_mask: pd.Series, nowcasting_train_mask: pd.Series, nowcasting_test_mask: pd.Series) -> tuple[pd.DataFrame, int, int]`.
- Produces: `run_fit_safely(fit: Callable[[], tuple[pd.DataFrame, int, int]]) -> tuple[pd.DataFrame | None, int, int, str, str]`.

- [ ] **Step 1: Add failing tests for common predictions and safe failure continuation**

```python
class FitRoutingTests(unittest.TestCase):
    def test_prediction_frame_requires_equal_valid_phase_vectors(self):
        result = sensitivity.prediction_frame([1, 3, 5], [2, 3, 4])
        self.assertEqual(result.columns.tolist(), ["actual_phase", "predicted_phase"])
        with self.assertRaisesRegex(ValueError, "same length"):
            sensitivity.prediction_frame([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "between one and five"):
            sensitivity.prediction_frame([1], [6])

    def test_safe_fit_records_known_non_estimable_error(self):
        def fail():
            raise ValueError("missing Ordered Probit class")

        predictions, n_train, n_test, status, reason = sensitivity.run_fit_safely(
            fail,
            n_train=20,
            n_test=5,
        )
        self.assertIsNone(predictions)
        self.assertEqual((n_train, n_test), (20, 5))
        self.assertEqual(status, "not_estimable")
        self.assertEqual(reason, "missing Ordered Probit class")
```

- [ ] **Step 2: Add a failing XGBoost routing test with the existing estimator seam**

Build a tiny prepared frame containing all required outcome columns and two numeric features. Pass a recording estimator factory to `fit_xgboost_task`; assert that Forecasting calls the existing `loco.fit_forecasting_split`, returns the dynamic train/test counts, and produces `actual_phase/predicted_phase`. Use `mock.patch.object(sensitivity.loco, "fit_forecasting_split")` to return a fixed frame with `overall_phase` and `overall_phase_pred`; do not fit real XGBoost in this unit test.

```python
    def test_xgboost_forecasting_routes_through_authoritative_split_helper(self):
        forecasting = pd.DataFrame(
            {
                "area_id": [1, 1, 2],
                "date": ["2021-01-01", "2022-01-01", "2022-01-01"],
                "country_code_3": ["AAA", "AAA", "BBB"],
            }
        )
        train = pd.Series([True, False, False])
        test = ~train
        returned = pd.DataFrame(
            {"overall_phase": [2, 4], "overall_phase_pred": [3, 4]}
        )
        with mock.patch.object(
            sensitivity.loco, "fit_forecasting_split", return_value=returned
        ) as fitted:
            result, n_train, n_test = sensitivity.fit_xgboost_task(
                task="Forecasting",
                forecasting=forecasting,
                nowcasting=forecasting.copy(),
                forecasting_train_mask=train,
                forecasting_test_mask=test,
                nowcasting_train_mask=train,
                nowcasting_test_mask=test,
                general_params={"max_depth": 1},
                phase3_params={"max_depth": 1},
            )
        fitted.assert_called_once()
        self.assertEqual((n_train, n_test), (1, 2))
        self.assertEqual(result["predicted_phase"].tolist(), [3, 4])
```

- [ ] **Step 3: Add failing baseline routing tests**

Use synthetic five-class training data for direct Ordered Probit and a synthetic full-rank cumulative-target frame for Ensemble OLS. Patch only `simple.fit_ordered_probit_arrays` in the routing test so the test checks preprocessing and argument flow without repeating the existing numerical-model tests.

```python
    def test_ordered_probit_uses_supplied_direct_features(self):
        phases = np.array([1, 2, 3, 4, 5] * 2 + [2, 4])
        data = pd.DataFrame(
            {
                "date": ["2021-01-01"] * 10 + ["2022-01-01"] * 2,
                "x": np.arange(12, dtype=float),
                "x__missing": [0, 1] * 6,
            }
        )
        for phase in range(1, 6):
            data[f"phase{phase}_percent"] = (phases == phase).astype(float)
        train = pd.Series([True] * 10 + [False] * 2)
        test = ~train
        returned = (
            np.array([2, 4]),
            np.full((2, 5), 0.2),
            {"converged": True},
        )
        with mock.patch.object(
            sensitivity.simple, "fit_ordered_probit_arrays", return_value=returned
        ) as fitted:
            result, n_train, n_test = sensitivity.fit_ordered_probit_task(
                data=data,
                features=["x", "x__missing"],
                train_mask=train,
                test_mask=test,
                task="Forecasting",
            )
        self.assertEqual(fitted.call_args.args[0].shape[0], 10)
        self.assertEqual((n_train, n_test), (10, 2))
        self.assertEqual(result["predicted_phase"].tolist(), [2, 4])
```

For Ensemble OLS, use this keyed Nowcasting fixture to prove that Layer 2 is added to Layer 1 before phase conversion:

```python
    def test_ensemble_ols_nowcasting_adds_layer1_and_residual(self):
        phases = np.array([1, 2, 3, 4, 5, 3, 1])
        forecasting = pd.DataFrame(
            {
                "area_id": np.arange(1, 8),
                "date": ["2021-01-01"] * 5 + ["2022-01-01"] * 2,
                "country_code_3": ["AAA"] * 7,
                "x": np.arange(7, dtype=float),
                "evaluation_phase": phases,
            }
        )
        for phase in range(1, 6):
            forecasting[f"phase{phase}_percent"] = (phases == phase).astype(float)
        forecasting = sensitivity.loco.add_cumulative_targets(forecasting)
        nowcasting = forecasting[["area_id", "date", "country_code_3"]].copy()
        nowcasting["z"] = np.arange(10, 17, dtype=float)
        train = pd.Series([True] * 5 + [False] * 2)
        test = ~train
        effects = []
        for phase in range(2, 6):
            layer1_test = np.zeros(2)
            layer1_train = np.zeros(5)
            residual_test = np.array([0.25, 0.0]) if phase == 3 else np.zeros(2)
            effects.extend(
                [
                    (layer1_test, {}),
                    (layer1_train, {}),
                    (residual_test, {}),
                ]
            )
        with mock.patch.object(
            sensitivity.simple, "fit_ols_arrays", side_effect=effects
        ) as fitted:
            result, n_train, n_test = sensitivity.fit_ensemble_ols_task(
                task="Nowcasting",
                forecasting=forecasting,
                nowcasting=nowcasting,
                layer1_features=["x"],
                layer2_features=["z"],
                forecasting_train_mask=train,
                forecasting_test_mask=test,
                nowcasting_train_mask=train,
                nowcasting_test_mask=test,
            )
        self.assertEqual(fitted.call_count, 12)
        self.assertEqual((n_train, n_test), (5, 2))
        self.assertEqual(result["predicted_phase"].tolist(), [3, 1])
```

The call order is Layer-1 test, Layer-1 train, and Layer-2 residual test for each of four cumulative targets. The Forecasting branch uses the first two calls per target and therefore makes eight calls.

- [ ] **Step 4: Run the focused tests and verify RED**

Expected: failures name the four missing fit interfaces.

- [ ] **Step 5: Implement prediction normalization and safe exception handling**

```python
KNOWN_NON_ESTIMABLE_ERRORS = (
    ValueError,
    RuntimeError,
    np.linalg.LinAlgError,
    xgb.core.XGBoostError,
)


def prediction_frame(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> pd.DataFrame:
    actual_values = np.asarray(actual, dtype=int)
    predicted_values = np.asarray(predicted, dtype=int)
    if len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted phase vectors must have the same length.")
    if not np.isin(actual_values, np.arange(1, 6)).all() or not np.isin(
        predicted_values, np.arange(1, 6)
    ).all():
        raise ValueError("Actual and predicted phases must lie between one and five.")
    return pd.DataFrame(
        {"actual_phase": actual_values, "predicted_phase": predicted_values}
    )


def run_fit_safely(
    fit: Callable[[], tuple[pd.DataFrame, int, int]],
    *,
    n_train: int,
    n_test: int,
) -> tuple[pd.DataFrame | None, int, int, str, str]:
    try:
        predictions, fitted_train, fitted_test = fit()
        return predictions, fitted_train, fitted_test, "generated", ""
    except KNOWN_NON_ESTIMABLE_ERRORS as error:
        return None, n_train, n_test, "not_estimable", str(error)
```

Do not catch `TypeError`, `KeyError`, `AssertionError`, or arbitrary `Exception`; programming and schema bugs must stop the run.

- [ ] **Step 6: Implement the XGBoost task wrapper**

`fit_xgboost_task` must call the existing `loco.fit_forecasting_split` or `loco.fit_nowcasting_split` with `split_id="missingness_sensitivity"`, `fold_column="condition_id"`, and the frozen parameter dictionaries. Convert only `overall_phase` and `overall_phase_pred` into `prediction_frame`. Do not copy an XGBoost training loop into the new generator.

```python
def fit_xgboost_task(
    *,
    task: str,
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    forecasting_train_mask: pd.Series,
    forecasting_test_mask: pd.Series,
    nowcasting_train_mask: pd.Series,
    nowcasting_test_mask: pd.Series,
    general_params: Mapping[str, object],
    phase3_params: Mapping[str, object],
) -> tuple[pd.DataFrame, int, int]:
    if task == "Forecasting":
        wide = loco.fit_forecasting_split(
            forecasting,
            forecasting_train_mask,
            forecasting_test_mask,
            "missingness_sensitivity",
            dict(general_params),
            dict(phase3_params),
            fold_column="condition_id",
        )
    elif task == "Nowcasting":
        wide = loco.fit_nowcasting_split(
            forecasting,
            nowcasting,
            forecasting_train_mask,
            forecasting_test_mask,
            nowcasting_train_mask,
            nowcasting_test_mask,
            "missingness_sensitivity",
            dict(general_params),
            dict(phase3_params),
            fold_column="condition_id",
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    return (
        prediction_frame(wide["overall_phase"], wide["overall_phase_pred"]),
        int(forecasting_train_mask.sum()),
        int(forecasting_test_mask.sum()),
    )
```

- [ ] **Step 7: Implement the direct Ordered Probit wrapper**

The wrapper must fit `simple.fit_numeric_preprocessor` on training rows only, transform training and test rows with the same object, and call `simple.fit_ordered_probit_arrays` with `optimizer="bfgs"` and `maxiter=1000`.

```python
def fit_ordered_probit_task(
    *,
    data: pd.DataFrame,
    features: Sequence[str],
    train_mask: pd.Series,
    test_mask: pd.Series,
    task: str,
) -> tuple[pd.DataFrame, int, int]:
    preprocessor = simple.fit_numeric_preprocessor(
        data.loc[train_mask],
        features,
        task=task,
        method="Ordered Probit",
        layer="direct",
    )
    outcome = simple.derive_evaluation_phase(data)
    predicted, _, _ = simple.fit_ordered_probit_arrays(
        preprocessor.transform(data.loc[train_mask]),
        outcome.loc[train_mask],
        preprocessor.transform(data.loc[test_mask]),
        optimizer="bfgs",
        maxiter=1000,
    )
    return (
        prediction_frame(outcome.loc[test_mask], predicted),
        int(train_mask.sum()),
        int(test_mask.sum()),
    )
```

- [ ] **Step 8: Implement the cumulative Ensemble OLS wrapper**

Use the same train-only preprocessing objects for all four cumulative targets. The exact sequence is:

1. fit one Layer-1 preprocessor from the supplied Layer-1 feature list;
2. transform Layer-1 train/test once;
3. for each phase 2 through 5, call `simple.fit_ols_arrays` for test predictions and again for in-sample train predictions;
4. for Forecasting, store the Layer-1 test prediction directly;
5. for Nowcasting, build the keyed in-sample residual from Forecasting, merge it one-to-one with the corresponding Nowcasting training rows, fit one Layer-2 OLS, predict test residuals, merge by `area_id/date/country_code_3`, and add Layer 1 plus residual;
6. round separate prediction columns to two decimals and call `simple._phase_from_rounded_predictions`;
7. return `prediction_frame` with the reconstructed test truth.

The public signature is:

```python
def fit_ensemble_ols_task(
    *,
    task: str,
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    layer1_features: Sequence[str],
    layer2_features: Sequence[str],
    forecasting_train_mask: pd.Series,
    forecasting_test_mask: pd.Series,
    nowcasting_train_mask: pd.Series,
    nowcasting_test_mask: pd.Series,
) -> tuple[pd.DataFrame, int, int]:
```

Copy only the orchestration necessary from `generate_simple_baseline_comparison.py:1444-1557`; all numerical preprocessing and fits must still call `simple.fit_numeric_preprocessor` and `simple.fit_ols_arrays`. Replace the existing adapter's hard-coded 4,405/1,170 merge checks with `int(forecasting_train_mask.sum())` and `int(forecasting_test_mask.sum())` in this new wrapper. Do not modify the existing adapter.

- [ ] **Step 9: Run the focused tests and verify GREEN**

Expected: routing, prediction normalization, residual addition, and failure continuation tests pass.

- [ ] **Step 10: Checkpoint the model wrappers**

Run the focused unittest module and `py_compile`. Do not run the 64 formal fits yet. Do not commit.

---

### Task 3: Build the Three Approved Experiment Grids

**Files:**

- Modify: `2.Source Code/generate_missingness_sensitivity.py`
- Modify: `tests/test_generate_missingness_sensitivity.py`

**Interfaces:**

- Produces: `source_feature_contract(bundle: simple.PreparedInputs) -> dict[str, object]`.
- Produces: `country_filtered_bundle(bundle: simple.PreparedInputs, removed: Sequence[str]) -> simple.PreparedInputs`.
- Produces: `run_feature_removal(bundle, general_params, phase3_params) -> list[dict[str, object]]`.
- Produces: `run_country_removal(bundle, general_params, phase3_params) -> list[dict[str, object]]`.
- Produces: `run_missing_indicators(bundle) -> list[dict[str, object]]`.

- [ ] **Step 1: Add failing live feature and country contract tests**

```python
class LiveSourceContractTests(unittest.TestCase):
    def test_live_feature_and_country_contract(self):
        bundle = sensitivity.simple.load_prepared_inputs()
        contract = sensitivity.source_feature_contract(bundle)
        self.assertEqual(len(contract["forecast_direct"]), 106)
        self.assertEqual(len(contract["nowcast_direct"]), 173)
        self.assertEqual(len(contract["layer1"]), 106)
        self.assertEqual(len(contract["layer2"]), 69)
        self.assertEqual(len(contract["countries"]), 29)
        self.assertEqual(
            set(contract["forecast_source_to_layer1"]),
            set(contract["forecast_direct"]),
        )
```

This test reads the real CSVs but does not fit a model or write an artifact.

- [ ] **Step 2: Add failing experiment-grid tests using patched fit wrappers**

Patch the three fit wrappers and run these exact grid assertions:

```python
class ExperimentGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = sensitivity.simple.load_prepared_inputs()

    @staticmethod
    def successful_fit():
        return sensitivity.prediction_frame([1, 3], [1, 3]), 4405, 1170

    def test_feature_and_country_grids_have_thirty_conditions_each(self):
        success = self.successful_fit()
        with mock.patch.object(
            sensitivity, "fit_xgboost_task", return_value=success
        ), mock.patch.object(
            sensitivity, "fit_ensemble_ols_task", return_value=success
        ), mock.patch.object(
            sensitivity, "fit_ordered_probit_task", return_value=success
        ):
            feature = sensitivity.run_feature_removal(self.bundle, {}, {})
            country = sensitivity.run_country_removal(self.bundle, {}, {})
        self.assertEqual(len(feature), 30)
        self.assertEqual(len(country), 30)
        zero_feature = [row for row in feature if row["threshold_percent"] == 0]
        self.assertTrue(all(row["removed_feature_count"] == 0 for row in zero_feature))
        expected_counts = {0: 0, 5: 2, 10: 3, 30: 9, 50: 15}
        for threshold, expected in expected_counts.items():
            rows = [row for row in country if row["threshold_percent"] == threshold]
            self.assertTrue(all(row["removed_country_count"] == expected for row in rows))
        grouped = pd.DataFrame(country).groupby(
            ["task", "threshold_percent"], observed=True
        )["removed_country_iso3"].nunique()
        self.assertTrue(grouped.eq(1).all())

    def test_indicator_grid_has_four_fits_and_never_calls_xgboost(self):
        success = self.successful_fit()
        with mock.patch.object(
            sensitivity, "fit_xgboost_task"
        ) as xgboost_fit, mock.patch.object(
            sensitivity, "fit_ensemble_ols_task", return_value=success
        ), mock.patch.object(
            sensitivity, "fit_ordered_probit_task", return_value=success
        ):
            rows = sensitivity.run_missing_indicators(self.bundle)
        self.assertEqual(len(rows), 4)
        xgboost_fit.assert_not_called()
        self.assertEqual(
            {(row["task"], row["model"]) for row in rows},
            {
                ("Forecasting", "Ensemble OLS"),
                ("Nowcasting", "Ensemble OLS"),
                ("Forecasting", "Ordered Probit"),
                ("Nowcasting", "Ordered Probit"),
            },
        )
```

Add this focused role-count assertion:

```python
    def test_indicator_grid_uses_each_models_source_roles(self):
        success = self.successful_fit()
        with mock.patch.object(
            sensitivity, "fit_ensemble_ols_task", return_value=success
        ) as ols_fit, mock.patch.object(
            sensitivity, "fit_ordered_probit_task", return_value=success
        ) as ordered_fit:
            sensitivity.run_missing_indicators(self.bundle)
        ols_calls = {call.kwargs["task"]: call.kwargs for call in ols_fit.call_args_list}
        ordered_calls = {
            call.kwargs["task"]: call.kwargs for call in ordered_fit.call_args_list
        }
        self.assertEqual(len(ols_calls["Forecasting"]["layer1_features"]), 212)
        self.assertEqual(len(ols_calls["Nowcasting"]["layer1_features"]), 212)
        self.assertEqual(len(ols_calls["Nowcasting"]["layer2_features"]), 138)
        self.assertEqual(len(ordered_calls["Forecasting"]["features"]), 212)
        self.assertEqual(len(ordered_calls["Nowcasting"]["features"]), 346)
```

- [ ] **Step 3: Run the focused tests and verify RED**

Expected: failures name the missing contract and experiment functions.

- [ ] **Step 4: Implement the live source-feature contract**

```python
def source_feature_contract(bundle: simple.PreparedInputs) -> dict[str, object]:
    forecast_direct = tuple(
        multinomial.select_feature_columns(bundle.raw_forecasting)
    )
    nowcast_direct = tuple(multinomial.select_feature_columns(bundle.raw_nowcasting))
    layer1 = tuple(
        feature
        for feature in loco.select_layer1_features(bundle.forecasting)
        if feature != "evaluation_phase"
    )
    layer2 = tuple(loco.NOWCAST_FEATURES)
    if (len(forecast_direct), len(nowcast_direct), len(layer1), len(layer2)) != (
        106,
        173,
        106,
        69,
    ):
        raise ValueError("Missingness sensitivity source-feature contract drifted.")
    source_to_layer1 = dict(zip(forecast_direct, layer1, strict=True))
    countries = tuple(sorted(bundle.forecasting["country_code_3"].astype(str).unique()))
    if len(countries) != 29:
        raise ValueError(f"Expected 29 source countries, found {len(countries)}.")
    return {
        "forecast_direct": forecast_direct,
        "nowcast_direct": nowcast_direct,
        "layer1": layer1,
        "layer2": layer2,
        "forecast_source_to_layer1": source_to_layer1,
        "countries": countries,
    }
```

The positional `zip` is valid because `loco.prepare_model_inputs` preserves source-column order and only renames two Forecasting columns. Prove the mapping with:

```python
    def test_forecast_source_to_layer1_mapping_preserves_values(self):
        bundle = sensitivity.simple.load_prepared_inputs()
        contract = sensitivity.source_feature_contract(bundle)
        raw = bundle.raw_forecasting.sort_values(
            sensitivity.loco.KEY_COLUMNS, kind="mergesort"
        ).reset_index(drop=True)
        prepared = bundle.forecasting.sort_values(
            sensitivity.loco.KEY_COLUMNS, kind="mergesort"
        ).reset_index(drop=True)
        for source, layer1 in contract["forecast_source_to_layer1"].items():
            left = raw[source].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            right = prepared[layer1].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
```

- [ ] **Step 5: Implement condition-local filtering without mutating the base bundle**

`country_filtered_bundle` must:

1. resolve removed ISO3 codes to `area_id` values from `bundle.forecasting`;
2. filter both raw source tables by those area IDs;
3. filter both prepared tables by `country_code_3`;
4. reset all four indexes;
5. recompute temporal masks with `simple.temporal_masks`;
6. verify Forecasting/Nowcasting key equality in train and test;
7. recompute the filtered test-set disagreement count between source `overall_phase` and `evaluation_phase`; and
8. construct a new `simple.PreparedInputs` object with the filtered frames, masks, test-key hash, and disagreement count, without calling any writer.

Use `simple.canonical_key_sha256` only inside the in-memory `PreparedInputs` contract required by that existing dataclass. Do not export the hash to the sensitivity metrics CSV.

```python
def country_filtered_bundle(
    bundle: simple.PreparedInputs,
    removed: Sequence[str],
) -> simple.PreparedInputs:
    removed_codes = tuple(removed)
    known_codes = set(bundle.forecasting["country_code_3"].astype(str))
    unknown = set(removed_codes).difference(known_codes)
    if unknown:
        raise ValueError(f"Unknown removed countries: {sorted(unknown)}")
    removed_area_ids = set(
        bundle.forecasting.loc[
            bundle.forecasting["country_code_3"].isin(removed_codes), "area_id"
        ].astype(int)
    )
    raw_forecasting = bundle.raw_forecasting.loc[
        ~bundle.raw_forecasting["area_id"].isin(removed_area_ids)
    ].reset_index(drop=True)
    raw_nowcasting = bundle.raw_nowcasting.loc[
        ~bundle.raw_nowcasting["area_id"].isin(removed_area_ids)
    ].reset_index(drop=True)
    forecasting = bundle.forecasting.loc[
        ~bundle.forecasting["country_code_3"].isin(removed_codes)
    ].reset_index(drop=True)
    nowcasting = bundle.nowcasting.loc[
        ~bundle.nowcasting["country_code_3"].isin(removed_codes)
    ].reset_index(drop=True)
    f_train, f_test = simple.temporal_masks(forecasting["date"])
    n_train, n_test = simple.temporal_masks(nowcasting["date"])
    for left, right, label in (
        (forecasting.loc[f_train], nowcasting.loc[n_train], "train"),
        (forecasting.loc[f_test], nowcasting.loc[n_test], "test"),
    ):
        left_keys = pd.MultiIndex.from_frame(left[loco.KEY_COLUMNS])
        right_keys = pd.MultiIndex.from_frame(right[loco.KEY_COLUMNS])
        if not left_keys.equals(right_keys):
            raise ValueError(f"Filtered Forecasting/Nowcasting {label} keys differ.")
    disagreement = int(
        forecasting.loc[f_test, "overall_phase"]
        .astype(int)
        .ne(forecasting.loc[f_test, "evaluation_phase"].astype(int))
        .sum()
    )
    return simple.PreparedInputs(
        raw_forecasting=raw_forecasting,
        raw_nowcasting=raw_nowcasting,
        forecasting=forecasting,
        nowcasting=nowcasting,
        forecasting_train_mask=f_train,
        forecasting_test_mask=f_test,
        nowcasting_train_mask=n_train,
        nowcasting_test_mask=n_test,
        test_key_sha256=simple.canonical_key_sha256(forecasting.loc[f_test]),
        source_label_disagreement_test_count=disagreement,
    )
```

- [ ] **Step 6: Implement feature-removal orchestration**

Precompute three rankings once from the unmodified pre-2022 data:

- `forecast_ranking` over `forecast_direct`;
- `layer2_ranking` over the 69 Nowcasting Layer-2 features; and
- `nowcast_direct_ranking` over the 173 direct Nowcasting features.

For each threshold and task-model pair:

- Ordered Probit receives a raw-table copy with its selected direct columns suppressed;
- Forecasting XGBoost and Ensemble OLS receive prepared Forecasting copies with the selected Forecasting source columns mapped through `forecast_source_to_layer1` and suppressed;
- Nowcasting XGBoost and Ensemble OLS receive both the suppressed prepared Forecasting Layer-1 copy and suppressed prepared Nowcasting Layer-2 copy;
- masks and rows remain unchanged from the full bundle; and
- `removed_feature_count` is 106-based for Forecasting, the sum of separate 106- and 69-based removals for cascading Nowcasting, and 173-based for direct Nowcasting Ordered Probit.

Call `run_fit_safely` once per model condition. Store raw records with the experiment, threshold, task, model, removal counts, train/test counts, status, reason, and optional prediction frame. Do not store feature names in output records.

- [ ] **Step 7: Implement country-removal orchestration**

Build separate Forecasting and Nowcasting country rankings:

```python
forecast_country_ranking = rank_countries(
    [(bundle.forecasting, contract["layer1"], bundle.forecasting_train_mask)]
)
nowcast_country_ranking = rank_countries(
    [
        (bundle.forecasting, contract["layer1"], bundle.forecasting_train_mask),
        (bundle.nowcasting, contract["layer2"], bundle.nowcasting_train_mask),
    ]
)
```

For each threshold, select `selection_count(threshold, 29)` ISO3 codes from the appropriate task ranking, construct one filtered bundle per task-threshold, and reuse that same filtered bundle for all three methods. Direct Ordered Probit must use the filtered raw task table; XGBoost and OLS must use the filtered prepared tables. Record the selected ISO3 codes as one semicolon-separated, ranking-ordered string.

- [ ] **Step 8: Implement missing-indicator orchestration**

Run exactly four fits on the unfiltered full sample:

- Forecasting Ordered Probit: 106 raw features plus 106 raw-feature indicators;
- Nowcasting Ordered Probit: 173 raw features plus 173 raw-feature indicators;
- Forecasting Ensemble OLS: 106 Layer-1 features plus 106 Layer-1 indicators; and
- Nowcasting Ensemble OLS: augmented Layer 1 plus 69 Layer-2 features and 69 Layer-2 indicators.

Indicators are computed from the untouched pre-imputation values. The XGBoost fit wrapper must not be called in this experiment.

- [ ] **Step 9: Run the focused tests and verify GREEN**

Expected: live contract and 30/30/4 grid tests pass with patched model fits.

- [ ] **Step 10: Checkpoint experiment construction**

Run the focused tests and `py_compile`. Confirm no file under `produced_graph` has changed. Do not commit.

---

### Task 4: Metrics Rows, Frozen XGBoost References, and Deltas

**Files:**

- Modify: `2.Source Code/generate_missingness_sensitivity.py`
- Modify: `tests/test_generate_missingness_sensitivity.py`

**Interfaces:**

- Produces: `METRIC_COLUMNS: tuple[str, ...]`.
- Produces: `record_metrics(record: Mapping[str, object]) -> dict[str, object]`.
- Produces: `load_frozen_xgboost_references(path: Path) -> list[dict[str, object]]`.
- Produces: `add_xgboost_deltas(metrics: pd.DataFrame) -> pd.DataFrame`.
- Produces: `validate_metrics(metrics: pd.DataFrame) -> None`.

- [ ] **Step 1: Add failing generated and non-estimable metric-row tests**

```python
class MetricTableTests(unittest.TestCase):
    def test_generated_record_uses_existing_pooled_metric_definitions(self):
        raw = {
            "experiment": "feature_removal",
            "threshold_percent": 10,
            "task": "Forecasting",
            "model": "Ensemble OLS",
            "removed_feature_count": 11,
            "removed_country_count": 0,
            "removed_country_iso3": "",
            "n_train": 4405,
            "n_test": 1170,
            "status": "generated",
            "reason": "",
            "predictions": sensitivity.prediction_frame([1, 3, 4], [1, 2, 4]),
        }
        row = sensitivity.record_metrics(raw)
        self.assertAlmostEqual(row["overall_accuracy"], 2 / 3)
        self.assertAlmostEqual(row["phase3plus_precision"], 1.0)
        self.assertAlmostEqual(row["phase3plus_recall"], 0.5)

    def test_not_estimable_record_has_nan_metrics_and_keeps_counts(self):
        raw = {
            "experiment": "country_removal",
            "threshold_percent": 50,
            "task": "Nowcasting",
            "model": "Ordered Probit",
            "removed_feature_count": 0,
            "removed_country_count": 15,
            "removed_country_iso3": "AAA;BBB",
            "n_train": 100,
            "n_test": 20,
            "status": "not_estimable",
            "reason": "missing Ordered Probit class",
            "predictions": None,
        }
        row = sensitivity.record_metrics(raw)
        self.assertTrue(math.isnan(row["overall_accuracy"]))
        self.assertEqual(row["n_test"], 20)
        self.assertEqual(row["reason"], "missing Ordered Probit class")
```

- [ ] **Step 2: Add failing frozen-reference and delta tests**

Use exact fixtures for frozen references and grouped deltas:

```python
    def test_frozen_xgboost_references_select_only_main_result_rows(self):
        source = pd.DataFrame(
            [
                {
                    "task": task,
                    "method": method,
                    "overall_accuracy": value,
                    "phase3plus_precision": value,
                    "phase3plus_recall": value,
                    "phase3above_r2": value,
                    "n_train": 4405,
                    "n_test": 1170,
                }
                for task, method, value in (
                    ("Forecasting", "Main result", 0.6),
                    ("Nowcasting", "Main result", 0.7),
                    ("Forecasting", "Ensemble OLS", 0.4),
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.csv"
            source.to_csv(path, index=False)
            references = sensitivity.load_frozen_xgboost_references(path)
        self.assertEqual(len(references), 2)
        self.assertEqual({row["task"] for row in references}, set(sensitivity.TASKS))
        self.assertTrue(all(row["model"] == "XGBoost" for row in references))
        self.assertTrue(all(row["status"] == "frozen_reference" for row in references))

    def test_deltas_use_same_experiment_task_and_threshold_xgboost(self):
        rows = []
        for experiment, threshold, xgb_value, baseline_value in (
            ("feature_removal", 10, 0.8, 0.7),
            ("country_removal", 10, 0.6, 0.55),
            ("missing_indicators", np.nan, 0.75, 0.65),
        ):
            for model, value in (("XGBoost", xgb_value), ("Ensemble OLS", baseline_value)):
                row = {column: np.nan for column in sensitivity.METRIC_COLUMNS}
                row.update(
                    {
                        "experiment": experiment,
                        "threshold_percent": threshold,
                        "task": "Forecasting",
                        "model": model,
                        "removed_feature_count": 0,
                        "removed_country_count": 0,
                        "removed_country_iso3": "",
                        "n_train": 4405,
                        "n_test": 1170,
                        "status": "frozen_reference" if experiment == "missing_indicators" and model == "XGBoost" else "generated",
                        "reason": "",
                    }
                )
                for metric in sensitivity.METRIC_NAMES:
                    row[metric] = value
                rows.append(row)
        result = sensitivity.add_xgboost_deltas(pd.DataFrame(rows))
        observed = result.loc[result["model"].eq("Ensemble OLS")].set_index("experiment")
        self.assertAlmostEqual(
            observed.loc["feature_removal", "delta_overall_accuracy_vs_xgboost"],
            -0.1,
        )
        self.assertAlmostEqual(
            observed.loc["country_removal", "delta_overall_accuracy_vs_xgboost"],
            -0.05,
        )
        self.assertAlmostEqual(
            observed.loc["missing_indicators", "delta_overall_accuracy_vs_xgboost"],
            -0.1,
        )
        xgboost = result.loc[result["model"].eq("XGBoost")]
        self.assertTrue(xgboost["delta_overall_accuracy_vs_xgboost"].eq(0.0).all())
```

```python
    def test_delta_is_nan_when_reference_metric_is_undefined(self):
        rows = []
        for model, value in (("XGBoost", np.nan), ("Ordered Probit", 0.4)):
            row = {column: np.nan for column in sensitivity.METRIC_COLUMNS}
            row.update(
                {
                    "experiment": "feature_removal",
                    "threshold_percent": 50,
                    "task": "Nowcasting",
                    "model": model,
                    "removed_feature_count": 88,
                    "removed_country_count": 0,
                    "removed_country_iso3": "",
                    "n_train": 4405,
                    "n_test": 1170,
                    "status": "generated",
                    "reason": "",
                    "overall_accuracy": value,
                }
            )
            rows.append(row)
        result = sensitivity.add_xgboost_deltas(pd.DataFrame(rows))
        delta = result.loc[
            result["model"].eq("Ordered Probit"),
            "delta_overall_accuracy_vs_xgboost",
        ].iloc[0]
        self.assertTrue(math.isnan(delta))
```

- [ ] **Step 3: Run the focused tests and verify RED**

Expected: failures name missing metrics interfaces.

- [ ] **Step 4: Implement the exact metrics schema**

```python
METRIC_NAMES = (
    "overall_accuracy",
    "phase3plus_precision",
    "phase3plus_recall",
    "phase3above_r2",
)
METRIC_COLUMNS = (
    "experiment",
    "threshold_percent",
    "task",
    "model",
    "removed_feature_count",
    "removed_country_count",
    "removed_country_iso3",
    *METRIC_NAMES,
    "n_train",
    "n_test",
    "status",
    "reason",
    "delta_overall_accuracy_vs_xgboost",
    "delta_phase3plus_precision_vs_xgboost",
    "delta_phase3plus_recall_vs_xgboost",
    "delta_phase3above_r2_vs_xgboost",
)
```

`record_metrics` must call `simple.calculate_pooled_metrics` for generated predictions and select only the four approved metrics. For `not_estimable`, fill those metrics with `NaN`. Do not copy confusion counts returned by the existing helper into the output row.

- [ ] **Step 5: Load exactly two frozen XGBoost reference rows**

Read `simple_baseline_comparison_metrics.csv`, require one `method == "Main result"` row for each task, and map:

- `overall_accuracy` to `overall_accuracy`;
- `phase3plus_precision` to `phase3plus_precision`;
- `phase3plus_recall` to `phase3plus_recall`; and
- `phase3above_r2` to `phase3above_r2`.

Set `experiment="missing_indicators"`, `threshold_percent=NaN`, both removal counts to zero, ISO3 to an empty string, `status="frozen_reference"`, and blank reason. Reject duplicate or missing task references.

- [ ] **Step 6: Implement grouped XGBoost deltas**

For feature/country rows, join the XGBoost reference by `(experiment, threshold_percent, task)`. For missing-indicator rows, join by `(experiment, task)`. Subtract reference values from each row and write the four delta columns. Use zero for finite XGBoost self-deltas and `NaN` when either side is not finite.

- [ ] **Step 7: Validate the completed 66-row table**

`validate_metrics` must require:

- exact `METRIC_COLUMNS` order;
- 30 feature-removal rows;
- 30 country-removal rows;
- six missing-indicator comparison rows;
- exactly one row per experiment/threshold/task/model condition;
- thresholds exactly `0/5/10/30/50` in both removal experiments;
- models exactly three per task-threshold;
- missing-indicator models exactly XGBoost, Ensemble OLS, and Ordered Probit per task;
- country removal counts exactly `0/2/3/9/15` by threshold;
- the semicolon-separated ISO3 count equals `removed_country_count`, and the value is identical across the three models within each task-threshold;
- positive `n_train` and `n_test` for generated rows;
- finite metrics for `generated` and `frozen_reference` rows except mathematically undefined precision, recall, or R-squared;
- all four metrics `NaN` for `not_estimable`; and
- non-empty reason only for `not_estimable`.

- [ ] **Step 8: Run the focused tests and verify GREEN**

Expected: metrics, frozen references, deltas, shape, and status tests pass.

- [ ] **Step 9: Checkpoint the result-table contract**

Run focused tests and `py_compile`. Do not commit.

---

### Task 5: One Sensitivity-Curve Figure in PDF and PNG and Isolated Atomic Publication

**Files:**

- Modify: `2.Source Code/generate_missingness_sensitivity.py`
- Modify: `tests/test_generate_missingness_sensitivity.py`

**Interfaces:**

- Produces: `apply_figure_style() -> None`.
- Produces: `create_sensitivity_figure(metrics: pd.DataFrame) -> matplotlib.figure.Figure`.
- Produces: `replace_with_retry(source: Path, destination: Path) -> None`.
- Produces: `write_outputs(metrics: pd.DataFrame, figure, output_dir: Path) -> dict[str, Path]`.
- Produces: `run_analysis(*, forecasting_path: Path, nowcasting_path: Path, country_lookup_path: Path, general_params_path: Path, phase3_params_path: Path, base_metrics_path: Path, output_dir: Path) -> dict[str, Path]`.
- Produces: `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`.
- Produces: `main(argv: Sequence[str] | None = None) -> None`.

- [ ] **Step 1: Add failing figure-contract tests**

Add one synthetic-table helper and exact figure assertions:

```python
def synthetic_removal_metrics() -> pd.DataFrame:
    rows = []
    removed_countries = {0: 0, 5: 2, 10: 3, 30: 9, 50: 15}
    for experiment in ("feature_removal", "country_removal"):
        for task in sensitivity.TASKS:
            for threshold in sensitivity.THRESHOLDS:
                for model_index, model in enumerate(sensitivity.MODELS):
                    value = 0.80 - threshold / 500 - model_index * 0.03
                    row = {column: np.nan for column in sensitivity.METRIC_COLUMNS}
                    row.update(
                        {
                            "experiment": experiment,
                            "threshold_percent": threshold,
                            "task": task,
                            "model": model,
                            "removed_feature_count": int(math.ceil(threshold / 100 * 106)),
                            "removed_country_count": removed_countries[threshold] if experiment == "country_removal" else 0,
                            "removed_country_iso3": (
                                ";".join(
                                    f"C{index:02d}"
                                    for index in range(removed_countries[threshold])
                                )
                                if experiment == "country_removal"
                                else ""
                            ),
                            "overall_accuracy": value,
                            "phase3plus_precision": value - 0.02,
                            "phase3plus_recall": value + 0.02,
                            "phase3above_r2": value - 0.30,
                            "n_train": 4405 - removed_countries[threshold] * 10,
                            "n_test": 1170 - removed_countries[threshold] * 5,
                            "status": "generated",
                            "reason": "",
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows, columns=sensitivity.METRIC_COLUMNS)


class FigureAndOutputTests(unittest.TestCase):
    def test_figure_is_the_fixed_four_by_four_grid(self):
        metrics = synthetic_removal_metrics()
        figure = sensitivity.create_sensitivity_figure(metrics)
        self.assertEqual(len(figure.axes), 16)
        for row in range(4):
            for axis in figure.axes[row * 4 : row * 4 + 3]:
                self.assertEqual(axis.get_ylim(), (0.0, 1.0))
                self.assertEqual(len(axis.lines), 3)
        r2_axes = figure.axes[3::4]
        self.assertEqual(len({axis.get_ylim() for axis in r2_axes}), 1)
        for axis in r2_axes:
            low, high = axis.get_ylim()
            self.assertLessEqual(low, 0.0)
            self.assertGreaterEqual(high, 0.0)
            self.assertEqual(len(axis.lines), 4)
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            list(sensitivity.MODELS),
        )
        country_axis = figure.axes[8]
        self.assertTrue(
            all("n=" in label.get_text() for label in country_axis.get_xticklabels())
        )
        sensitivity.plt.close(figure)

    def test_not_estimable_metric_is_not_replaced_with_a_point(self):
        metrics = synthetic_removal_metrics()
        selector = (
            metrics["experiment"].eq("feature_removal")
            & metrics["task"].eq("Forecasting")
            & metrics["model"].eq("Ordered Probit")
            & metrics["threshold_percent"].eq(30)
        )
        metrics.loc[selector, "overall_accuracy"] = np.nan
        figure = sensitivity.create_sensitivity_figure(metrics)
        line = figure.axes[0].lines[2]
        self.assertTrue(np.isnan(line.get_ydata()[3]))
        sensitivity.plt.close(figure)
```

- [ ] **Step 2: Add failing output and CLI tests**

In a `TemporaryDirectory`, run exact output and CLI checks:

```python
    def test_write_outputs_creates_only_csv_pdf_and_png(self):
        metrics = synthetic_removal_metrics()
        indicator_rows = []
        for task in sensitivity.TASKS:
            for model in sensitivity.MODELS:
                row = metrics.iloc[0].copy()
                row["experiment"] = "missing_indicators"
                row["threshold_percent"] = np.nan
                row["task"] = task
                row["model"] = model
                row["removed_feature_count"] = 0
                row["removed_country_count"] = 0
                row["n_train"] = 4405
                row["n_test"] = 1170
                row["status"] = "frozen_reference" if model == "XGBoost" else "generated"
                indicator_rows.append(row)
        complete = pd.concat(
            [metrics, pd.DataFrame(indicator_rows)], ignore_index=True
        )[list(sensitivity.METRIC_COLUMNS)]
        complete = sensitivity.add_xgboost_deltas(complete)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "sentinel.txt"
            sentinel.write_bytes(b"frozen")
            output_dir = root / "sensitivity"
            figure = sensitivity.create_sensitivity_figure(metrics)
            paths = sensitivity.write_outputs(complete, figure, output_dir)
            sensitivity.plt.close(figure)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    sensitivity.METRICS_FILENAME,
                    sensitivity.FIGURE_FILENAME,
                    sensitivity.FIGURE_PNG_FILENAME,
                },
            )
            self.assertEqual(sentinel.read_bytes(), b"frozen")
            self.assertEqual(
                pd.read_csv(paths["metrics_csv"]).columns.tolist(),
                list(sensitivity.METRIC_COLUMNS),
            )
            pdf = paths["figure_pdf"].read_bytes()
            self.assertEqual(pdf[:4], b"%PDF")
            self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", pdf)), 1)
            self.assertNotIn(b"/Subtype /Image", pdf)
            self.assertEqual(
                paths["figure_png"].read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
            )

    def test_cli_has_paths_but_no_model_or_threshold_controls(self):
        parser_result = sensitivity.parse_args([])
        self.assertEqual(parser_result.output_dir, sensitivity.DEFAULT_OUTPUT_DIR)
        self.assertFalse(hasattr(parser_result, "thresholds"))
        self.assertFalse(hasattr(parser_result, "models"))
        self.assertFalse(hasattr(parser_result, "optimizer"))
        self.assertFalse(hasattr(parser_result, "random_state"))
```

- [ ] **Step 3: Run the focused tests and verify RED**

Expected: failures name the missing figure, output, and CLI interfaces.

- [ ] **Step 4: Implement the restrained Python figure style**

Use the established simple-baseline typography with editable PDF text:

```python
def apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
```

Use fixed method colors across every panel. Do not use red/green to encode method quality and do not smooth curves.

- [ ] **Step 5: Implement the 4-by-4 quantitative grid**

Use:

```python
FIGURE_ROWS = (
    ("feature_removal", "Forecasting"),
    ("feature_removal", "Nowcasting"),
    ("country_removal", "Forecasting"),
    ("country_removal", "Nowcasting"),
)
FIGURE_METRICS = (
    ("overall_accuracy", "Five-class accuracy"),
    ("phase3plus_precision", "Phase 3+ precision"),
    ("phase3plus_recall", "Phase 3+ recall"),
    ("phase3above_r2", "Phase 3+ R-squared"),
)
MODEL_COLORS = {
    "XGBoost": "#4C78A8",
    "Ensemble OLS": "#B279A2",
    "Ordered Probit": "#F28E2B",
}
```

Use five equally spaced x positions for the ordered sensitivity thresholds so irregular numeric gaps do not make adjacent labels overlap. Feature rows label those positions with threshold percentages. Country rows label them with the actual removed-country count and retained `n_test` in the form `"{removed}\nn={n_test}"`; assert that all three methods have identical values at each task-threshold. Sort before plotting. Matplotlib must receive `NaN` values unchanged so lines break at non-estimable conditions. Accuracy, precision, and recall use `[0, 1]`; derive one shared R-squared y-range from all finite removal-grid R-squared values, include zero with a small margin, and draw a zero-reference line in every R-squared panel.

- [ ] **Step 6: Implement three-file staged publication**

Create the supplied output directory, stage all three files in a child `TemporaryDirectory`, write the CSV with `float_format="%.17g"`, reload and call `validate_metrics`, save one PDF with `metadata={"CreationDate": None, "ModDate": None}` and one matching 300 dpi PNG, validate both headers, then replace only the three owned destination files with the existing five-attempt `os.replace` retry pattern. Never enumerate, delete, move, or replace sibling files outside the supplied output directory.

- [ ] **Step 7: Implement formal orchestration and fixed CLI**

`run_analysis` must:

1. identify a formal run only when `output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()`;
2. for a formal run, call `frozen_main_result.assert_frozen_environment(("matplotlib", "statsmodels", "patsy"))`;
3. validate the default source, lookup, parameter, and base-metrics files against the existing frozen hashes exposed by `selected_main.EXPECTED_FILE_SHA256`;
4. call `simple.load_prepared_inputs(forecasting_path, nowcasting_path, country_lookup_path, enforce_formal_counts=formal_run)`;
5. load XGBoost parameters through `loco.load_hyperparameters(general_params_path, phase3_params_path, random_state=None, estimator_n_jobs=None)`;
6. run feature removal, country removal, and missing indicators sequentially;
7. convert raw records to metrics, append the two frozen XGBoost indicator references, add deltas, and validate the 66-row table;
8. create the figure from the first 60 removal rows; and
9. write only the three approved output artifacts.

The CLI contains exactly:

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-input", type=Path, default=simple.DEFAULT_FORECASTING_INPUT)
    parser.add_argument("--nowcasting-input", type=Path, default=simple.DEFAULT_NOWCASTING_INPUT)
    parser.add_argument("--country-lookup", type=Path, default=simple.DEFAULT_COUNTRY_LOOKUP)
    parser.add_argument("--general-params", type=Path, default=DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=DEFAULT_PHASE3_PARAMS)
    parser.add_argument("--base-metrics", type=Path, default=DEFAULT_BASE_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)
```

`main(argv)` forwards these values to `run_analysis` and prints the three resulting paths. It exposes no scenario or model tuning controls.

- [ ] **Step 8: Run the focused tests and verify GREEN**

Expected: figure, vector-PDF, PNG, three-file output, sentinel-preservation, and CLI tests pass.

- [ ] **Step 9: Checkpoint the complete non-formal implementation**

Run:

```bash
python -B -m py_compile "2.Source Code/generate_missingness_sensitivity.py" tests/test_generate_missingness_sensitivity.py
python -B -m unittest discover -s tests -p "test_generate_missingness_sensitivity.py" -v
```

Expected: exit status 0. Do not commit.

---

### Task 6: Live Regression Checks, Formal Generation, and Final Verification

**Files:**

- Modify: `tests/test_generate_missingness_sensitivity.py`
- Generate: `2.Source Code/produced_graph/missingness_sensitivity/missingness_sensitivity_metrics.csv`
- Generate: `2.Source Code/produced_graph/missingness_sensitivity/missingness_sensitivity_curves.pdf`
- Generate: `2.Source Code/produced_graph/missingness_sensitivity/missingness_sensitivity_curves.png`

**Interfaces:**

- Verifies: the 0% conditions reproduce frozen main XGBoost and existing OLS/Ordered Probit metrics.
- Verifies: all formal outputs are isolated and internally consistent.

- [ ] **Step 1: Add the live 0% regression test behind an explicit environment gate**

At the start of the test, call `frozen_main_result.assert_frozen_environment(("matplotlib", "statsmodels", "patsy"))` inside `try/except RuntimeError`; call `self.skipTest(str(error))` on mismatch. In the matching Windows environment, run only the six 0% task-model combinations and compare:

- XGBoost with the two `method=Main result` rows in `simple_baseline_comparison_metrics.csv`;
- Ensemble OLS with the two `method=Ensemble OLS` rows; and
- Ordered Probit with the two `method=Ordered Probit` rows.

Require exact `n_train=4405`, `n_test=1170`, and metric agreement at absolute tolerance `1e-12`. Also require the 0% feature-removal and 0% country-removal rows to agree with each other.

- [ ] **Step 2: Run focused tests in the audited Windows environment**

Activate the environment recorded by the simple-baseline source audit and run:

```bash
python -B -m unittest discover -s tests -p "test_generate_missingness_sensitivity.py" -v
```

Expected: all tests pass, including the live 0% regression test.

- [ ] **Step 3: Run package syntax and structural checks**

```bash
python -B -m py_compile run_replication.py "2.Source Code/generate_missingness_sensitivity.py" tests/test_generate_missingness_sensitivity.py
python -B run_replication.py --check-only
```

Expected: exit status 0. The new standalone sensitivity generator does not need registration in `run_replication.py` because the spec does not add it to the default replication path.

- [ ] **Step 4: Run the complete historical tests only in their matching environment**

```bash
python -B -m unittest discover -s tests -p "test_*.py"
```

Expected: all tests whose frozen environment matches pass. Record any intentional environment-gated skips; do not weaken existing tests.

- [ ] **Step 5: Snapshot the protected result files outside the generator**

Before formal execution, calculate SHA-256 values for the existing main and simple-baseline files that the study treats as frozen. This is verification evidence only; do not add these hashes to the new metrics CSV or create an audit file.

- [ ] **Step 6: Run the formal sensitivity generator once**

From the package root in the audited Windows Python 3.11.3 environment:

```bash
python -B "2.Source Code/generate_missingness_sensitivity.py"
```

Expected: sequential progress for 64 fitted model conditions and two frozen XGBoost references, then exactly three paths printed under `produced_graph/missingness_sensitivity/`.

- [ ] **Step 7: Validate the formal metrics CSV**

Reload the CSV and verify:

- exactly 66 rows and exact `METRIC_COLUMNS` order;
- 30 feature-removal, 30 country-removal, and six missing-indicator rows;
- all 29 countries participated in each task ranking before deletion;
- country removal counts are `0/2/3/9/15`;
- the same task-threshold country set and `n_test` are used by all three models;
- 0% conditions reproduce the frozen/existing metrics at `1e-12` tolerance;
- missing-indicator XGBoost rows exactly match the frozen main rows;
- finite deltas equal metric minus the correct XGBoost reference;
- every non-estimable row has four missing metrics and a non-empty reason; and
- no deletion set changed because another model failed.

- [ ] **Step 8: Validate the formal PDF and figure-data identity**

Using Python/matplotlib only, recreate the figure in a temporary directory from the saved CSV and require the plotted line coordinates and country tick labels to match the formal figure-building inputs. Confirm 16 required axes, one shared legend, one shared R-squared y-range containing zero, and zero-reference lines. Confirm the formal PDF is one page, vector-rendered, and has embedded editable text; confirm the PNG is readable and matches the approved 300 dpi rendering contract.

- [ ] **Step 9: Confirm the frozen files remained byte-identical**

Recalculate the protected-file SHA-256 values and compare them with Step 5. Any difference fails the sensitivity run; do not restore, overwrite, or refresh a protected artifact automatically.

- [ ] **Step 10: Final checkpoint**

Report:

- focused and full test outcomes;
- formal environment versions;
- runtime;
- metrics row counts by experiment/status;
- output paths;
- 0% regression results;
- protected-file before/after equality; and
- any `not_estimable` conditions.

Do not change manuscript text or interpret/adopt the sensitivity result without a separate user decision. Do not commit.

---

## Plan Self-Review

- **Spec coverage:** Tasks 1-4 implement all three approved missingness experiments and the fixed failure policy; Task 5 implements the three allowed output artifacts; Task 6 verifies frozen-result isolation and 0% comparability.
- **Minimality:** Only one generator and one test module are added. Existing frozen generators, shared modules, notebooks, and artifacts are not modified.
- **No leakage:** Every ranking, imputer, scaler, and rank selection uses pre-2022 training rows only. Country deletion is applied after ranking and before model fitting to both train and test.
- **Type consistency:** Experiment functions return raw record dictionaries; `record_metrics` converts them to the fixed metrics schema; `add_xgboost_deltas` and the figure consume only that schema.
- **Ponytail exclusions:** No audit CSV, prediction CSV, feature manifest, run manifest, hash column, confusion count, indicator count, extra figure format, concurrency layer, or reusable framework appears in the implementation scope.
- **Execution gate:** The user authorized and completed implementation under this plan. Commit, merge, push, manuscript changes, and adoption of a new main result still require separate authorization.
