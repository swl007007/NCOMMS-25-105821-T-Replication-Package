"""Generate a unified comparison of four simple food-crisis baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache-simple-baselines")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import qr
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

import generate_leave_one_country_out_robustness as loco
import generate_multinomial_baseline_comparison as multinomial
import generate_persistence_baseline_comparison as persistence
import main_result_figure1_v1 as frozen_main_result


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
DEFAULT_FORECASTING_INPUT = SOURCE_DATA_DIR / "Forecasting_Analysis_010825.csv"
DEFAULT_NOWCASTING_INPUT = SOURCE_DATA_DIR / "Nowcasting_Analysis_010825.csv"
DEFAULT_COUNTRY_LOOKUP = SOURCE_DATA_DIR / "area_country_lookup.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
DEFAULT_CUTOFF_DATE = "2022-01-01"
DEFAULT_ORDERED_OPTIMIZER = "bfgs"
DEFAULT_ORDERED_MAXITER = 1000

EXPECTED_SOURCE_ROWS = 5575
EXPECTED_TRAIN_ROWS = 4405
EXPECTED_TEST_ROWS = 1170
EXPECTED_TEST_AREAS = 646
EXPECTED_TEST_COUNTRIES = 27
EXPECTED_ENSEMBLE_OLS_LAYER1_SOURCE_FEATURES = 106
EXPECTED_ENSEMBLE_OLS_LAYER2_SOURCE_FEATURES = 69

TASK_ORDER = ("Nowcasting", "Forecasting")
METHOD_ORDER = ("Persistence", "Multinomial", "Ordered Probit", "Ensemble OLS")
METRIC_ORDER = ("phase3plus_precision", "phase3plus_recall")
MAIN_RESULT_METHOD = "Main result"
TARGET_DEFINITION = "highest cumulative IPC phase with population share >= 0.20"
PHASE3ABOVE_R2_DEFINITION = (
    "sklearn r2_score on binary actual and predicted indicators for IPC phase >= 3"
)

MAIN_RESULT_REFERENCES: Mapping[str, Mapping[str, object]] = (
    frozen_main_result.classification_references()
)

METHOD_COLORS = {
    "Persistence": "#0072B2",
    "Multinomial": "#E69F00",
    "Ordered Probit": "#009E73",
    "Ensemble OLS": "#CC79A7",
}

OUTPUT_FILENAMES = {
    "predictions_csv": "simple_baseline_comparison_predictions.csv",
    "metrics_csv": "simple_baseline_comparison_metrics.csv",
    "feature_manifest_csv": "simple_baseline_comparison_feature_manifest.csv",
    "model_audit_csv": "simple_baseline_comparison_model_audit.csv",
    "source_audit_csv": "simple_baseline_comparison_source_audit.csv",
    "figure_jpg": "phase3plus_precision_recall_simple_baseline_comparison.jpg",
    "figure_png": "phase3plus_precision_recall_simple_baseline_comparison.png",
    "figure_pdf": "phase3plus_precision_recall_simple_baseline_comparison.pdf",
}

CORE_PREDICTION_COLUMNS = [
    "task",
    "method",
    "area_id",
    "date",
    "source_dataset",
    "source_row_index",
    "source_key_area_id",
    "source_key_date",
    "actual_phase",
    "predicted_phase",
    "actual_phase3plus",
    "predicted_phase3plus",
    "test_key_sha256",
]

PERSISTENCE_DETAIL_COLUMNS = [
    "persistence_issue_date",
    "persistence_source_method",
    "persistence_source_area_id",
    "persistence_source_country_code_3",
    "persistence_source_date",
    "persistence_distance_km",
    "persistence_target_gap_months",
    "persistence_issue_gap_months",
]

ORDERED_PROBABILITY_COLUMNS = [f"ordered_probability_phase{phase}" for phase in range(1, 6)]

OLS_DETAIL_COLUMNS = [
    column
    for phase in range(2, 6)
    for column in (
        f"phase{phase}_actual_cumulative",
        f"phase{phase}_pred_raw",
        f"phase{phase}_pred_rounded",
        f"phase{phase}_layer1_pred",
        f"phase{phase}_residual_pred",
    )
]

PREDICTION_COLUMNS = [
    *CORE_PREDICTION_COLUMNS,
    *PERSISTENCE_DETAIL_COLUMNS,
    *ORDERED_PROBABILITY_COLUMNS,
    *OLS_DETAIL_COLUMNS,
]

METRIC_COLUMNS = [
    "task",
    "method",
    "method_role",
    "overall_accuracy",
    "phase3plus_accuracy",
    "phase3plus_precision",
    "phase3plus_recall",
    "phase3above_r2",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "n_train",
    "n_test",
    "test_key_sha256",
    "target_definition",
    "metric_source",
    "fit_status",
    "converged",
]

FEATURE_MANIFEST_COLUMNS = [
    "task",
    "method",
    "layer",
    "feature",
    "source_order",
    "training_observed_count",
    "training_missing_count",
    "training_missing_rate",
    "training_median",
    "scaling_center",
    "scaling_scale",
    "retained",
    "retained_order",
    "exclusion_reason",
    "duplicate_of_feature",
    "rank_policy",
]

MODEL_AUDIT_COLUMNS = [
    "task",
    "method",
    "fit_status",
    "source_dataset",
    "n_train",
    "n_test",
    "source_feature_count",
    "retained_feature_count",
    "layer1_retained_feature_count",
    "layer2_retained_feature_count",
    "optimizer",
    "converged",
    "max_abs_gradient",
    "parameters_finite",
    "raw_threshold_parameters_json",
    "transformed_cutpoints_json",
    "probabilities_valid",
    "design_rank_min",
    "design_condition_number_max",
    "out_of_range_prediction_cell_count",
    "cumulative_order_violation_row_count",
    "fit_diagnostics_json",
    "notes",
]


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"Input data are missing required columns: {missing}")


def derive_evaluation_phase(data: pd.DataFrame) -> pd.Series:
    """Return the approved cumulative-share reconstructed five-level phase."""
    return multinomial.derive_phase_labels(data).rename("evaluation_phase")


def temporal_masks(
    dates: pd.Series,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
) -> tuple[pd.Series, pd.Series]:
    """Return the approved fixed temporal train/test masks."""
    return multinomial.temporal_masks(dates, cutoff_date)


def _normalize_canonical_dates(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in [
        name
        for name in result.columns
        if name == "date" or name.endswith("_date") or name.endswith("_month")
    ]:
        present = result[column].notna()
        if not present.any():
            continue
        parsed = pd.to_datetime(result.loc[present, column], errors="raise")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise ValueError(f"Canonical date column {column} must be timezone-naive.")
        if not parsed.eq(parsed.dt.normalize()).all():
            raise ValueError(f"Canonical date column {column} must contain midnight values.")
        normalized = pd.Series(pd.NA, index=result.index, dtype="string")
        normalized.loc[present] = parsed.dt.strftime("%Y-%m-%d").to_numpy()
        result[column] = normalized
    return result


def canonical_dataframe_sha256(
    data: pd.DataFrame,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
) -> str:
    """Hash a schema-ordered and key-sorted canonical CSV representation."""
    keys = tuple(key_columns)
    if not keys:
        raise ValueError("Canonical hashes require at least one key column.")
    columns = list(dict.fromkeys([*keys, *value_columns]))
    _require_columns(data, columns)
    ordered = data.loc[:, columns].copy()
    if ordered[list(keys)].isna().any().any():
        raise ValueError("Canonical hash keys contain missing values.")
    ordered = _normalize_canonical_dates(ordered)
    if ordered.duplicated(list(keys)).any():
        raise ValueError("Canonical hash keys are not unique.")
    ordered = ordered.sort_values(list(keys), kind="mergesort")
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_key_sha256(
    data: pd.DataFrame,
    key_columns: Sequence[str] = ("area_id", "date"),
) -> str:
    """Hash only the canonical observation keys."""
    return canonical_dataframe_sha256(data, key_columns, key_columns)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PreparedInputs:
    raw_forecasting: pd.DataFrame
    raw_nowcasting: pd.DataFrame
    forecasting: pd.DataFrame
    nowcasting: pd.DataFrame
    forecasting_train_mask: pd.Series
    forecasting_test_mask: pd.Series
    nowcasting_train_mask: pd.Series
    nowcasting_test_mask: pd.Series
    test_key_sha256: str
    source_label_disagreement_test_count: int


def load_prepared_inputs(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
    *,
    enforce_formal_counts: bool = True,
) -> PreparedInputs:
    """Load, key-align, target-prepare, and validate the two source tables."""
    raw_forecasting = pd.read_csv(forecasting_path)
    raw_nowcasting = pd.read_csv(nowcasting_path)
    lookup = loco.load_country_lookup(country_lookup_path)
    forecasting, nowcasting = loco.prepare_model_inputs(
        raw_forecasting, raw_nowcasting, lookup
    )
    forecasting = loco.add_cumulative_targets(forecasting)
    forecasting["evaluation_phase"] = derive_evaluation_phase(forecasting).to_numpy()
    nowcasting["evaluation_phase"] = derive_evaluation_phase(nowcasting).to_numpy()

    forecast_train, forecast_test = temporal_masks(forecasting["date"], cutoff_date)
    nowcast_train, nowcast_test = temporal_masks(nowcasting["date"], cutoff_date)
    forecast_test_keys = pd.MultiIndex.from_frame(
        forecasting.loc[forecast_test, loco.KEY_COLUMNS]
    )
    nowcast_test_keys = pd.MultiIndex.from_frame(
        nowcasting.loc[nowcast_test, loco.KEY_COLUMNS]
    )
    if set(forecast_test_keys) != set(nowcast_test_keys):
        raise ValueError("Forecasting and Nowcasting temporal-test keys differ.")

    forecast_disagreement = int(
        forecasting.loc[forecast_test, "overall_phase"]
        .astype(int)
        .ne(forecasting.loc[forecast_test, "evaluation_phase"].astype(int))
        .sum()
    )
    nowcast_disagreement = int(
        nowcasting.loc[nowcast_test, "overall_phase"]
        .astype(int)
        .ne(nowcasting.loc[nowcast_test, "evaluation_phase"].astype(int))
        .sum()
    )
    if forecast_disagreement != nowcast_disagreement:
        raise ValueError("Source-label disagreement counts differ across task tables.")

    test_hash = canonical_key_sha256(forecasting.loc[forecast_test])
    if canonical_key_sha256(nowcasting.loc[nowcast_test]) != test_hash:
        raise ValueError("Forecasting and Nowcasting test-key hashes differ.")

    if enforce_formal_counts:
        checks = {
            "Forecasting source rows": (len(forecasting), EXPECTED_SOURCE_ROWS),
            "Nowcasting source rows": (len(nowcasting), EXPECTED_SOURCE_ROWS),
            "Forecasting train rows": (int(forecast_train.sum()), EXPECTED_TRAIN_ROWS),
            "Nowcasting train rows": (int(nowcast_train.sum()), EXPECTED_TRAIN_ROWS),
            "Forecasting test rows": (int(forecast_test.sum()), EXPECTED_TEST_ROWS),
            "Nowcasting test rows": (int(nowcast_test.sum()), EXPECTED_TEST_ROWS),
            "source-label test disagreements": (forecast_disagreement, 5),
        }
        failures = {
            label: (actual, expected)
            for label, (actual, expected) in checks.items()
            if actual != expected
        }
        if failures:
            raise ValueError(f"Formal source snapshot counts changed: {failures}")
        test_rows = forecasting.loc[forecast_test]
        if test_rows["area_id"].nunique() != EXPECTED_TEST_AREAS:
            raise ValueError("Formal temporal test no longer contains 646 areas.")
        if test_rows["country_code_3"].nunique() != EXPECTED_TEST_COUNTRIES:
            raise ValueError("Formal temporal test no longer contains 27 countries.")

    return PreparedInputs(
        raw_forecasting=raw_forecasting,
        raw_nowcasting=raw_nowcasting,
        forecasting=forecasting,
        nowcasting=nowcasting,
        forecasting_train_mask=forecast_train,
        forecasting_test_mask=forecast_test,
        nowcasting_train_mask=nowcast_train,
        nowcasting_test_mask=nowcast_test,
        test_key_sha256=test_hash,
        source_label_disagreement_test_count=forecast_disagreement,
    )


@dataclass
class NumericPreprocessor:
    feature_columns: tuple[str, ...]
    eligible_features: tuple[str, ...]
    retained_features: tuple[str, ...]
    retained_positions: tuple[int, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    manifest: pd.DataFrame
    matrix_rank: int
    condition_number: float

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        _require_columns(data, self.feature_columns)
        numeric = data.loc[:, self.eligible_features].replace(
            [np.inf, -np.inf], np.nan
        )
        non_numeric = [
            column
            for column in self.eligible_features
            if not pd.api.types.is_numeric_dtype(numeric[column])
        ]
        if non_numeric:
            raise ValueError(f"Preprocessor features must be numeric: {non_numeric}")
        imputed = self.imputer.transform(numeric)
        scaled = self.scaler.transform(imputed)
        transformed = np.asarray(scaled[:, self.retained_positions], dtype=float)
        if not np.isfinite(transformed).all():
            raise ValueError("Transformed feature matrix contains nonfinite values.")
        return transformed


def fit_numeric_preprocessor(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    task: str,
    method: str,
    layer: str,
) -> NumericPreprocessor:
    """Fit approved train-only imputation, scaling, and deterministic rank pruning."""
    features = tuple(feature_columns)
    if not features:
        raise ValueError("Numeric preprocessing requires at least one feature.")
    _require_columns(train, features)
    raw = train.loc[:, features].replace([np.inf, -np.inf], np.nan)
    non_numeric = [
        column for column in features if not pd.api.types.is_numeric_dtype(raw[column])
    ]
    if non_numeric:
        raise ValueError(f"Preprocessor features must be numeric: {non_numeric}")

    finite_counts = raw.notna().sum()
    eligible = tuple(column for column in features if finite_counts[column] > 0)
    if not eligible:
        raise ValueError("No feature has a finite training value.")

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw.loc[:, eligible])
    scaler = StandardScaler()
    scaled = np.asarray(scaler.fit_transform(imputed), dtype=float)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled training matrix contains nonfinite values.")

    zero_variance_positions = {
        index
        for index in range(scaled.shape[1])
        if np.ptp(scaled[:, index]) <= 1e-12
    }
    candidate_positions = [
        index for index in range(scaled.shape[1]) if index not in zero_variance_positions
    ]
    unique_positions: list[int] = []
    duplicate_of: dict[int, int] = {}
    for index in candidate_positions:
        duplicate = next(
            (
                previous
                for previous in unique_positions
                if np.allclose(
                    scaled[:, index],
                    scaled[:, previous],
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            None,
        )
        if duplicate is None:
            unique_positions.append(index)
        else:
            duplicate_of[index] = duplicate

    if not unique_positions:
        raise ValueError("All training features are constant or duplicate.")
    unique_matrix = scaled[:, unique_positions]
    _, upper, pivots = qr(unique_matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(upper))
    leading = float(diagonal[0]) if diagonal.size else 0.0
    tolerance = (
        np.finfo(float).eps * max(unique_matrix.shape) * leading
        if leading
        else 0.0
    )
    rank = int((diagonal > tolerance).sum())
    if rank < 1:
        raise ValueError("Training feature matrix has zero numerical rank.")
    selected_unique_local = set(int(position) for position in pivots[:rank])
    retained_positions = tuple(
        sorted(unique_positions[position] for position in selected_unique_local)
    )
    retained_features = tuple(eligible[position] for position in retained_positions)
    retained_matrix = scaled[:, retained_positions]
    if np.linalg.matrix_rank(retained_matrix) != retained_matrix.shape[1]:
        raise ValueError("Pivoted-QR preprocessing did not produce a full-rank matrix.")
    condition_number = float(np.linalg.cond(retained_matrix))

    median_by_feature = dict(zip(eligible, imputer.statistics_, strict=True))
    center_by_feature = dict(zip(eligible, scaler.mean_, strict=True))
    scale_by_feature = dict(zip(eligible, scaler.scale_, strict=True))
    retained_order = {position: order for order, position in enumerate(retained_positions)}
    rows: list[dict[str, object]] = []
    eligible_position = {column: index for index, column in enumerate(eligible)}
    for source_order, column in enumerate(features):
        observed = int(finite_counts[column])
        position = eligible_position.get(column)
        reason: str | None = None
        retained = False
        duplicate_feature: str | None = None
        if position is None:
            reason = "all_missing_in_training"
        elif position in zero_variance_positions:
            reason = "zero_variance_after_imputation"
        elif position in duplicate_of:
            reason = "exact_duplicate_after_scaling"
            duplicate_feature = eligible[duplicate_of[position]]
        elif position not in retained_positions:
            reason = "pivoted_qr_rank_dependent"
        else:
            retained = True
        rows.append(
            {
                "task": task,
                "method": method,
                "layer": layer,
                "feature": column,
                "source_order": source_order,
                "training_observed_count": observed,
                "training_missing_count": int(len(train) - observed),
                "training_missing_rate": float((len(train) - observed) / len(train)),
                "training_median": (
                    float(median_by_feature[column]) if position is not None else np.nan
                ),
                "scaling_center": (
                    float(center_by_feature[column]) if position is not None else np.nan
                ),
                "scaling_scale": (
                    float(scale_by_feature[column]) if position is not None else np.nan
                ),
                "retained": retained,
                "retained_order": (
                    retained_order[position] if position in retained_order else np.nan
                ),
                "exclusion_reason": reason,
                "duplicate_of_feature": duplicate_feature,
                "rank_policy": "training_only_pivoted_qr",
            }
        )
    manifest = pd.DataFrame(rows)
    return NumericPreprocessor(
        feature_columns=features,
        eligible_features=eligible,
        retained_features=retained_features,
        retained_positions=retained_positions,
        imputer=imputer,
        scaler=scaler,
        manifest=manifest,
        matrix_rank=rank,
        condition_number=condition_number,
    )


def add_ols_intercept(matrix: np.ndarray) -> np.ndarray:
    """Add exactly one intercept to an already full-rank feature matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("OLS feature matrix must be two-dimensional with multiple rows.")
    if not np.isfinite(values).all():
        raise ValueError("OLS feature matrix contains nonfinite values.")
    if np.linalg.matrix_rank(values) != values.shape[1]:
        raise ValueError("OLS feature matrix must be full rank before adding an intercept.")
    design = sm.add_constant(values, prepend=True, has_constant="add")
    if design.shape[1] != values.shape[1] + 1:
        raise ValueError("OLS design did not gain exactly one intercept.")
    if not np.allclose(design[:, 0], 1.0):
        raise ValueError("OLS intercept column is not constant one.")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("OLS design is not full rank after adding the intercept.")
    return np.asarray(design, dtype=float)


def fit_ols_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray | pd.Series,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one full-rank statsmodels OLS regression and predict the test matrix."""
    train = np.asarray(x_train, dtype=float)
    test = np.asarray(x_test, dtype=float)
    outcome = np.asarray(y_train, dtype=float)
    if train.shape[0] != outcome.shape[0]:
        raise ValueError("OLS features and outcome have different training row counts.")
    if not np.isfinite(outcome).all() or not np.isfinite(test).all():
        raise ValueError("OLS inputs contain nonfinite values.")
    train_design = add_ols_intercept(train)
    test_design = sm.add_constant(test, prepend=True, has_constant="add")
    result = sm.OLS(outcome, train_design, missing="raise", hasconst=True).fit()
    predictions = np.asarray(result.predict(test_design), dtype=float)
    parameters = np.asarray(result.params, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("OLS produced nonfinite predictions.")
    audit = {
        "design_rank": int(np.linalg.matrix_rank(train_design)),
        "design_column_count": int(train_design.shape[1]),
        "condition_number": float(np.linalg.cond(train_design)),
        "parameters_finite": bool(np.isfinite(parameters).all()),
        "parameter_count": int(len(parameters)),
        "r_squared": float(result.rsquared),
        "residual_sum_squares": float(np.square(result.resid).sum()),
        "df_resid": float(result.df_resid),
    }
    if not audit["parameters_finite"]:
        raise ValueError("OLS produced nonfinite parameters.")
    return predictions, audit


def fit_ordered_probit_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray | pd.Series,
    x_test: np.ndarray,
    *,
    optimizer: str = DEFAULT_ORDERED_OPTIMIZER,
    maxiter: int = DEFAULT_ORDERED_MAXITER,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit the approved five-class Ordered Probit and validate formal diagnostics."""
    train = np.asarray(x_train, dtype=float)
    test = np.asarray(x_test, dtype=float)
    outcome = np.asarray(y_train, dtype=int)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Ordered Probit train/test matrices have incompatible shapes.")
    if train.shape[0] != outcome.shape[0]:
        raise ValueError("Ordered Probit features and outcome have different row counts.")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Ordered Probit inputs contain nonfinite values.")
    if set(np.unique(outcome)) != {1, 2, 3, 4, 5}:
        raise ValueError("Ordered Probit training outcome must retain all five ordered classes.")
    if np.any(np.ptp(train, axis=0) <= 1e-12):
        raise ValueError("Ordered Probit design contains an implicit constant feature.")
    if np.linalg.matrix_rank(train) != train.shape[1]:
        raise ValueError("Ordered Probit design must be full rank.")

    model = OrderedModel(outcome, train, distr="probit")
    if not np.array_equal(np.asarray(model.labels), np.arange(1, 6)):
        raise ValueError(f"Unexpected Ordered Probit class mapping: {model.labels}")
    result = model.fit(
        method=optimizer,
        maxiter=maxiter,
        disp=False,
        full_output=True,
    )
    parameters = np.asarray(result.params, dtype=float)
    mle_retvals = dict(result.mle_retvals)
    converged = bool(mle_retvals.get("converged", False))
    raw_threshold_parameters = parameters[model.k_vars :]
    transformed_thresholds = np.asarray(
        model.transform_threshold_params(raw_threshold_parameters), dtype=float
    )
    finite_cutpoints = transformed_thresholds[1:-1]
    ordered_cutpoints = bool(
        np.isfinite(finite_cutpoints).all()
        and np.all(np.diff(finite_cutpoints) > 0)
    )
    probabilities = np.asarray(model.predict(parameters, exog=test), dtype=float)
    probability_valid = bool(
        probabilities.shape == (len(test), 5)
        and np.isfinite(probabilities).all()
        and np.all(probabilities >= -1e-12)
        and np.all(probabilities <= 1.0 + 1e-12)
        and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)
    )
    gradient = np.asarray(mle_retvals.get("gopt", np.nan), dtype=float)
    max_abs_gradient = (
        float(np.max(np.abs(gradient))) if np.isfinite(gradient).any() else np.nan
    )
    audit = {
        "optimizer": optimizer,
        "maxiter": int(maxiter),
        "converged": converged,
        "warnflag": int(mle_retvals.get("warnflag", -1)),
        "function_calls": int(mle_retvals.get("fcalls", -1)),
        "gradient_calls": int(mle_retvals.get("gcalls", -1)),
        "max_abs_gradient": max_abs_gradient,
        "parameters_finite": bool(np.isfinite(parameters).all()),
        "raw_threshold_parameters": raw_threshold_parameters.tolist(),
        "transformed_cutpoints": finite_cutpoints.tolist(),
        "transformed_cutpoints_strictly_ordered": ordered_cutpoints,
        "probabilities_valid": probability_valid,
    }
    failures = [
        name
        for name, passed in (
            ("optimizer convergence", converged),
            ("finite parameters", audit["parameters_finite"]),
            ("ordered transformed cutpoints", ordered_cutpoints),
            ("five-class probabilities", probability_valid),
        )
        if not passed
    ]
    if failures:
        raise RuntimeError(
            "Ordered Probit failed the approved formal checks: "
            + ", ".join(failures)
            + f"; diagnostics={audit}"
        )
    predicted = np.asarray(model.labels)[np.argmax(probabilities, axis=1)].astype(int)
    return predicted, probabilities, audit


def calculate_pooled_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    """Calculate pooled five-class and binary Phase 3+ metrics."""
    _require_columns(predictions, ["actual_phase", "predicted_phase"])
    actual = predictions["actual_phase"].to_numpy(dtype=int)
    predicted = predictions["predicted_phase"].to_numpy(dtype=int)
    actual_positive = actual >= 3
    predicted_positive = predicted >= 3
    true_positive = int((actual_positive & predicted_positive).sum())
    false_positive = int((~actual_positive & predicted_positive).sum())
    false_negative = int((actual_positive & ~predicted_positive).sum())
    true_negative = int((~actual_positive & ~predicted_positive).sum())
    predicted_positive_count = true_positive + false_positive
    actual_positive_count = true_positive + false_negative
    precision = (
        true_positive / predicted_positive_count
        if predicted_positive_count
        else np.nan
    )
    recall = (
        true_positive / actual_positive_count if actual_positive_count else np.nan
    )
    return {
        "overall_accuracy": float(accuracy_score(actual, predicted)),
        "phase3plus_accuracy": float(
            accuracy_score(actual_positive, predicted_positive)
        ),
        "phase3plus_precision": float(precision),
        "phase3plus_recall": float(recall),
        "phase3above_r2": float(
            r2_score(
                actual_positive.astype(int),
                predicted_positive.astype(int),
            )
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _validate_figure_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = [
        "task",
        "method",
        "method_role",
        *METRIC_ORDER,
        "n_test",
        "test_key_sha256",
    ]
    _require_columns(metrics, required)
    if metrics.duplicated(["task", "method"]).any():
        raise ValueError("Figure metrics contain duplicate task-method rows.")
    expected = {
        (task, method)
        for task in TASK_ORDER
        for method in (*METHOD_ORDER, MAIN_RESULT_METHOD)
    }
    actual = set(zip(metrics["task"], metrics["method"], strict=False))
    if actual != expected:
        raise ValueError(
            f"Figure metrics have missing or unexpected task-method rows: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    values = metrics.loc[:, METRIC_ORDER].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Figure metrics contain undefined or nonfinite values.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Figure classification metrics must lie between zero and one.")
    if not metrics["n_test"].eq(EXPECTED_TEST_ROWS).all():
        raise ValueError("Figure metrics must all use the 1,170-row evaluation population.")
    if metrics["test_key_sha256"].nunique() != 1:
        raise ValueError("Figure metrics do not share one test-key hash.")
    task_rank = {task: index for index, task in enumerate(TASK_ORDER)}
    method_rank = {
        method: index
        for index, method in enumerate((*METHOD_ORDER, MAIN_RESULT_METHOD))
    }
    return (
        metrics.assign(
            _task=metrics["task"].map(task_rank),
            _method=metrics["method"].map(method_rank),
        )
        .sort_values(["_task", "_method"], kind="mergesort")
        .drop(columns=["_task", "_method"])
        .reset_index(drop=True)
    )


def create_simple_baseline_comparison_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Create the approved 2 x 2 four-bar comparison with original references."""
    plotting = _validate_figure_metrics(metrics).set_index(["task", "method"])
    loco.apply_figure_style()
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), squeeze=False)
    x = np.arange(len(METHOD_ORDER), dtype=float)
    display_names = ["Persistence", "Multinomial", "Ordered\nProbit", "Ensemble\nOLS"]
    panel_index = 0

    for row, task in enumerate(TASK_ORDER):
        for column, metric in enumerate(METRIC_ORDER):
            axis = axes[row, column]
            values = np.asarray(
                [float(plotting.loc[(task, method), metric]) for method in METHOD_ORDER]
            )
            bars = axis.bar(
                x,
                values,
                width=0.68,
                color=[METHOD_COLORS[method] for method in METHOD_ORDER],
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            reference = float(plotting.loc[(task, MAIN_RESULT_METHOD), metric])
            axis.axhline(
                reference,
                color="#6F6F6F",
                linestyle="--",
                linewidth=0.9,
                zorder=1,
            )
            marker_x = 3.58
            axis.scatter(
                [marker_x],
                [reference],
                marker="D",
                s=35,
                facecolors="white",
                edgecolors="#5F5F5F",
                linewidth=1.0,
                zorder=4,
            )
            for bar, value in zip(bars, values, strict=True):
                offset = 0.018
                vertical_alignment = "bottom"
                label_y = value + offset
                if value > 0.955:
                    label_y = value - offset
                    vertical_alignment = "top"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:.3f}",
                    ha="center",
                    va=vertical_alignment,
                    fontsize=6,
                    color="#222222",
                    zorder=5,
                )
            axis.text(
                marker_x,
                reference - 0.035,
                f"{reference:.3f}",
                ha="center",
                va="top",
                fontsize=6,
                color="#555555",
                zorder=5,
            )
            axis.set_ylim(0.0, 1.0)
            axis.set_xlim(-0.55, 3.92)
            axis.set_xticks(x, display_names)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row == 0:
                axis.set_title(
                    "Phase 3+ precision" if column == 0 else "Phase 3+ recall",
                    loc="left",
                    pad=5,
                    fontweight="normal",
                )
            if column == 0:
                axis.set_ylabel(task, labelpad=10)
            axis.text(
                -0.15,
                1.10,
                chr(ord("a") + panel_index),
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="top",
                clip_on=False,
            )
            panel_index += 1

    handles = [
        mpl.patches.Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="white",
            label=method,
        )
        for method in METHOD_ORDER
    ]
    handles.append(
        mpl.lines.Line2D(
            [],
            [],
            color="#6F6F6F",
            linestyle="--",
            linewidth=0.9,
            marker="D",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="#5F5F5F",
            label=MAIN_RESULT_METHOD,
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    figure.suptitle(
        "Simple baselines versus the main result: "
        "2022 temporal evaluation, 1,170 area-date rows",
        x=0.10,
        ha="left",
        fontsize=9,
        fontweight="normal",
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.19,
        top=0.90,
        wspace=0.24,
        hspace=0.38,
    )
    return figure


def save_simple_baseline_comparison_figure(
    figure: plt.Figure,
    output_dir: Path,
) -> dict[str, Path]:
    """Save the formal comparison figure in three publication formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        key: output_dir / OUTPUT_FILENAMES[key]
        for key in ("figure_jpg", "figure_png", "figure_pdf")
    }
    figure.savefig(paths["figure_jpg"], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(paths["figure_png"], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        paths["figure_pdf"],
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    return paths


def _truth_table(bundle: PreparedInputs) -> pd.DataFrame:
    truth = bundle.forecasting.loc[
        bundle.forecasting_test_mask,
        [*loco.KEY_COLUMNS, "evaluation_phase"],
    ].copy()
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    truth = truth.rename(columns={"evaluation_phase": "canonical_actual_phase"})
    if truth.duplicated(loco.KEY_COLUMNS).any() or len(truth) != EXPECTED_TEST_ROWS:
        raise ValueError("Canonical truth table has invalid temporal-test keys.")
    return truth


def _finalize_prediction_frame(
    frame: pd.DataFrame,
    bundle: PreparedInputs,
) -> pd.DataFrame:
    """Attach canonical truth, common indicators, group hash, and exact schema."""
    required = [
        "task",
        "method",
        "area_id",
        "date",
        "source_dataset",
        "source_row_index",
        "source_key_area_id",
        "source_key_date",
        "predicted_phase",
    ]
    _require_columns(frame, required)
    result = frame.copy()
    result["area_id"] = result["area_id"].astype(int)
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    result["source_key_date"] = pd.to_datetime(
        result["source_key_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if result.duplicated(["area_id", "date"]).any():
        raise ValueError(
            f"{result['task'].iloc[0]} {result['method'].iloc[0]} has duplicate test keys."
        )
    result = result.merge(
        _truth_table(bundle),
        on=loco.KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(result) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"{result['task'].iloc[0]} {result['method'].iloc[0]} lost test rows."
        )
    if "actual_phase" in result.columns:
        supplied = result["actual_phase"].astype(int)
        canonical = result["canonical_actual_phase"].astype(int)
        if not supplied.eq(canonical).all():
            bad = result.loc[
                ~supplied.eq(canonical), ["area_id", "date", "actual_phase", "canonical_actual_phase"]
            ].head()
            raise ValueError(f"Adapter actual phases differ from canonical truth: {bad}")
    result["actual_phase"] = result.pop("canonical_actual_phase").astype(int)
    result["predicted_phase"] = result["predicted_phase"].astype(int)
    if not result["predicted_phase"].between(1, 5).all():
        raise ValueError("Predicted phases must lie between one and five.")
    result["actual_phase3plus"] = result["actual_phase"].ge(3).astype(int)
    result["predicted_phase3plus"] = result["predicted_phase"].ge(3).astype(int)
    group_hash = canonical_key_sha256(result)
    if group_hash != bundle.test_key_sha256:
        raise ValueError(
            f"{result['task'].iloc[0]} {result['method'].iloc[0]} test-key hash differs."
        )
    result["test_key_sha256"] = group_hash
    for column in PREDICTION_COLUMNS:
        if column not in result:
            result[column] = np.nan
    return (
        result.loc[:, PREDICTION_COLUMNS]
        .sort_values(["date", "area_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _adapter_manifest(
    data: pd.DataFrame,
    train_mask: pd.Series,
    features: Sequence[str],
    *,
    task: str,
    method: str,
    layer: str,
    rank_policy: str,
) -> pd.DataFrame:
    """Describe an authoritative adapter feature set without changing its fit."""
    raw = data.loc[:, list(features)].replace([np.inf, -np.inf], np.nan)
    train = raw.loc[train_mask]
    observed = train.notna().sum()
    eligible = [column for column in features if observed[column] > 0]
    medians: dict[str, float] = {}
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    if eligible:
        imputer = SimpleImputer(strategy="median")
        imputed = imputer.fit_transform(train[eligible])
        scaler = StandardScaler()
        scaler.fit(imputed)
        medians = dict(zip(eligible, imputer.statistics_, strict=True))
        centers = dict(zip(eligible, scaler.mean_, strict=True))
        scales = dict(zip(eligible, scaler.scale_, strict=True))
    rows = []
    retained_order = {column: index for index, column in enumerate(eligible)}
    for source_order, column in enumerate(features):
        is_retained = column in retained_order
        rows.append(
            {
                "task": task,
                "method": method,
                "layer": layer,
                "feature": column,
                "source_order": source_order,
                "training_observed_count": int(observed[column]),
                "training_missing_count": int(len(train) - observed[column]),
                "training_missing_rate": float(
                    (len(train) - observed[column]) / len(train)
                ),
                "training_median": (
                    float(medians[column]) if is_retained else np.nan
                ),
                "scaling_center": (
                    float(centers[column]) if is_retained else np.nan
                ),
                "scaling_scale": (
                    float(scales[column]) if is_retained else np.nan
                ),
                "retained": is_retained,
                "retained_order": (
                    retained_order[column] if is_retained else np.nan
                ),
                "exclusion_reason": (
                    None if is_retained else "all_missing_in_training"
                ),
                "duplicate_of_feature": None,
                "rank_policy": rank_policy,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS)


def _persistence_manifest(task: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": task,
                "method": "Persistence",
                "layer": "historical_source_rule",
                "feature": "(not applicable)",
                "source_order": 0,
                "training_observed_count": np.nan,
                "training_missing_count": np.nan,
                "training_missing_rate": np.nan,
                "training_median": np.nan,
                "scaling_center": np.nan,
                "scaling_scale": np.nan,
                "retained": False,
                "retained_order": np.nan,
                "exclusion_reason": "not_feature_based",
                "duplicate_of_feature": None,
                "rank_policy": "not_applicable_authoritative_adapter",
            }
        ],
        columns=FEATURE_MANIFEST_COLUMNS,
    )


def _empty_model_audit_row(task: str, method: str) -> dict[str, object]:
    row = {column: np.nan for column in MODEL_AUDIT_COLUMNS}
    row.update({"task": task, "method": method})
    return row


def fit_persistence_adapter(
    bundle: PreparedInputs,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Run and normalize the existing Persistence baseline without changing it."""
    countries = persistence.load_country_lookup(country_lookup_path)
    predictions: dict[str, pd.DataFrame] = {}
    manifests = []
    audits = []
    for horizon, task in (("forecasting", "Forecasting"), ("nowcasting", "Nowcasting")):
        existing = persistence.build_persistence_predictions(
            bundle.raw_forecasting,
            countries,
            horizon,
            target_start=DEFAULT_CUTOFF_DATE,
        )
        frame = pd.DataFrame(
            {
                "task": task,
                "method": "Persistence",
                "area_id": existing["area_id"].astype(int),
                "date": existing["target_date"],
                "source_dataset": DEFAULT_FORECASTING_INPUT.name,
                "source_row_index": existing["target_index"].astype(int),
                "source_key_area_id": existing["area_id"].astype(int),
                "source_key_date": existing["target_date"],
                "actual_phase": existing["actual_phase"].astype(int),
                "predicted_phase": existing["predicted_phase"].astype(int),
                "persistence_issue_date": existing["issue_date"],
                "persistence_source_method": existing["source_method"],
                "persistence_source_area_id": existing["source_area_id"].astype(int),
                "persistence_source_country_code_3": existing[
                    "source_country_code_3"
                ],
                "persistence_source_date": existing["source_date"],
                "persistence_distance_km": existing["distance_km"].astype(float),
                "persistence_target_gap_months": existing[
                    "target_gap_months"
                ].astype(int),
                "persistence_issue_gap_months": existing[
                    "issue_gap_months"
                ].astype(int),
            }
        )
        predictions[task] = _finalize_prediction_frame(frame, bundle)
        manifests.append(_persistence_manifest(task))
        counts = existing["source_method"].value_counts()
        row = _empty_model_audit_row(task, "Persistence")
        row.update(
            {
                "fit_status": "generated",
                "source_dataset": DEFAULT_FORECASTING_INPUT.name,
                "n_train": EXPECTED_TRAIN_ROWS,
                "n_test": EXPECTED_TEST_ROWS,
                "converged": True,
                "notes": (
                    "Authoritative historical source rule; own/same-country/global "
                    f"counts={int(counts.get('own_history', 0))}/"
                    f"{int(counts.get('same_country_neighbor', 0))}/"
                    f"{int(counts.get('global_neighbor', 0))}"
                ),
            }
        )
        audits.append(row)
    return (
        predictions,
        pd.concat(manifests, ignore_index=True),
        pd.DataFrame(audits, columns=MODEL_AUDIT_COLUMNS),
    )


def fit_multinomial_adapter(
    bundle: PreparedInputs,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Load and normalize the frozen existing Multinomial prediction artifacts."""
    predictions: dict[str, pd.DataFrame] = {}
    manifests = []
    audits = []
    metric_artifact = pd.read_csv(DEFAULT_OUTPUT_DIR / "multinomial_baseline_metrics.csv")
    task_sources = {
        "Forecasting": (bundle.raw_forecasting, DEFAULT_FORECASTING_INPUT.name),
        "Nowcasting": (bundle.raw_nowcasting, DEFAULT_NOWCASTING_INPUT.name),
    }
    for task in ("Forecasting", "Nowcasting"):
        data, source_name = task_sources[task]
        prediction_path = (
            DEFAULT_OUTPUT_DIR
            / f"multinomial_baseline_{task.lower()}_predictions.csv"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Frozen Multinomial prediction artifact is missing: {prediction_path}"
            )
        existing = pd.read_csv(prediction_path, float_precision="round_trip")
        record = metric_artifact.loc[
            metric_artifact["task"].eq(task)
            & metric_artifact["model"].eq("Multinomial logistic baseline")
        ]
        if len(record) != 1:
            raise ValueError(f"Frozen Multinomial metric row is missing for {task}.")
        record = record.iloc[0]
        frame = pd.DataFrame(
            {
                "task": task,
                "method": "Multinomial",
                "area_id": existing["area_id"].astype(int),
                "date": existing["date"],
                "source_dataset": source_name,
                "source_row_index": existing["source_index"].astype(int),
                "source_key_area_id": existing["area_id"].astype(int),
                "source_key_date": existing["date"],
                "actual_phase": existing["overall_phase_evaluation"].astype(int),
                "predicted_phase": existing["overall_phase_pred"].astype(int),
            }
        )
        predictions[task] = _finalize_prediction_frame(frame, bundle)
        train_mask, _ = temporal_masks(data["date"])
        features = multinomial.select_feature_columns(data)
        manifests.append(
            _adapter_manifest(
                data,
                train_mask,
                features,
                task=task,
                method="Multinomial",
                layer="direct",
                rank_policy="not_applied_authoritative_adapter",
            )
        )
        row = _empty_model_audit_row(task, "Multinomial")
        row.update(
            {
                "fit_status": "generated",
                "source_dataset": source_name,
                "n_train": int(record["n_train"]),
                "n_test": int(record["n_test"]),
                "source_feature_count": int(record["feature_count"]),
                "retained_feature_count": int(record["feature_count"]),
                "optimizer": "frozen_existing_prediction_artifact",
                "converged": True,
                "parameters_finite": True,
                "notes": (
                    "Frozen authoritative prediction artifact; avoids current "
                    "scikit-learn refit drift while retaining the documented "
                    "SimpleImputer/StandardScaler/logistic contract."
                ),
            }
        )
        audits.append(row)
    return (
        predictions,
        pd.concat(manifests, ignore_index=True),
        pd.DataFrame(audits, columns=MODEL_AUDIT_COLUMNS),
    )


def fit_ordered_probit_adapter(
    bundle: PreparedInputs,
    *,
    optimizer: str = DEFAULT_ORDERED_OPTIMIZER,
    maxiter: int = DEFAULT_ORDERED_MAXITER,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Fit one approved direct Ordered Probit for each task."""
    predictions: dict[str, pd.DataFrame] = {}
    manifests = []
    audits = []
    task_sources = {
        "Forecasting": (bundle.raw_forecasting, DEFAULT_FORECASTING_INPUT.name),
        "Nowcasting": (bundle.raw_nowcasting, DEFAULT_NOWCASTING_INPUT.name),
    }
    for task in ("Forecasting", "Nowcasting"):
        data, source_name = task_sources[task]
        train_mask, test_mask = temporal_masks(data["date"])
        features = multinomial.select_feature_columns(data)
        preprocessor = fit_numeric_preprocessor(
            data.loc[train_mask],
            features,
            task=task,
            method="Ordered Probit",
            layer="direct",
        )
        x_train = preprocessor.transform(data.loc[train_mask])
        x_test = preprocessor.transform(data.loc[test_mask])
        outcome = derive_evaluation_phase(data)
        predicted, probabilities, fit_audit = fit_ordered_probit_arrays(
            x_train,
            outcome.loc[train_mask],
            x_test,
            optimizer=optimizer,
            maxiter=maxiter,
        )
        test_rows = data.loc[test_mask]
        frame = pd.DataFrame(
            {
                "task": task,
                "method": "Ordered Probit",
                "area_id": test_rows["area_id"].to_numpy(dtype=int),
                "date": pd.to_datetime(test_rows["date"]).dt.strftime("%Y-%m-%d"),
                "source_dataset": source_name,
                "source_row_index": test_rows.index.to_numpy(dtype=int),
                "source_key_area_id": test_rows["area_id"].to_numpy(dtype=int),
                "source_key_date": pd.to_datetime(test_rows["date"]).dt.strftime(
                    "%Y-%m-%d"
                ),
                "actual_phase": outcome.loc[test_mask].to_numpy(dtype=int),
                "predicted_phase": predicted,
            }
        )
        for phase in range(1, 6):
            frame[f"ordered_probability_phase{phase}"] = probabilities[:, phase - 1]
        predictions[task] = _finalize_prediction_frame(frame, bundle)
        manifests.append(preprocessor.manifest)
        row = _empty_model_audit_row(task, "Ordered Probit")
        row.update(
            {
                "fit_status": "generated",
                "source_dataset": source_name,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "source_feature_count": len(features),
                "retained_feature_count": len(preprocessor.retained_features),
                "optimizer": optimizer,
                "converged": fit_audit["converged"],
                "max_abs_gradient": fit_audit["max_abs_gradient"],
                "parameters_finite": fit_audit["parameters_finite"],
                "raw_threshold_parameters_json": json.dumps(
                    fit_audit["raw_threshold_parameters"], separators=(",", ":")
                ),
                "transformed_cutpoints_json": json.dumps(
                    fit_audit["transformed_cutpoints"], separators=(",", ":")
                ),
                "probabilities_valid": fit_audit["probabilities_valid"],
                "design_rank_min": preprocessor.matrix_rank,
                "design_condition_number_max": preprocessor.condition_number,
                "fit_diagnostics_json": json.dumps(
                    fit_audit, sort_keys=True, separators=(",", ":")
                ),
                "notes": "Direct five-class cumulative-share reconstructed phase outcome.",
            }
        )
        audits.append(row)
    return (
        predictions,
        pd.concat(manifests, ignore_index=True),
        pd.DataFrame(audits, columns=MODEL_AUDIT_COLUMNS),
    )


def _phase_from_rounded_predictions(data: pd.DataFrame) -> np.ndarray:
    conditions = [
        data["phase5_pred_rounded"].ge(0.20),
        data["phase4_pred_rounded"].ge(0.20),
        data["phase3_pred_rounded"].ge(0.20),
        data["phase2_pred_rounded"].ge(0.20),
    ]
    return np.select(conditions, [5, 4, 3, 2], default=1).astype(int)


def fit_ensemble_ols_adapter(
    bundle: PreparedInputs,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Fit the approved four-target Forecasting and two-layer Nowcasting OLS."""
    forecasting = bundle.forecasting
    nowcasting = bundle.nowcasting
    f_train = bundle.forecasting_train_mask
    f_test = bundle.forecasting_test_mask
    n_train = bundle.nowcasting_train_mask
    n_test = bundle.nowcasting_test_mask
    layer1_features = tuple(
        feature
        for feature in loco.select_layer1_features(forecasting)
        if feature != "evaluation_phase"
    )
    layer2_features = tuple(loco.NOWCAST_FEATURES)
    if len(layer1_features) != EXPECTED_ENSEMBLE_OLS_LAYER1_SOURCE_FEATURES:
        raise ValueError(
            "Ensemble OLS Layer 1 source-feature contract drifted: "
            f"expected {EXPECTED_ENSEMBLE_OLS_LAYER1_SOURCE_FEATURES}, "
            f"found {len(layer1_features)}."
        )
    if len(layer2_features) != EXPECTED_ENSEMBLE_OLS_LAYER2_SOURCE_FEATURES:
        raise ValueError(
            "Ensemble OLS Layer 2 source-feature contract drifted: "
            f"expected {EXPECTED_ENSEMBLE_OLS_LAYER2_SOURCE_FEATURES}, "
            f"found {len(layer2_features)}."
        )

    layer1_preprocessor = fit_numeric_preprocessor(
        forecasting.loc[f_train],
        layer1_features,
        task="Forecasting",
        method="Ensemble OLS",
        layer="layer1_shared",
    )
    layer2_preprocessor = fit_numeric_preprocessor(
        nowcasting.loc[n_train],
        layer2_features,
        task="Nowcasting",
        method="Ensemble OLS",
        layer="layer2_residual",
    )
    x1_train = layer1_preprocessor.transform(forecasting.loc[f_train])
    x1_test = layer1_preprocessor.transform(forecasting.loc[f_test])

    test_rows = forecasting.loc[f_test]
    base_common = pd.DataFrame(
        {
            "area_id": test_rows["area_id"].to_numpy(dtype=int),
            "date": pd.to_datetime(test_rows["date"]).dt.strftime("%Y-%m-%d"),
            "source_dataset": DEFAULT_FORECASTING_INPUT.name,
            "source_row_index": test_rows.index.to_numpy(dtype=int),
            "source_key_area_id": test_rows["area_id"].to_numpy(dtype=int),
            "source_key_date": pd.to_datetime(test_rows["date"]).dt.strftime(
                "%Y-%m-%d"
            ),
            "actual_phase": test_rows["evaluation_phase"].to_numpy(dtype=int),
        }
    )

    forecast_frame = base_common.copy()
    forecast_fit_audits = []
    layer1_train_predictions: dict[int, np.ndarray] = {}
    layer1_test_predictions: dict[int, np.ndarray] = {}
    for phase, target_column in loco.CUMULATIVE_TARGETS.items():
        y_train = forecasting.loc[f_train, target_column].to_numpy(dtype=float)
        predictions, fit_audit = fit_ols_arrays(x1_train, y_train, x1_test)
        train_predictions, _ = fit_ols_arrays(x1_train, y_train, x1_train)
        layer1_train_predictions[phase] = train_predictions
        layer1_test_predictions[phase] = predictions
        forecast_frame[f"phase{phase}_actual_cumulative"] = forecasting.loc[
            f_test, target_column
        ].to_numpy(dtype=float)
        forecast_frame[f"phase{phase}_pred_raw"] = predictions
        forecast_frame[f"phase{phase}_pred_rounded"] = np.round(predictions, 2)
        forecast_frame[f"phase{phase}_layer1_pred"] = predictions
        forecast_frame[f"phase{phase}_residual_pred"] = np.nan
        forecast_fit_audits.append({"phase": phase, **fit_audit})
    forecast_frame["predicted_phase"] = _phase_from_rounded_predictions(
        forecast_frame
    )
    forecast_frame.insert(0, "method", "Ensemble OLS")
    forecast_frame.insert(0, "task", "Forecasting")

    keys = [*loco.KEY_COLUMNS, "country_code_3"]
    nowcast_frame = base_common.copy()
    nowcast_fit_audits = []
    for phase, target_column in loco.CUMULATIVE_TARGETS.items():
        residual_frame = forecasting.loc[f_train, keys].copy()
        residual_frame["layer1_residual"] = (
            forecasting.loc[f_train, target_column].to_numpy(dtype=float)
            - layer1_train_predictions[phase]
        )
        keyed_train = nowcasting.loc[
            n_train, [*keys, *layer2_features]
        ].merge(
            residual_frame,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(keyed_train) != EXPECTED_TRAIN_ROWS:
            raise ValueError("Ensemble OLS Nowcasting lost residual training rows.")
        x2_train = layer2_preprocessor.transform(keyed_train)
        now_test_rows = nowcasting.loc[n_test, [*keys, *layer2_features]]
        x2_test = layer2_preprocessor.transform(now_test_rows)
        residual_predictions, fit_audit = fit_ols_arrays(
            x2_train,
            keyed_train["layer1_residual"].to_numpy(dtype=float),
            x2_test,
        )
        residual_prediction_frame = now_test_rows[keys].copy()
        residual_prediction_frame["residual_prediction"] = residual_predictions
        phase_frame = forecasting.loc[f_test, keys].copy()
        phase_frame["layer1_prediction"] = layer1_test_predictions[phase]
        phase_frame = phase_frame.merge(
            residual_prediction_frame,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(phase_frame) != EXPECTED_TEST_ROWS:
            raise ValueError("Ensemble OLS Nowcasting lost test residual rows.")
        raw = (
            phase_frame["layer1_prediction"].to_numpy(dtype=float)
            + phase_frame["residual_prediction"].to_numpy(dtype=float)
        )
        nowcast_frame[f"phase{phase}_actual_cumulative"] = forecasting.loc[
            f_test, target_column
        ].to_numpy(dtype=float)
        nowcast_frame[f"phase{phase}_pred_raw"] = raw
        nowcast_frame[f"phase{phase}_pred_rounded"] = np.round(raw, 2)
        nowcast_frame[f"phase{phase}_layer1_pred"] = phase_frame[
            "layer1_prediction"
        ].to_numpy(dtype=float)
        nowcast_frame[f"phase{phase}_residual_pred"] = phase_frame[
            "residual_prediction"
        ].to_numpy(dtype=float)
        nowcast_fit_audits.append({"phase": phase, **fit_audit})
    nowcast_frame["predicted_phase"] = _phase_from_rounded_predictions(nowcast_frame)
    nowcast_frame.insert(0, "method", "Ensemble OLS")
    nowcast_frame.insert(0, "task", "Nowcasting")

    predictions = {
        "Forecasting": _finalize_prediction_frame(forecast_frame, bundle),
        "Nowcasting": _finalize_prediction_frame(nowcast_frame, bundle),
    }

    manifests = []
    for task in ("Forecasting", "Nowcasting"):
        copied = layer1_preprocessor.manifest.copy()
        copied["task"] = task
        copied["layer"] = "layer1_shared"
        manifests.append(copied)
    manifests.append(layer2_preprocessor.manifest)

    audits = []
    for task, frame, fit_audits in (
        ("Forecasting", predictions["Forecasting"], forecast_fit_audits),
        ("Nowcasting", predictions["Nowcasting"], nowcast_fit_audits),
    ):
        raw_columns = [f"phase{phase}_pred_raw" for phase in range(2, 6)]
        raw_values = frame[raw_columns].to_numpy(dtype=float)
        range_violations = int(((raw_values < 0.0) | (raw_values > 1.0)).sum())
        order_violations = int(
            (
                (frame["phase2_pred_raw"] < frame["phase3_pred_raw"])
                | (frame["phase3_pred_raw"] < frame["phase4_pred_raw"])
                | (frame["phase4_pred_raw"] < frame["phase5_pred_raw"])
            ).sum()
        )
        row = _empty_model_audit_row(task, "Ensemble OLS")
        row.update(
            {
                "fit_status": "generated",
                "source_dataset": (
                    DEFAULT_FORECASTING_INPUT.name
                    if task == "Forecasting"
                    else f"{DEFAULT_FORECASTING_INPUT.name} + {DEFAULT_NOWCASTING_INPUT.name}"
                ),
                "n_train": EXPECTED_TRAIN_ROWS,
                "n_test": EXPECTED_TEST_ROWS,
                "source_feature_count": len(layer1_features)
                + (len(layer2_features) if task == "Nowcasting" else 0),
                "retained_feature_count": len(layer1_preprocessor.retained_features)
                + (
                    len(layer2_preprocessor.retained_features)
                    if task == "Nowcasting"
                    else 0
                ),
                "layer1_retained_feature_count": len(
                    layer1_preprocessor.retained_features
                ),
                "layer2_retained_feature_count": (
                    len(layer2_preprocessor.retained_features)
                    if task == "Nowcasting"
                    else 0
                ),
                "converged": True,
                "parameters_finite": bool(
                    all(audit["parameters_finite"] for audit in fit_audits)
                ),
                "design_rank_min": int(
                    min(audit["design_rank"] for audit in fit_audits)
                ),
                "design_condition_number_max": float(
                    max(audit["condition_number"] for audit in fit_audits)
                ),
                "out_of_range_prediction_cell_count": range_violations,
                "cumulative_order_violation_row_count": order_violations,
                "fit_diagnostics_json": json.dumps(
                    fit_audits, sort_keys=True, separators=(",", ":")
                ),
                "notes": (
                    "Unconstrained raw cumulative predictions; two-decimal copies "
                    "used only for phase conversion."
                ),
            }
        )
        audits.append(row)
    return (
        predictions,
        pd.concat(manifests, ignore_index=True)[FEATURE_MANIFEST_COLUMNS],
        pd.DataFrame(audits, columns=MODEL_AUDIT_COLUMNS),
    )


def build_combined_predictions(
    prediction_frames: Mapping[tuple[str, str], pd.DataFrame],
    *,
    expected_tasks: Sequence[str] = TASK_ORDER,
    expected_methods: Sequence[str] = METHOD_ORDER,
) -> pd.DataFrame:
    """Combine normalized groups and enforce one identical test population."""
    expected = {
        (task, method) for task in expected_tasks for method in expected_methods
    }
    if set(prediction_frames) != expected:
        raise ValueError(
            f"Prediction groups differ from selection: missing="
            f"{sorted(expected - set(prediction_frames))}, "
            f"unexpected={sorted(set(prediction_frames) - expected)}"
        )
    combined = pd.concat(
        [prediction_frames[key] for key in expected],
        ignore_index=True,
    )
    if combined.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Combined predictions have an unexpected schema.")
    if combined.duplicated(["task", "method", "area_id", "date"]).any():
        raise ValueError("Combined predictions contain duplicate group observation keys.")
    hashes = []
    truth_hashes = []
    for (task, method), group in combined.groupby(["task", "method"], sort=False):
        if len(group) != EXPECTED_TEST_ROWS:
            raise ValueError(f"{task} {method} does not contain 1,170 predictions.")
        key_hash = canonical_key_sha256(group)
        if not group["test_key_sha256"].eq(key_hash).all():
            raise ValueError(f"{task} {method} stores an incorrect test-key hash.")
        hashes.append(key_hash)
        truth_hashes.append(
            canonical_dataframe_sha256(
                group,
                ("area_id", "date"),
                ("area_id", "date", "actual_phase"),
            )
        )
    if len(set(hashes)) != 1 or len(set(truth_hashes)) != 1:
        raise ValueError("Baseline groups do not share one key and truth population.")
    task_rank = {task: index for index, task in enumerate(TASK_ORDER)}
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    return (
        combined.assign(
            _task=combined["task"].map(task_rank),
            _method=combined["method"].map(method_rank),
        )
        .sort_values(["_task", "_method", "date", "area_id"], kind="mergesort")
        .drop(columns=["_task", "_method"])
        .reset_index(drop=True)
    )


def _reference_metrics(
    task: str,
    actual_phase: pd.Series,
    test_key_hash: str,
) -> dict[str, object]:
    reference = MAIN_RESULT_REFERENCES[task]
    actual = actual_phase.to_numpy(dtype=int)
    actual_positive_count = int((actual >= 3).sum())
    actual_negative_count = int((actual < 3).sum())
    true_positive = int(
        round(float(reference["phase3plus_recall"]) * actual_positive_count)
    )
    predicted_positive_count = int(
        round(true_positive / float(reference["phase3plus_precision"]))
    )
    false_positive = predicted_positive_count - true_positive
    false_negative = actual_positive_count - true_positive
    true_negative = actual_negative_count - false_positive
    if min(true_positive, false_positive, false_negative, true_negative) < 0:
        raise ValueError(f"Main-result reference counts are invalid for {task}.")
    reconstructed_precision = true_positive / (true_positive + false_positive)
    reconstructed_recall = true_positive / (true_positive + false_negative)
    if not np.isclose(
        reconstructed_precision,
        float(reference["phase3plus_precision"]),
        rtol=0.0,
        atol=1e-15,
    ) or not np.isclose(
        reconstructed_recall,
        float(reference["phase3plus_recall"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(f"Main-result reference rates do not reconstruct exactly for {task}.")
    actual_positive_indicator = np.concatenate(
        [
            np.ones(true_positive + false_negative, dtype=int),
            np.zeros(true_negative + false_positive, dtype=int),
        ]
    )
    predicted_positive_indicator = np.concatenate(
        [
            np.ones(true_positive, dtype=int),
            np.zeros(false_negative, dtype=int),
            np.ones(false_positive, dtype=int),
            np.zeros(true_negative, dtype=int),
        ]
    )
    return {
        "task": task,
        "method": MAIN_RESULT_METHOD,
        "method_role": "main_result_reference",
        "overall_accuracy": float(reference["overall_accuracy"]),
        "phase3plus_accuracy": float(
            (true_positive + true_negative) / len(actual)
        ),
        "phase3plus_precision": reconstructed_precision,
        "phase3plus_recall": reconstructed_recall,
        "phase3above_r2": float(
            r2_score(actual_positive_indicator, predicted_positive_indicator)
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "n_train": EXPECTED_TRAIN_ROWS,
        "n_test": len(actual),
        "test_key_sha256": test_key_hash,
        "target_definition": TARGET_DEFINITION,
        "metric_source": reference["metric_source"],
        "fit_status": "frozen_main_result_reference",
        "converged": np.nan,
    }


def build_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build eight generated metric rows plus two approved original references."""
    _require_columns(predictions, PREDICTION_COLUMNS)
    records = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            group = predictions.loc[
                predictions["task"].eq(task) & predictions["method"].eq(method)
            ]
            if len(group) != EXPECTED_TEST_ROWS:
                raise ValueError(f"Metrics group {task} {method} is incomplete.")
            metrics = calculate_pooled_metrics(group)
            records.append(
                {
                    "task": task,
                    "method": method,
                    "method_role": "baseline",
                    **metrics,
                    "n_train": EXPECTED_TRAIN_ROWS,
                    "n_test": EXPECTED_TEST_ROWS,
                    "test_key_sha256": group["test_key_sha256"].iloc[0],
                    "target_definition": TARGET_DEFINITION,
                    "metric_source": "Generated from unified row-level predictions",
                    "fit_status": "generated",
                    "converged": True,
                }
            )
        truth_group = predictions.loc[
            predictions["task"].eq(task)
            & predictions["method"].eq(METHOD_ORDER[0])
        ]
        records.append(
            _reference_metrics(
                task,
                truth_group["actual_phase"],
                truth_group["test_key_sha256"].iloc[0],
            )
        )
    metrics = pd.DataFrame(records, columns=METRIC_COLUMNS)
    task_rank = {task: index for index, task in enumerate(TASK_ORDER)}
    method_rank = {
        method: index
        for index, method in enumerate((*METHOD_ORDER, MAIN_RESULT_METHOD))
    }
    return (
        metrics.assign(
            _task=metrics["task"].map(task_rank),
            _method=metrics["method"].map(method_rank),
        )
        .sort_values(["_task", "_method"], kind="mergesort")
        .drop(columns=["_task", "_method"])
        .reset_index(drop=True)
    )


def validate_run_configuration(
    tasks: Sequence[str] | None,
    methods: Sequence[str] | None,
    output_dir: Path,
    cutoff_date: str,
    ordered_optimizer: str,
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    selected_tasks = tuple(TASK_ORDER if tasks is None else tasks)
    selected_methods = tuple(METHOD_ORDER if methods is None else methods)
    if len(set(selected_tasks)) != len(selected_tasks) or not set(
        selected_tasks
    ).issubset(TASK_ORDER):
        raise ValueError("Tasks contain duplicates or unknown values.")
    if len(set(selected_methods)) != len(selected_methods) or not set(
        selected_methods
    ).issubset(METHOD_ORDER):
        raise ValueError("Methods contain duplicates or unknown values.")
    if not selected_tasks or not selected_methods:
        raise ValueError("At least one task and method are required.")
    production_run = (
        selected_tasks == TASK_ORDER
        and selected_methods == METHOD_ORDER
        and Path(output_dir).resolve() == DEFAULT_OUTPUT_DIR.resolve()
        and cutoff_date == DEFAULT_CUTOFF_DATE
        and ordered_optimizer == DEFAULT_ORDERED_OPTIMIZER
    )
    if not production_run and Path(output_dir).resolve() == DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("Diagnostic subset or alternate optimizer requires a non-default output.")
    return selected_tasks, selected_methods, production_run


def _write_dataframe(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
        na_rep="",
    )


def _validate_staged_artifacts(
    staged_paths: Mapping[str, Path],
    *,
    production_run: bool,
) -> None:
    predictions = pd.read_csv(
        staged_paths["predictions_csv"], float_precision="round_trip"
    )
    metrics = pd.read_csv(staged_paths["metrics_csv"], float_precision="round_trip")
    manifest = pd.read_csv(staged_paths["feature_manifest_csv"])
    model_audit = pd.read_csv(staged_paths["model_audit_csv"])
    if predictions.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Staged predictions have an unexpected schema.")
    if metrics.columns.tolist() != METRIC_COLUMNS:
        raise ValueError("Staged metrics have an unexpected schema.")
    if manifest.columns.tolist() != FEATURE_MANIFEST_COLUMNS:
        raise ValueError("Staged feature manifest has an unexpected schema.")
    if model_audit.columns.tolist() != MODEL_AUDIT_COLUMNS:
        raise ValueError("Staged model audit has an unexpected schema.")
    if predictions.duplicated(["task", "method", "area_id", "date"]).any():
        raise ValueError("Staged predictions contain duplicate group keys.")
    if production_run:
        if len(predictions) != EXPECTED_TEST_ROWS * len(TASK_ORDER) * len(METHOD_ORDER):
            raise ValueError("Formal staged prediction count is incorrect.")
        if len(metrics) != len(TASK_ORDER) * (len(METHOD_ORDER) + 1):
            raise ValueError("Formal staged metric count is incorrect.")
        recomputed = build_metrics_table(predictions)
        numeric = [
            "overall_accuracy",
            "phase3plus_accuracy",
            "phase3plus_precision",
            "phase3plus_recall",
            "phase3above_r2",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "n_train",
            "n_test",
        ]
        pd.testing.assert_frame_equal(
            metrics[["task", "method", *numeric]].reset_index(drop=True),
            recomputed[["task", "method", *numeric]].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
        for key in ("figure_jpg", "figure_png", "figure_pdf"):
            if key not in staged_paths:
                raise ValueError(f"Formal staged artifacts are missing {key}.")


def _build_source_audit(
    *,
    staged_paths: Mapping[str, Path],
    bundle: PreparedInputs,
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    production_run: bool,
    ordered_optimizer: str,
    ordered_maxiter: int,
) -> pd.DataFrame:
    spatial_path = DEFAULT_OUTPUT_DIR / "spatial_feature_comparison_metrics.csv"
    same_environment_spatial_reference: dict[str, object] = {}
    if spatial_path.is_file():
        spatial_metrics = pd.read_csv(spatial_path)
        selected = spatial_metrics.loc[
            spatial_metrics["condition"].eq("baseline_with_lat_lon"),
            [
                "model",
                "overall_accuracy",
                "phase3plus_precision",
                "phase3plus_recall",
                "phase3plus_r2",
            ],
        ]
        same_environment_spatial_reference = {
            row.model: {
                "overall_accuracy": float(row.overall_accuracy),
                "phase3plus_precision": float(row.phase3plus_precision),
                "phase3plus_recall": float(row.phase3plus_recall),
                "phase3plus_r2": float(row.phase3plus_r2),
            }
            for row in selected.itertuples(index=False)
        }
    table1_alternative = {
        "Nowcasting": dict(frozen_main_result.TABLE1_NOWCASTING_ALTERNATIVE)
    }
    record: dict[str, object] = {
        "run_status": "complete",
        "production_run": production_run,
        "cutoff_date": DEFAULT_CUTOFF_DATE,
        "source_rows": len(bundle.forecasting),
        "train_rows": int(bundle.forecasting_train_mask.sum()),
        "test_rows": int(bundle.forecasting_test_mask.sum()),
        "test_areas": int(
            bundle.forecasting.loc[bundle.forecasting_test_mask, "area_id"].nunique()
        ),
        "test_countries": int(
            bundle.forecasting.loc[
                bundle.forecasting_test_mask, "country_code_3"
            ].nunique()
        ),
        "test_key_sha256": bundle.test_key_sha256,
        "source_label_disagreement_test_count": (
            bundle.source_label_disagreement_test_count
        ),
        "ordered_optimizer": ordered_optimizer,
        "ordered_maxiter": ordered_maxiter,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "statsmodels_version": statsmodels.__version__,
        "matplotlib_version": mpl.__version__,
        "forecasting_input_path": str(Path(forecasting_path)),
        "forecasting_input_sha256": file_sha256(forecasting_path),
        "nowcasting_input_path": str(Path(nowcasting_path)),
        "nowcasting_input_sha256": file_sha256(nowcasting_path),
        "country_lookup_path": str(Path(country_lookup_path)),
        "country_lookup_sha256": file_sha256(country_lookup_path),
        "generator_path": str(Path(__file__)),
        "generator_sha256": file_sha256(Path(__file__)),
        "main_result_reference_id": frozen_main_result.FREEZE_ID,
        "main_result_environment_id": frozen_main_result.ENVIRONMENT[
            "environment_id"
        ],
        "main_result_freeze_source_path": str(Path(frozen_main_result.__file__)),
        "main_result_freeze_source_sha256": file_sha256(
            Path(frozen_main_result.__file__)
        ),
        "selected_main_result_references_json": json.dumps(
            MAIN_RESULT_REFERENCES, sort_keys=True, separators=(",", ":")
        ),
        "phase3above_r2_definition": PHASE3ABOVE_R2_DEFINITION,
        "nonselected_table1_reference_json": json.dumps(
            table1_alternative, sort_keys=True, separators=(",", ":")
        ),
        "same_environment_spatial_baseline_json": json.dumps(
            same_environment_spatial_reference,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "consumer_contract": (
            "Accept only run_status=complete and verify every listed artifact hash."
        ),
    }
    lineage_files = {
        "persistence_metrics": DEFAULT_OUTPUT_DIR / "persistence_baseline_metrics.csv",
        "persistence_forecasting_predictions": DEFAULT_OUTPUT_DIR
        / "persistence_baseline_forecasting_predictions.csv",
        "persistence_nowcasting_predictions": DEFAULT_OUTPUT_DIR
        / "persistence_baseline_nowcasting_predictions.csv",
        "multinomial_metrics": DEFAULT_OUTPUT_DIR / "multinomial_baseline_metrics.csv",
        "multinomial_forecasting_predictions": DEFAULT_OUTPUT_DIR
        / "multinomial_baseline_forecasting_predictions.csv",
        "multinomial_nowcasting_predictions": DEFAULT_OUTPUT_DIR
        / "multinomial_baseline_nowcasting_predictions.csv",
    }
    for name, path in lineage_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required existing baseline lineage is missing: {path}")
        record[f"{name}_path"] = str(path)
        record[f"{name}_sha256"] = file_sha256(path)
    if spatial_path.is_file():
        record["same_environment_spatial_baseline_path"] = str(spatial_path)
        record["same_environment_spatial_baseline_sha256"] = file_sha256(spatial_path)
    for name, path in staged_paths.items():
        record[f"{name}_path"] = Path(path).name
        record[f"{name}_sha256"] = file_sha256(path)
    return pd.DataFrame([record])


def _replace_with_retry(source: Path, destination: Path) -> None:
    last_error: OSError | None = None
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            last_error = error
            if attempt == 4:
                break
            time.sleep(0.25)
    assert last_error is not None
    raise last_error


def write_artifacts(
    *,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    model_audit: pd.DataFrame,
    figure: plt.Figure | None,
    output_dir: Path,
    bundle: PreparedInputs,
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    production_run: bool,
    ordered_optimizer: str,
    ordered_maxiter: int,
) -> dict[str, Path]:
    """Stage, validate, hash, and publish each artifact with audit last."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".simple_baseline_staging_",
        dir=output_dir,
    ) as staging_name:
        staging = Path(staging_name)
        staged_paths = {
            "predictions_csv": staging / OUTPUT_FILENAMES["predictions_csv"],
            "metrics_csv": staging / OUTPUT_FILENAMES["metrics_csv"],
            "feature_manifest_csv": staging
            / OUTPUT_FILENAMES["feature_manifest_csv"],
            "model_audit_csv": staging / OUTPUT_FILENAMES["model_audit_csv"],
        }
        _write_dataframe(predictions, staged_paths["predictions_csv"])
        _write_dataframe(metrics, staged_paths["metrics_csv"])
        _write_dataframe(
            feature_manifest.loc[:, FEATURE_MANIFEST_COLUMNS],
            staged_paths["feature_manifest_csv"],
        )
        _write_dataframe(
            model_audit.loc[:, MODEL_AUDIT_COLUMNS],
            staged_paths["model_audit_csv"],
        )
        if figure is not None:
            staged_paths.update(save_simple_baseline_comparison_figure(figure, staging))
        _validate_staged_artifacts(staged_paths, production_run=production_run)

        source_audit = _build_source_audit(
            staged_paths=staged_paths,
            bundle=bundle,
            forecasting_path=forecasting_path,
            nowcasting_path=nowcasting_path,
            country_lookup_path=country_lookup_path,
            production_run=production_run,
            ordered_optimizer=ordered_optimizer,
            ordered_maxiter=ordered_maxiter,
        )
        audit_path = staging / OUTPUT_FILENAMES["source_audit_csv"]
        _write_dataframe(source_audit, audit_path)
        reloaded_audit = pd.read_csv(audit_path)
        if (
            len(reloaded_audit) != 1
            or reloaded_audit["run_status"].iloc[0] != "complete"
        ):
            raise ValueError("Staged source audit is not a complete one-row marker.")
        staged_paths["source_audit_csv"] = audit_path

        publication_order = [
            name for name in staged_paths if name != "source_audit_csv"
        ] + ["source_audit_csv"]
        final_paths = {
            name: output_dir / Path(staged_paths[name]).name
            for name in publication_order
        }
        backup_dir = staging / "previous_artifacts"
        backup_dir.mkdir()
        existing = {
            name: path for name, path in final_paths.items() if path.exists()
        }
        for name, path in existing.items():
            shutil.copy2(path, backup_dir / path.name)
        published: list[str] = []
        try:
            for name in publication_order:
                _replace_with_retry(staged_paths[name], final_paths[name])
                published.append(name)
        except BaseException:
            for name in reversed(published):
                final_paths[name].unlink(missing_ok=True)
            for name, original in existing.items():
                shutil.copy2(backup_dir / original.name, original)
            raise
        return final_paths


def _assert_existing_adapter_metrics(
    metrics: pd.DataFrame,
    *,
    method: str,
) -> None:
    filename = (
        "persistence_baseline_metrics.csv"
        if method == "Persistence"
        else "multinomial_baseline_metrics.csv"
    )
    path = DEFAULT_OUTPUT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Existing adapter metric artifact is missing: {path}")
    existing = pd.read_csv(path)
    existing_label = (
        "Persistence baseline"
        if method == "Persistence"
        else "Multinomial logistic baseline"
    )
    expected = existing.loc[existing["model"].eq(existing_label)].set_index("task")
    actual = metrics.loc[
        metrics["method"].eq(method) & metrics["method_role"].eq("baseline")
    ].set_index("task")
    for task in ("Forecasting", "Nowcasting"):
        for column in ("phase3plus_precision", "phase3plus_recall", "overall_accuracy"):
            if not np.isclose(
                float(actual.loc[task, column]),
                float(expected.loc[task, column]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{method} {task} no longer reproduces existing {column}."
                )


def run_analysis(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
    tasks: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    ordered_optimizer: str = DEFAULT_ORDERED_OPTIMIZER,
    ordered_maxiter: int = DEFAULT_ORDERED_MAXITER,
) -> dict[str, Path]:
    """Fit selected baselines and publish the approved formal artifact family."""
    selected_tasks, selected_methods, production_run = validate_run_configuration(
        tasks,
        methods,
        output_dir,
        cutoff_date,
        ordered_optimizer,
    )
    if production_run:
        frozen_main_result.assert_frozen_environment(
            ("matplotlib", "statsmodels", "patsy")
        )
    if ordered_maxiter < 1:
        raise ValueError("ordered_maxiter must be positive.")
    bundle = load_prepared_inputs(
        forecasting_path,
        nowcasting_path,
        country_lookup_path,
        cutoff_date,
        enforce_formal_counts=production_run,
    )

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    manifests = []
    audits = []
    if "Persistence" in selected_methods:
        print("Fitting Persistence baselines...", flush=True)
        group, manifest, audit = fit_persistence_adapter(bundle, country_lookup_path)
        for task in selected_tasks:
            frames[(task, "Persistence")] = group[task]
        manifests.append(manifest.loc[manifest["task"].isin(selected_tasks)])
        audits.append(audit.loc[audit["task"].isin(selected_tasks)])
    if "Multinomial" in selected_methods:
        print("Fitting Multinomial baselines...", flush=True)
        group, manifest, audit = fit_multinomial_adapter(bundle)
        for task in selected_tasks:
            frames[(task, "Multinomial")] = group[task]
        manifests.append(manifest.loc[manifest["task"].isin(selected_tasks)])
        audits.append(audit.loc[audit["task"].isin(selected_tasks)])
    if "Ordered Probit" in selected_methods:
        print("Fitting Ordered Probit baselines...", flush=True)
        group, manifest, audit = fit_ordered_probit_adapter(
            bundle,
            optimizer=ordered_optimizer,
            maxiter=ordered_maxiter,
        )
        for task in selected_tasks:
            frames[(task, "Ordered Probit")] = group[task]
        manifests.append(manifest.loc[manifest["task"].isin(selected_tasks)])
        audits.append(audit.loc[audit["task"].isin(selected_tasks)])
    if "Ensemble OLS" in selected_methods:
        print("Fitting Ensemble OLS baselines...", flush=True)
        group, manifest, audit = fit_ensemble_ols_adapter(bundle)
        for task in selected_tasks:
            frames[(task, "Ensemble OLS")] = group[task]
        manifests.append(manifest.loc[manifest["task"].isin(selected_tasks)])
        audits.append(audit.loc[audit["task"].isin(selected_tasks)])

    predictions = build_combined_predictions(
        frames,
        expected_tasks=selected_tasks,
        expected_methods=selected_methods,
    )
    if not production_run:
        raise ValueError(
            "Diagnostic subset fitting is supported for model checks, but publication "
            "requires the complete formal task-method grid."
        )
    metrics = build_metrics_table(predictions)
    _assert_existing_adapter_metrics(metrics, method="Persistence")
    _assert_existing_adapter_metrics(metrics, method="Multinomial")
    feature_manifest = pd.concat(manifests, ignore_index=True)[
        FEATURE_MANIFEST_COLUMNS
    ]
    model_audit = pd.concat(audits, ignore_index=True)[MODEL_AUDIT_COLUMNS]
    figure = create_simple_baseline_comparison_figure(metrics)
    try:
        return write_artifacts(
            predictions=predictions,
            metrics=metrics,
            feature_manifest=feature_manifest,
            model_audit=model_audit,
            figure=figure,
            output_dir=output_dir,
            bundle=bundle,
            forecasting_path=forecasting_path,
            nowcasting_path=nowcasting_path,
            country_lookup_path=country_lookup_path,
            production_run=production_run,
            ordered_optimizer=ordered_optimizer,
            ordered_maxiter=ordered_maxiter,
        )
    finally:
        plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-input", type=Path, default=DEFAULT_FORECASTING_INPUT)
    parser.add_argument("--nowcasting-input", type=Path, default=DEFAULT_NOWCASTING_INPUT)
    parser.add_argument("--country-lookup", type=Path, default=DEFAULT_COUNTRY_LOOKUP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff-date", default=DEFAULT_CUTOFF_DATE)
    parser.add_argument("--tasks", nargs="+", choices=TASK_ORDER)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER)
    parser.add_argument(
        "--ordered-optimizer",
        default=DEFAULT_ORDERED_OPTIMIZER,
        help="Formal contract uses bfgs; alternatives are diagnostic only.",
    )
    parser.add_argument("--ordered-maxiter", type=int, default=DEFAULT_ORDERED_MAXITER)
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    paths = run_analysis(
        forecasting_path=arguments.forecasting_input,
        nowcasting_path=arguments.nowcasting_input,
        country_lookup_path=arguments.country_lookup,
        output_dir=arguments.output_dir,
        cutoff_date=arguments.cutoff_date,
        tasks=arguments.tasks,
        methods=arguments.methods,
        ordered_optimizer=arguments.ordered_optimizer,
        ordered_maxiter=arguments.ordered_maxiter,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
