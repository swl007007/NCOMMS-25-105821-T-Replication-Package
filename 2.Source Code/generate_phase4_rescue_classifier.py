"""Generate the frozen Phase-4 rescue experiment and comparison artifacts.

This generator implements the contract frozen in
``docs/superpowers/specs/2026-08-14-phase4-rescue-classifier-design.md``.
Formal fitting and publication are intentionally restricted to the verified
Figure-1 Windows/XGBoost lineage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Iterable, Mapping, Sequence
import warnings

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache-phase4-rescue")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import patches as mpl_patches
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, log_loss, r2_score
import xgboost as xgb

from generate_leave_one_country_out_robustness import (
    CUMULATIVE_TARGETS,
    KEY_COLUMNS,
    NOWCAST_FEATURES,
    OUTCOME_COLUMNS,
    PHASE_SHARE_COLUMNS,
    add_cumulative_targets,
    load_country_lookup,
    load_hyperparameters,
    prepare_model_inputs,
    select_layer1_features,
)
import main_result_figure1_v1 as frozen_main


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE_DIR = REPO_ROOT / "2.Source Code"
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
PRODUCED_GRAPH_DIR = SOURCE_CODE_DIR / "produced_graph"
DEFAULT_OUTPUT_DIR = PRODUCED_GRAPH_DIR / "phase4_rescue_classifier"

FORECASTING_INPUT = SOURCE_DATA_DIR / "Forecasting_Analysis_010825.csv"
NOWCASTING_INPUT = SOURCE_DATA_DIR / "Nowcasting_Analysis_010825.csv"
COUNTRY_LOOKUP = SOURCE_DATA_DIR / "area_country_lookup.csv"
GENERAL_PARAMS = SOURCE_CODE_DIR / "forecasting_hyperparameters.json"
PHASE3_PARAMS = SOURCE_CODE_DIR / "forecasting_hyperparameters_p3.json"
FREEZE_SOURCE = SOURCE_CODE_DIR / "main_result_figure1_v1.py"
SPATIAL_PREDICTIONS = PRODUCED_GRAPH_DIR / "spatial_feature_comparison_predictions.csv"
SPATIAL_METRICS = PRODUCED_GRAPH_DIR / "spatial_feature_comparison_metrics.csv"
SPATIAL_SOURCE_AUDIT = PRODUCED_GRAPH_DIR / "spatial_feature_comparison_source_audit.csv"

FREEZE_ID = "main-result-figure1-v1"
REFERENCE_ENVIRONMENT_ID = "windows_py3113_xgb203_defaultthreads_no_explicit_seed"
EVALUATION_POPULATION_ID = "temporal_test_2022_reconstructed_phase_1170"
CUTOFF = pd.Timestamp("2022-01-01")
EXPECTED_SOURCE_ROWS = 5575
EXPECTED_PRE2022_ROWS = 4405
EXPECTED_BENCHMARK_ROWS = 1170
EXPECTED_BENCHMARK_AREAS = 646
EXPECTED_BENCHMARK_COUNTRIES = 27
EXPECTED_OOF_ROWS_PER_TASK = 3760
EXPECTED_OOF_ROWS = 7520
EXPECTED_DIRECT_ROWS = 2518
EXPECTED_DIRECT_PHASE3 = 2149
EXPECTED_DIRECT_PHASE4 = 369
DIRECT_THRESHOLD = 0.5
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_RANDOM_STATE = 0
ALL_PHASES = (1, 2, 3, 4, 5)
TASK_ORDER = ("Forecasting", "Nowcasting")
METHOD_ORDER = (
    "frozen_base",
    "single_score",
    "direct_phase34_xgboost",
    "xgboost",
)
MAIN_METHOD_ORDER = (
    "frozen_base",
    "direct_phase34_xgboost",
    "xgboost",
)

EXPECTED_SOURCE_HASHES: Mapping[Path, str] = {
    FREEZE_SOURCE: "a33b9b2a8ef72569efe017049fdd831ada9411e44d3e59cf4d840a7a7fc01fe2",
    SPATIAL_PREDICTIONS: "bdef15d0b1facf892dcbfdb019e119463cb93ee79bd384915ff6c45fa9f1a7a7",
    SPATIAL_METRICS: "e414b5d53c6634a69dbb6471b1c4a32cee7c77516b164cabb6078bf8fde06f2f",
    SPATIAL_SOURCE_AUDIT: "54bc43f15dade7415b456ced87b2f88fc68f662250711e559a482386090247b6",
    FORECASTING_INPUT: "60f17f079ab569060a29fd8c7af0d3f6feaab558f7a29a104c9a0e702172dc0e",
    NOWCASTING_INPUT: "d541f7c0c21c1878fcf8a74c99c953ddda200d0dfeda7365e7ff5934af8f550f",
    COUNTRY_LOOKUP: "01b58c577b53cbadd1258766244c071d8ef44759da92015566c5507d894f85a7",
    GENERAL_PARAMS: "3742300661466f22eba8198e5d4f9c2a277615ba59562fbf27f251ecc932dd76",
    PHASE3_PARAMS: "cdc0e55aa15bdda932465088568b9ee717d226208ffb32076f438fb274e6317b",
}

EXPECTED_ARTIFACTS = (
    "phase4_rescue_oof_predictions.csv",
    "phase4_rescue_model_selection.csv",
    "phase4_rescue_threshold_search.csv",
    "phase4_rescue_benchmark_predictions.csv",
    "phase4_rescue_metrics.csv",
    "phase4_rescue_feature_manifest.csv",
    "phase4_rescue_model_audit.csv",
    "phase4_rescue_bootstrap_draws.csv",
    "phase4_rescue_bootstrap_summary.csv",
    "phase4_rescue_confusion_matrices.csv",
    "phase4_rescue_main_comparison_metrics.csv",
    "phase4_rescue_source_audit.csv",
    "phase4_rescue_forecasting_model.json",
    "phase4_rescue_nowcasting_model.json",
    "phase4_rescue_direct_forecasting_model.json",
    "phase4_rescue_direct_nowcasting_layer1_model.json",
    "phase4_rescue_direct_nowcasting_layer2_model.json",
    "phase4_rescue_configuration.json",
    "phase4_rescue_main_comparison.pdf",
    "phase4_rescue_main_comparison.png",
)

FORECAST_RESCUE_FEATURES = (
    "phase2_pred_raw",
    "phase3_pred_raw",
    "phase4_pred_raw",
    "phase5_pred_raw",
    "phase3_margin_020",
    "phase4_margin_020",
    "phase3_minus_phase4",
)
NOWCAST_RESCUE_FEATURES = (
    "phase2_pred_raw",
    "phase3_pred_raw",
    "phase4_pred_raw",
    "phase5_pred_raw",
    "phase2_layer1_pred",
    "phase3_layer1_pred",
    "phase4_layer1_pred",
    "phase5_layer1_pred",
    "phase3_residual_pred",
    "phase4_residual_pred",
    "phase3_margin_020",
    "phase4_margin_020",
    "phase3_minus_phase4",
    "layer1_phase3_minus_phase4",
    "residual_phase3_minus_phase4",
)

OOF_COLUMNS = (
    "task", "base_oof_fold", "area_id", "date", "country_code_3",
    "source_overall_phase", "reconstructed_overall_phase", "auxiliary_target",
    "base_overall_phase_pred", "in_auxiliary_gate", "phase2_pred_raw",
    "phase3_pred_raw", "phase4_pred_raw", "phase5_pred_raw",
    "phase2_pred_rounded", "phase3_pred_rounded", "phase4_pred_rounded",
    "phase5_pred_rounded", "phase2_layer1_pred", "phase3_layer1_pred",
    "phase4_layer1_pred", "phase5_layer1_pred", "phase3_residual_pred",
    "phase4_residual_pred", "phase3_margin_020", "phase4_margin_020",
    "phase3_minus_phase4", "layer1_phase3_minus_phase4",
    "residual_phase3_minus_phase4",
)
MODEL_SELECTION_COLUMNS = (
    "task", "candidate_id", "scope", "meta_fold", "max_depth",
    "min_child_weight", "learning_rate", "reg_lambda", "n_train",
    "n_validation", "validation_positive_count", "pr_auc", "log_loss",
    "selected",
)
THRESHOLD_COLUMNS = (
    "task", "method", "threshold_rank", "threshold", "threshold_source",
    "evaluation_status", "no_rescue_sentinel", "gate_rows",
    "gate_positive_count", "gate_prevalence", "predicted_positive_count",
    "true_positive", "false_positive", "false_positive_actual_phase3",
    "phase4_precision", "phase4_recall", "accuracy", "balanced_accuracy",
    "ordinal_mae", "precision_gate_passed", "accuracy_gate_passed",
    "balanced_accuracy_gate_passed", "ordinal_mae_gate_passed",
    "recall_gate_passed", "eligible", "selected",
)
BENCHMARK_COLUMNS = (
    "task", "area_id", "date", "country_code_3", "source_overall_phase",
    "reconstructed_overall_phase", "phase2_test", "phase3_test", "phase4_test",
    "phase5_test", "phase2_pred_raw", "phase3_pred_raw", "phase4_pred_raw",
    "phase5_pred_raw", "phase2_pred_rounded", "phase3_pred_rounded",
    "phase4_pred_rounded", "phase5_pred_rounded", "phase2_layer1_pred",
    "phase3_layer1_pred", "phase4_layer1_pred", "phase5_layer1_pred",
    "phase3_residual_pred", "phase4_residual_pred", "phase3_margin_020",
    "phase4_margin_020", "phase3_minus_phase4",
    "layer1_phase3_minus_phase4", "residual_phase3_minus_phase4",
    "base_overall_phase_pred", "single_score_threshold", "single_score_triggered",
    "single_score_overall_phase_pred", "direct_layer1_phase4_score",
    "direct_layer2_residual_score", "direct_phase4_score", "direct_threshold",
    "direct_triggered", "direct_overall_phase_pred", "auxiliary_phase4_score",
    "xgboost_threshold", "xgboost_triggered", "rescued_overall_phase_pred",
)
METRIC_COLUMNS = (
    "split", "task", "method", "evaluation_status", "n_rows", "n_gate",
    "actual_phase4_count", "predicted_phase4_count", "phase4_true_positive",
    "phase4_false_positive", "phase4_false_negative", "phase4_precision",
    "phase4_recall", "phase4_f1", "pr_auc", "accuracy", "balanced_accuracy",
    "macro_f1", "ordinal_mae", "phase3_precision", "phase3_recall",
    "phase3_f1", "phase3plus_precision", "phase3plus_recall",
    "phase3plus_r2", "changed_3_to_4", "correct_rescues", "false_promotions",
    "country_macro_accuracy", "country_macro_balanced_accuracy",
    "country_macro_macro_f1", "country_macro_ordinal_mae", "accepted",
)
FEATURE_MANIFEST_COLUMNS = (
    "task", "method", "model_component", "feature_order", "feature_name",
    "source_columns", "formula", "dtype", "allow_missing",
)
MODEL_AUDIT_COLUMNS = (
    "task", "method", "model_component", "estimator_class", "objective",
    "target_definition", "run_status", "selected_candidate_id",
    "selected_threshold", "no_rescue_selected", "training_rows",
    "training_positive_count", "scale_pos_weight", "feature_count",
    "feature_order_sha256", "training_key_sha256", "target_sha256",
    "training_matrix_sha256", "training_missingness_sha256",
    "training_matrix_with_target_sha256", "benchmark_rows",
    "benchmark_key_sha256", "benchmark_matrix_sha256",
    "benchmark_missingness_sha256", "model_path", "model_sha256",
)
BOOTSTRAP_DRAW_COLUMNS = (
    "task", "method", "bootstrap_id", "sampled_country_count", "accuracy_delta",
    "balanced_accuracy_delta", "macro_f1_delta", "ordinal_mae_delta",
    "phase4_precision_delta", "phase4_recall_delta",
)
BOOTSTRAP_SUMMARY_COLUMNS = (
    "task", "method", "repetitions", "ci_level", "interval_method",
    "accuracy_delta_lower", "accuracy_delta_upper", "balanced_accuracy_delta_lower",
    "balanced_accuracy_delta_upper", "macro_f1_delta_lower", "macro_f1_delta_upper",
    "ordinal_mae_delta_lower", "ordinal_mae_delta_upper",
    "phase4_precision_delta_lower", "phase4_precision_delta_upper",
    "phase4_recall_delta_lower", "phase4_recall_delta_upper",
)
CONFUSION_COLUMNS = (
    "task", "method", "actual_phase", "predicted_phase", "count",
    "actual_row_total", "actual_row_share",
)
MAIN_METRIC_COLUMNS = (
    "task", "method", "display_status", "n_rows", "phase3plus_precision",
    "phase3plus_recall", "phase3plus_r2", "accuracy",
    "phase3plus_precision_delta_from_base", "phase3plus_recall_delta_from_base",
    "phase3plus_r2_delta_from_base", "accuracy_delta_from_base",
)
SOURCE_AUDIT_COLUMNS = (
    "run_status", "freeze_id", "reference_environment_id", "evaluation_population_id",
    "source_rows", "pre2022_rows", "oof_rows", "benchmark_rows",
    "benchmark_areas", "benchmark_countries", "benchmark_key_sha256",
    "direct_training_rows", "direct_phase3_count", "direct_phase4_count",
    "direct_training_key_sha256", "direct_target_sha256",
    "direct_forecasting_training_matrix_sha256",
    "direct_forecasting_training_missingness_sha256",
    "direct_nowcasting_layer1_training_matrix_sha256",
    "direct_nowcasting_layer1_training_missingness_sha256",
    "direct_nowcasting_layer2_training_matrix_sha256",
    "direct_nowcasting_layer2_training_missingness_sha256", "direct_threshold",
    "freeze_source_path", "freeze_source_sha256", "spatial_predictions_path",
    "spatial_predictions_sha256", "spatial_metrics_path", "spatial_metrics_sha256",
    "spatial_source_audit_path", "spatial_source_audit_sha256",
    "forecasting_input_path", "forecasting_input_sha256", "nowcasting_input_path",
    "nowcasting_input_sha256", "country_lookup_path", "country_lookup_sha256",
    "general_params_path", "general_params_sha256", "phase3_params_path",
    "phase3_params_sha256", "generator_path", "generator_sha256",
    "platform_family", "python_version", "numpy_version", "pandas_version",
    "scipy_version", "sklearn_version", "xgboost_version", "matplotlib_version",
    "xgboost_dll_sha256", "base_random_state_override", "base_n_jobs_override",
    "auxiliary_random_state", "auxiliary_n_jobs", "direct_random_state",
    "direct_n_jobs", "outer_workers", "bootstrap_repetitions",
    "bootstrap_random_state", "figure_backend", "figure_width_inches",
    "figure_height_inches", "figure_png_dpi", "protected_manifest_sha256_before",
    "protected_manifest_sha256_after", "protected_manifest_match",
    "artifact_manifest_json", "artifact_manifest_sha256",
)


@dataclass(frozen=True)
class TemporalFold:
    fold_id: str
    training_years: tuple[int, ...]
    validation_year: int


BASE_FOLDS = (
    TemporalFold("B1", (2017,), 2018),
    TemporalFold("B2", (2017, 2018), 2019),
    TemporalFold("B3", (2017, 2018, 2019), 2020),
    TemporalFold("B4", (2017, 2018, 2019, 2020), 2021),
)
META_FOLDS = (
    TemporalFold("A1", (2018,), 2019),
    TemporalFold("A2", (2018, 2019), 2020),
    TemporalFold("A3", (2018, 2019, 2020), 2021),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_sha256(value: object) -> str:
    return bytes_sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    )


def relative_path(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def canonical_key_frame(data: pd.DataFrame) -> pd.DataFrame:
    required = ["area_id", "date"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Key frame is missing columns: {missing}")
    frame = data[required].copy()
    frame["area_id"] = frame["area_id"].astype(int)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if frame.isna().any().any() or frame.duplicated(required).any():
        raise ValueError("Canonical keys must be complete and unique.")
    return frame.sort_values(required, kind="mergesort").reset_index(drop=True)


def canonical_key_sha256(data: pd.DataFrame) -> str:
    payload = canonical_key_frame(data).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    return bytes_sha256(payload)


def canonical_series_sha256(series: pd.Series, name: str = "value") -> str:
    frame = pd.DataFrame({name: series.reset_index(drop=True)})
    payload = frame.to_csv(
        index=False, float_format="%.17g", na_rep="<NA>", lineterminator="\n"
    ).encode("utf-8")
    return bytes_sha256(payload)


def canonical_matrix_sha256(matrix: pd.DataFrame) -> str:
    payload = matrix.reset_index(drop=True).to_csv(
        index=False, float_format="%.17g", na_rep="<NA>", lineterminator="\n"
    ).encode("utf-8")
    return bytes_sha256(payload)


def canonical_missingness_sha256(matrix: pd.DataFrame) -> str:
    missingness = matrix.isna().astype(np.uint8)
    return canonical_matrix_sha256(missingness)


def matrix_with_target_sha256(matrix: pd.DataFrame, target: pd.Series) -> str:
    combined = matrix.reset_index(drop=True).copy()
    combined["__target__"] = target.reset_index(drop=True)
    return canonical_matrix_sha256(combined)


def manifest_sha256(hashes: Mapping[str, str]) -> str:
    payload = "".join(f"{name}\t{hashes[name]}\n" for name in sorted(hashes))
    return bytes_sha256(payload.encode("utf-8"))


def verify_frozen_source_hashes() -> None:
    failures: list[str] = []
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        if not path.is_file():
            failures.append(f"missing {relative_path(path)}")
            continue
        actual = file_sha256(path)
        if actual != expected:
            failures.append(f"{relative_path(path)}={actual} expected={expected}")
    if failures:
        raise RuntimeError("Frozen source hash gate failed: " + "; ".join(failures))


def validate_generation_target(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Formal output directory must be absent or empty: {output_dir}")


def protected_artifact_manifest_sha256(excluded: Path | None = None) -> str:
    hashes: dict[str, str] = {}
    excluded_roots = {DEFAULT_OUTPUT_DIR.resolve()}
    if excluded is not None:
        excluded_roots.add(Path(excluded).resolve())
    if not PRODUCED_GRAPH_DIR.exists():
        return manifest_sha256(hashes)
    for path in sorted(PRODUCED_GRAPH_DIR.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file():
            continue
        is_excluded = False
        for excluded_root in excluded_roots:
            try:
                path.resolve().relative_to(excluded_root)
                is_excluded = True
                break
            except ValueError:
                pass
        if is_excluded:
            continue
        hashes[path.relative_to(PRODUCED_GRAPH_DIR).as_posix()] = file_sha256(path)
    return manifest_sha256(hashes)


def xgboost_dll_sha256() -> str:
    library_path = Path(xgb.core._LIB._name)
    if not library_path.is_file():
        raise RuntimeError(f"Loaded XGBoost library does not exist: {library_path}")
    return file_sha256(library_path)


def assert_formal_environment() -> dict[str, object]:
    record = frozen_main.assert_frozen_environment(required_extensions=("matplotlib",))
    actual_dll = xgboost_dll_sha256()
    expected_dll = str(frozen_main.ENVIRONMENT["xgboost_dll_sha256"])
    if actual_dll != expected_dll:
        raise RuntimeError(f"XGBoost DLL hash mismatch: {actual_dll} expected={expected_dll}")
    if matplotlib.get_backend().lower() != "agg":
        raise RuntimeError(f"Formal figure backend must be Agg, got {matplotlib.get_backend()}")
    return {**record, "xgboost_dll_sha256": actual_dll}


def _require_columns(data: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_unique_keys(data: pd.DataFrame, name: str) -> None:
    _require_columns(data, KEY_COLUMNS, name)
    if data[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} contains missing keys.")
    duplicates = data.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise ValueError(f"{name} contains duplicate keys: {data.loc[duplicates, list(KEY_COLUMNS)].head().to_dict('records')}")


def reconstruct_overall_phase(data: pd.DataFrame) -> pd.Series:
    cumulative = add_cumulative_targets(data)
    conditions = [
        cumulative["phase5_worse"].ge(0.20),
        cumulative["phase4_worse"].ge(0.20),
        cumulative["phase3_worse"].ge(0.20),
        cumulative["phase2_worse"].ge(0.20),
    ]
    values = np.select(conditions, [5, 4, 3, 2], default=1).astype(int)
    return pd.Series(values, index=data.index, name="reconstructed_overall_phase")


def phase_from_rounded_predictions(data: pd.DataFrame) -> np.ndarray:
    conditions = [
        data["phase5_pred_rounded"].ge(0.20),
        data["phase4_pred_rounded"].ge(0.20),
        data["phase3_pred_rounded"].ge(0.20),
        data["phase2_pred_rounded"].ge(0.20),
    ]
    return np.select(conditions, [5, 4, 3, 2], default=1).astype(int)


def build_auxiliary_target(reconstructed_phase: pd.Series) -> pd.Series:
    return reconstructed_phase.eq(4).astype(np.uint8).rename("auxiliary_target")


def apply_phase4_rescue(
    base_phase: Sequence[int] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
    threshold: float,
) -> np.ndarray:
    base = np.asarray(base_phase, dtype=int)
    values = np.asarray(score, dtype=float)
    if base.shape != values.shape:
        raise ValueError("base_phase and score must have identical shapes.")
    result = base.copy()
    trigger = (base == 3) & np.isfinite(values) & (values >= float(threshold))
    result[trigger] = 4
    assert_postclassification_invariants(base, result)
    return result


def combine_direct_nowcasting_scores(
    layer1_score: Sequence[float] | np.ndarray,
    residual_score: Sequence[float] | np.ndarray,
) -> np.ndarray:
    layer1 = np.asarray(layer1_score, dtype=float)
    residual = np.asarray(residual_score, dtype=float)
    if layer1.shape != residual.shape:
        raise ValueError("Direct Nowcasting score components must have identical shapes.")
    return layer1 + residual


def assert_postclassification_invariants(base: Sequence[int], rescued: Sequence[int]) -> None:
    base_array = np.asarray(base, dtype=int)
    rescued_array = np.asarray(rescued, dtype=int)
    if base_array.shape != rescued_array.shape:
        raise ValueError("Base and rescued phase arrays must have identical shapes.")
    changed = base_array != rescued_array
    if np.any(changed & ~((base_array == 3) & (rescued_array == 4))):
        raise ValueError("The only permitted post-classification change is Phase 3 to Phase 4.")
    if not np.array_equal(base_array >= 3, rescued_array >= 3):
        raise ValueError("Phase 3+ binary predictions changed after rescue.")


def load_prepared_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    forecasting_raw = pd.read_csv(FORECASTING_INPUT, parse_dates=["date"])
    nowcasting_raw = pd.read_csv(NOWCASTING_INPUT, parse_dates=["date"])
    if len(forecasting_raw) != EXPECTED_SOURCE_ROWS or len(nowcasting_raw) != EXPECTED_SOURCE_ROWS:
        raise ValueError("Unexpected source row count.")
    forecasting_raw = forecasting_raw.loc[forecasting_raw["phase1_percent"].notna()].copy()
    nowcasting_raw = nowcasting_raw.loc[nowcasting_raw["phase1_percent"].notna()].copy()
    if len(forecasting_raw) != EXPECTED_SOURCE_ROWS or len(nowcasting_raw) != EXPECTED_SOURCE_ROWS:
        raise ValueError("The established phase1_percent population filter changed source rows.")
    lookup = load_country_lookup(COUNTRY_LOOKUP)
    forecasting, nowcasting = prepare_model_inputs(forecasting_raw, nowcasting_raw, lookup)
    forecasting = add_cumulative_targets(forecasting)
    nowcasting = add_cumulative_targets(nowcasting)
    layer1_features = select_layer1_features(forecasting)
    if len(layer1_features) != 106:
        raise ValueError(f"Expected 106 Layer-1 features, found {len(layer1_features)}.")
    forecasting["reconstructed_overall_phase"] = reconstruct_overall_phase(forecasting)
    nowcasting["reconstructed_overall_phase"] = reconstruct_overall_phase(nowcasting)
    for name, frame in (("Forecasting", forecasting), ("Nowcasting", nowcasting)):
        _require_unique_keys(frame, name)
        if len(frame.loc[frame["date"] < CUTOFF]) != EXPECTED_PRE2022_ROWS:
            raise ValueError(f"{name} pre-2022 row count differs from {EXPECTED_PRE2022_ROWS}.")
        if len(frame.loc[frame["date"] >= CUTOFF]) != EXPECTED_BENCHMARK_ROWS:
            raise ValueError(f"{name} benchmark row count differs from {EXPECTED_BENCHMARK_ROWS}.")
    left = forecasting.set_index(list(KEY_COLUMNS)).sort_index()
    right = nowcasting.set_index(list(KEY_COLUMNS)).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Prepared task key sets differ.")
    for column in [*OUTCOME_COLUMNS, "reconstructed_overall_phase"]:
        equal = left[column].eq(right[column]) | (left[column].isna() & right[column].isna())
        if not bool(equal.all()):
            raise ValueError(f"Prepared task outcomes differ for {column}.")
    if tuple(NOWCAST_FEATURES) != tuple(nowcasting.loc[:, list(NOWCAST_FEATURES)].columns):
        raise ValueError("Nowcasting Layer-2 feature order drifted.")
    return forecasting, nowcasting, layer1_features


def build_direct_phase34_population(
    forecasting: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    population = forecasting.loc[
        (forecasting["date"] < CUTOFF)
        & forecasting["reconstructed_overall_phase"].isin([3, 4])
    ].copy()
    population = population.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    _require_unique_keys(population, "Direct Phase-3/4 population")
    counts = population["reconstructed_overall_phase"].value_counts().to_dict()
    if len(population) != EXPECTED_DIRECT_ROWS or counts.get(3, 0) != EXPECTED_DIRECT_PHASE3 or counts.get(4, 0) != EXPECTED_DIRECT_PHASE4:
        raise ValueError(f"Direct population contract failed: rows={len(population)} counts={counts}")
    target = build_auxiliary_target(population["reconstructed_overall_phase"])
    return population, target


def get_temporal_folds() -> tuple[TemporalFold, ...]:
    return BASE_FOLDS


def validate_temporal_folds(data: pd.DataFrame, folds: Sequence[TemporalFold]) -> None:
    years = pd.to_datetime(data["date"]).dt.year
    validation_rows = 0
    seen_validation_years: set[int] = set()
    for fold in folds:
        if fold.validation_year in fold.training_years:
            raise ValueError(f"Fold {fold.fold_id} trains on its validation year.")
        if max(fold.training_years) >= fold.validation_year:
            raise ValueError(f"Fold {fold.fold_id} is not strictly forward.")
        train_mask = years.isin(fold.training_years)
        validation_mask = years.eq(fold.validation_year)
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"Fold {fold.fold_id} lacks train or validation rows.")
        if data.loc[train_mask, "date"].max() >= data.loc[validation_mask, "date"].min():
            raise ValueError(f"Fold {fold.fold_id} violates temporal ordering.")
        if fold.validation_year in seen_validation_years:
            raise ValueError("Validation years are duplicated across folds.")
        seen_validation_years.add(fold.validation_year)
        validation_rows += int(validation_mask.sum())
    if validation_rows != EXPECTED_OOF_ROWS_PER_TASK:
        raise ValueError(f"Temporal folds cover {validation_rows} OOF rows, expected {EXPECTED_OOF_ROWS_PER_TASK}.")


def _xgb_regressor(parameters: Mapping[str, object]) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**dict(parameters))


def _format_fold_predictions(frame: pd.DataFrame, task: str, fold_id: str) -> pd.DataFrame:
    result = frame.copy()
    result["task"] = task
    result["base_oof_fold"] = fold_id
    result["auxiliary_target"] = build_auxiliary_target(result["reconstructed_overall_phase"])
    result["base_overall_phase_pred"] = phase_from_rounded_predictions(result)
    result["in_auxiliary_gate"] = result["base_overall_phase_pred"].eq(3)
    result["phase3_margin_020"] = result["phase3_pred_raw"] - 0.20
    result["phase4_margin_020"] = result["phase4_pred_raw"] - 0.20
    result["phase3_minus_phase4"] = result["phase3_pred_raw"] - result["phase4_pred_raw"]
    if task == "Nowcasting":
        result["layer1_phase3_minus_phase4"] = result["phase3_layer1_pred"] - result["phase4_layer1_pred"]
        result["residual_phase3_minus_phase4"] = result["phase3_residual_pred"] - result["phase4_residual_pred"]
    else:
        for column in (
            "phase2_layer1_pred", "phase3_layer1_pred", "phase4_layer1_pred",
            "phase5_layer1_pred", "phase3_residual_pred", "phase4_residual_pred",
            "layer1_phase3_minus_phase4", "residual_phase3_minus_phase4",
        ):
            result[column] = np.nan
    return result.loc[:, list(OOF_COLUMNS)]


def _fit_forecasting_fold(
    data: pd.DataFrame,
    fold: TemporalFold,
    layer1_features: Sequence[str],
    general_params: Mapping[str, object],
    phase3_params: Mapping[str, object],
) -> pd.DataFrame:
    years = data["date"].dt.year
    fold_data = data.loc[years.isin((*fold.training_years, fold.validation_year))].copy()
    train = fold_data["date"].dt.year.isin(fold.training_years)
    validation = fold_data["date"].dt.year.eq(fold.validation_year)
    rows = fold_data.loc[validation, [*KEY_COLUMNS, "country_code_3", "overall_phase", "reconstructed_overall_phase"]].copy()
    rows = rows.rename(columns={"overall_phase": "source_overall_phase"})
    for phase, target_column in CUMULATIVE_TARGETS.items():
        parameters = general_params if phase == 2 else phase3_params
        model = _xgb_regressor(parameters)
        model.fit(fold_data.loc[train, list(layer1_features)], fold_data.loc[train, target_column])
        raw = np.asarray(model.predict(fold_data.loc[validation, list(layer1_features)]))
        rows[f"phase{phase}_pred_raw"] = raw
        rows[f"phase{phase}_pred_rounded"] = pd.Series(raw).round(2).to_numpy()
    return _format_fold_predictions(rows, "Forecasting", fold.fold_id)


def _fit_nowcasting_fold(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    fold: TemporalFold,
    layer1_features: Sequence[str],
    general_params: Mapping[str, object],
    phase3_params: Mapping[str, object],
) -> pd.DataFrame:
    forecast_years = forecasting["date"].dt.year
    nowcast_years = nowcasting["date"].dt.year
    selected_years = (*fold.training_years, fold.validation_year)
    forecast_fold = forecasting.loc[forecast_years.isin(selected_years)].copy()
    nowcast_fold = nowcasting.loc[nowcast_years.isin(selected_years)].copy()
    train = forecast_fold["date"].dt.year.isin(fold.training_years)
    validation = forecast_fold["date"].dt.year.eq(fold.validation_year)
    now_train = nowcast_fold["date"].dt.year.isin(fold.training_years)
    now_validation = nowcast_fold["date"].dt.year.eq(fold.validation_year)
    train_keys = canonical_key_frame(forecast_fold.loc[train])
    now_train_keys = canonical_key_frame(nowcast_fold.loc[now_train])
    validation_keys = canonical_key_frame(forecast_fold.loc[validation])
    now_validation_keys = canonical_key_frame(nowcast_fold.loc[now_validation])
    if not train_keys.equals(now_train_keys) or not validation_keys.equals(now_validation_keys):
        raise ValueError(f"Task keys differ inside {fold.fold_id}.")
    keys = [*KEY_COLUMNS, "country_code_3"]
    rows = forecast_fold.loc[validation, [*keys, "overall_phase", "reconstructed_overall_phase"]].copy()
    rows = rows.rename(columns={"overall_phase": "source_overall_phase"})
    for phase, target_column in CUMULATIVE_TARGETS.items():
        parameters = general_params if phase == 2 else phase3_params
        layer1 = _xgb_regressor(parameters)
        layer1.fit(forecast_fold.loc[train, list(layer1_features)], forecast_fold.loc[train, target_column])
        layer1_train_score = np.asarray(layer1.predict(forecast_fold.loc[train, list(layer1_features)]))
        layer1_validation_score = np.asarray(layer1.predict(forecast_fold.loc[validation, list(layer1_features)]))
        residual_frame = forecast_fold.loc[train, keys].copy()
        residual_frame["residual_target"] = forecast_fold.loc[train, target_column].to_numpy() - layer1_train_score
        keyed_train = nowcast_fold.loc[now_train, [*keys, *NOWCAST_FEATURES]].merge(
            residual_frame, on=keys, how="inner", validate="one_to_one"
        )
        if len(keyed_train) != int(train.sum()):
            raise ValueError(f"Nowcasting {fold.fold_id} lost residual training rows.")
        layer2 = _xgb_regressor(parameters)
        layer2.fit(keyed_train[list(NOWCAST_FEATURES)], keyed_train["residual_target"])
        keyed_validation = nowcast_fold.loc[now_validation, [*keys, *NOWCAST_FEATURES]].copy()
        residual_score = np.asarray(layer2.predict(keyed_validation[list(NOWCAST_FEATURES)]))
        score_frame = forecast_fold.loc[validation, keys].copy()
        score_frame[f"phase{phase}_layer1_pred"] = layer1_validation_score
        residual_score_frame = keyed_validation[keys].copy()
        residual_score_frame[f"phase{phase}_residual_pred"] = residual_score
        score_frame = score_frame.merge(residual_score_frame, on=keys, how="inner", validate="one_to_one")
        score_frame[f"phase{phase}_pred_raw"] = score_frame[f"phase{phase}_layer1_pred"] + score_frame[f"phase{phase}_residual_pred"]
        score_frame[f"phase{phase}_pred_rounded"] = score_frame[f"phase{phase}_pred_raw"].round(2)
        rows = rows.merge(score_frame, on=keys, how="inner", validate="one_to_one")
    return _format_fold_predictions(rows, "Nowcasting", fold.fold_id)


def generate_base_oof_predictions(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    layer1_features: Sequence[str],
) -> pd.DataFrame:
    validate_temporal_folds(forecasting.loc[forecasting["date"] < CUTOFF], BASE_FOLDS)
    general_params, phase3_params = load_hyperparameters(
        GENERAL_PARAMS, PHASE3_PARAMS, random_state=None, estimator_n_jobs=None
    )
    frames: list[pd.DataFrame] = []
    for fold in BASE_FOLDS:
        frames.append(_fit_forecasting_fold(forecasting, fold, layer1_features, general_params, phase3_params))
        frames.append(_fit_nowcasting_fold(forecasting, nowcasting, fold, layer1_features, general_params, phase3_params))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        oof = pd.concat(frames, ignore_index=True)
    oof["date"] = pd.to_datetime(oof["date"], errors="raise")
    if len(oof) != EXPECTED_OOF_ROWS:
        raise ValueError(f"OOF artifact has {len(oof)} rows, expected {EXPECTED_OOF_ROWS}.")
    if oof.duplicated(["task", "base_oof_fold", *KEY_COLUMNS]).any():
        raise ValueError("OOF artifact contains duplicate task/fold keys.")
    for task in TASK_ORDER:
        task_rows = oof.loc[oof["task"].eq(task)]
        if len(task_rows) != EXPECTED_OOF_ROWS_PER_TASK:
            raise ValueError(f"{task} OOF rows differ from {EXPECTED_OOF_ROWS_PER_TASK}.")
        if task_rows["date"].ge(CUTOFF).any():
            raise ValueError("2022 rows entered OOF data.")
        feature_columns = FORECAST_RESCUE_FEATURES if task == "Forecasting" else NOWCAST_RESCUE_FEATURES
        matrix = task_rows.loc[task_rows["in_auxiliary_gate"], list(feature_columns)]
        if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError(f"{task} gated OOF rescue features are missing or nonfinite.")
        by_year = task_rows.assign(year=task_rows["date"].dt.year).groupby("year", observed=True)
        for year in (2019, 2020, 2021):
            gated = by_year.get_group(year).loc[lambda frame: frame["in_auxiliary_gate"]]
            if gated["auxiliary_target"].nunique() != 2:
                raise ValueError(f"{task} auxiliary validation year {year} lacks class support.")
    return oof.sort_values(["task", "base_oof_fold", *KEY_COLUMNS], kind="mergesort").reset_index(drop=True)


def get_feature_contract(task: str, method: str, model_component: str) -> tuple[str, ...]:
    if method == "xgboost" and model_component == "rescue_classifier":
        if task == "Forecasting":
            return FORECAST_RESCUE_FEATURES
        if task == "Nowcasting":
            return NOWCAST_RESCUE_FEATURES
    if method == "direct_phase34_xgboost" and model_component in {
        "direct_forecasting_classifier",
        "direct_nowcasting_layer1_classifier",
    }:
        raise ValueError("Native Layer-1 feature order requires the prepared input selector.")
    if (
        task == "Nowcasting"
        and method == "direct_phase34_xgboost"
        and model_component == "direct_nowcasting_layer2_regressor"
    ):
        return tuple(NOWCAST_FEATURES)
    raise ValueError(f"Unknown feature contract: {(task, method, model_component)}")


def _validate_native_matrix(matrix: pd.DataFrame, name: str) -> None:
    if matrix.empty:
        raise ValueError(f"{name} is empty.")
    non_numeric = [column for column in matrix if not pd.api.types.is_numeric_dtype(matrix[column])]
    if non_numeric:
        raise ValueError(f"{name} contains nonnumeric columns: {non_numeric}")
    values = matrix.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{name} contains positive or negative infinity.")


def build_feature_manifest(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    layer1_features: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_features(
        task: str,
        method: str,
        component: str,
        features: Sequence[str],
        source_frame: pd.DataFrame | None,
        allow_missing: bool,
    ) -> None:
        for order, feature in enumerate(features, start=1):
            if source_frame is None:
                dtype = "float64"
                source = feature
                formula = {
                    "phase3_margin_020": "phase3_pred_raw - 0.20",
                    "phase4_margin_020": "phase4_pred_raw - 0.20",
                    "phase3_minus_phase4": "phase3_pred_raw - phase4_pred_raw",
                    "layer1_phase3_minus_phase4": "phase3_layer1_pred - phase4_layer1_pred",
                    "residual_phase3_minus_phase4": "phase3_residual_pred - phase4_residual_pred",
                }.get(feature, "saved leakage-safe base score")
            else:
                dtype = str(source_frame[feature].dtype)
                source = feature
                formula = "native predictor"
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "model_component": component,
                    "feature_order": order,
                    "feature_name": feature,
                    "source_columns": source,
                    "formula": formula,
                    "dtype": dtype,
                    "allow_missing": allow_missing,
                }
            )

    add_features("Forecasting", "xgboost", "rescue_classifier", FORECAST_RESCUE_FEATURES, None, False)
    add_features("Nowcasting", "xgboost", "rescue_classifier", NOWCAST_RESCUE_FEATURES, None, False)
    add_features(
        "Forecasting", "direct_phase34_xgboost", "direct_forecasting_classifier",
        layer1_features, forecasting, True,
    )
    add_features(
        "Nowcasting", "direct_phase34_xgboost", "direct_nowcasting_layer1_classifier",
        layer1_features, forecasting, True,
    )
    add_features(
        "Nowcasting", "direct_phase34_xgboost", "direct_nowcasting_layer2_regressor",
        NOWCAST_FEATURES, nowcasting, True,
    )
    manifest = pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS)
    if len(manifest) != 303 or manifest.duplicated(
        ["task", "method", "model_component", "feature_order"]
    ).any():
        raise ValueError("Feature manifest does not satisfy the frozen 303-row contract.")
    return manifest


def get_auxiliary_parameter_candidates() -> tuple[dict[str, object], ...]:
    candidates: list[dict[str, object]] = []
    for max_depth, min_child_weight, learning_rate, reg_lambda in itertools.product(
        (1, 2, 3), (5, 20), (0.03, 0.10), (1, 10)
    ):
        candidates.append(
            {
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "learning_rate": learning_rate,
                "reg_lambda": reg_lambda,
            }
        )
    if len(candidates) != 24:
        raise AssertionError("Auxiliary parameter grid must contain exactly 24 candidates.")
    return tuple(candidates)


def _candidate_id(index: int) -> str:
    return f"C{index:03d}"


def _auxiliary_parameters(candidate: Mapping[str, object], scale_pos_weight: float) -> dict[str, object]:
    return {
        "objective": "binary:logistic",
        "n_estimators": 200,
        "max_depth": int(candidate["max_depth"]),
        "min_child_weight": int(candidate["min_child_weight"]),
        "learning_rate": float(candidate["learning_rate"]),
        "reg_lambda": float(candidate["reg_lambda"]),
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "gamma": 0,
        "reg_alpha": 0,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": 0,
        "n_jobs": 1,
    }


def _class_weight(target: pd.Series) -> float:
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("Binary training fold requires both classes.")
    return negatives / positives


def _safe_pr_auc(target: Sequence[int], score: Sequence[float]) -> float:
    target_array = np.asarray(target, dtype=int)
    if np.unique(target_array).size < 2:
        return math.nan
    return float(average_precision_score(target_array, np.asarray(score, dtype=float)))


def _safe_log_loss(target: Sequence[int], score: Sequence[float]) -> float:
    target_array = np.asarray(target, dtype=int)
    if np.unique(target_array).size < 2:
        return math.nan
    return float(log_loss(target_array, np.asarray(score, dtype=float), labels=[0, 1]))


def fit_and_select_auxiliary_models(
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    selection_rows: list[dict[str, object]] = []
    task_results: dict[str, dict[str, object]] = {}
    candidates = get_auxiliary_parameter_candidates()
    for task in TASK_ORDER:
        feature_columns = list(
            FORECAST_RESCUE_FEATURES if task == "Forecasting" else NOWCAST_RESCUE_FEATURES
        )
        task_oof = oof.loc[oof["task"].eq(task) & oof["in_auxiliary_gate"]].copy()
        task_oof["year"] = task_oof["date"].dt.year
        task_oof = task_oof.sort_values(["year", *KEY_COLUMNS], kind="mergesort").reset_index(drop=True)
        prediction_by_candidate: dict[str, pd.DataFrame] = {}
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_id = _candidate_id(candidate_index)
            fold_predictions: list[pd.DataFrame] = []
            for fold in META_FOLDS:
                train = task_oof["year"].isin(fold.training_years)
                validation = task_oof["year"].eq(fold.validation_year)
                train_target = task_oof.loc[train, "auxiliary_target"].astype(int)
                validation_target = task_oof.loc[validation, "auxiliary_target"].astype(int)
                if train_target.nunique() != 2 or validation_target.nunique() != 2:
                    raise ValueError(f"{task} {fold.fold_id} lacks binary class support.")
                model = xgb.XGBClassifier(
                    **_auxiliary_parameters(candidate, _class_weight(train_target))
                )
                model.fit(task_oof.loc[train, feature_columns], train_target)
                score = np.asarray(model.predict_proba(task_oof.loc[validation, feature_columns])[:, 1])
                fold_frame = task_oof.loc[
                    validation,
                    ["area_id", "date", "country_code_3", "reconstructed_overall_phase",
                     "auxiliary_target", "base_overall_phase_pred", "phase3_pred_rounded",
                     "phase4_pred_raw"],
                ].copy()
                fold_frame["meta_fold"] = fold.fold_id
                fold_frame["auxiliary_phase4_score"] = score
                fold_predictions.append(fold_frame)
                selection_rows.append(
                    {
                        "task": task,
                        "candidate_id": candidate_id,
                        "scope": "fold",
                        "meta_fold": fold.fold_id,
                        **candidate,
                        "n_train": int(train.sum()),
                        "n_validation": int(validation.sum()),
                        "validation_positive_count": int(validation_target.sum()),
                        "pr_auc": _safe_pr_auc(validation_target, score),
                        "log_loss": _safe_log_loss(validation_target, score),
                        "selected": False,
                    }
                )
            pooled = pd.concat(fold_predictions, ignore_index=True).sort_values(
                ["date", "area_id"], kind="mergesort"
            ).reset_index(drop=True)
            if pooled.duplicated(list(KEY_COLUMNS)).any():
                raise ValueError(f"{task} {candidate_id} pooled meta predictions duplicate keys.")
            prediction_by_candidate[candidate_id] = pooled
            selection_rows.append(
                {
                    "task": task,
                    "candidate_id": candidate_id,
                    "scope": "pooled",
                    "meta_fold": "A1-A3",
                    **candidate,
                    "n_train": int(sum(task_oof["year"].isin(fold.training_years).sum() for fold in META_FOLDS)),
                    "n_validation": len(pooled),
                    "validation_positive_count": int(pooled["auxiliary_target"].sum()),
                    "pr_auc": _safe_pr_auc(pooled["auxiliary_target"], pooled["auxiliary_phase4_score"]),
                    "log_loss": _safe_log_loss(pooled["auxiliary_target"], pooled["auxiliary_phase4_score"]),
                    "selected": False,
                }
            )
        task_selection = pd.DataFrame(selection_rows)
        pooled_rows = task_selection.loc[
            task_selection["task"].eq(task) & task_selection["scope"].eq("pooled")
        ].copy()
        pooled_rows["lexical"] = pooled_rows.apply(
            lambda row: json.dumps(
                {
                    "learning_rate": row["learning_rate"],
                    "max_depth": row["max_depth"],
                    "min_child_weight": row["min_child_weight"],
                    "reg_lambda": row["reg_lambda"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            axis=1,
        )
        selected_row = pooled_rows.sort_values(
            ["pr_auc", "log_loss", "max_depth", "min_child_weight", "reg_lambda", "learning_rate", "lexical"],
            ascending=[False, True, True, False, False, True, True],
            kind="mergesort",
        ).iloc[0]
        selected_id = str(selected_row["candidate_id"])
        selected_candidate = next(
            candidate for index, candidate in enumerate(candidates, start=1)
            if _candidate_id(index) == selected_id
        )
        task_results[task] = {
            "candidate_id": selected_id,
            "candidate": dict(selected_candidate),
            "meta_predictions": prediction_by_candidate[selected_id],
            "candidate_pr_auc": float(selected_row["pr_auc"]),
            "candidate_log_loss": float(selected_row["log_loss"]),
            "training_oof": task_oof,
            "feature_columns": feature_columns,
        }
    selection = pd.DataFrame(selection_rows, columns=MODEL_SELECTION_COLUMNS)
    for task, result in task_results.items():
        selected = (
            selection["task"].eq(task)
            & selection["scope"].eq("pooled")
            & selection["candidate_id"].eq(result["candidate_id"])
        )
        selection.loc[selected, "selected"] = True
    if len(selection) != 192 or selection.duplicated(["task", "candidate_id", "scope", "meta_fold"]).any():
        raise ValueError("Model-selection artifact does not satisfy the 192-row contract.")
    return selection.sort_values(["task", "candidate_id", "scope", "meta_fold"], kind="mergesort").reset_index(drop=True), task_results


def _balanced_accuracy(actual: Sequence[int], predicted: Sequence[int]) -> float:
    actual_array = np.asarray(actual, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    recalls: list[float] = []
    for phase in sorted(np.unique(actual_array)):
        mask = actual_array == phase
        recalls.append(float(np.mean(predicted_array[mask] == phase)))
    return float(np.mean(recalls)) if recalls else math.nan


def _macro_f1(actual: Sequence[int], predicted: Sequence[int]) -> float:
    actual_array = np.asarray(actual, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    values: list[float] = []
    for phase in ALL_PHASES:
        tp = int(np.sum((actual_array == phase) & (predicted_array == phase)))
        fp = int(np.sum((actual_array != phase) & (predicted_array == phase)))
        fn = int(np.sum((actual_array == phase) & (predicted_array != phase)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return float(np.mean(values))


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return math.nan if denominator == 0 else float(numerator / denominator)


def _phase_class_metrics(actual: np.ndarray, predicted: np.ndarray, phase: int) -> tuple[int, int, int, float, float, float]:
    tp = int(np.sum((actual == phase) & (predicted == phase)))
    fp = int(np.sum((actual != phase) & (predicted == phase)))
    fn = int(np.sum((actual == phase) & (predicted != phase)))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    return tp, fp, fn, precision, recall, f1


def select_promotion_threshold(
    task: str,
    method: str,
    validation: pd.DataFrame,
    score_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    required = ["reconstructed_overall_phase", "base_overall_phase_pred", score_column]
    _require_columns(validation, required, f"{task} threshold validation")
    if not validation["base_overall_phase_pred"].eq(3).all():
        raise ValueError("Threshold selection must use only the complete Phase-3 gate.")
    actual = validation["reconstructed_overall_phase"].to_numpy(dtype=int)
    base = validation["base_overall_phase_pred"].to_numpy(dtype=int)
    scores = validation[score_column].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Threshold-selection scores must be finite.")
    positives = actual == 4
    prevalence = float(np.mean(positives))
    base_accuracy = float(np.mean(base == actual))
    base_balanced = _balanced_accuracy(actual, base)
    base_mae = float(np.mean(np.abs(actual - base)))
    sentinel = float(np.nextafter(np.max(scores), np.inf))
    thresholds = [sentinel, *sorted(np.unique(scores).tolist(), reverse=True)]
    rows: list[dict[str, object]] = []
    for rank, threshold in enumerate(thresholds, start=1):
        promoted = scores >= threshold
        predicted = base.copy()
        predicted[promoted] = 4
        tp = int(np.sum(promoted & positives))
        fp = int(np.sum(promoted & ~positives))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, int(np.sum(positives)))
        accuracy = float(np.mean(predicted == actual))
        balanced = _balanced_accuracy(actual, predicted)
        mae = float(np.mean(np.abs(actual - predicted)))
        is_sentinel = rank == 1
        precision_gate = bool(np.isfinite(precision) and precision >= max(0.30, 2 * prevalence))
        accuracy_gate = bool(accuracy > base_accuracy)
        balanced_gate = bool(balanced >= base_balanced)
        mae_gate = bool(mae <= base_mae)
        recall_gate = bool(np.isfinite(recall) and recall > 0)
        eligible = bool(
            not is_sentinel
            and precision_gate
            and accuracy_gate
            and balanced_gate
            and mae_gate
            and recall_gate
        )
        rows.append(
            {
                "task": task,
                "method": method,
                "threshold_rank": rank,
                "threshold": threshold,
                "threshold_source": "pre2022_meta_validation",
                "evaluation_status": "evaluated",
                "no_rescue_sentinel": is_sentinel,
                "gate_rows": len(validation),
                "gate_positive_count": int(positives.sum()),
                "gate_prevalence": prevalence,
                "predicted_positive_count": int(promoted.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_positive_actual_phase3": int(np.sum(promoted & (actual == 3))),
                "phase4_precision": precision,
                "phase4_recall": recall,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "ordinal_mae": mae,
                "precision_gate_passed": precision_gate,
                "accuracy_gate_passed": accuracy_gate,
                "balanced_accuracy_gate_passed": balanced_gate,
                "ordinal_mae_gate_passed": mae_gate,
                "recall_gate_passed": recall_gate,
                "eligible": eligible,
                "selected": False,
            }
        )
    search = pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)
    eligible = search.loc[search["eligible"]].copy()
    if eligible.empty:
        selected_index = search.index[search["no_rescue_sentinel"]][0]
        search.loc[selected_index, "evaluation_status"] = "selected_no_rescue"
    else:
        selected_index = eligible.sort_values(
            ["phase4_recall", "balanced_accuracy", "phase4_precision", "predicted_positive_count", "threshold"],
            ascending=[False, False, False, True, False],
            kind="mergesort",
        ).index[0]
        search.loc[selected_index, "evaluation_status"] = "selected"
    search.loc[selected_index, "selected"] = True
    return search, search.loc[selected_index].copy()


def build_threshold_searches(
    task_results: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.Series]]]:
    frames: list[pd.DataFrame] = []
    selected: dict[str, dict[str, pd.Series]] = {}
    for task in TASK_ORDER:
        meta = task_results[task]["meta_predictions"].copy()
        xgb_search, xgb_selected = select_promotion_threshold(
            task, "xgboost", meta, "auxiliary_phase4_score"
        )
        score_search, score_selected = select_promotion_threshold(
            task, "single_score", meta, "phase4_pred_raw"
        )
        direct_row = {column: np.nan for column in THRESHOLD_COLUMNS}
        direct_row.update(
            {
                "task": task,
                "method": "direct_phase34_xgboost",
                "threshold_rank": 1,
                "threshold": DIRECT_THRESHOLD,
                "threshold_source": "fixed_predeclared",
                "evaluation_status": "not_selected",
                "no_rescue_sentinel": False,
                "selected": False,
            }
        )
        frames.extend([score_search, pd.DataFrame([direct_row], columns=THRESHOLD_COLUMNS), xgb_search])
        selected[task] = {"single_score": score_selected, "xgboost": xgb_selected}
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["task", "method", "threshold_rank"]).any():
        raise ValueError("Threshold-search artifact contains duplicate keys.")
    for task in TASK_ORDER:
        for method in ("single_score", "xgboost"):
            if int(result.loc[result["task"].eq(task) & result["method"].eq(method), "selected"].sum()) != 1:
                raise ValueError(f"{task}/{method} does not have exactly one selected threshold.")
        direct = result.loc[result["task"].eq(task) & result["method"].eq("direct_phase34_xgboost")]
        if len(direct) != 1 or bool(direct.iloc[0]["selected"]) or float(direct.iloc[0]["threshold"]) != 0.5:
            raise ValueError("Direct threshold row violates its fixed unselected contract.")
    return result.sort_values(["task", "method", "threshold_rank"], kind="mergesort").reset_index(drop=True), selected


def refit_selected_auxiliary_models(
    task_results: dict[str, dict[str, object]],
    selected_thresholds: dict[str, dict[str, pd.Series]],
    staging_dir: Path,
) -> dict[str, dict[str, object]]:
    fitted: dict[str, dict[str, object]] = {}
    for task in TASK_ORDER:
        result = task_results[task]
        training = result["training_oof"].copy()
        features = list(result["feature_columns"])
        matrix = training[features].reset_index(drop=True)
        target = training["auxiliary_target"].astype(int).reset_index(drop=True)
        scale_pos_weight = _class_weight(target)
        selected_threshold = selected_thresholds[task]["xgboost"]
        no_rescue = bool(selected_threshold["no_rescue_sentinel"])
        model_path = staging_dir / (
            "phase4_rescue_forecasting_model.json"
            if task == "Forecasting"
            else "phase4_rescue_nowcasting_model.json"
        )
        model: xgb.XGBClassifier | None = None
        if no_rescue:
            model_path.write_text(
                json.dumps(
                    {
                        "candidate_id": result["candidate_id"],
                        "freeze_id": FREEZE_ID,
                        "model_type": "no_rescue_sentinel",
                        "selected_threshold": float(selected_threshold["threshold"]),
                        "task": task,
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
        else:
            model = xgb.XGBClassifier(
                **_auxiliary_parameters(result["candidate"], scale_pos_weight)
            )
            model.fit(matrix, target)
            model.save_model(model_path)
        fitted[task] = {
            **result,
            "matrix": matrix,
            "target": target,
            "scale_pos_weight": scale_pos_weight,
            "selected_threshold": float(selected_threshold["threshold"]),
            "no_rescue": no_rescue,
            "model": model,
            "model_path": model_path,
        }
    return fitted


def _direct_parameters(estimator: str) -> dict[str, object]:
    with PHASE3_PARAMS.open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    parameters["random_state"] = 0
    parameters["n_jobs"] = 1
    if estimator == "classifier":
        parameters["objective"] = "binary:logistic"
    elif estimator == "regressor":
        parameters["objective"] = "reg:squarederror"
    else:
        raise ValueError(f"Unknown direct estimator class: {estimator}")
    return parameters


def fit_direct_baselines(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    layer1_features: Sequence[str],
    staging_dir: Path,
) -> dict[str, object]:
    direct_population, target = build_direct_phase34_population(forecasting)
    keys = direct_population[list(KEY_COLUMNS)].copy()
    forecasting_matrix = direct_population[list(layer1_features)].reset_index(drop=True)
    _validate_native_matrix(forecasting_matrix, "Direct Forecasting training matrix")
    nowcast_keyed = nowcasting.merge(
        direct_population[list(KEY_COLUMNS)], on=list(KEY_COLUMNS), how="inner", validate="one_to_one"
    ).sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    if canonical_key_frame(nowcast_keyed).equals(canonical_key_frame(direct_population)) is False:
        raise ValueError("Direct Nowcasting training keys differ from the Phase-3/4 population.")
    nowcasting_matrix = nowcast_keyed[list(NOWCAST_FEATURES)].reset_index(drop=True)
    _validate_native_matrix(nowcasting_matrix, "Direct Nowcasting Layer-2 training matrix")

    forecasting_model = xgb.XGBClassifier(**_direct_parameters("classifier"))
    forecasting_model.fit(forecasting_matrix, target)
    forecasting_model_path = staging_dir / "phase4_rescue_direct_forecasting_model.json"
    forecasting_model.save_model(forecasting_model_path)

    nowcasting_layer1_model = xgb.XGBClassifier(**_direct_parameters("classifier"))
    nowcasting_layer1_model.fit(forecasting_matrix, target)
    layer1_training_score = np.asarray(nowcasting_layer1_model.predict_proba(forecasting_matrix)[:, 1])
    residual_target = target.to_numpy(dtype=float) - layer1_training_score
    nowcasting_layer2_model = xgb.XGBRegressor(**_direct_parameters("regressor"))
    nowcasting_layer2_model.fit(nowcasting_matrix, residual_target)
    layer1_path = staging_dir / "phase4_rescue_direct_nowcasting_layer1_model.json"
    layer2_path = staging_dir / "phase4_rescue_direct_nowcasting_layer2_model.json"
    nowcasting_layer1_model.save_model(layer1_path)
    nowcasting_layer2_model.save_model(layer2_path)

    return {
        "population": direct_population,
        "target": target.reset_index(drop=True),
        "keys": keys.reset_index(drop=True),
        "forecasting_matrix": forecasting_matrix,
        "nowcasting_layer1_matrix": forecasting_matrix.copy(),
        "nowcasting_layer2_matrix": nowcasting_matrix,
        "forecasting_model": forecasting_model,
        "nowcasting_layer1_model": nowcasting_layer1_model,
        "nowcasting_layer2_model": nowcasting_layer2_model,
        "forecasting_model_path": forecasting_model_path,
        "nowcasting_layer1_model_path": layer1_path,
        "nowcasting_layer2_model_path": layer2_path,
        "nowcasting_residual_target": pd.Series(residual_target, name="residual_target"),
    }


def load_and_validate_frozen_base_predictions() -> dict[str, pd.DataFrame]:
    predictions = pd.read_csv(SPATIAL_PREDICTIONS, float_precision="round_trip", parse_dates=["date"])
    audit = pd.read_csv(SPATIAL_SOURCE_AUDIT, float_precision="round_trip")
    audit_rows = audit.loc[
        audit["condition"].eq("baseline_with_lat_lon") & audit["model"].isin(TASK_ORDER)
    ]
    if len(audit_rows) != 2:
        raise ValueError("Spatial source audit lacks two formal baseline rows.")
    if not audit_rows["production_run"].astype(bool).all():
        raise ValueError("Selected spatial baseline was not recorded as a production run.")
    if not audit_rows["freeze_id"].eq(FREEZE_ID).all() or not audit_rows["reference_environment_id"].eq(REFERENCE_ENVIRONMENT_ID).all():
        raise ValueError("Selected spatial baseline freeze/environment identity drifted.")
    if not audit_rows["metric_reference_applied"].astype(bool).all():
        raise ValueError("Selected spatial baseline did not apply the frozen metric reference.")

    result: dict[str, pd.DataFrame] = {}
    required = [
        "area_id", "date", "country_code_3", "source_overall_phase", "overall_phase",
        "overall_phase_pred", "phase2_test", "phase3_test", "phase4_test", "phase5_test",
        "phase2_pred_raw", "phase3_pred_raw", "phase4_pred_raw", "phase5_pred_raw",
        "phase2_pred_rounded", "phase3_pred_rounded", "phase4_pred_rounded", "phase5_pred_rounded",
        "phase2_layer1_pred", "phase3_layer1_pred", "phase4_layer1_pred", "phase5_layer1_pred",
        "phase2_residual_pred", "phase3_residual_pred", "phase4_residual_pred", "phase5_residual_pred",
    ]
    for task in TASK_ORDER:
        frame = predictions.loc[
            predictions["condition"].eq("baseline_with_lat_lon") & predictions["model"].eq(task)
        ].copy()
        _require_columns(frame, required, f"Frozen {task} predictions")
        _require_unique_keys(frame, f"Frozen {task} predictions")
        frame = frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
        if len(frame) != EXPECTED_BENCHMARK_ROWS:
            raise ValueError(f"Frozen {task} predictions do not contain 1,170 rows.")
        recomputed = phase_from_rounded_predictions(frame)
        if not np.array_equal(recomputed, frame["overall_phase_pred"].to_numpy(dtype=int)):
            raise ValueError(f"Frozen {task} saved phase does not match rounded cumulative scores.")
        if task == "Nowcasting":
            for phase in range(2, 6):
                generated = frame[f"phase{phase}_layer1_pred"] + frame[f"phase{phase}_residual_pred"]
                if not np.allclose(
                    generated.to_numpy(dtype=float), frame[f"phase{phase}_pred_raw"].to_numpy(dtype=float),
                    rtol=0, atol=5e-8,
                ):
                    raise ValueError(f"Frozen Nowcasting Phase-{phase} score identity failed.")
        frame["phase3_margin_020"] = frame["phase3_pred_raw"] - 0.20
        frame["phase4_margin_020"] = frame["phase4_pred_raw"] - 0.20
        frame["phase3_minus_phase4"] = frame["phase3_pred_raw"] - frame["phase4_pred_raw"]
        if task == "Nowcasting":
            frame["layer1_phase3_minus_phase4"] = frame["phase3_layer1_pred"] - frame["phase4_layer1_pred"]
            frame["residual_phase3_minus_phase4"] = frame["phase3_residual_pred"] - frame["phase4_residual_pred"]
        else:
            frame["layer1_phase3_minus_phase4"] = np.nan
            frame["residual_phase3_minus_phase4"] = np.nan
        result[task] = frame
    if canonical_key_sha256(result["Forecasting"]) != canonical_key_sha256(result["Nowcasting"]):
        raise ValueError("Frozen task benchmark keys differ.")
    expected_key_hash = "288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2"
    if canonical_key_sha256(result["Forecasting"]) != expected_key_hash:
        raise ValueError("Frozen benchmark key hash drifted.")
    return result


def validate_frozen_benchmark_outcomes(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    frozen_base: Mapping[str, pd.DataFrame],
) -> None:
    outcome_mapping = {
        "source_overall_phase": "overall_phase",
        "overall_phase": "reconstructed_overall_phase",
        "phase2_test": "phase2_worse",
        "phase3_test": "phase3_worse",
        "phase4_test": "phase4_worse",
        "phase5_test": "phase5_worse",
    }
    for task, source in (("Forecasting", forecasting), ("Nowcasting", nowcasting)):
        source_test = source.loc[source["date"] >= CUTOFF].sort_values(
            list(KEY_COLUMNS), kind="mergesort"
        ).reset_index(drop=True)
        frozen = frozen_base[task].sort_values(
            list(KEY_COLUMNS), kind="mergesort"
        ).reset_index(drop=True)
        if not canonical_key_frame(source_test).equals(canonical_key_frame(frozen)):
            raise ValueError(f"Frozen {task} benchmark keys differ from source outcomes.")
        comparisons = {"country_code_3": "country_code_3", **outcome_mapping}
        for frozen_column, source_column in comparisons.items():
            left = frozen[frozen_column]
            right = source_test[source_column]
            equal = left.eq(right) | (left.isna() & right.isna())
            if not bool(equal.all()):
                bad = source_test.loc[~equal, list(KEY_COLUMNS)].head().to_dict("records")
                raise ValueError(
                    f"Frozen {task} benchmark outcome differs for {frozen_column}: {bad}"
                )


def load_direct_benchmark_matrices(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    layer1_features: Sequence[str],
    frozen_base: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    forecast_test = forecasting.loc[forecasting["date"] >= CUTOFF].sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    nowcast_test = nowcasting.loc[nowcasting["date"] >= CUTOFF].sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    if len(forecast_test) != EXPECTED_BENCHMARK_ROWS or len(nowcast_test) != EXPECTED_BENCHMARK_ROWS:
        raise ValueError("Direct benchmark source matrices do not contain 1,170 rows.")
    reference_keys = canonical_key_frame(frozen_base["Forecasting"])
    if not canonical_key_frame(forecast_test).equals(reference_keys):
        raise ValueError("Direct Forecasting benchmark keys differ from the frozen base.")
    if not canonical_key_frame(nowcast_test).equals(reference_keys):
        raise ValueError("Direct Nowcasting benchmark keys differ from the frozen base.")
    validate_frozen_benchmark_outcomes(forecasting, nowcasting, frozen_base)
    forecasting_matrix = forecast_test[list(layer1_features)].reset_index(drop=True)
    nowcasting_layer1_matrix = forecast_test[list(layer1_features)].reset_index(drop=True)
    nowcasting_layer2_matrix = nowcast_test[list(NOWCAST_FEATURES)].reset_index(drop=True)
    _validate_native_matrix(forecasting_matrix, "Direct Forecasting benchmark matrix")
    _validate_native_matrix(nowcasting_layer1_matrix, "Direct Nowcasting Layer-1 benchmark matrix")
    _validate_native_matrix(nowcasting_layer2_matrix, "Direct Nowcasting Layer-2 benchmark matrix")
    return {
        "keys": forecast_test[list(KEY_COLUMNS)].copy(),
        "forecasting_matrix": forecasting_matrix,
        "nowcasting_layer1_matrix": nowcasting_layer1_matrix,
        "nowcasting_layer2_matrix": nowcasting_layer2_matrix,
    }


def build_benchmark_predictions(
    frozen_base: Mapping[str, pd.DataFrame],
    direct_models: Mapping[str, object],
    direct_benchmark: Mapping[str, object],
    rescue_models: Mapping[str, Mapping[str, object]],
    selected_thresholds: Mapping[str, Mapping[str, pd.Series]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    outputs: list[pd.DataFrame] = []
    rescue_benchmark_matrices: dict[str, pd.DataFrame] = {}

    forecast_direct_all = np.asarray(
        direct_models["forecasting_model"].predict_proba(
            direct_benchmark["forecasting_matrix"]
        )[:, 1]
    )
    nowcast_direct_layer1_all = np.asarray(
        direct_models["nowcasting_layer1_model"].predict_proba(
            direct_benchmark["nowcasting_layer1_matrix"]
        )[:, 1]
    )
    nowcast_direct_residual_all = np.asarray(
        direct_models["nowcasting_layer2_model"].predict(
            direct_benchmark["nowcasting_layer2_matrix"]
        )
    )
    nowcast_direct_all = combine_direct_nowcasting_scores(
        nowcast_direct_layer1_all, nowcast_direct_residual_all
    )

    for task in TASK_ORDER:
        base = frozen_base[task].copy().sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
        gate = base["overall_phase_pred"].eq(3).to_numpy()
        single_threshold = float(selected_thresholds[task]["single_score"]["threshold"])
        single_score = base["phase4_pred_raw"].to_numpy(dtype=float)
        single_trigger = gate & (single_score >= single_threshold)
        single_phase = base["overall_phase_pred"].to_numpy(dtype=int).copy()
        single_phase[single_trigger] = 4
        assert_postclassification_invariants(base["overall_phase_pred"], single_phase)

        if task == "Forecasting":
            direct_layer1_all = forecast_direct_all
            direct_residual_all = np.full(len(base), np.nan)
            direct_score_all = forecast_direct_all
        else:
            direct_layer1_all = nowcast_direct_layer1_all
            direct_residual_all = nowcast_direct_residual_all
            direct_score_all = nowcast_direct_all
        direct_trigger = gate & (direct_score_all >= DIRECT_THRESHOLD)
        direct_phase = base["overall_phase_pred"].to_numpy(dtype=int).copy()
        direct_phase[direct_trigger] = 4
        assert_postclassification_invariants(base["overall_phase_pred"], direct_phase)

        feature_columns = list(
            FORECAST_RESCUE_FEATURES if task == "Forecasting" else NOWCAST_RESCUE_FEATURES
        )
        rescue_matrix = base.loc[gate, feature_columns].reset_index(drop=True)
        if rescue_matrix.isna().any().any() or not np.isfinite(rescue_matrix.to_numpy(dtype=float)).all():
            raise ValueError(f"Frozen {task} rescue benchmark matrix is missing or nonfinite.")
        rescue_benchmark_matrices[task] = rescue_matrix
        rescue = rescue_models[task]
        rescue_score = np.full(len(base), np.nan)
        rescue_trigger = np.zeros(len(base), dtype=bool)
        rescued_phase = base["overall_phase_pred"].to_numpy(dtype=int).copy()
        if not bool(rescue["no_rescue"]):
            model = rescue["model"]
            if model is None:
                raise ValueError(f"{task} selected a rescue but has no fitted model.")
            gate_score = np.asarray(model.predict_proba(rescue_matrix)[:, 1])
            rescue_score[gate] = gate_score
            threshold = float(rescue["selected_threshold"])
            rescue_trigger = gate & (rescue_score >= threshold)
            rescued_phase[rescue_trigger] = 4
        assert_postclassification_invariants(base["overall_phase_pred"], rescued_phase)

        output = pd.DataFrame(
            {
                "task": task,
                "area_id": base["area_id"].astype(int),
                "date": base["date"],
                "country_code_3": base["country_code_3"],
                "source_overall_phase": base["source_overall_phase"].astype(int),
                "reconstructed_overall_phase": base["overall_phase"].astype(int),
            }
        )
        for phase in range(2, 6):
            output[f"phase{phase}_test"] = base[f"phase{phase}_test"]
            output[f"phase{phase}_pred_raw"] = base[f"phase{phase}_pred_raw"]
            output[f"phase{phase}_pred_rounded"] = base[f"phase{phase}_pred_rounded"]
            output[f"phase{phase}_layer1_pred"] = base[f"phase{phase}_layer1_pred"]
        output["phase3_residual_pred"] = base["phase3_residual_pred"]
        output["phase4_residual_pred"] = base["phase4_residual_pred"]
        output["phase3_margin_020"] = base["phase3_margin_020"]
        output["phase4_margin_020"] = base["phase4_margin_020"]
        output["phase3_minus_phase4"] = base["phase3_minus_phase4"]
        output["layer1_phase3_minus_phase4"] = base["layer1_phase3_minus_phase4"]
        output["residual_phase3_minus_phase4"] = base["residual_phase3_minus_phase4"]
        output["base_overall_phase_pred"] = base["overall_phase_pred"].astype(int)
        output["single_score_threshold"] = single_threshold
        output["single_score_triggered"] = single_trigger
        output["single_score_overall_phase_pred"] = single_phase
        output["direct_layer1_phase4_score"] = np.where(gate, direct_layer1_all, np.nan)
        output["direct_layer2_residual_score"] = np.where(gate, direct_residual_all, np.nan)
        output["direct_phase4_score"] = np.where(gate, direct_score_all, np.nan)
        output["direct_threshold"] = DIRECT_THRESHOLD
        output["direct_triggered"] = direct_trigger
        output["direct_overall_phase_pred"] = direct_phase
        output["auxiliary_phase4_score"] = rescue_score
        output["xgboost_threshold"] = float(rescue["selected_threshold"])
        output["xgboost_triggered"] = rescue_trigger
        output["rescued_overall_phase_pred"] = rescued_phase
        output = output.loc[:, list(BENCHMARK_COLUMNS)]
        if task == "Forecasting":
            forecast_na = [
                "phase2_layer1_pred", "phase3_layer1_pred", "phase4_layer1_pred",
                "phase5_layer1_pred", "phase3_residual_pred", "phase4_residual_pred",
                "layer1_phase3_minus_phase4", "residual_phase3_minus_phase4",
                "direct_layer2_residual_score",
            ]
            if output[forecast_na].notna().any().any():
                raise ValueError("Forecasting-only placeholder fields must remain NA.")
        outputs.append(output)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        benchmark = pd.concat(outputs, ignore_index=True)
    if len(benchmark) != 2340 or benchmark.duplicated(["task", *KEY_COLUMNS]).any():
        raise ValueError("Benchmark prediction artifact violates its 2,340-row key contract.")
    return benchmark.sort_values(["task", *KEY_COLUMNS], kind="mergesort").reset_index(drop=True), rescue_benchmark_matrices


def build_meta_validation_frames(
    task_results: Mapping[str, Mapping[str, object]],
    selected_thresholds: Mapping[str, Mapping[str, pd.Series]],
    forecasting: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    truth = forecasting.loc[
        forecasting["date"].dt.year.isin((2019, 2020, 2021)),
        [*KEY_COLUMNS, "country_code_3", "phase3_worse"],
    ].copy()
    truth = truth.rename(columns={"phase3_worse": "phase3_test"})
    frames: dict[str, pd.DataFrame] = {}
    for task in TASK_ORDER:
        meta = task_results[task]["meta_predictions"].copy()
        meta = meta.merge(truth, on=[*KEY_COLUMNS, "country_code_3"], how="left", validate="one_to_one")
        if meta["phase3_test"].isna().any():
            raise ValueError(f"{task} meta-validation rows lost Phase-3 actual shares.")
        base = meta["base_overall_phase_pred"].to_numpy(dtype=int)
        single_threshold = float(selected_thresholds[task]["single_score"]["threshold"])
        xgb_threshold = float(selected_thresholds[task]["xgboost"]["threshold"])
        meta["single_score_overall_phase_pred"] = apply_phase4_rescue(
            base, meta["phase4_pred_raw"], single_threshold
        )
        meta["rescued_overall_phase_pred"] = apply_phase4_rescue(
            base, meta["auxiliary_phase4_score"], xgb_threshold
        )
        frames[task] = meta.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    return frames


def calculate_metric_record(
    split: str,
    task: str,
    method: str,
    frame: pd.DataFrame,
    prediction_column: str,
    base_column: str,
    score_column: str | None,
    evaluation_status: str,
    accepted: object = np.nan,
) -> dict[str, object]:
    actual = frame["reconstructed_overall_phase"].to_numpy(dtype=int)
    predicted = frame[prediction_column].to_numpy(dtype=int)
    base = frame[base_column].to_numpy(dtype=int)
    assert_postclassification_invariants(base, predicted)
    gate = base == 3
    p4_tp, p4_fp, p4_fn, p4_precision, p4_recall, p4_f1 = _phase_class_metrics(actual, predicted, 4)
    _, _, _, p3_precision, p3_recall, p3_f1 = _phase_class_metrics(actual, predicted, 3)
    actual_positive = actual >= 3
    predicted_positive = predicted >= 3
    phase3plus_tp = int(np.sum(actual_positive & predicted_positive))
    phase3plus_fp = int(np.sum(~actual_positive & predicted_positive))
    phase3plus_fn = int(np.sum(actual_positive & ~predicted_positive))
    phase3plus_precision = _safe_divide(phase3plus_tp, phase3plus_tp + phase3plus_fp)
    phase3plus_recall = _safe_divide(phase3plus_tp, phase3plus_tp + phase3plus_fn)
    if len(frame) < 2 or frame["phase3_test"].nunique(dropna=False) < 2:
        phase3plus_r2 = math.nan
    else:
        phase3plus_r2 = float(r2_score(frame["phase3_test"], frame["phase3_pred_rounded"]))
    score_pr_auc = math.nan
    if score_column is not None:
        score_mask = gate & frame[score_column].notna().to_numpy()
        if score_mask.any():
            score_pr_auc = _safe_pr_auc(actual[score_mask] == 4, frame.loc[score_mask, score_column])
    country_records: list[dict[str, float]] = []
    for _, group in frame.assign(__predicted=predicted).groupby("country_code_3", sort=True, observed=True):
        group_actual = group["reconstructed_overall_phase"].to_numpy(dtype=int)
        group_predicted = group["__predicted"].to_numpy(dtype=int)
        country_records.append(
            {
                "accuracy": float(np.mean(group_actual == group_predicted)),
                "balanced_accuracy": _balanced_accuracy(group_actual, group_predicted),
                "macro_f1": _macro_f1(group_actual, group_predicted),
                "ordinal_mae": float(np.mean(np.abs(group_actual - group_predicted))),
            }
        )
    country = pd.DataFrame(country_records)
    changed = base != predicted
    return {
        "split": split,
        "task": task,
        "method": method,
        "evaluation_status": evaluation_status,
        "n_rows": len(frame),
        "n_gate": int(gate.sum()),
        "actual_phase4_count": int(np.sum(actual == 4)),
        "predicted_phase4_count": int(np.sum(predicted == 4)),
        "phase4_true_positive": p4_tp,
        "phase4_false_positive": p4_fp,
        "phase4_false_negative": p4_fn,
        "phase4_precision": p4_precision,
        "phase4_recall": p4_recall,
        "phase4_f1": p4_f1,
        "pr_auc": score_pr_auc,
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": _balanced_accuracy(actual, predicted),
        "macro_f1": _macro_f1(actual, predicted),
        "ordinal_mae": float(np.mean(np.abs(actual - predicted))),
        "phase3_precision": p3_precision,
        "phase3_recall": p3_recall,
        "phase3_f1": p3_f1,
        "phase3plus_precision": phase3plus_precision,
        "phase3plus_recall": phase3plus_recall,
        "phase3plus_r2": phase3plus_r2,
        "changed_3_to_4": int(changed.sum()),
        "correct_rescues": int(np.sum(changed & (actual == 4))),
        "false_promotions": int(np.sum(changed & (actual != 4))),
        "country_macro_accuracy": float(country["accuracy"].mean()),
        "country_macro_balanced_accuracy": float(country["balanced_accuracy"].mean()),
        "country_macro_macro_f1": float(country["macro_f1"].mean()),
        "country_macro_ordinal_mae": float(country["ordinal_mae"].mean()),
        "accepted": accepted,
    }


def _not_applicable_metric_row(split: str, task: str, method: str) -> dict[str, object]:
    row = {column: np.nan for column in METRIC_COLUMNS}
    row.update(
        {
            "split": split,
            "task": task,
            "method": method,
            "evaluation_status": "not_applicable",
            "accepted": np.nan,
        }
    )
    return row


def calculate_metrics(
    meta_frames: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    task_results: Mapping[str, Mapping[str, object]],
    rescue_models: Mapping[str, Mapping[str, object]],
    selected_thresholds: Mapping[str, Mapping[str, pd.Series]],
) -> tuple[pd.DataFrame, dict[str, bool], dict[str, dict[str, dict[str, object]]]]:
    rows: list[dict[str, object]] = []
    indexed: dict[str, dict[str, dict[str, object]]] = {task: {} for task in TASK_ORDER}
    for task in TASK_ORDER:
        meta = meta_frames[task]
        meta_specs = (
            ("frozen_base", "base_overall_phase_pred", None, "reference"),
            (
                "single_score", "single_score_overall_phase_pred", "phase4_pred_raw",
                "selected_no_rescue"
                if bool(selected_thresholds[task]["single_score"]["no_rescue_sentinel"])
                else "selected",
            ),
            (
                "xgboost", "rescued_overall_phase_pred", "auxiliary_phase4_score",
                "selected_no_rescue" if rescue_models[task]["no_rescue"] else "selected",
            ),
        )
        for method, prediction, score, status in meta_specs:
            record = calculate_metric_record(
                "meta_validation", task, method, meta, prediction,
                "base_overall_phase_pred", score, status,
                accepted=(False if method == "xgboost" and rescue_models[task]["no_rescue"] else np.nan),
            )
            rows.append(record)
            indexed[task][f"meta_{method}"] = record
        rows.append(_not_applicable_metric_row("meta_validation", task, "direct_phase34_xgboost"))

        task_benchmark = benchmark.loc[benchmark["task"].eq(task)].copy().reset_index(drop=True)
        benchmark_specs = (
            ("frozen_base", "base_overall_phase_pred", None, "frozen_reference"),
            (
                "single_score", "single_score_overall_phase_pred", "phase4_pred_raw",
                "diagnostic_no_rescue"
                if bool(selected_thresholds[task]["single_score"]["no_rescue_sentinel"])
                else "diagnostic",
            ),
            ("direct_phase34_xgboost", "direct_overall_phase_pred", "direct_phase4_score", "reporting_only"),
            (
                "xgboost", "rescued_overall_phase_pred", "auxiliary_phase4_score",
                "no_rescue" if rescue_models[task]["no_rescue"] else "pending_acceptance",
            ),
        )
        for method, prediction, score, status in benchmark_specs:
            record = calculate_metric_record(
                "benchmark", task, method, task_benchmark, prediction,
                "base_overall_phase_pred", score, status,
            )
            rows.append(record)
            indexed[task][f"benchmark_{method}"] = record

    accepted_by_task: dict[str, bool] = {}
    references = frozen_main.classification_references()
    for task in TASK_ORDER:
        meta_xgb = indexed[task]["meta_xgboost"]
        meta_single = indexed[task]["meta_single_score"]
        base = indexed[task]["benchmark_frozen_base"]
        comparator = indexed[task]["benchmark_single_score"]
        candidate = indexed[task]["benchmark_xgboost"]
        reference = references[task]
        invariant = (
            candidate["phase3plus_precision"] == reference["phase3plus_precision"]
            and candidate["phase3plus_recall"] == reference["phase3plus_recall"]
            and math.isclose(
                candidate["phase3plus_r2"], reference["phase3plus_r2"],
                rel_tol=0.0, abs_tol=1e-12,
            )
        )
        prebenchmark = (
            not bool(rescue_models[task]["no_rescue"])
            and float(task_results[task]["candidate_pr_auc"]) > float(meta_single["pr_auc"])
        )
        comparator_gate = (
            candidate["balanced_accuracy"] >= comparator["balanced_accuracy"]
            and candidate["ordinal_mae"] <= comparator["ordinal_mae"]
            and (
                candidate["phase4_recall"] > comparator["phase4_recall"]
                or (
                    candidate["phase4_recall"] == comparator["phase4_recall"]
                    and candidate["phase4_precision"] > comparator["phase4_precision"]
                )
            )
        )
        accepted = bool(
            prebenchmark
            and candidate["phase4_recall"] >= 0.20
            and candidate["phase4_precision"] >= 0.30
            and candidate["accuracy"] > base["accuracy"]
            and candidate["balanced_accuracy"] > base["balanced_accuracy"]
            and candidate["macro_f1"] >= base["macro_f1"]
            and candidate["ordinal_mae"] <= base["ordinal_mae"]
            and (base["phase3_recall"] - candidate["phase3_recall"]) <= 0.02
            and invariant
            and comparator_gate
        )
        accepted_by_task[task] = accepted
        for row in rows:
            if row["task"] == task and row["method"] == "xgboost":
                if row["split"] == "meta_validation":
                    row["accepted"] = bool(prebenchmark)
                else:
                    row["accepted"] = accepted
                    row["evaluation_status"] = (
                        "no_rescue" if rescue_models[task]["no_rescue"]
                        else ("accepted" if accepted else "rejected")
                    )
        if not invariant:
            raise ValueError(f"{task} Phase-3+ invariant metrics do not match the frozen reference.")
        task_benchmark = benchmark.loc[benchmark["task"].eq(task)]
        for prediction_column in (
            "single_score_overall_phase_pred", "direct_overall_phase_pred", "rescued_overall_phase_pred"
        ):
            if not np.array_equal(
                task_benchmark["base_overall_phase_pred"].to_numpy(dtype=int) >= 3,
                task_benchmark[prediction_column].to_numpy(dtype=int) >= 3,
            ):
                raise ValueError(f"{task}/{prediction_column} changed the Phase-3+ binary array.")
    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    if len(metrics) != 16 or metrics.duplicated(["split", "task", "method"]).any():
        raise ValueError("Metric cube does not satisfy the frozen 16-row contract.")
    return metrics.sort_values(["split", "task", "method"], kind="mergesort").reset_index(drop=True), accepted_by_task, indexed


def build_confusion_source(benchmark: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = {
        "frozen_base": "base_overall_phase_pred",
        "single_score": "single_score_overall_phase_pred",
        "direct_phase34_xgboost": "direct_overall_phase_pred",
        "xgboost": "rescued_overall_phase_pred",
    }
    rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        frame = benchmark.loc[benchmark["task"].eq(task)]
        actual = frame["reconstructed_overall_phase"].to_numpy(dtype=int)
        for method in METHOD_ORDER:
            predicted = frame[prediction_columns[method]].to_numpy(dtype=int)
            for actual_phase in ALL_PHASES:
                row_total = int(np.sum(actual == actual_phase))
                for predicted_phase in ALL_PHASES:
                    count = int(np.sum((actual == actual_phase) & (predicted == predicted_phase)))
                    rows.append(
                        {
                            "task": task,
                            "method": method,
                            "actual_phase": actual_phase,
                            "predicted_phase": predicted_phase,
                            "count": count,
                            "actual_row_total": row_total,
                            "actual_row_share": _safe_divide(count, row_total),
                        }
                    )
    result = pd.DataFrame(rows, columns=CONFUSION_COLUMNS)
    if len(result) != 200 or result.duplicated(
        ["task", "method", "actual_phase", "predicted_phase"]
    ).any():
        raise ValueError("Confusion source does not satisfy the frozen 200-cell contract.")
    for (_, _), group in result.groupby(["task", "method"], observed=True):
        if int(group["count"].sum()) != EXPECTED_BENCHMARK_ROWS:
            raise ValueError("A confusion matrix does not sum to 1,170 rows.")
    return result


def build_main_comparison_metrics(
    metrics: pd.DataFrame,
    accepted_by_task: Mapping[str, bool],
    rescue_models: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    benchmark = metrics.loc[metrics["split"].eq("benchmark")].set_index(["task", "method"])
    for task in TASK_ORDER:
        base = benchmark.loc[(task, "frozen_base")]
        for method in MAIN_METHOD_ORDER:
            current = benchmark.loc[(task, method)]
            if method == "frozen_base":
                status = "frozen_reference"
            elif method == "direct_phase34_xgboost":
                status = "reporting_only"
            elif rescue_models[task]["no_rescue"]:
                status = "no_rescue"
            else:
                status = "accepted" if accepted_by_task[task] else "rejected"
            precision_delta = float(current["phase3plus_precision"] - base["phase3plus_precision"])
            recall_delta = float(current["phase3plus_recall"] - base["phase3plus_recall"])
            r2_delta = float(current["phase3plus_r2"] - base["phase3plus_r2"])
            if precision_delta != 0.0 or recall_delta != 0.0 or r2_delta != 0.0:
                raise ValueError(f"{task}/{method} violates exact Main-metric invariants.")
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "display_status": status,
                    "n_rows": int(current["n_rows"]),
                    "phase3plus_precision": current["phase3plus_precision"],
                    "phase3plus_recall": current["phase3plus_recall"],
                    "phase3plus_r2": current["phase3plus_r2"],
                    "accuracy": current["accuracy"],
                    "phase3plus_precision_delta_from_base": 0.0,
                    "phase3plus_recall_delta_from_base": 0.0,
                    "phase3plus_r2_delta_from_base": 0.0,
                    "accuracy_delta_from_base": float(current["accuracy"] - base["accuracy"]),
                }
            )
    result = pd.DataFrame(rows, columns=MAIN_METRIC_COLUMNS)
    if len(result) != 6 or result.duplicated(["task", "method"]).any():
        raise ValueError("Main comparison metric source must contain exactly six rows.")
    return result


def _bootstrap_metric_bundle(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    _, _, _, p4_precision, p4_recall, _ = _phase_class_metrics(actual, predicted, 4)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": _balanced_accuracy(actual, predicted),
        "macro_f1": _macro_f1(actual, predicted),
        "ordinal_mae": float(np.mean(np.abs(actual - predicted))),
        "phase4_precision": p4_precision,
        "phase4_recall": p4_recall,
    }


def generate_bootstrap(
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_columns = {
        "single_score": "single_score_overall_phase_pred",
        "direct_phase34_xgboost": "direct_overall_phase_pred",
        "xgboost": "rescued_overall_phase_pred",
    }
    draw_rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        frame = benchmark.loc[benchmark["task"].eq(task)].reset_index(drop=True)
        countries = np.array(sorted(frame["country_code_3"].unique().tolist()), dtype=object)
        if len(countries) != EXPECTED_BENCHMARK_COUNTRIES:
            raise ValueError(f"{task} benchmark has {len(countries)} countries, expected 27.")
        country_indices = {
            country: frame.index[frame["country_code_3"].eq(country)].to_numpy(dtype=int)
            for country in countries
        }
        rng = np.random.RandomState(BOOTSTRAP_RANDOM_STATE)
        samples = [rng.choice(countries, size=len(countries), replace=True) for _ in range(BOOTSTRAP_REPETITIONS)]
        actual_all = frame["reconstructed_overall_phase"].to_numpy(dtype=int)
        base_all = frame["base_overall_phase_pred"].to_numpy(dtype=int)
        for bootstrap_id, sample in enumerate(samples, start=1):
            indices = np.concatenate([country_indices[country] for country in sample])
            actual = actual_all[indices]
            base_metrics = _bootstrap_metric_bundle(actual, base_all[indices])
            for method, column in prediction_columns.items():
                predicted = frame[column].to_numpy(dtype=int)[indices]
                current = _bootstrap_metric_bundle(actual, predicted)
                draw_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "bootstrap_id": bootstrap_id,
                        "sampled_country_count": len(sample),
                        "accuracy_delta": current["accuracy"] - base_metrics["accuracy"],
                        "balanced_accuracy_delta": current["balanced_accuracy"] - base_metrics["balanced_accuracy"],
                        "macro_f1_delta": current["macro_f1"] - base_metrics["macro_f1"],
                        "ordinal_mae_delta": current["ordinal_mae"] - base_metrics["ordinal_mae"],
                        "phase4_precision_delta": current["phase4_precision"] - base_metrics["phase4_precision"],
                        "phase4_recall_delta": current["phase4_recall"] - base_metrics["phase4_recall"],
                    }
                )
    draws = pd.DataFrame(draw_rows, columns=BOOTSTRAP_DRAW_COLUMNS)
    if len(draws) != 12000 or draws.duplicated(["task", "method", "bootstrap_id"]).any():
        raise ValueError("Bootstrap draws do not satisfy the 12,000-row contract.")
    summary_rows: list[dict[str, object]] = []
    metric_names = (
        "accuracy_delta", "balanced_accuracy_delta", "macro_f1_delta",
        "ordinal_mae_delta", "phase4_precision_delta", "phase4_recall_delta",
    )
    for (task, method), group in draws.groupby(["task", "method"], sort=True, observed=True):
        row: dict[str, object] = {
            "task": task,
            "method": method,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "ci_level": 0.95,
            "interval_method": "paired_country_cluster_percentile",
        }
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                lower = upper = math.nan
            else:
                lower, upper = np.percentile(finite, [2.5, 97.5])
            stem = metric.removesuffix("_delta")
            row[f"{stem}_delta_lower"] = lower
            row[f"{stem}_delta_upper"] = upper
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows, columns=BOOTSTRAP_SUMMARY_COLUMNS)
    if len(summary) != 6 or summary.duplicated(["task", "method"]).any():
        raise ValueError("Bootstrap summary does not satisfy the six-row contract.")
    return draws, summary


def build_model_audit(
    rescue_models: Mapping[str, Mapping[str, object]],
    direct_models: Mapping[str, object],
    direct_benchmark: Mapping[str, object],
    rescue_benchmark_matrices: Mapping[str, pd.DataFrame],
    layer1_features: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        rescue = rescue_models[task]
        training_oof = rescue["training_oof"].sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
        benchmark_keys = load_and_validate_frozen_base_predictions()[task].loc[
            lambda frame: frame["overall_phase_pred"].eq(3), list(KEY_COLUMNS)
        ].reset_index(drop=True)
        matrix = rescue["matrix"]
        benchmark_matrix = rescue_benchmark_matrices[task]
        rows.append(
            {
                "task": task,
                "method": "xgboost",
                "model_component": "rescue_classifier",
                "estimator_class": "XGBClassifier" if not rescue["no_rescue"] else "no_rescue_sentinel",
                "objective": "binary:logistic",
                "target_definition": "1[reconstructed_overall_phase == 4] within OOF base Phase-3 gate",
                "run_status": "no_rescue_sentinel" if rescue["no_rescue"] else "fitted",
                "selected_candidate_id": rescue["candidate_id"],
                "selected_threshold": rescue["selected_threshold"],
                "no_rescue_selected": rescue["no_rescue"],
                "training_rows": len(matrix),
                "training_positive_count": int(rescue["target"].sum()),
                "scale_pos_weight": rescue["scale_pos_weight"],
                "feature_count": len(rescue["feature_columns"]),
                "feature_order_sha256": json_sha256(list(rescue["feature_columns"])),
                "training_key_sha256": canonical_key_sha256(training_oof),
                "target_sha256": canonical_series_sha256(rescue["target"], "auxiliary_target"),
                "training_matrix_sha256": canonical_matrix_sha256(matrix),
                "training_missingness_sha256": canonical_missingness_sha256(matrix),
                "training_matrix_with_target_sha256": matrix_with_target_sha256(matrix, rescue["target"]),
                "benchmark_rows": len(benchmark_matrix),
                "benchmark_key_sha256": canonical_key_sha256(benchmark_keys),
                "benchmark_matrix_sha256": canonical_matrix_sha256(benchmark_matrix),
                "benchmark_missingness_sha256": canonical_missingness_sha256(benchmark_matrix),
                "model_path": rescue["model_path"].name,
                "model_sha256": file_sha256(rescue["model_path"]),
            }
        )

    direct_keys = direct_models["keys"]
    direct_target = direct_models["target"]
    benchmark_keys = direct_benchmark["keys"]
    direct_components = (
        (
            "Forecasting", "direct_forecasting_classifier", "XGBClassifier", "binary:logistic",
            "1[reconstructed_overall_phase == 4] among pre-2022 reconstructed Phase 3/4",
            direct_models["forecasting_matrix"], direct_target,
            direct_benchmark["forecasting_matrix"], direct_models["forecasting_model_path"], layer1_features,
        ),
        (
            "Nowcasting", "direct_nowcasting_layer1_classifier", "XGBClassifier", "binary:logistic",
            "1[reconstructed_overall_phase == 4] among pre-2022 reconstructed Phase 3/4",
            direct_models["nowcasting_layer1_matrix"], direct_target,
            direct_benchmark["nowcasting_layer1_matrix"], direct_models["nowcasting_layer1_model_path"], layer1_features,
        ),
        (
            "Nowcasting", "direct_nowcasting_layer2_regressor", "XGBRegressor", "reg:squarederror",
            "binary_target - direct Nowcasting Layer-1 training score",
            direct_models["nowcasting_layer2_matrix"], direct_models["nowcasting_residual_target"],
            direct_benchmark["nowcasting_layer2_matrix"], direct_models["nowcasting_layer2_model_path"], NOWCAST_FEATURES,
        ),
    )
    for task, component, estimator, objective, target_definition, matrix, target, benchmark_matrix, model_path, features in direct_components:
        rows.append(
            {
                "task": task,
                "method": "direct_phase34_xgboost",
                "model_component": component,
                "estimator_class": estimator,
                "objective": objective,
                "target_definition": target_definition,
                "run_status": "fitted_reporting_only",
                "selected_candidate_id": np.nan,
                "selected_threshold": DIRECT_THRESHOLD,
                "no_rescue_selected": False,
                "training_rows": len(matrix),
                "training_positive_count": int(direct_target.sum()),
                "scale_pos_weight": 1.0,
                "feature_count": len(features),
                "feature_order_sha256": json_sha256(list(features)),
                "training_key_sha256": canonical_key_sha256(direct_keys),
                "target_sha256": canonical_series_sha256(pd.Series(target), "target"),
                "training_matrix_sha256": canonical_matrix_sha256(matrix),
                "training_missingness_sha256": canonical_missingness_sha256(matrix),
                "training_matrix_with_target_sha256": matrix_with_target_sha256(matrix, pd.Series(target)),
                "benchmark_rows": len(benchmark_matrix),
                "benchmark_key_sha256": canonical_key_sha256(benchmark_keys),
                "benchmark_matrix_sha256": canonical_matrix_sha256(benchmark_matrix),
                "benchmark_missingness_sha256": canonical_missingness_sha256(benchmark_matrix),
                "model_path": model_path.name,
                "model_sha256": file_sha256(model_path),
            }
        )
    audit = pd.DataFrame(rows, columns=MODEL_AUDIT_COLUMNS)
    if len(audit) != 5 or audit.duplicated(["task", "method", "model_component"]).any():
        raise ValueError("Model audit does not satisfy the five-row contract.")
    return audit


def _method_display_name(method: str) -> str:
    return {
        "frozen_base": "Frozen Figure-1 base",
        "single_score": "Single-score comparator",
        "direct_phase34_xgboost": "Direct Phase-3/4 XGBoost",
        "xgboost": "Deployment-aligned rescue",
    }[method]


def _table_method_display_name(method: str) -> str:
    return {
        "frozen_base": "Frozen base",
        "direct_phase34_xgboost": "Direct P3/4 XGB",
        "xgboost": "Rescue XGB",
    }[method]


def render_main_comparison_figure(
    confusion: pd.DataFrame,
    main_metrics: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(12.0, 6.5), constrained_layout=False, facecolor="white")
    grid = figure.add_gridspec(
        2, 4, left=0.055, right=0.985, bottom=0.105, top=0.91,
        wspace=0.34, hspace=0.42, width_ratios=(1.0, 1.0, 1.0, 1.85),
    )
    cmap = mpl.colormaps["Blues"]
    norm = mpl_colors.Normalize(vmin=0.0, vmax=1.0)
    matrix_methods = MAIN_METHOD_ORDER
    metric_labels = (
        ("phase3plus_precision", "P3+ precision"),
        ("phase3plus_recall", "P3+ recall"),
        ("phase3plus_r2", "P3+ R²"),
        ("accuracy", "5-class accuracy"),
    )
    delta_columns = {
        "phase3plus_precision": "phase3plus_precision_delta_from_base",
        "phase3plus_recall": "phase3plus_recall_delta_from_base",
        "phase3plus_r2": "phase3plus_r2_delta_from_base",
        "accuracy": "accuracy_delta_from_base",
    }
    table_layout_contracts: list[tuple[object, object, object]] = []
    for task_index, task in enumerate(TASK_ORDER):
        for method_index, method in enumerate(matrix_methods):
            axis = figure.add_subplot(grid[task_index, method_index])
            cells = confusion.loc[
                confusion["task"].eq(task) & confusion["method"].eq(method)
            ].sort_values(["actual_phase", "predicted_phase"], kind="mergesort")
            if len(cells) != 25:
                raise ValueError(f"Figure source lacks 25 cells for {task}/{method}.")
            for cell in cells.itertuples(index=False):
                x = int(cell.predicted_phase) - 1
                y = int(cell.actual_phase) - 1
                share = float(cell.actual_row_share)
                color = cmap(norm(share))
                axis.add_patch(
                    mpl_patches.Rectangle(
                        (x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=0.8
                    )
                )
                text_color = "white" if share >= 0.58 else "#1f1f1f"
                axis.text(
                    x + 0.5, y + 0.5, str(int(cell.count)),
                    ha="center", va="center", color=text_color, fontsize=6.7,
                )
            axis.set_xlim(0, 5)
            axis.set_ylim(5, 0)
            axis.set_aspect("equal")
            axis.set_xticks(np.arange(5) + 0.5, labels=[str(value) for value in ALL_PHASES])
            axis.set_yticks(np.arange(5) + 0.5, labels=[str(value) for value in ALL_PHASES])
            axis.tick_params(length=0, pad=1.5)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_title(_method_display_name(method), pad=4)
            axis.set_xlabel("Predicted phase", labelpad=2)
            if method_index == 0:
                axis.set_ylabel(f"{task}\nActual phase", labelpad=3, fontweight="bold")
            else:
                axis.set_ylabel("Actual phase", labelpad=2)

        table_axis = figure.add_subplot(grid[task_index, 3])
        table_axis.axis("off")
        metric_rows = main_metrics.loc[main_metrics["task"].eq(task)].set_index("method")
        cell_text: list[list[str]] = []
        row_labels: list[str] = []
        for method in MAIN_METHOD_ORDER:
            row = metric_rows.loc[method]
            row_labels.append(_table_method_display_name(method))
            values: list[str] = []
            for metric, _ in metric_labels:
                value = float(row[metric])
                delta = float(row[delta_columns[metric]])
                values.append(f"{value:.4f}\nΔ {delta:+.4f}")
            cell_text.append(values)
        table = table_axis.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=[label for _, label in metric_labels],
            cellLoc="center",
            rowLoc="left",
            loc="center",
            bbox=[0.29, 0.17, 0.69, 0.66],
            colWidths=[0.17, 0.16, 0.14, 0.22],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.3)
        for (row, column), cell in table.get_celld().items():
            cell.set_linewidth(0.45)
            cell.set_edgecolor("#bdbdbd")
            if row == 0:
                cell.set_facecolor("#e8eef5")
                cell.set_text_props(weight="bold", fontsize=5.8)
            elif column == -1:
                cell.set_facecolor("#f4f4f4")
                cell.set_text_props(weight="bold", ha="left")
                cell.set_width(0.27)
            else:
                cell.set_facecolor("white")
        table_axis.set_title(
            f"{task}: Main metrics (Δ vs frozen base)",
            fontsize=8.0, fontweight="bold", pad=5,
        )
        note = table_axis.text(
            0.02, 0.035,
            "P3+ precision, recall and R² are invariant;\nonly five-class accuracy may change.",
            transform=table_axis.transAxes, fontsize=6.2, color="#4d4d4d", va="bottom",
        )
        table_layout_contracts.append((table_axis, table, note))

    figure.suptitle(
        "Phase-4 post-classification rescue on the frozen 2022 benchmark",
        fontsize=10.5, fontweight="bold", y=0.972,
    )
    figure.text(
        0.5, 0.935,
        "Counts are shown in cells; color is the percentage within each reconstructed actual-phase row (common 0–100% scale).",
        ha="center", va="center", fontsize=7.0, color="#4d4d4d",
    )
    color_axis = figure.add_axes([0.08, 0.035, 0.55, 0.018])
    color_axis.set_xlim(0, 100)
    color_axis.set_ylim(0, 1)
    for start in range(0, 100, 5):
        midpoint = (start + 2.5) / 100.0
        color_axis.add_patch(
            mpl_patches.Rectangle(
                (start, 0), 5, 1, facecolor=cmap(norm(midpoint)),
                edgecolor="none", linewidth=0,
            )
        )
    color_axis.set_xticks([0, 25, 50, 75, 100])
    color_axis.set_yticks([])
    color_axis.tick_params(axis="x", length=2, width=0.5, pad=1)
    color_axis.set_xlabel("Actual-class row percentage (%)", labelpad=1.5)
    for spine in color_axis.spines.values():
        spine.set_linewidth(0.5)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_bbox = figure.bbox
    for table_axis, table, note in table_layout_contracts:
        axis_bbox = table_axis.get_window_extent(renderer)
        for cell in table.get_celld().values():
            bbox = cell.get_window_extent(renderer)
            if (
                bbox.x0 < axis_bbox.x0 - 1.0
                or bbox.x1 > axis_bbox.x1 + 1.0
                or bbox.y0 < axis_bbox.y0 - 1.0
                or bbox.y1 > axis_bbox.y1 + 1.0
            ):
                raise ValueError("Main comparison metric table exceeds its layout axis.")
        note_bbox = note.get_window_extent(renderer)
        if (
            note_bbox.x0 < axis_bbox.x0 - 1.0
            or note_bbox.x1 > axis_bbox.x1 + 1.0
            or note_bbox.y0 < axis_bbox.y0 - 1.0
            or note_bbox.y1 > axis_bbox.y1 + 1.0
        ):
            raise ValueError("Main comparison metric note is clipped by its layout axis.")
        title_bbox = table_axis.title.get_window_extent(renderer)
        if (
            title_bbox.x0 < figure_bbox.x0
            or title_bbox.x1 > figure_bbox.x1
            or title_bbox.y0 < figure_bbox.y0
            or title_bbox.y1 > figure_bbox.y1
        ):
            raise ValueError("Main comparison metric title is clipped by the figure canvas.")
    pdf_metadata = {
        "Title": "Phase-4 rescue classifier main comparison",
        "Author": "",
        "Subject": "Frozen 2022 food-crisis classification comparison",
        "Keywords": "",
        "Creator": "NCOMMS Phase-4 rescue generator",
        "CreationDate": None,
        "ModDate": None,
    }
    png_metadata = {"Software": "NCOMMS Phase-4 rescue generator"}
    figure.savefig(output_pdf, format="pdf", metadata=pdf_metadata, facecolor="white")
    figure.savefig(output_png, format="png", dpi=600, metadata=png_metadata, facecolor="white")
    plt.close(figure)


def _write_csv(data: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    frame = data.loc[:, list(columns)].copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="<NA>",
        lineterminator="\n",
    )


def _write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_png_contract(path: Path) -> tuple[int, int, float | None]:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Main comparison PNG has an invalid signature.")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    offset = 8
    dpi: float | None = None
    while offset + 12 <= len(payload):
        length = int.from_bytes(payload[offset:offset + 4], "big")
        chunk_type = payload[offset + 4:offset + 8]
        chunk_data = payload[offset + 8:offset + 8 + length]
        if chunk_type == b"pHYs" and len(chunk_data) == 9 and chunk_data[8] == 1:
            pixels_per_meter = int.from_bytes(chunk_data[:4], "big")
            dpi = pixels_per_meter * 0.0254
            break
        offset += 12 + length
    return width, height, dpi


def _configuration_payload(
    rescue_models: Mapping[str, Mapping[str, object]],
    selected_thresholds: Mapping[str, Mapping[str, pd.Series]],
    accepted_by_task: Mapping[str, bool],
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "freeze_id": FREEZE_ID,
        "environment": {
            "environment_id": REFERENCE_ENVIRONMENT_ID,
            "platform_family": "Windows",
            "python_version": "3.11.3",
            "numpy_version": "1.26.4",
            "pandas_version": "2.2.3",
            "scipy_version": "1.17.1",
            "scikit_learn_version": "1.5.2",
            "xgboost_version": "2.0.3",
            "matplotlib_version": "3.10.1",
            "xgboost_dll_sha256": frozen_main.ENVIRONMENT["xgboost_dll_sha256"],
        },
        "decisions": {
            "truth": "reconstructed_fixed_0.20_phase",
            "positive_class": "exact_phase4",
            "allowed_action": "phase3_to_phase4_only",
            "benchmark_role": "fixed_reused_2022_benchmark",
            "direct_baseline_role": "retrospective_reporting_only_diagnostic_comparator",
        },
        "folds": {
            "base": [
                {"fold_id": fold.fold_id, "training_years": list(fold.training_years), "validation_year": fold.validation_year}
                for fold in BASE_FOLDS
            ],
            "auxiliary": [
                {"fold_id": fold.fold_id, "training_years": list(fold.training_years), "validation_year": fold.validation_year}
                for fold in META_FOLDS
            ],
        },
        "features": {
            "Forecasting": list(FORECAST_RESCUE_FEATURES),
            "Nowcasting": list(NOWCAST_RESCUE_FEATURES),
            "direct_forecasting_count": 106,
            "direct_nowcasting_layer1_count": 106,
            "direct_nowcasting_layer2": list(NOWCAST_FEATURES),
        },
        "parameter_grid": {
            "candidates": list(get_auxiliary_parameter_candidates()),
            "fixed": {
                "objective": "binary:logistic", "n_estimators": 200,
                "subsample": 0.8, "colsample_bytree": 1.0, "gamma": 0,
                "reg_alpha": 0, "random_state": 0, "n_jobs": 1,
            },
        },
        "direct_baseline": {
            "threshold": DIRECT_THRESHOLD,
            "parameters_sha256": EXPECTED_SOURCE_HASHES[PHASE3_PARAMS],
            "classifier_objective": "binary:logistic",
            "residual_objective": "reg:squarederror",
            "random_state": 0,
            "n_jobs": 1,
            "native_nan_preserved": True,
        },
        "selected_models": {
            task: {
                "candidate_id": rescue_models[task]["candidate_id"],
                "candidate": rescue_models[task]["candidate"],
                "no_rescue": bool(rescue_models[task]["no_rescue"]),
                "model_file": rescue_models[task]["model_path"].name,
            }
            for task in TASK_ORDER
        },
        "thresholds": {
            task: {
                method: {
                    "threshold": float(selected_thresholds[task][method]["threshold"]),
                    "no_rescue_sentinel": bool(selected_thresholds[task][method]["no_rescue_sentinel"]),
                }
                for method in ("single_score", "xgboost")
            }
            for task in TASK_ORDER
        },
        "acceptance_gate": {
            "task_results": {task: bool(accepted_by_task[task]) for task in TASK_ORDER},
            "bootstrap_is_hard_gate": False,
            "manuscript_adoption_authorized": False,
        },
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "random_state": BOOTSTRAP_RANDOM_STATE,
            "cluster_unit": "country_code_3",
            "ci_level": 0.95,
            "interval_method": "percentile",
        },
        "main_comparison": {
            "tasks": list(TASK_ORDER),
            "methods": list(MAIN_METHOD_ORDER),
            "metrics": ["phase3plus_precision", "phase3plus_recall", "phase3plus_r2", "accuracy"],
        },
        "figure_exports": {
            "backend": "Agg",
            "width_inches": 12.0,
            "height_inches": 6.5,
            "font": "DejaVu Sans",
            "pdf": "phase4_rescue_main_comparison.pdf",
            "png": "phase4_rescue_main_comparison.png",
            "png_dpi": 600,
        },
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
    }


def _source_audit_row(
    environment: Mapping[str, object],
    protected_before: str,
    protected_after: str,
    direct_models: Mapping[str, object],
    model_audit: pd.DataFrame,
    benchmark: pd.DataFrame,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    direct_audit = model_audit.set_index(["task", "model_component"])
    forecasting_direct = direct_audit.loc[("Forecasting", "direct_forecasting_classifier")]
    nowcasting_layer1 = direct_audit.loc[("Nowcasting", "direct_nowcasting_layer1_classifier")]
    nowcasting_layer2 = direct_audit.loc[("Nowcasting", "direct_nowcasting_layer2_regressor")]
    manifest_json = json.dumps(dict(sorted(artifact_hashes.items())), sort_keys=True, separators=(",", ":"))
    package_versions = environment["package_versions"]
    return {
        "run_status": "complete",
        "freeze_id": FREEZE_ID,
        "reference_environment_id": REFERENCE_ENVIRONMENT_ID,
        "evaluation_population_id": EVALUATION_POPULATION_ID,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "pre2022_rows": EXPECTED_PRE2022_ROWS,
        "oof_rows": EXPECTED_OOF_ROWS,
        "benchmark_rows": len(benchmark),
        "benchmark_areas": int(benchmark["area_id"].nunique()),
        "benchmark_countries": int(benchmark["country_code_3"].nunique()),
        "benchmark_key_sha256": canonical_key_sha256(benchmark.loc[benchmark["task"].eq("Forecasting")]),
        "direct_training_rows": EXPECTED_DIRECT_ROWS,
        "direct_phase3_count": EXPECTED_DIRECT_PHASE3,
        "direct_phase4_count": EXPECTED_DIRECT_PHASE4,
        "direct_training_key_sha256": canonical_key_sha256(direct_models["keys"]),
        "direct_target_sha256": canonical_series_sha256(direct_models["target"], "auxiliary_target"),
        "direct_forecasting_training_matrix_sha256": forecasting_direct["training_matrix_sha256"],
        "direct_forecasting_training_missingness_sha256": forecasting_direct["training_missingness_sha256"],
        "direct_nowcasting_layer1_training_matrix_sha256": nowcasting_layer1["training_matrix_sha256"],
        "direct_nowcasting_layer1_training_missingness_sha256": nowcasting_layer1["training_missingness_sha256"],
        "direct_nowcasting_layer2_training_matrix_sha256": nowcasting_layer2["training_matrix_sha256"],
        "direct_nowcasting_layer2_training_missingness_sha256": nowcasting_layer2["training_missingness_sha256"],
        "direct_threshold": DIRECT_THRESHOLD,
        "freeze_source_path": relative_path(FREEZE_SOURCE),
        "freeze_source_sha256": file_sha256(FREEZE_SOURCE),
        "spatial_predictions_path": relative_path(SPATIAL_PREDICTIONS),
        "spatial_predictions_sha256": file_sha256(SPATIAL_PREDICTIONS),
        "spatial_metrics_path": relative_path(SPATIAL_METRICS),
        "spatial_metrics_sha256": file_sha256(SPATIAL_METRICS),
        "spatial_source_audit_path": relative_path(SPATIAL_SOURCE_AUDIT),
        "spatial_source_audit_sha256": file_sha256(SPATIAL_SOURCE_AUDIT),
        "forecasting_input_path": relative_path(FORECASTING_INPUT),
        "forecasting_input_sha256": file_sha256(FORECASTING_INPUT),
        "nowcasting_input_path": relative_path(NOWCASTING_INPUT),
        "nowcasting_input_sha256": file_sha256(NOWCASTING_INPUT),
        "country_lookup_path": relative_path(COUNTRY_LOOKUP),
        "country_lookup_sha256": file_sha256(COUNTRY_LOOKUP),
        "general_params_path": relative_path(GENERAL_PARAMS),
        "general_params_sha256": file_sha256(GENERAL_PARAMS),
        "phase3_params_path": relative_path(PHASE3_PARAMS),
        "phase3_params_sha256": file_sha256(PHASE3_PARAMS),
        "generator_path": relative_path(Path(__file__)),
        "generator_sha256": file_sha256(Path(__file__)),
        "platform_family": environment["platform_family"],
        "python_version": environment["python_version"],
        "numpy_version": package_versions["numpy"],
        "pandas_version": package_versions["pandas"],
        "scipy_version": package_versions["scipy"],
        "sklearn_version": package_versions["scikit-learn"],
        "xgboost_version": package_versions["xgboost"],
        "matplotlib_version": package_versions["matplotlib"],
        "xgboost_dll_sha256": environment["xgboost_dll_sha256"],
        "base_random_state_override": np.nan,
        "base_n_jobs_override": np.nan,
        "auxiliary_random_state": 0,
        "auxiliary_n_jobs": 1,
        "direct_random_state": 0,
        "direct_n_jobs": 1,
        "outer_workers": 1,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_random_state": BOOTSTRAP_RANDOM_STATE,
        "figure_backend": "Agg",
        "figure_width_inches": 12.0,
        "figure_height_inches": 6.5,
        "figure_png_dpi": 600,
        "protected_manifest_sha256_before": protected_before,
        "protected_manifest_sha256_after": protected_after,
        "protected_manifest_match": protected_before == protected_after,
        "artifact_manifest_json": manifest_json,
        "artifact_manifest_sha256": manifest_sha256(artifact_hashes),
    }


def validate_artifact_contract(directory: Path) -> None:
    files = sorted(path.name for path in directory.iterdir() if path.is_file())
    directories = [path.name for path in directory.iterdir() if path.is_dir()]
    if directories or set(files) != set(EXPECTED_ARTIFACTS):
        raise ValueError(f"Formal basename contract failed: files={files}, directories={directories}")
    read_options = {"na_values": ["<NA>"], "keep_default_na": True, "float_precision": "round_trip"}
    checks = {
        "phase4_rescue_oof_predictions.csv": (EXPECTED_OOF_ROWS, ["task", "base_oof_fold", "area_id", "date"]),
        "phase4_rescue_model_selection.csv": (192, ["task", "candidate_id", "scope", "meta_fold"]),
        "phase4_rescue_benchmark_predictions.csv": (2340, ["task", "area_id", "date"]),
        "phase4_rescue_metrics.csv": (16, ["split", "task", "method"]),
        "phase4_rescue_feature_manifest.csv": (303, ["task", "method", "model_component", "feature_order"]),
        "phase4_rescue_model_audit.csv": (5, ["task", "method", "model_component"]),
        "phase4_rescue_bootstrap_draws.csv": (12000, ["task", "method", "bootstrap_id"]),
        "phase4_rescue_bootstrap_summary.csv": (6, ["task", "method"]),
        "phase4_rescue_confusion_matrices.csv": (200, ["task", "method", "actual_phase", "predicted_phase"]),
        "phase4_rescue_main_comparison_metrics.csv": (6, ["task", "method"]),
        "phase4_rescue_source_audit.csv": (1, []),
    }
    for name, (expected_rows, keys) in checks.items():
        frame = pd.read_csv(directory / name, **read_options)
        if len(frame) != expected_rows:
            raise ValueError(f"{name} has {len(frame)} rows, expected {expected_rows}.")
        if keys and frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate contract keys.")
    threshold = pd.read_csv(directory / "phase4_rescue_threshold_search.csv", **read_options)
    if threshold.duplicated(["task", "method", "threshold_rank"]).any():
        raise ValueError("Threshold search contains duplicate keys.")
    for task in TASK_ORDER:
        for method in ("single_score", "xgboost"):
            selected = threshold.loc[threshold["task"].eq(task) & threshold["method"].eq(method), "selected"]
            if int(selected.astype(bool).sum()) != 1:
                raise ValueError("Threshold selected-row contract failed.")
        direct = threshold.loc[threshold["task"].eq(task) & threshold["method"].eq("direct_phase34_xgboost")]
        if len(direct) != 1 or bool(direct.iloc[0]["selected"]):
            raise ValueError("Direct threshold row must be fixed and unselected.")
    main = pd.read_csv(directory / "phase4_rescue_main_comparison_metrics.csv", **read_options)
    invariant_columns = [
        "phase3plus_precision_delta_from_base", "phase3plus_recall_delta_from_base",
        "phase3plus_r2_delta_from_base",
    ]
    if not (main[invariant_columns].to_numpy(dtype=float) == 0.0).all():
        raise ValueError("Main comparison invariant deltas are not exact zero.")
    configuration = json.loads((directory / "phase4_rescue_configuration.json").read_text(encoding="utf-8"))
    required_configuration_keys = {
        "schema_version", "freeze_id", "environment", "decisions", "folds", "features",
        "parameter_grid", "direct_baseline", "selected_models", "thresholds",
        "acceptance_gate", "bootstrap", "main_comparison", "figure_exports", "artifact_hashes",
    }
    if set(configuration) != required_configuration_keys or len(configuration["artifact_hashes"]) != 18:
        raise ValueError("Configuration JSON contract failed.")
    for name, expected_hash in configuration["artifact_hashes"].items():
        if file_sha256(directory / name) != expected_hash:
            raise ValueError(f"Configuration artifact hash mismatch for {name}.")
    source_audit = pd.read_csv(directory / "phase4_rescue_source_audit.csv", **read_options).iloc[0]
    manifest = json.loads(source_audit["artifact_manifest_json"])
    if len(manifest) != 19 or source_audit["artifact_manifest_sha256"] != manifest_sha256(manifest):
        raise ValueError("Source-audit artifact manifest contract failed.")
    for name, expected_hash in manifest.items():
        if file_sha256(directory / name) != expected_hash:
            raise ValueError(f"Source-audit artifact hash mismatch for {name}.")
    pdf_payload = (directory / "phase4_rescue_main_comparison.pdf").read_bytes()
    if not pdf_payload.startswith(b"%PDF") or b"/Subtype /Image" in pdf_payload:
        raise ValueError("Main comparison PDF is not a vector PDF contract output.")
    if b"CreationDate" in pdf_payload or b"ModDate" in pdf_payload:
        raise ValueError("Main comparison PDF contains nondeterministic date metadata.")
    width, height, dpi = _read_png_contract(directory / "phase4_rescue_main_comparison.png")
    if (width, height) != (7200, 3900) or dpi is None or abs(dpi - 600.0) > 1.0:
        raise ValueError(f"Main PNG contract failed: width={width} height={height} dpi={dpi}")


def _replace_with_retry(source: Path, destination: Path, attempts: int = 20) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.25)


def finalize_source_audit(directory: Path, protected_after: str) -> None:
    path = directory / "phase4_rescue_source_audit.csv"
    read_options = {
        "na_values": ["<NA>"],
        "keep_default_na": True,
        "float_precision": "round_trip",
    }
    source_audit = pd.read_csv(path, **read_options)
    if len(source_audit) != 1:
        raise ValueError("Source audit must contain exactly one row before finalization.")
    protected_before = str(source_audit.loc[0, "protected_manifest_sha256_before"])
    source_audit.loc[0, "protected_manifest_sha256_after"] = str(protected_after)
    source_audit.loc[0, "protected_manifest_match"] = protected_before == str(protected_after)
    temporary = directory / ".phase4_rescue_source_audit.csv.tmp"
    try:
        _write_csv(source_audit, temporary, SOURCE_AUDIT_COLUMNS)
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_formal_artifacts(
    staging_dir: Path,
    environment: Mapping[str, object],
    protected_before: str,
    oof: pd.DataFrame,
    model_selection: pd.DataFrame,
    threshold_search: pd.DataFrame,
    benchmark: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    model_audit: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    confusion: pd.DataFrame,
    main_metrics: pd.DataFrame,
    rescue_models: Mapping[str, Mapping[str, object]],
    selected_thresholds: Mapping[str, Mapping[str, pd.Series]],
    accepted_by_task: Mapping[str, bool],
    direct_models: Mapping[str, object],
) -> None:
    csv_payloads = (
        (oof, "phase4_rescue_oof_predictions.csv", OOF_COLUMNS),
        (model_selection, "phase4_rescue_model_selection.csv", MODEL_SELECTION_COLUMNS),
        (threshold_search, "phase4_rescue_threshold_search.csv", THRESHOLD_COLUMNS),
        (benchmark, "phase4_rescue_benchmark_predictions.csv", BENCHMARK_COLUMNS),
        (metrics, "phase4_rescue_metrics.csv", METRIC_COLUMNS),
        (feature_manifest, "phase4_rescue_feature_manifest.csv", FEATURE_MANIFEST_COLUMNS),
        (model_audit, "phase4_rescue_model_audit.csv", MODEL_AUDIT_COLUMNS),
        (bootstrap_draws, "phase4_rescue_bootstrap_draws.csv", BOOTSTRAP_DRAW_COLUMNS),
        (bootstrap_summary, "phase4_rescue_bootstrap_summary.csv", BOOTSTRAP_SUMMARY_COLUMNS),
        (confusion, "phase4_rescue_confusion_matrices.csv", CONFUSION_COLUMNS),
        (main_metrics, "phase4_rescue_main_comparison_metrics.csv", MAIN_METRIC_COLUMNS),
    )
    for frame, name, columns in csv_payloads:
        _write_csv(frame, staging_dir / name, columns)
    render_main_comparison_figure(
        confusion,
        main_metrics,
        staging_dir / "phase4_rescue_main_comparison.pdf",
        staging_dir / "phase4_rescue_main_comparison.png",
    )
    preconfiguration_names = [
        name for name in EXPECTED_ARTIFACTS
        if name not in {"phase4_rescue_configuration.json", "phase4_rescue_source_audit.csv"}
    ]
    artifact_hashes_18 = {name: file_sha256(staging_dir / name) for name in preconfiguration_names}
    configuration = _configuration_payload(
        rescue_models, selected_thresholds, accepted_by_task, artifact_hashes_18
    )
    _write_json(configuration, staging_dir / "phase4_rescue_configuration.json")
    artifact_hashes_19 = {
        name: file_sha256(staging_dir / name)
        for name in EXPECTED_ARTIFACTS
        if name != "phase4_rescue_source_audit.csv"
    }
    source_row = _source_audit_row(
        environment,
        protected_before,
        protected_before,
        direct_models,
        model_audit,
        benchmark,
        artifact_hashes_19,
    )
    source_audit = pd.DataFrame([source_row], columns=SOURCE_AUDIT_COLUMNS)
    _write_csv(source_audit, staging_dir / "phase4_rescue_source_audit.csv", SOURCE_AUDIT_COLUMNS)
    validate_artifact_contract(staging_dir)


def run_generation(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_dir = Path(output_dir)
    validate_generation_target(output_dir)
    verify_frozen_source_hashes()
    environment = assert_formal_environment()
    protected_before = protected_artifact_manifest_sha256(output_dir if output_dir.exists() else None)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(prefix=".phase4_rescue_classifier-staging-", dir=output_dir.parent)
    )
    published = False
    try:
        forecasting, nowcasting, layer1_features = load_prepared_inputs()
        oof = generate_base_oof_predictions(forecasting, nowcasting, layer1_features)
        feature_manifest = build_feature_manifest(forecasting, nowcasting, layer1_features)
        model_selection, task_results = fit_and_select_auxiliary_models(oof)
        threshold_search, selected_thresholds = build_threshold_searches(task_results)
        rescue_models = refit_selected_auxiliary_models(
            task_results, selected_thresholds, staging_path
        )
        direct_models = fit_direct_baselines(
            forecasting, nowcasting, layer1_features, staging_path
        )
        frozen_base = load_and_validate_frozen_base_predictions()
        direct_benchmark = load_direct_benchmark_matrices(
            forecasting, nowcasting, layer1_features, frozen_base
        )
        benchmark, rescue_benchmark_matrices = build_benchmark_predictions(
            frozen_base, direct_models, direct_benchmark, rescue_models, selected_thresholds
        )
        meta_frames = build_meta_validation_frames(
            task_results, selected_thresholds, forecasting
        )
        metrics, accepted_by_task, _ = calculate_metrics(
            meta_frames, benchmark, task_results, rescue_models, selected_thresholds
        )
        confusion = build_confusion_source(benchmark)
        main_metrics = build_main_comparison_metrics(
            metrics, accepted_by_task, rescue_models
        )
        bootstrap_draws, bootstrap_summary = generate_bootstrap(benchmark)
        model_audit = build_model_audit(
            rescue_models,
            direct_models,
            direct_benchmark,
            rescue_benchmark_matrices,
            layer1_features,
        )
        write_formal_artifacts(
            staging_path,
            environment,
            protected_before,
            oof,
            model_selection,
            threshold_search,
            benchmark,
            metrics,
            feature_manifest,
            model_audit,
            bootstrap_draws,
            bootstrap_summary,
            confusion,
            main_metrics,
            rescue_models,
            selected_thresholds,
            accepted_by_task,
            direct_models,
        )
        if output_dir.exists():
            output_dir.rmdir()
        _replace_with_retry(staging_path, output_dir)
        published = True
        protected_after = protected_artifact_manifest_sha256(output_dir)
        if protected_after != protected_before:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise RuntimeError("Protected produced_graph artifacts changed during generation.")
        finalize_source_audit(output_dir, protected_after)
        validate_artifact_contract(output_dir)
        return {
            "output_dir": output_dir,
            "accepted_by_task": accepted_by_task,
            "selected_thresholds": {
                task: {
                    method: float(selected_thresholds[task][method]["threshold"])
                    for method in ("single_score", "xgboost")
                }
                for task in TASK_ORDER
            },
            "no_rescue_by_task": {
                task: bool(rescue_models[task]["no_rescue"]) for task in TASK_ORDER
            },
        }
    except BaseException:
        if staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)
        if published and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    result = run_generation(arguments.output_dir)
    print(f"output_dir: {result['output_dir']}")
    print("accepted_by_task: " + json.dumps(result["accepted_by_task"], sort_keys=True))
    print("selected_thresholds: " + json.dumps(result["selected_thresholds"], sort_keys=True))
    print("no_rescue_by_task: " + json.dumps(result["no_rescue_by_task"], sort_keys=True))


if __name__ == "__main__":
    main()
