"""Generate the isolated Direct Phase-3 versus Phase-4/5 rescue experiment.

The implementation follows the frozen R1-R12 contract in
docs/superpowers/specs/2026-08-17-direct-phase3-vs-phase45-rescue-design.md.
The completed Phase-4 rescue experiment is read-only evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterable, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache-direct-phase45-rescue"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import patches as mpl_patches
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score
import xgboost as xgb

import generate_phase4_rescue_classifier as predecessor


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE_DIR = REPO_ROOT / "2.Source Code"
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
PRODUCED_GRAPH_DIR = SOURCE_CODE_DIR / "produced_graph"
DEFAULT_OUTPUT_DIR = PRODUCED_GRAPH_DIR / "direct_phase3_vs_phase45_rescue"
PREDECESSOR_DIR = PRODUCED_GRAPH_DIR / "phase4_rescue_classifier"
PREDECESSOR_OOF = PREDECESSOR_DIR / "phase4_rescue_oof_predictions.csv"
PREDECESSOR_BENCHMARK = PREDECESSOR_DIR / "phase4_rescue_benchmark_predictions.csv"
PREDECESSOR_SOURCE_AUDIT = PREDECESSOR_DIR / "phase4_rescue_source_audit.csv"
PREDECESSOR_GENERATOR = SOURCE_CODE_DIR / "generate_phase4_rescue_classifier.py"
CONTEMPORANEOUS_RESCUE_DIR = (
    PRODUCED_GRAPH_DIR / "direct_phase3_vs_phase45_rescue_contemporaneous"
)
CONTEMPORANEOUS_RESCUE_PREFIX = (
    "direct_phase3_vs_phase45_rescue_contemporaneous_"
)
CONTEMPORANEOUS_GATE_METRICS = (
    CONTEMPORANEOUS_RESCUE_DIR
    / f"{CONTEMPORANEOUS_RESCUE_PREFIX}gate_pooled_metrics.csv"
)
CONTEMPORANEOUS_BINARY_CONFUSION = (
    CONTEMPORANEOUS_RESCUE_DIR
    / f"{CONTEMPORANEOUS_RESCUE_PREFIX}binary_confusion_matrices.csv"
)
CONTEMPORANEOUS_FIVE_CLASS_CONFUSION = (
    CONTEMPORANEOUS_RESCUE_DIR
    / f"{CONTEMPORANEOUS_RESCUE_PREFIX}five_class_confusion_matrices.csv"
)
CONTEMPORANEOUS_CONFIGURATION = (
    CONTEMPORANEOUS_RESCUE_DIR
    / f"{CONTEMPORANEOUS_RESCUE_PREFIX}configuration.json"
)
CONTEMPORANEOUS_SOURCE_AUDIT = (
    CONTEMPORANEOUS_RESCUE_DIR
    / f"{CONTEMPORANEOUS_RESCUE_PREFIX}source_audit.csv"
)

CUTOFF = predecessor.CUTOFF
NOWCAST_FEATURES = predecessor.NOWCAST_FEATURES
KEY_COLUMNS = predecessor.KEY_COLUMNS
TemporalFold = predecessor.TemporalFold
BASE_FOLDS = predecessor.BASE_FOLDS

FREEZE_ID = "main-result-figure1-v1"
EXPERIMENT_ID = "direct_phase3_vs_phase45_rescue_r1_r12"
REFERENCE_ENVIRONMENT_ID = "windows_py3113_xgb203_direct_seed0_njobs1"
EVALUATION_POPULATION_ID = "temporal_test_2022_reconstructed_phase_1170"
PREFIX = "direct_phase3_vs_phase45_rescue_"

EXPECTED_PREDECESSOR_SOURCE_AUDIT_SHA256 = (
    "d8e9f1181e9df8a890a038c4e2005259b959bad839313a40adf62860ebc32066"
)
EXPECTED_PREDECESSOR_OOF_SHA256 = (
    "dea864ab8a1e3c5785748b3b8ebec8eb3151585f21d664e1169fd116de1b7cd9"
)
EXPECTED_PREDECESSOR_BENCHMARK_SHA256 = (
    "27a51295b1512503c71f60eb0313925d5cc1b01b93f048c34909cd0e1a3be132"
)
EXPECTED_PREDECESSOR_GENERATOR_SHA256 = (
    "6a7a6318e1450cd630ac0f3a76aa4a96548d01c1147b4ca07e37a0cc21a9cbc1"
)
EXPECTED_BENCHMARK_KEY_SHA256 = (
    "288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2"
)

EXPECTED_SOURCE_ROWS = 5575
EXPECTED_PRE2022_ROWS = 4405
EXPECTED_BENCHMARK_ROWS = 1170
EXPECTED_BENCHMARK_AREAS = 646
EXPECTED_BENCHMARK_COUNTRIES = 27
EXPECTED_PREDECESSOR_OOF_ROWS = 7520
EXPECTED_OOF_ROWS_PER_TASK = 3760
EXPECTED_PHASE345_ROWS = 2530
EXPECTED_PHASE3_ROWS = 2149
EXPECTED_PHASE4_ROWS = 369
EXPECTED_PHASE5_ROWS = 12
EXPECTED_PHASE45_ROWS = 381
EXPECTED_OOF_ROWS = 22560
EXPECTED_BENCHMARK_ROWS_LONG = 11700
EXPECTED_OOF_STABILITY_ROWS = 30
EXPECTED_COUNTRY_METRIC_ROWS = 270
EXPECTED_BINARY_CONFUSION_ROWS = 80
EXPECTED_FIVE_CLASS_CONFUSION_ROWS = 250
EXPECTED_FEATURE_MANIFEST_ROWS = 281
EXPECTED_MODEL_AUDIT_ROWS = 45
EXPECTED_BOOTSTRAP_DRAW_ROWS = 16000
EXPECTED_BOOTSTRAP_SUMMARY_ROWS = 8
MAX_THRESHOLD_FRONTIER_ROWS = 15507

BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_RANDOM_STATE = 0
ALL_PHASES = (1, 2, 3, 4, 5)
TASK_ORDER = ("Forecasting", "Nowcasting")
CANDIDATE_ORDER = ("unweighted", "sqrt_balance", "full_balance")
METHOD_ORDER = (
    "frozen_base",
    "legacy_direct_exact_phase4_050",
    "direct_phase45_unweighted",
    "direct_phase45_sqrt_balance",
    "direct_phase45_full_balance",
)
CANDIDATE_METHODS = {
    candidate: f"direct_phase45_{candidate}" for candidate in CANDIDATE_ORDER
}
METHOD_CANDIDATES = {method: candidate for candidate, method in CANDIDATE_METHODS.items()}
FIGURE_TASK_ORDER = (*TASK_ORDER, "Contemporaneous")
FIGURE_METHOD_ORDER = tuple(CANDIDATE_METHODS[candidate] for candidate in CANDIDATE_ORDER)
FIGURE_POPULATIONS = {
    "Forecasting": {
        "scope": "benchmark",
        "n": EXPECTED_BENCHMARK_ROWS,
        "protocol": "fixed 2022 temporal holdout",
    },
    "Nowcasting": {
        "scope": "benchmark",
        "n": EXPECTED_BENCHMARK_ROWS,
        "protocol": "fixed 2022 temporal holdout",
    },
    "Contemporaneous": {
        "scope": "full_oof",
        "n": EXPECTED_SOURCE_ROWS,
        "protocol": "seed-0 random five-fold row-CV full OOF",
    },
}

CSV_BASENAMES = (
    f"{PREFIX}oof_predictions.csv",
    f"{PREFIX}threshold_frontier.csv",
    f"{PREFIX}selected_policies.csv",
    f"{PREFIX}oof_stability.csv",
    f"{PREFIX}benchmark_predictions.csv",
    f"{PREFIX}gate_pooled_metrics.csv",
    f"{PREFIX}benchmark_pooled_metrics.csv",
    f"{PREFIX}country_metrics.csv",
    f"{PREFIX}benchmark_country_macro_metrics.csv",
    f"{PREFIX}binary_confusion_matrices.csv",
    f"{PREFIX}five_class_confusion_matrices.csv",
    f"{PREFIX}feature_manifest.csv",
    f"{PREFIX}model_audit.csv",
    f"{PREFIX}bootstrap_draws.csv",
    f"{PREFIX}bootstrap_summary.csv",
    f"{PREFIX}source_audit.csv",
)
MODEL_BASENAMES = tuple(
    f"{PREFIX}{candidate}_{component}_model.json"
    for candidate in CANDIDATE_ORDER
    for component in ("forecasting", "nowcasting_layer1", "nowcasting_layer2")
)
FIGURE_BASENAMES = (
    f"{PREFIX}main_comparison.pdf",
    f"{PREFIX}main_comparison.png",
    f"{PREFIX}binary_confusion_atlas.pdf",
    f"{PREFIX}binary_confusion_atlas.png",
)
CONFIGURATION_BASENAME = f"{PREFIX}configuration.json"
EXPECTED_ARTIFACTS = (
    *CSV_BASENAMES[:-1],
    CONFIGURATION_BASENAME,
    *MODEL_BASENAMES,
    *FIGURE_BASENAMES,
    CSV_BASENAMES[-1],
)

OOF_COLUMNS = (
    "task",
    "candidate_id",
    "base_oof_fold",
    "validation_year",
    "area_id",
    "date",
    "country_code_3",
    "source_overall_phase",
    "reconstructed_overall_phase",
    "severe_rescue_target",
    "base_overall_phase_pred",
    "in_base_phase3_gate",
    "direct_layer1_score",
    "direct_layer2_residual_score",
    "direct_phase45_score",
    "fold_training_years",
    "training_rows",
    "training_negative_count",
    "training_positive_count",
    "class_ratio",
    "positive_row_weight",
)
THRESHOLD_COLUMNS = (
    "task",
    "candidate_id",
    "threshold_rank",
    "threshold",
    "threshold_source",
    "is_above_max_reference",
    "gate_rows",
    "positive_support",
    "negative_support",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "phase45_precision",
    "phase45_recall",
    "phase45_f1",
    "phase45_f2",
    "true_rescue_actual_phase4",
    "true_rescue_actual_phase5",
    "false_promotion_actual_phase1",
    "false_promotion_actual_phase2",
    "false_promotion_actual_phase3",
    "false_promotion_actual_phase12",
    "total_promotions",
    "false_promotions_per_100_gate_rows",
    "within_candidate_selected",
    "primary_selected",
    "selection_status",
)
SELECTED_POLICY_COLUMNS = (
    "task",
    "candidate_id",
    "method",
    "selection_status",
    "threshold_rank",
    "threshold",
    "is_above_max_reference",
    "phase45_precision",
    "phase45_recall",
    "phase45_f1",
    "phase45_f2",
    "false_promotion_actual_phase12",
    "total_promotions",
    "primary_selected",
    "final_training_negative_count",
    "final_training_positive_count",
    "final_class_ratio",
    "final_positive_row_weight",
)
OOF_STABILITY_COLUMNS = (
    "task",
    "candidate_id",
    "method",
    "period_id",
    "validation_year",
    "threshold",
    "gate_rows",
    "positive_support",
    "negative_support",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "phase45_precision",
    "phase45_recall",
    "phase45_f1",
    "phase45_f2",
    "true_rescue_actual_phase4",
    "true_rescue_actual_phase5",
    "false_promotion_actual_phase1",
    "false_promotion_actual_phase2",
    "false_promotion_actual_phase3",
    "total_promotions",
)
BENCHMARK_COLUMNS = (
    "task",
    "method",
    "candidate_id",
    "method_role",
    "selection_status",
    "primary_selected",
    "area_id",
    "date",
    "country_code_3",
    "source_overall_phase",
    "reconstructed_overall_phase",
    "severe_rescue_target",
    "phase2_test",
    "phase3_test",
    "phase4_test",
    "phase5_test",
    "phase2_pred_rounded",
    "phase3_pred_rounded",
    "phase4_pred_rounded",
    "phase5_pred_rounded",
    "base_overall_phase_pred",
    "in_base_phase3_gate",
    "direct_layer1_score",
    "direct_layer2_residual_score",
    "direct_phase45_score",
    "threshold",
    "triggered",
    "final_overall_phase_pred",
)
GATE_METRIC_COLUMNS = (
    "task",
    "method",
    "method_role",
    "selection_status",
    "primary_selected",
    "n_rows",
    "positive_support",
    "negative_support",
    "actual_phase4_support",
    "actual_phase5_support",
    "predicted_positive_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "phase45_precision",
    "phase45_recall",
    "phase45_f1",
    "phase45_f2",
    "phase45_pr_auc",
    "true_rescue_actual_phase4",
    "true_rescue_actual_phase5",
    "false_promotion_actual_phase1",
    "false_promotion_actual_phase2",
    "false_promotion_actual_phase3",
    "false_promotion_actual_phase12",
    "total_promotions",
    "false_promotions_per_100_gate_rows",
)
BENCHMARK_METRIC_CORE_COLUMNS = (
    "n_rows",
    "n_gate",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "ordinal_mae",
    "phase3_precision",
    "phase3_recall",
    "phase3_f1",
    "phase4_precision",
    "phase4_recall",
    "phase4_f1",
    "phase5_precision",
    "phase5_recall",
    "phase5_f1",
    "phase45_precision",
    "phase45_recall",
    "phase45_f1",
    "phase45_f2",
    "phase3plus_precision",
    "phase3plus_recall",
    "phase3plus_r2",
    "changed_3_to_4",
    "exact_phase4_rescues",
    "actual_phase5_promoted_to_phase4",
    "false_promotion_actual_phase1",
    "false_promotion_actual_phase2",
    "false_promotion_actual_phase3",
)
BENCHMARK_METRIC_COLUMNS = (
    "task",
    "method",
    "method_role",
    "selection_status",
    "primary_selected",
    *BENCHMARK_METRIC_CORE_COLUMNS,
    "phase3plus_precision_delta_from_base",
    "phase3plus_recall_delta_from_base",
    "phase3plus_r2_delta_from_base",
)
COUNTRY_METRIC_COLUMNS = (
    "task",
    "method",
    "country_code_3",
    "method_role",
    "selection_status",
    "primary_selected",
    *BENCHMARK_METRIC_CORE_COLUMNS,
)
MACRO_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "ordinal_mae",
    "phase3_precision",
    "phase3_recall",
    "phase3_f1",
    "phase4_precision",
    "phase4_recall",
    "phase4_f1",
    "phase5_precision",
    "phase5_recall",
    "phase5_f1",
    "phase45_precision",
    "phase45_recall",
    "phase45_f1",
    "phase45_f2",
    "phase3plus_precision",
    "phase3plus_recall",
    "phase3plus_r2",
)
COUNTRY_MACRO_COLUMNS = (
    "task",
    "method",
    "method_role",
    "selection_status",
    "primary_selected",
    "countries_total",
    *tuple(
        item
        for metric in MACRO_METRICS
        for item in (metric, f"{metric}_countries_defined")
    ),
)
BINARY_CONFUSION_COLUMNS = (
    "population_scope",
    "task",
    "method",
    "actual_binary",
    "predicted_binary",
    "count",
    "actual_row_total",
    "actual_row_share",
)
FIVE_CLASS_CONFUSION_COLUMNS = (
    "task",
    "method",
    "actual_phase",
    "predicted_phase",
    "count",
    "actual_row_total",
    "actual_row_share",
)
FEATURE_MANIFEST_COLUMNS = (
    "task",
    "model_component",
    "feature_order",
    "feature_name",
    "source_dtype",
    "feature_definition",
    "native_missingness_preserved",
    "feature_order_sha256",
)
MODEL_AUDIT_COLUMNS = (
    "task",
    "candidate_id",
    "fit_scope",
    "fold_id",
    "model_component",
    "estimator_class",
    "objective",
    "target_definition",
    "run_status",
    "training_years",
    "validation_year",
    "training_rows",
    "training_negative_count",
    "training_positive_count",
    "class_ratio",
    "negative_row_weight",
    "positive_row_weight",
    "scale_pos_weight",
    "sample_weight_sha256",
    "sample_weight_sum",
    "feature_count",
    "feature_order_sha256",
    "parameter_sha256",
    "random_state",
    "n_jobs",
    "training_key_sha256",
    "target_sha256",
    "training_matrix_sha256",
    "training_missingness_sha256",
    "training_matrix_with_target_sha256",
    "layer1_training_score_sha256",
    "residual_target_sha256",
    "validation_rows",
    "validation_key_sha256",
    "validation_matrix_sha256",
    "validation_missingness_sha256",
    "validation_score_sha256",
    "model_path",
    "model_sha256",
)
BOOTSTRAP_METRICS = (
    "phase45_precision",
    "phase45_recall",
    "phase45_f1",
    "phase45_f2",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "ordinal_mae",
)
BOOTSTRAP_DRAW_COLUMNS = (
    "task",
    "method",
    "bootstrap_id",
    "sampled_country_count",
    "sample_row_count",
    "sampled_countries_sha256",
    *tuple(f"{metric}_delta" for metric in BOOTSTRAP_METRICS),
)
BOOTSTRAP_SUMMARY_COLUMNS = (
    "task",
    "method",
    "repetitions",
    "ci_level",
    "interval_method",
    *tuple(
        item
        for metric in BOOTSTRAP_METRICS
        for item in (
            f"{metric}_delta_lower",
            f"{metric}_delta_upper",
            f"{metric}_finite_draws",
        )
    ),
)
SOURCE_AUDIT_COLUMNS = (
    "run_status",
    "experiment_id",
    "freeze_id",
    "reference_environment_id",
    "evaluation_population_id",
    "source_rows",
    "pre2022_rows",
    "direct_training_rows",
    "direct_phase3_count",
    "direct_phase4_count",
    "direct_phase5_count",
    "direct_phase45_count",
    "oof_rows",
    "benchmark_rows",
    "benchmark_areas",
    "benchmark_countries",
    "benchmark_key_sha256",
    "direct_training_key_sha256",
    "direct_target_sha256",
    "predecessor_directory",
    "predecessor_source_audit_sha256",
    "predecessor_artifact_manifest_json",
    "predecessor_artifact_manifest_sha256",
    "contemporaneous_rescue_directory",
    "contemporaneous_rescue_artifact_manifest_json",
    "contemporaneous_rescue_artifact_manifest_sha256",
    "generator_path",
    "generator_sha256",
    "platform_family",
    "python_version",
    "numpy_version",
    "pandas_version",
    "scipy_version",
    "sklearn_version",
    "xgboost_version",
    "matplotlib_version",
    "xgboost_dll_sha256",
    "direct_random_state",
    "direct_n_jobs",
    "outer_workers",
    "bootstrap_repetitions",
    "bootstrap_random_state",
    "selected_policies_json",
    "protected_manifest_sha256_before",
    "protected_manifest_sha256_after",
    "protected_manifest_match",
    "artifact_manifest_json",
    "artifact_manifest_sha256",
)

CSV_SCHEMAS = {
    f"{PREFIX}oof_predictions.csv": OOF_COLUMNS,
    f"{PREFIX}threshold_frontier.csv": THRESHOLD_COLUMNS,
    f"{PREFIX}selected_policies.csv": SELECTED_POLICY_COLUMNS,
    f"{PREFIX}oof_stability.csv": OOF_STABILITY_COLUMNS,
    f"{PREFIX}benchmark_predictions.csv": BENCHMARK_COLUMNS,
    f"{PREFIX}gate_pooled_metrics.csv": GATE_METRIC_COLUMNS,
    f"{PREFIX}benchmark_pooled_metrics.csv": BENCHMARK_METRIC_COLUMNS,
    f"{PREFIX}country_metrics.csv": COUNTRY_METRIC_COLUMNS,
    f"{PREFIX}benchmark_country_macro_metrics.csv": COUNTRY_MACRO_COLUMNS,
    f"{PREFIX}binary_confusion_matrices.csv": BINARY_CONFUSION_COLUMNS,
    f"{PREFIX}five_class_confusion_matrices.csv": FIVE_CLASS_CONFUSION_COLUMNS,
    f"{PREFIX}feature_manifest.csv": FEATURE_MANIFEST_COLUMNS,
    f"{PREFIX}model_audit.csv": MODEL_AUDIT_COLUMNS,
    f"{PREFIX}bootstrap_draws.csv": BOOTSTRAP_DRAW_COLUMNS,
    f"{PREFIX}bootstrap_summary.csv": BOOTSTRAP_SUMMARY_COLUMNS,
    f"{PREFIX}source_audit.csv": SOURCE_AUDIT_COLUMNS,
}


def file_sha256(path: Path) -> str:
    return predecessor.file_sha256(Path(path))


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_sha256(value: object) -> str:
    return bytes_sha256(
        (
            json.dumps(
                json_safe(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def manifest_sha256(hashes: Mapping[str, str]) -> str:
    payload = "".join(f"{name}\t{hashes[name]}\n" for name in sorted(hashes))
    return bytes_sha256(payload.encode("utf-8"))


def relative_path(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value is pd.NA:
        return None
    return value


def canonical_key_frame(data: pd.DataFrame) -> pd.DataFrame:
    return predecessor.canonical_key_frame(data)


def canonical_key_sha256(data: pd.DataFrame) -> str:
    return predecessor.canonical_key_sha256(data)


def canonical_series_sha256(series: pd.Series, name: str = "value") -> str:
    return predecessor.canonical_series_sha256(series, name)


def canonical_matrix_sha256(matrix: pd.DataFrame) -> str:
    return predecessor.canonical_matrix_sha256(matrix)


def canonical_missingness_sha256(matrix: pd.DataFrame) -> str:
    return predecessor.canonical_missingness_sha256(matrix)


def matrix_with_target_sha256(matrix: pd.DataFrame, target: pd.Series) -> str:
    return predecessor.matrix_with_target_sha256(matrix, target)


def _require_columns(data: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    predecessor._require_columns(data, columns, name)


def _require_unique_keys(data: pd.DataFrame, name: str) -> None:
    predecessor._require_unique_keys(data, name)


def validate_generation_target(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Formal output directory must be absent or empty: {output_dir}"
        )


def protected_artifact_manifest_sha256(
    excluded_paths: Sequence[Path] = (),
) -> str:
    excluded_roots = {Path(path).resolve() for path in excluded_paths}
    hashes: dict[str, str] = {}
    if not PRODUCED_GRAPH_DIR.exists():
        return manifest_sha256(hashes)
    for path in sorted(
        PRODUCED_GRAPH_DIR.rglob("*"), key=lambda value: value.as_posix()
    ):
        if not path.is_file():
            continue
        excluded = False
        for root in excluded_roots:
            try:
                path.resolve().relative_to(root)
                excluded = True
                break
            except ValueError:
                continue
        if excluded:
            continue
        hashes[path.relative_to(PRODUCED_GRAPH_DIR).as_posix()] = file_sha256(path)
    return manifest_sha256(hashes)


def assert_formal_environment() -> dict[str, object]:
    return predecessor.assert_formal_environment()


def verify_predecessor_artifacts() -> dict[str, str]:
    required = (
        PREDECESSOR_SOURCE_AUDIT,
        PREDECESSOR_OOF,
        PREDECESSOR_BENCHMARK,
        PREDECESSOR_GENERATOR,
    )
    missing = [relative_path(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing frozen predecessor evidence: {missing}")
    fixed = {
        PREDECESSOR_SOURCE_AUDIT: EXPECTED_PREDECESSOR_SOURCE_AUDIT_SHA256,
        PREDECESSOR_OOF: EXPECTED_PREDECESSOR_OOF_SHA256,
        PREDECESSOR_BENCHMARK: EXPECTED_PREDECESSOR_BENCHMARK_SHA256,
        PREDECESSOR_GENERATOR: EXPECTED_PREDECESSOR_GENERATOR_SHA256,
    }
    failures = [
        f"{relative_path(path)}={file_sha256(path)} expected={expected}"
        for path, expected in fixed.items()
        if file_sha256(path) != expected
    ]
    if failures:
        raise RuntimeError("Frozen predecessor hash gate failed: " + "; ".join(failures))
    audit = pd.read_csv(
        PREDECESSOR_SOURCE_AUDIT,
        na_values=["<NA>"],
        keep_default_na=True,
        float_precision="round_trip",
    )
    if len(audit) != 1:
        raise RuntimeError("Frozen predecessor source audit must contain one row.")
    row = audit.iloc[0]
    if (
        str(row["run_status"]) != "complete"
        or str(row["freeze_id"]) != FREEZE_ID
        or not bool(row["protected_manifest_match"])
    ):
        raise RuntimeError("Frozen predecessor source-audit status drifted.")
    manifest = json.loads(str(row["artifact_manifest_json"]))
    expected_files = set(manifest) | {PREDECESSOR_SOURCE_AUDIT.name}
    observed_files = {
        path.name for path in PREDECESSOR_DIR.iterdir() if path.is_file()
    }
    observed_dirs = [path.name for path in PREDECESSOR_DIR.iterdir() if path.is_dir()]
    if observed_dirs or observed_files != expected_files:
        raise RuntimeError("Frozen predecessor directory manifest drifted.")
    for name, expected_hash in manifest.items():
        actual = file_sha256(PREDECESSOR_DIR / name)
        if actual != expected_hash:
            raise RuntimeError(
                f"Frozen predecessor artifact hash mismatch for {name}: "
                f"{actual} expected={expected_hash}"
            )
    if manifest_sha256(manifest) != str(row["artifact_manifest_sha256"]):
        raise RuntimeError("Frozen predecessor manifest digest drifted.")
    return dict(sorted(manifest.items()))


def verify_frozen_sources() -> dict[str, str]:
    predecessor.verify_frozen_source_hashes()
    return verify_predecessor_artifacts()


def load_contemporaneous_figure_evidence() -> dict[str, object]:
    required = (
        CONTEMPORANEOUS_GATE_METRICS,
        CONTEMPORANEOUS_BINARY_CONFUSION,
        CONTEMPORANEOUS_FIVE_CLASS_CONFUSION,
        CONTEMPORANEOUS_CONFIGURATION,
        CONTEMPORANEOUS_SOURCE_AUDIT,
    )
    missing = [relative_path(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Missing Contemporaneous Phase-4/5 rescue figure evidence: {missing}"
        )
    configuration = json.loads(
        CONTEMPORANEOUS_CONFIGURATION.read_text(encoding="utf-8")
    )
    decisions = configuration.get("decisions", {})
    if (
        configuration.get("experiment_id")
        != "direct_phase3_vs_phase45_rescue_contemporaneous_random5fold"
        or decisions.get("evaluation_protocol") != "random_5fold_row_cv"
        or decisions.get("evaluation_population") != "random_5fold_full_oof_5575"
        or decisions.get("allowed_action") != "base_phase3_only_to_phase4"
        or decisions.get("direct_comparison_with_temporal_holdout_authorized")
        is not False
    ):
        raise RuntimeError("Contemporaneous rescue configuration contract drifted.")
    configuration_manifest = configuration.get("artifact_hashes", {})
    if not configuration_manifest:
        raise RuntimeError("Contemporaneous rescue configuration manifest is empty.")
    for name, expected_hash in configuration_manifest.items():
        path = CONTEMPORANEOUS_RESCUE_DIR / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise RuntimeError(
                f"Contemporaneous rescue configuration hash mismatch for {name}."
            )

    source_audit = pd.read_csv(
        CONTEMPORANEOUS_SOURCE_AUDIT,
        na_values=["<NA>"],
        keep_default_na=True,
        float_precision="round_trip",
    )
    if len(source_audit) != 1:
        raise RuntimeError("Contemporaneous rescue source audit must contain one row.")
    audit_row = source_audit.iloc[0]
    if (
        str(audit_row["run_status"]) != "complete"
        or str(audit_row["evaluation_protocol"]) != "random_5fold_row_cv"
        or str(audit_row["evaluation_population"])
        != "random_5fold_full_oof_5575"
        or not bool(audit_row["protected_manifest_match"])
    ):
        raise RuntimeError("Contemporaneous rescue source-audit status drifted.")
    audit_manifest = json.loads(str(audit_row["artifact_manifest_json"]))
    if manifest_sha256(audit_manifest) != str(
        audit_row["artifact_manifest_sha256"]
    ):
        raise RuntimeError("Contemporaneous rescue audit-manifest digest drifted.")
    for name, expected_hash in audit_manifest.items():
        path = CONTEMPORANEOUS_RESCUE_DIR / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise RuntimeError(
                f"Contemporaneous rescue source-audit hash mismatch for {name}."
            )
    full_manifest = {
        **audit_manifest,
        CONTEMPORANEOUS_SOURCE_AUDIT.name: file_sha256(
            CONTEMPORANEOUS_SOURCE_AUDIT
        ),
    }
    observed_files = {
        path.name for path in CONTEMPORANEOUS_RESCUE_DIR.iterdir() if path.is_file()
    }
    observed_directories = [
        path.name for path in CONTEMPORANEOUS_RESCUE_DIR.iterdir() if path.is_dir()
    ]
    if observed_directories or observed_files != set(full_manifest):
        raise RuntimeError("Contemporaneous rescue directory manifest drifted.")

    gate_metrics = pd.read_csv(
        CONTEMPORANEOUS_GATE_METRICS,
        na_values=["<NA>"],
        keep_default_na=True,
        float_precision="round_trip",
    )
    binary_confusion = pd.read_csv(
        CONTEMPORANEOUS_BINARY_CONFUSION,
        na_values=["<NA>"],
        keep_default_na=True,
        float_precision="round_trip",
    )
    five_confusion = pd.read_csv(
        CONTEMPORANEOUS_FIVE_CLASS_CONFUSION,
        na_values=["<NA>"],
        keep_default_na=True,
        float_precision="round_trip",
    )
    if tuple(gate_metrics.columns) != tuple(GATE_METRIC_COLUMNS):
        raise RuntimeError("Contemporaneous gate-metric schema drifted.")
    if tuple(binary_confusion.columns) != tuple(BINARY_CONFUSION_COLUMNS):
        raise RuntimeError("Contemporaneous binary-confusion schema drifted.")
    if tuple(five_confusion.columns) != tuple(FIVE_CLASS_CONFUSION_COLUMNS):
        raise RuntimeError("Contemporaneous five-class-confusion schema drifted.")
    expected_methods = set(FIGURE_METHOD_ORDER)
    if (
        len(gate_metrics) != 3
        or set(gate_metrics["task"]) != {"Contemporaneous"}
        or set(gate_metrics["method"]) != expected_methods
        or int(gate_metrics["primary_selected"].astype(bool).sum()) != 1
        or gate_metrics.duplicated(["task", "method"]).any()
    ):
        raise RuntimeError("Contemporaneous gate-metric row contract drifted.")
    if (
        len(binary_confusion) != 24
        or set(binary_confusion["task"]) != {"Contemporaneous"}
        or set(binary_confusion["method"]) != expected_methods
        or set(binary_confusion["population_scope"]) != {"gate", "full_oof"}
        or binary_confusion.duplicated(
            [
                "population_scope",
                "task",
                "method",
                "actual_binary",
                "predicted_binary",
            ]
        ).any()
    ):
        raise RuntimeError("Contemporaneous binary-confusion row contract drifted.")
    if (
        len(five_confusion) != 75
        or set(five_confusion["task"]) != {"Contemporaneous"}
        or set(five_confusion["method"]) != expected_methods
        or five_confusion.duplicated(
            ["task", "method", "actual_phase", "predicted_phase"]
        ).any()
    ):
        raise RuntimeError("Contemporaneous five-class-confusion row contract drifted.")
    for method in FIGURE_METHOD_ORDER:
        five_group = five_confusion.loc[five_confusion["method"].eq(method)]
        if int(five_group["count"].sum()) != EXPECTED_SOURCE_ROWS:
            raise RuntimeError(
                f"Contemporaneous/{method} five-class support drifted."
            )
        full_binary = binary_confusion.loc[
            binary_confusion["method"].eq(method)
            & binary_confusion["population_scope"].eq("full_oof")
        ]
        if int(full_binary["count"].sum()) != EXPECTED_SOURCE_ROWS:
            raise RuntimeError(
                f"Contemporaneous/{method} binary support drifted."
            )
    return {
        "gate_metrics": gate_metrics,
        "binary_confusion": binary_confusion,
        "five_confusion": five_confusion,
        "configuration": configuration,
        "artifact_manifest": dict(sorted(full_manifest.items())),
    }


def load_prepared_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return predecessor.load_prepared_inputs()


def build_severe_rescue_target(reconstructed_phase: pd.Series) -> pd.Series:
    phase = pd.Series(reconstructed_phase).astype(int)
    return phase.isin([4, 5]).astype(np.uint8).rename("severe_rescue_target")


def build_phase345_training_population(
    forecasting: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    population = forecasting.loc[
        (pd.to_datetime(forecasting["date"]) < CUTOFF)
        & forecasting["reconstructed_overall_phase"].isin([3, 4, 5])
    ].copy()
    population = population.sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    _require_unique_keys(population, "Direct Phase-3/4/5 training population")
    counts = (
        population["reconstructed_overall_phase"].value_counts().sort_index().to_dict()
    )
    expected = {3: EXPECTED_PHASE3_ROWS, 4: EXPECTED_PHASE4_ROWS, 5: EXPECTED_PHASE5_ROWS}
    if len(population) != EXPECTED_PHASE345_ROWS or counts != expected:
        raise ValueError(
            f"Direct Phase-3/4/5 population contract failed: "
            f"rows={len(population)} counts={counts}"
        )
    target = build_severe_rescue_target(population["reconstructed_overall_phase"])
    if int(target.sum()) != EXPECTED_PHASE45_ROWS:
        raise ValueError("Direct Phase-4/5 target support drifted.")
    return population, target.reset_index(drop=True)


def build_fold_training_population(
    forecasting: pd.DataFrame, training_years: Sequence[int]
) -> tuple[pd.DataFrame, pd.Series]:
    years = pd.to_datetime(forecasting["date"]).dt.year
    population = forecasting.loc[
        years.isin(tuple(training_years))
        & forecasting["reconstructed_overall_phase"].isin([3, 4, 5])
    ].copy()
    population = population.sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    _require_unique_keys(population, "Fold Direct Phase-3/4/5 training population")
    target = build_severe_rescue_target(population["reconstructed_overall_phase"])
    if int((target == 0).sum()) == 0 or int((target == 1).sum()) == 0:
        raise ValueError("A Direct training fold lacks negative or positive support.")
    return population, target.reset_index(drop=True)


def candidate_positive_weight(candidate_id: str, class_ratio: float) -> float:
    ratio = float(class_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"Invalid class ratio: {class_ratio}")
    if candidate_id == "unweighted":
        return 1.0
    if candidate_id == "sqrt_balance":
        return math.sqrt(ratio)
    if candidate_id == "full_balance":
        return ratio
    raise ValueError(f"Unknown candidate: {candidate_id}")


def build_candidate_sample_weight(
    target: pd.Series, candidate_id: str
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    values = pd.Series(target).astype(np.uint8).reset_index(drop=True)
    negative_count = int((values == 0).sum())
    positive_count = int((values == 1).sum())
    if negative_count == 0 or positive_count == 0:
        raise ValueError("Candidate weighting requires both classes.")
    class_ratio = negative_count / positive_count
    positive_weight = candidate_positive_weight(candidate_id, class_ratio)
    weights = np.where(values.to_numpy(dtype=int) == 1, positive_weight, 1.0)
    return weights.astype(float), {
        "candidate_id": candidate_id,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "class_ratio": float(class_ratio),
        "negative_row_weight": 1.0,
        "positive_row_weight": float(positive_weight),
    }


def _direct_parameters(estimator: str) -> dict[str, object]:
    parameters = predecessor._direct_parameters(estimator)
    parameters["random_state"] = 0
    parameters["n_jobs"] = 1
    parameters["scale_pos_weight"] = 1
    return parameters


def fit_direct_candidate_models(
    task: str,
    layer1_matrix: pd.DataFrame,
    layer2_matrix: pd.DataFrame | None,
    target: pd.Series,
    sample_weight: Sequence[float],
    classifier_factory=xgb.XGBClassifier,
    regressor_factory=xgb.XGBRegressor,
    classifier_parameters: Mapping[str, object] | None = None,
    regressor_parameters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if task not in TASK_ORDER:
        raise ValueError(f"Unknown task: {task}")
    classifier_params = dict(
        _direct_parameters("classifier")
        if classifier_parameters is None
        else classifier_parameters
    )
    if classifier_params.get("scale_pos_weight") != 1:
        raise ValueError("scale_pos_weight must remain 1 with row-level weights.")
    weights = np.asarray(sample_weight, dtype=float)
    if len(weights) != len(layer1_matrix) or len(target) != len(layer1_matrix):
        raise ValueError("Training matrix, target, and sample weights must align.")
    layer1_model = classifier_factory(**classifier_params)
    layer1_model.fit(layer1_matrix, target, sample_weight=weights)
    layer1_training_score = np.asarray(
        layer1_model.predict_proba(layer1_matrix)[:, 1], dtype=float
    )
    result: dict[str, object] = {
        "layer1_model": layer1_model,
        "layer2_model": None,
        "layer1_training_score": layer1_training_score,
        "residual_target": None,
    }
    if task == "Forecasting":
        if layer2_matrix is not None:
            raise ValueError("Forecasting must not receive a Layer-2 matrix.")
        return result
    if layer2_matrix is None or len(layer2_matrix) != len(layer1_matrix):
        raise ValueError("Nowcasting requires an aligned Layer-2 matrix.")
    regressor_params = dict(
        _direct_parameters("regressor")
        if regressor_parameters is None
        else regressor_parameters
    )
    if regressor_params.get("scale_pos_weight") != 1:
        raise ValueError("Layer-2 scale_pos_weight must remain 1.")
    residual_target = target.to_numpy(dtype=float) - layer1_training_score
    layer2_model = regressor_factory(**regressor_params)
    layer2_model.fit(layer2_matrix, residual_target, sample_weight=weights)
    result["layer2_model"] = layer2_model
    result["residual_target"] = residual_target
    return result


def apply_phase45_rescue(
    base_phase: Sequence[int] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
    threshold: float,
) -> np.ndarray:
    result = predecessor.apply_phase4_rescue(base_phase, score, threshold)
    predecessor.assert_postclassification_invariants(base_phase, result)
    return result


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return math.nan if denominator == 0 else float(numerator / denominator)


def binary_metric_bundle(
    actual_binary: Sequence[int], predicted_binary: Sequence[int]
) -> dict[str, float | int]:
    actual = np.asarray(actual_binary, dtype=int)
    predicted = np.asarray(predicted_binary, dtype=int)
    if actual.shape != predicted.shape:
        raise ValueError("Binary actual and predicted arrays must align.")
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "f2": safe_divide(5 * tp, 5 * tp + 4 * fn + fp),
    }


def build_binary_confusion_cells(
    actual_binary: Sequence[int], predicted_binary: Sequence[int]
) -> pd.DataFrame:
    actual = np.asarray(actual_binary, dtype=int)
    predicted = np.asarray(predicted_binary, dtype=int)
    rows: list[dict[str, object]] = []
    for actual_value in (0, 1):
        row_total = int(np.sum(actual == actual_value))
        for predicted_value in (0, 1):
            count = int(
                np.sum((actual == actual_value) & (predicted == predicted_value))
            )
            rows.append(
                {
                    "actual_binary": actual_value,
                    "predicted_binary": predicted_value,
                    "count": count,
                    "actual_row_total": row_total,
                    "actual_row_share": safe_divide(count, row_total),
                }
            )
    return pd.DataFrame(rows)


def build_five_class_confusion_cells(
    actual_phase: Sequence[int], predicted_phase: Sequence[int]
) -> pd.DataFrame:
    actual = np.asarray(actual_phase, dtype=int)
    predicted = np.asarray(predicted_phase, dtype=int)
    rows: list[dict[str, object]] = []
    for actual_value in ALL_PHASES:
        row_total = int(np.sum(actual == actual_value))
        for predicted_value in ALL_PHASES:
            count = int(
                np.sum((actual == actual_value) & (predicted == predicted_value))
            )
            rows.append(
                {
                    "actual_phase": actual_value,
                    "predicted_phase": predicted_value,
                    "count": count,
                    "actual_row_total": row_total,
                    "actual_row_share": safe_divide(count, row_total),
                }
            )
    return pd.DataFrame(rows)


def _safe_pr_auc(actual_binary: Sequence[int], score: Sequence[float]) -> float:
    actual = np.asarray(actual_binary, dtype=int)
    values = np.asarray(score, dtype=float)
    finite = np.isfinite(values)
    if not finite.any() or len(np.unique(actual[finite])) < 2:
        return math.nan
    return float(average_precision_score(actual[finite], values[finite]))


def _frontier_metrics(
    actual_phase: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, object]:
    promoted = scores >= float(threshold)
    severe = np.isin(actual_phase, [4, 5])
    metrics = binary_metric_bundle(severe.astype(int), promoted.astype(int))
    false_promotions = promoted & ~severe
    return {
        "gate_rows": len(actual_phase),
        "positive_support": int(severe.sum()),
        "negative_support": int((~severe).sum()),
        **{
            "true_positive": metrics["true_positive"],
            "false_positive": metrics["false_positive"],
            "false_negative": metrics["false_negative"],
            "true_negative": metrics["true_negative"],
            "phase45_precision": metrics["precision"],
            "phase45_recall": metrics["recall"],
            "phase45_f1": metrics["f1"],
            "phase45_f2": metrics["f2"],
        },
        "true_rescue_actual_phase4": int(np.sum(promoted & (actual_phase == 4))),
        "true_rescue_actual_phase5": int(np.sum(promoted & (actual_phase == 5))),
        "false_promotion_actual_phase1": int(
            np.sum(false_promotions & (actual_phase == 1))
        ),
        "false_promotion_actual_phase2": int(
            np.sum(false_promotions & (actual_phase == 2))
        ),
        "false_promotion_actual_phase3": int(
            np.sum(false_promotions & (actual_phase == 3))
        ),
        "false_promotion_actual_phase12": int(
            np.sum(false_promotions & np.isin(actual_phase, [1, 2]))
        ),
        "total_promotions": int(promoted.sum()),
        "false_promotions_per_100_gate_rows": (
            100.0 * int(false_promotions.sum()) / len(actual_phase)
            if len(actual_phase)
            else math.nan
        ),
    }


def build_threshold_frontier(oof_candidate: pd.DataFrame) -> pd.DataFrame:
    required = (
        "task",
        "candidate_id",
        "reconstructed_overall_phase",
        "base_overall_phase_pred",
        "in_base_phase3_gate",
        "direct_phase45_score",
    )
    _require_columns(oof_candidate, required, "Candidate OOF predictions")
    if oof_candidate["task"].nunique() != 1 or oof_candidate["candidate_id"].nunique() != 1:
        raise ValueError("A threshold frontier must contain one task and candidate.")
    gate = oof_candidate.loc[
        oof_candidate["in_base_phase3_gate"].astype(bool)
    ].copy()
    if gate.empty or not gate["base_overall_phase_pred"].eq(3).all():
        raise ValueError("Threshold selection requires the complete OOF Phase-3 gate.")
    scores = gate["direct_phase45_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Threshold-selection scores must be finite.")
    actual = gate["reconstructed_overall_phase"].to_numpy(dtype=int)
    sentinel = float(np.nextafter(np.max(scores), np.inf))
    thresholds = [sentinel, *sorted(np.unique(scores).tolist(), reverse=True)]
    task = str(gate["task"].iloc[0])
    candidate = str(gate["candidate_id"].iloc[0])
    rows: list[dict[str, object]] = []
    for rank, threshold in enumerate(thresholds, start=1):
        rows.append(
            {
                "task": task,
                "candidate_id": candidate,
                "threshold_rank": rank,
                "threshold": float(threshold),
                "threshold_source": "pooled_pre2022_oof_base_phase3_gate",
                "is_above_max_reference": rank == 1,
                **_frontier_metrics(actual, scores, threshold),
                "within_candidate_selected": False,
                "primary_selected": False,
                "selection_status": "evaluated",
            }
        )
    result = pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)
    if result.duplicated(["task", "candidate_id", "threshold_rank"]).any():
        raise ValueError("Threshold frontier contains duplicate keys.")
    return result


def _selection_sort_frame(data: pd.DataFrame, include_candidate: bool) -> pd.DataFrame:
    frame = data.copy()
    frame["__f2"] = frame["phase45_f2"].fillna(-np.inf)
    frame["__recall"] = frame["phase45_recall"].fillna(-np.inf)
    frame["__precision"] = frame["phase45_precision"].fillna(-np.inf)
    frame["__candidate_order"] = frame["candidate_id"].map(
        {candidate: index for index, candidate in enumerate(CANDIDATE_ORDER)}
    )
    columns = [
        "__f2",
        "__recall",
        "__precision",
        "false_promotion_actual_phase12",
        "total_promotions",
    ]
    ascending = [False, False, False, True, True]
    if include_candidate:
        columns.append("__candidate_order")
        ascending.append(True)
    columns.append("threshold")
    ascending.append(False)
    return frame.sort_values(columns, ascending=ascending, kind="mergesort")


def select_task_policies(
    task_frontier: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if task_frontier["task"].nunique() != 1:
        raise ValueError("Policy selection must operate on one task.")
    if set(task_frontier["candidate_id"].unique()) != set(CANDIDATE_ORDER):
        raise ValueError("Policy selection requires all three candidates.")
    task = str(task_frontier["task"].iloc[0])
    frontier = task_frontier.copy().reset_index(drop=True)
    triggered = frontier.loc[frontier["total_promotions"].gt(0)]
    best_triggered_f2 = (
        float(triggered["phase45_f2"].fillna(0.0).max())
        if not triggered.empty
        else 0.0
    )
    no_effective = best_triggered_f2 == 0.0
    selected_indices: dict[str, int] = {}
    if no_effective:
        for candidate in CANDIDATE_ORDER:
            rows = frontier.loc[
                frontier["candidate_id"].eq(candidate)
                & frontier["is_above_max_reference"].astype(bool)
            ]
            if len(rows) != 1:
                raise ValueError("Each candidate requires one above-maximum reference.")
            selected_indices[candidate] = int(rows.index[0])
        primary_index: int | None = None
        status = "no_effective_rescue"
    else:
        for candidate in CANDIDATE_ORDER:
            rows = frontier.loc[frontier["candidate_id"].eq(candidate)]
            selected_indices[candidate] = int(
                _selection_sort_frame(rows, include_candidate=False).index[0]
            )
        primary_index = int(
            _selection_sort_frame(frontier, include_candidate=True).index[0]
        )
        status = "selected"
    frontier["selection_status"] = status
    frontier.loc[list(selected_indices.values()), "within_candidate_selected"] = True
    if primary_index is not None:
        frontier.loc[primary_index, "primary_selected"] = True
    policy_rows: list[dict[str, object]] = []
    for candidate in CANDIDATE_ORDER:
        row = frontier.loc[selected_indices[candidate]]
        policy_rows.append(
            {
                "task": task,
                "candidate_id": candidate,
                "method": CANDIDATE_METHODS[candidate],
                "selection_status": status,
                "threshold_rank": int(row["threshold_rank"]),
                "threshold": float(row["threshold"]),
                "is_above_max_reference": bool(row["is_above_max_reference"]),
                "phase45_precision": row["phase45_precision"],
                "phase45_recall": row["phase45_recall"],
                "phase45_f1": row["phase45_f1"],
                "phase45_f2": row["phase45_f2"],
                "false_promotion_actual_phase12": int(
                    row["false_promotion_actual_phase12"]
                ),
                "total_promotions": int(row["total_promotions"]),
                "primary_selected": bool(
                    primary_index is not None and selected_indices[candidate] == primary_index
                ),
                "final_training_negative_count": np.nan,
                "final_training_positive_count": np.nan,
                "final_class_ratio": np.nan,
                "final_positive_row_weight": np.nan,
            }
        )
    policies = pd.DataFrame(policy_rows, columns=SELECTED_POLICY_COLUMNS)
    return frontier.loc[:, THRESHOLD_COLUMNS], policies


def load_frozen_oof_gate_evidence() -> pd.DataFrame:
    verify_predecessor_artifacts()
    source = pd.read_csv(
        PREDECESSOR_OOF,
        parse_dates=["date"],
        float_precision="round_trip",
        na_values=["<NA>"],
        keep_default_na=True,
    )
    required = (
        "task",
        "base_oof_fold",
        "area_id",
        "date",
        "country_code_3",
        "source_overall_phase",
        "reconstructed_overall_phase",
        "base_overall_phase_pred",
        "in_auxiliary_gate",
    )
    _require_columns(source, required, "Frozen predecessor OOF evidence")
    if len(source) != EXPECTED_PREDECESSOR_OOF_ROWS:
        raise ValueError("Frozen predecessor OOF row count drifted.")
    if source.duplicated(["task", *KEY_COLUMNS]).any():
        raise ValueError("Frozen predecessor OOF evidence has duplicate keys.")
    result = source.loc[:, list(required)].rename(
        columns={"in_auxiliary_gate": "in_base_phase3_gate"}
    )
    result["in_base_phase3_gate"] = result["in_base_phase3_gate"].astype(bool)
    if not np.array_equal(
        result["in_base_phase3_gate"].to_numpy(dtype=bool),
        result["base_overall_phase_pred"].to_numpy(dtype=int) == 3,
    ):
        raise ValueError("Frozen OOF gate does not equal base Phase 3.")
    result["severe_rescue_target"] = build_severe_rescue_target(
        result["reconstructed_overall_phase"]
    )
    fold_year = {fold.fold_id: fold.validation_year for fold in BASE_FOLDS}
    if set(result["base_oof_fold"].unique()) != set(fold_year):
        raise ValueError("Frozen OOF fold IDs drifted.")
    mapped = result["base_oof_fold"].map(fold_year).to_numpy(dtype=int)
    if not np.array_equal(mapped, result["date"].dt.year.to_numpy(dtype=int)):
        raise ValueError("Frozen OOF fold/year mapping drifted.")
    counts = result.groupby("task", observed=True).size().to_dict()
    if counts != {task: EXPECTED_OOF_ROWS_PER_TASK for task in TASK_ORDER}:
        raise ValueError(f"Frozen OOF task counts drifted: {counts}")
    return result.sort_values(
        ["task", "base_oof_fold", *KEY_COLUMNS], kind="mergesort"
    ).reset_index(drop=True)


def load_frozen_benchmark_evidence() -> pd.DataFrame:
    verify_predecessor_artifacts()
    source = pd.read_csv(
        PREDECESSOR_BENCHMARK,
        parse_dates=["date"],
        float_precision="round_trip",
        na_values=["<NA>"],
        keep_default_na=True,
    )
    required = (
        "task",
        "area_id",
        "date",
        "country_code_3",
        "source_overall_phase",
        "reconstructed_overall_phase",
        "phase2_test",
        "phase3_test",
        "phase4_test",
        "phase5_test",
        "phase2_pred_rounded",
        "phase3_pred_rounded",
        "phase4_pred_rounded",
        "phase5_pred_rounded",
        "base_overall_phase_pred",
        "direct_layer1_phase4_score",
        "direct_layer2_residual_score",
        "direct_phase4_score",
        "direct_threshold",
        "direct_triggered",
        "direct_overall_phase_pred",
    )
    _require_columns(source, required, "Frozen predecessor benchmark evidence")
    if len(source) != 2 * EXPECTED_BENCHMARK_ROWS:
        raise ValueError("Frozen predecessor benchmark row count drifted.")
    if source.duplicated(["task", *KEY_COLUMNS]).any():
        raise ValueError("Frozen predecessor benchmark has duplicate keys.")
    if not source["direct_threshold"].eq(0.5).all():
        raise ValueError("Legacy Direct exact-Phase-4 threshold drifted.")
    result = source.loc[:, list(required)].copy()
    result["in_base_phase3_gate"] = result["base_overall_phase_pred"].eq(3)
    result["severe_rescue_target"] = build_severe_rescue_target(
        result["reconstructed_overall_phase"]
    )
    recomputed = apply_phase45_rescue(
        result["base_overall_phase_pred"],
        result["direct_phase4_score"].fillna(-np.inf),
        0.5,
    )
    if not np.array_equal(
        recomputed, result["direct_overall_phase_pred"].to_numpy(dtype=int)
    ):
        raise ValueError("Legacy Direct exact-Phase-4 predictions drifted.")
    for task in TASK_ORDER:
        frame = result.loc[result["task"].eq(task)]
        if len(frame) != EXPECTED_BENCHMARK_ROWS:
            raise ValueError(f"Frozen {task} benchmark does not contain 1,170 rows.")
        if canonical_key_sha256(frame) != EXPECTED_BENCHMARK_KEY_SHA256:
            raise ValueError(f"Frozen {task} benchmark key hash drifted.")
    return result.sort_values(["task", *KEY_COLUMNS], kind="mergesort").reset_index(
        drop=True
    )


def build_feature_manifest(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    layer1_features: Sequence[str],
) -> pd.DataFrame:
    contracts = (
        (
            "Forecasting",
            "direct_forecasting_classifier",
            forecasting,
            tuple(layer1_features),
        ),
        (
            "Nowcasting",
            "direct_nowcasting_layer1_classifier",
            forecasting,
            tuple(layer1_features),
        ),
        (
            "Nowcasting",
            "direct_nowcasting_layer2_regressor",
            nowcasting,
            tuple(NOWCAST_FEATURES),
        ),
    )
    rows: list[dict[str, object]] = []
    for task, component, source, features in contracts:
        order_hash = json_sha256(list(features))
        for order, feature in enumerate(features, start=1):
            rows.append(
                {
                    "task": task,
                    "model_component": component,
                    "feature_order": order,
                    "feature_name": feature,
                    "source_dtype": str(source[feature].dtype),
                    "feature_definition": "native_source_column",
                    "native_missingness_preserved": True,
                    "feature_order_sha256": order_hash,
                }
            )
    result = pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS)
    if len(result) != EXPECTED_FEATURE_MANIFEST_ROWS or result.duplicated(
        ["task", "model_component", "feature_order"]
    ).any():
        raise ValueError("Feature manifest violates the 281-row contract.")
    return result


def _align_source_to_reference(
    source: pd.DataFrame, reference: pd.DataFrame, name: str
) -> pd.DataFrame:
    _require_unique_keys(source, f"{name} source")
    _require_unique_keys(reference, f"{name} reference")
    keys = reference.loc[:, list(KEY_COLUMNS)].copy()
    aligned = keys.merge(
        source,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all() or len(aligned) != len(reference):
        raise ValueError(f"{name} failed complete one-to-one key alignment.")
    aligned = aligned.drop(columns="_merge")
    if not canonical_key_frame(aligned).equals(canonical_key_frame(reference)):
        raise ValueError(f"{name} key order drifted after alignment.")
    return aligned.reset_index(drop=True)


def _validate_native_matrix(matrix: pd.DataFrame, name: str) -> None:
    predecessor._validate_native_matrix(matrix, name)


def _temporary_model_sha256(model: object, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save_model(str(path))
        return file_sha256(path)
    finally:
        if path.exists():
            path.unlink()


def _model_audit_row(
    *,
    task: str,
    candidate_id: str,
    fit_scope: str,
    fold_id: str,
    model_component: str,
    estimator_class: str,
    objective: str,
    target_definition: str,
    run_status: str,
    training_years: Sequence[int],
    validation_year: int | float,
    training_keys: pd.DataFrame,
    binary_target: pd.Series,
    model_target: pd.Series,
    training_matrix: pd.DataFrame,
    sample_weight: np.ndarray,
    weight_audit: Mapping[str, object],
    features: Sequence[str],
    parameters: Mapping[str, object],
    validation_keys: pd.DataFrame,
    validation_matrix: pd.DataFrame,
    validation_score: Sequence[float],
    model: object,
    model_hash_path: Path | None,
    published_model_path: Path | None = None,
    layer1_training_score: Sequence[float] | None = None,
    residual_target: Sequence[float] | None = None,
) -> dict[str, object]:
    if published_model_path is not None:
        model_sha = file_sha256(published_model_path)
        model_path_value: object = published_model_path.name
    else:
        if model_hash_path is None:
            raise ValueError("Fold model audit requires a temporary hash path.")
        model_sha = _temporary_model_sha256(model, model_hash_path)
        model_path_value = np.nan
    return {
        "task": task,
        "candidate_id": candidate_id,
        "fit_scope": fit_scope,
        "fold_id": fold_id,
        "model_component": model_component,
        "estimator_class": estimator_class,
        "objective": objective,
        "target_definition": target_definition,
        "run_status": run_status,
        "training_years": "|".join(str(year) for year in training_years),
        "validation_year": validation_year,
        "training_rows": len(training_matrix),
        "training_negative_count": int((binary_target == 0).sum()),
        "training_positive_count": int((binary_target == 1).sum()),
        "class_ratio": float(weight_audit["class_ratio"]),
        "negative_row_weight": float(weight_audit["negative_row_weight"]),
        "positive_row_weight": float(weight_audit["positive_row_weight"]),
        "scale_pos_weight": float(parameters["scale_pos_weight"]),
        "sample_weight_sha256": canonical_series_sha256(
            pd.Series(sample_weight), "sample_weight"
        ),
        "sample_weight_sum": float(np.sum(sample_weight)),
        "feature_count": len(features),
        "feature_order_sha256": json_sha256(list(features)),
        "parameter_sha256": json_sha256(dict(parameters)),
        "random_state": int(parameters["random_state"]),
        "n_jobs": int(parameters["n_jobs"]),
        "training_key_sha256": canonical_key_sha256(training_keys),
        "target_sha256": canonical_series_sha256(model_target, "model_target"),
        "training_matrix_sha256": canonical_matrix_sha256(training_matrix),
        "training_missingness_sha256": canonical_missingness_sha256(
            training_matrix
        ),
        "training_matrix_with_target_sha256": matrix_with_target_sha256(
            training_matrix, model_target
        ),
        "layer1_training_score_sha256": (
            canonical_series_sha256(
                pd.Series(layer1_training_score), "layer1_training_score"
            )
            if layer1_training_score is not None
            else np.nan
        ),
        "residual_target_sha256": (
            canonical_series_sha256(pd.Series(residual_target), "residual_target")
            if residual_target is not None
            else np.nan
        ),
        "validation_rows": len(validation_matrix),
        "validation_key_sha256": canonical_key_sha256(validation_keys),
        "validation_matrix_sha256": canonical_matrix_sha256(validation_matrix),
        "validation_missingness_sha256": canonical_missingness_sha256(
            validation_matrix
        ),
        "validation_score_sha256": canonical_series_sha256(
            pd.Series(validation_score), "validation_score"
        ),
        "model_path": model_path_value,
        "model_sha256": model_sha,
    }


def generate_oof_predictions(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    layer1_features: Sequence[str],
    frozen_oof: pd.DataFrame,
    model_hash_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predecessor.validate_temporal_folds(forecasting, BASE_FOLDS)
    rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    classifier_parameters = _direct_parameters("classifier")
    regressor_parameters = _direct_parameters("regressor")
    for fold in BASE_FOLDS:
        training, target = build_fold_training_population(
            forecasting, fold.training_years
        )
        training_keys = training.loc[:, list(KEY_COLUMNS)].copy()
        layer1_training = training.loc[:, list(layer1_features)].reset_index(drop=True)
        _validate_native_matrix(
            layer1_training, f"{fold.fold_id} Layer-1 training matrix"
        )
        now_training = _align_source_to_reference(
            nowcasting, training_keys, f"{fold.fold_id} Nowcasting training"
        )
        layer2_training = now_training.loc[:, list(NOWCAST_FEATURES)].reset_index(
            drop=True
        )
        _validate_native_matrix(
            layer2_training, f"{fold.fold_id} Nowcasting Layer-2 training matrix"
        )

        year_mask = pd.to_datetime(forecasting["date"]).dt.year.eq(
            fold.validation_year
        )
        forecast_validation_source = forecasting.loc[year_mask].copy()
        forecast_validation_source = forecast_validation_source.sort_values(
            list(KEY_COLUMNS), kind="mergesort"
        ).reset_index(drop=True)
        now_validation_source = _align_source_to_reference(
            nowcasting,
            forecast_validation_source.loc[:, list(KEY_COLUMNS)],
            f"{fold.fold_id} Nowcasting validation",
        )
        layer1_validation = forecast_validation_source.loc[
            :, list(layer1_features)
        ].reset_index(drop=True)
        layer2_validation = now_validation_source.loc[
            :, list(NOWCAST_FEATURES)
        ].reset_index(drop=True)
        _validate_native_matrix(
            layer1_validation, f"{fold.fold_id} Layer-1 validation matrix"
        )
        _validate_native_matrix(
            layer2_validation, f"{fold.fold_id} Layer-2 validation matrix"
        )

        for task in TASK_ORDER:
            evidence = frozen_oof.loc[
                frozen_oof["task"].eq(task)
                & frozen_oof["base_oof_fold"].eq(fold.fold_id)
            ].copy()
            evidence = evidence.sort_values(
                list(KEY_COLUMNS), kind="mergesort"
            ).reset_index(drop=True)
            expected_source = (
                forecast_validation_source
                if task == "Forecasting"
                else now_validation_source
            )
            if not canonical_key_frame(evidence).equals(
                canonical_key_frame(expected_source)
            ):
                raise ValueError(f"{task}/{fold.fold_id} OOF key alignment failed.")
            for column in ("country_code_3", "reconstructed_overall_phase"):
                left = evidence[column].reset_index(drop=True)
                right = expected_source[column].reset_index(drop=True)
                equal = left.eq(right) | (left.isna() & right.isna())
                if not bool(equal.all()):
                    raise ValueError(
                        f"{task}/{fold.fold_id} OOF evidence drifted for {column}."
                    )

        for candidate in CANDIDATE_ORDER:
            sample_weight, weight_audit = build_candidate_sample_weight(
                target, candidate
            )
            forecast_fit = fit_direct_candidate_models(
                task="Forecasting",
                layer1_matrix=layer1_training,
                layer2_matrix=None,
                target=target,
                sample_weight=sample_weight,
                classifier_parameters=classifier_parameters,
            )
            forecast_score = np.asarray(
                forecast_fit["layer1_model"].predict_proba(layer1_validation)[:, 1],
                dtype=float,
            )
            nowcast_fit = fit_direct_candidate_models(
                task="Nowcasting",
                layer1_matrix=layer1_training,
                layer2_matrix=layer2_training,
                target=target,
                sample_weight=sample_weight,
                classifier_parameters=classifier_parameters,
                regressor_parameters=regressor_parameters,
            )
            nowcast_layer1_score = np.asarray(
                nowcast_fit["layer1_model"].predict_proba(layer1_validation)[:, 1],
                dtype=float,
            )
            nowcast_residual_score = np.asarray(
                nowcast_fit["layer2_model"].predict(layer2_validation), dtype=float
            )
            nowcast_score = predecessor.combine_direct_nowcasting_scores(
                nowcast_layer1_score, nowcast_residual_score
            )
            score_contracts = {
                "Forecasting": (
                    forecast_score,
                    forecast_score,
                    np.full(len(forecast_score), np.nan),
                ),
                "Nowcasting": (
                    nowcast_score,
                    nowcast_layer1_score,
                    nowcast_residual_score,
                ),
            }
            for task in TASK_ORDER:
                evidence = frozen_oof.loc[
                    frozen_oof["task"].eq(task)
                    & frozen_oof["base_oof_fold"].eq(fold.fold_id)
                ].copy()
                evidence = evidence.sort_values(
                    list(KEY_COLUMNS), kind="mergesort"
                ).reset_index(drop=True)
                combined_score, layer1_score, residual_score = score_contracts[task]
                output = evidence.loc[
                    :,
                    [
                        "task",
                        "base_oof_fold",
                        "area_id",
                        "date",
                        "country_code_3",
                        "source_overall_phase",
                        "reconstructed_overall_phase",
                        "severe_rescue_target",
                        "base_overall_phase_pred",
                        "in_base_phase3_gate",
                    ],
                ].copy()
                output.insert(1, "candidate_id", candidate)
                output.insert(3, "validation_year", fold.validation_year)
                output["direct_layer1_score"] = layer1_score
                output["direct_layer2_residual_score"] = residual_score
                output["direct_phase45_score"] = combined_score
                output["fold_training_years"] = "|".join(
                    str(year) for year in fold.training_years
                )
                output["training_rows"] = len(training)
                output["training_negative_count"] = weight_audit["negative_count"]
                output["training_positive_count"] = weight_audit["positive_count"]
                output["class_ratio"] = weight_audit["class_ratio"]
                output["positive_row_weight"] = weight_audit["positive_row_weight"]
                rows.append(output.loc[:, list(OOF_COLUMNS)])

            audit_rows.append(
                _model_audit_row(
                    task="Forecasting",
                    candidate_id=candidate,
                    fit_scope="oof_fold",
                    fold_id=fold.fold_id,
                    model_component="direct_forecasting_classifier",
                    estimator_class="XGBClassifier",
                    objective="binary:logistic",
                    target_definition=(
                        "1[reconstructed_overall_phase in {4,5}] among "
                        "reconstructed Phase 3/4/5"
                    ),
                    run_status="fitted_temporary_hash_only",
                    training_years=fold.training_years,
                    validation_year=fold.validation_year,
                    training_keys=training_keys,
                    binary_target=target,
                    model_target=target,
                    training_matrix=layer1_training,
                    sample_weight=sample_weight,
                    weight_audit=weight_audit,
                    features=layer1_features,
                    parameters=classifier_parameters,
                    validation_keys=forecast_validation_source,
                    validation_matrix=layer1_validation,
                    validation_score=forecast_score,
                    model=forecast_fit["layer1_model"],
                    model_hash_path=(
                        model_hash_dir
                        / f"{candidate}_{fold.fold_id}_forecasting.json"
                    ),
                )
            )
            audit_rows.append(
                _model_audit_row(
                    task="Nowcasting",
                    candidate_id=candidate,
                    fit_scope="oof_fold",
                    fold_id=fold.fold_id,
                    model_component="direct_nowcasting_layer1_classifier",
                    estimator_class="XGBClassifier",
                    objective="binary:logistic",
                    target_definition=(
                        "1[reconstructed_overall_phase in {4,5}] among "
                        "reconstructed Phase 3/4/5"
                    ),
                    run_status="fitted_temporary_hash_only",
                    training_years=fold.training_years,
                    validation_year=fold.validation_year,
                    training_keys=training_keys,
                    binary_target=target,
                    model_target=target,
                    training_matrix=layer1_training,
                    sample_weight=sample_weight,
                    weight_audit=weight_audit,
                    features=layer1_features,
                    parameters=classifier_parameters,
                    validation_keys=forecast_validation_source,
                    validation_matrix=layer1_validation,
                    validation_score=nowcast_layer1_score,
                    model=nowcast_fit["layer1_model"],
                    model_hash_path=(
                        model_hash_dir
                        / f"{candidate}_{fold.fold_id}_nowcasting_layer1.json"
                    ),
                    layer1_training_score=nowcast_fit["layer1_training_score"],
                )
            )
            residual_target = pd.Series(
                nowcast_fit["residual_target"], name="residual_target"
            )
            audit_rows.append(
                _model_audit_row(
                    task="Nowcasting",
                    candidate_id=candidate,
                    fit_scope="oof_fold",
                    fold_id=fold.fold_id,
                    model_component="direct_nowcasting_layer2_regressor",
                    estimator_class="XGBRegressor",
                    objective="reg:squarederror",
                    target_definition=(
                        "severe_rescue_target - direct Nowcasting Layer-1 "
                        "training score"
                    ),
                    run_status="fitted_temporary_hash_only",
                    training_years=fold.training_years,
                    validation_year=fold.validation_year,
                    training_keys=training_keys,
                    binary_target=target,
                    model_target=residual_target,
                    training_matrix=layer2_training,
                    sample_weight=sample_weight,
                    weight_audit=weight_audit,
                    features=NOWCAST_FEATURES,
                    parameters=regressor_parameters,
                    validation_keys=now_validation_source,
                    validation_matrix=layer2_validation,
                    validation_score=nowcast_residual_score,
                    model=nowcast_fit["layer2_model"],
                    model_hash_path=(
                        model_hash_dir
                        / f"{candidate}_{fold.fold_id}_nowcasting_layer2.json"
                    ),
                    layer1_training_score=nowcast_fit["layer1_training_score"],
                    residual_target=nowcast_fit["residual_target"],
                )
            )
    oof = pd.concat(rows, ignore_index=True)
    oof = oof.sort_values(
        ["task", "candidate_id", "base_oof_fold", *KEY_COLUMNS],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(oof) != EXPECTED_OOF_ROWS or oof.duplicated(
        ["task", "candidate_id", "base_oof_fold", *KEY_COLUMNS]
    ).any():
        raise ValueError("OOF predictions violate the 22,560-row contract.")
    if pd.to_datetime(oof["date"]).dt.year.max() > 2021:
        raise ValueError("OOF selection opened the 2022 benchmark.")
    audit = pd.DataFrame(audit_rows, columns=MODEL_AUDIT_COLUMNS)
    if len(audit) != 36 or audit.duplicated(
        ["task", "candidate_id", "fit_scope", "fold_id", "model_component"]
    ).any():
        raise ValueError("OOF model audit violates the 36-row contract.")
    return oof, audit


def build_thresholds_and_policies(
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_frontiers: list[pd.DataFrame] = []
    all_policies: list[pd.DataFrame] = []
    for task in TASK_ORDER:
        candidate_frontiers = [
            build_threshold_frontier(
                oof.loc[
                    oof["task"].eq(task) & oof["candidate_id"].eq(candidate)
                ]
            )
            for candidate in CANDIDATE_ORDER
        ]
        task_frontier, task_policies = select_task_policies(
            pd.concat(candidate_frontiers, ignore_index=True)
        )
        all_frontiers.append(task_frontier)
        all_policies.append(task_policies)
    frontier = pd.concat(all_frontiers, ignore_index=True).sort_values(
        ["task", "candidate_id", "threshold_rank"], kind="mergesort"
    ).reset_index(drop=True)
    policies = pd.concat(all_policies, ignore_index=True).sort_values(
        ["task", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(frontier) > MAX_THRESHOLD_FRONTIER_ROWS or frontier.duplicated(
        ["task", "candidate_id", "threshold_rank"]
    ).any():
        raise ValueError("Threshold frontier violates its formula-governed contract.")
    if len(policies) != 6 or policies.duplicated(["task", "candidate_id"]).any():
        raise ValueError("Selected policies violate the six-row contract.")
    return frontier, policies


def build_oof_stability(
    oof: pd.DataFrame, policies: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policy_index = policies.set_index(["task", "candidate_id"])
    for task in TASK_ORDER:
        for candidate in CANDIDATE_ORDER:
            frame = oof.loc[
                oof["task"].eq(task)
                & oof["candidate_id"].eq(candidate)
                & oof["in_base_phase3_gate"].astype(bool)
            ].copy()
            threshold = float(policy_index.loc[(task, candidate), "threshold"])
            periods = [("pooled", np.nan, frame)]
            periods.extend(
                (
                    str(year),
                    year,
                    frame.loc[pd.to_datetime(frame["date"]).dt.year.eq(year)],
                )
                for year in (2018, 2019, 2020, 2021)
            )
            for period_id, validation_year, period in periods:
                actual = period["reconstructed_overall_phase"].to_numpy(dtype=int)
                scores = period["direct_phase45_score"].to_numpy(dtype=float)
                metrics = _frontier_metrics(actual, scores, threshold)
                rows.append(
                    {
                        "task": task,
                        "candidate_id": candidate,
                        "method": CANDIDATE_METHODS[candidate],
                        "period_id": period_id,
                        "validation_year": validation_year,
                        "threshold": threshold,
                        **{
                            column: metrics[column]
                            for column in OOF_STABILITY_COLUMNS
                            if column in metrics
                        },
                    }
                )
    result = pd.DataFrame(rows, columns=OOF_STABILITY_COLUMNS)
    if len(result) != EXPECTED_OOF_STABILITY_ROWS or result.duplicated(
        ["task", "candidate_id", "period_id"]
    ).any():
        raise ValueError("OOF stability violates the 30-row contract.")
    return result


def _save_published_model(model: object, path: Path) -> None:
    model.save_model(str(path))
    if not path.is_file():
        raise RuntimeError(f"Model was not saved: {path}")


def refit_all_candidates(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    layer1_features: Sequence[str],
    benchmark_evidence: pd.DataFrame,
    policies: pd.DataFrame,
    staging_dir: Path,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame, pd.DataFrame]:
    population, target = build_phase345_training_population(forecasting)
    training_keys = population.loc[:, list(KEY_COLUMNS)].copy()
    layer1_training = population.loc[:, list(layer1_features)].reset_index(drop=True)
    now_training = _align_source_to_reference(
        nowcasting, training_keys, "Final Nowcasting training"
    )
    layer2_training = now_training.loc[:, list(NOWCAST_FEATURES)].reset_index(drop=True)
    _validate_native_matrix(layer1_training, "Final Layer-1 training matrix")
    _validate_native_matrix(layer2_training, "Final Layer-2 training matrix")

    reference = benchmark_evidence.loc[
        benchmark_evidence["task"].eq("Forecasting"), list(KEY_COLUMNS)
    ].copy()
    forecast_test_source = forecasting.loc[
        pd.to_datetime(forecasting["date"]) >= CUTOFF
    ].copy()
    forecast_test = _align_source_to_reference(
        forecast_test_source, reference, "Final Forecasting benchmark"
    )
    now_test = _align_source_to_reference(
        nowcasting, reference, "Final Nowcasting benchmark"
    )
    layer1_test = forecast_test.loc[:, list(layer1_features)].reset_index(drop=True)
    layer2_test = now_test.loc[:, list(NOWCAST_FEATURES)].reset_index(drop=True)
    _validate_native_matrix(layer1_test, "Final Layer-1 benchmark matrix")
    _validate_native_matrix(layer2_test, "Final Layer-2 benchmark matrix")

    bundles: dict[str, dict[str, object]] = {}
    audit_rows: list[dict[str, object]] = []
    policy_output = policies.copy()
    classifier_parameters = _direct_parameters("classifier")
    regressor_parameters = _direct_parameters("regressor")
    training_years = tuple(
        sorted(pd.to_datetime(population["date"]).dt.year.unique().tolist())
    )
    for candidate in CANDIDATE_ORDER:
        sample_weight, weight_audit = build_candidate_sample_weight(
            target, candidate
        )
        policy_mask = policy_output["candidate_id"].eq(candidate)
        policy_output.loc[
            policy_mask, "final_training_negative_count"
        ] = weight_audit["negative_count"]
        policy_output.loc[
            policy_mask, "final_training_positive_count"
        ] = weight_audit["positive_count"]
        policy_output.loc[policy_mask, "final_class_ratio"] = weight_audit[
            "class_ratio"
        ]
        policy_output.loc[
            policy_mask, "final_positive_row_weight"
        ] = weight_audit["positive_row_weight"]

        forecast_fit = fit_direct_candidate_models(
            task="Forecasting",
            layer1_matrix=layer1_training,
            layer2_matrix=None,
            target=target,
            sample_weight=sample_weight,
            classifier_parameters=classifier_parameters,
        )
        nowcast_fit = fit_direct_candidate_models(
            task="Nowcasting",
            layer1_matrix=layer1_training,
            layer2_matrix=layer2_training,
            target=target,
            sample_weight=sample_weight,
            classifier_parameters=classifier_parameters,
            regressor_parameters=regressor_parameters,
        )
        forecast_path = (
            staging_dir / f"{PREFIX}{candidate}_forecasting_model.json"
        )
        nowcast_layer1_path = (
            staging_dir / f"{PREFIX}{candidate}_nowcasting_layer1_model.json"
        )
        nowcast_layer2_path = (
            staging_dir / f"{PREFIX}{candidate}_nowcasting_layer2_model.json"
        )
        _save_published_model(forecast_fit["layer1_model"], forecast_path)
        _save_published_model(nowcast_fit["layer1_model"], nowcast_layer1_path)
        _save_published_model(nowcast_fit["layer2_model"], nowcast_layer2_path)

        forecast_score = np.asarray(
            forecast_fit["layer1_model"].predict_proba(layer1_test)[:, 1],
            dtype=float,
        )
        nowcast_layer1_score = np.asarray(
            nowcast_fit["layer1_model"].predict_proba(layer1_test)[:, 1],
            dtype=float,
        )
        nowcast_residual_score = np.asarray(
            nowcast_fit["layer2_model"].predict(layer2_test), dtype=float
        )
        nowcast_score = predecessor.combine_direct_nowcasting_scores(
            nowcast_layer1_score, nowcast_residual_score
        )
        bundles[candidate] = {
            "Forecasting": {
                "score": forecast_score,
                "layer1_score": forecast_score,
                "residual_score": np.full(len(forecast_score), np.nan),
                "model_paths": (forecast_path,),
            },
            "Nowcasting": {
                "score": nowcast_score,
                "layer1_score": nowcast_layer1_score,
                "residual_score": nowcast_residual_score,
                "model_paths": (nowcast_layer1_path, nowcast_layer2_path),
            },
        }

        target_definition = (
            "1[reconstructed_overall_phase in {4,5}] among reconstructed "
            "Phase 3/4/5"
        )
        audit_rows.append(
            _model_audit_row(
                task="Forecasting",
                candidate_id=candidate,
                fit_scope="final_refit",
                fold_id="final_pre2022",
                model_component="direct_forecasting_classifier",
                estimator_class="XGBClassifier",
                objective="binary:logistic",
                target_definition=target_definition,
                run_status="fitted_published",
                training_years=training_years,
                validation_year=np.nan,
                training_keys=training_keys,
                binary_target=target,
                model_target=target,
                training_matrix=layer1_training,
                sample_weight=sample_weight,
                weight_audit=weight_audit,
                features=layer1_features,
                parameters=classifier_parameters,
                validation_keys=forecast_test,
                validation_matrix=layer1_test,
                validation_score=forecast_score,
                model=forecast_fit["layer1_model"],
                model_hash_path=None,
                published_model_path=forecast_path,
            )
        )
        audit_rows.append(
            _model_audit_row(
                task="Nowcasting",
                candidate_id=candidate,
                fit_scope="final_refit",
                fold_id="final_pre2022",
                model_component="direct_nowcasting_layer1_classifier",
                estimator_class="XGBClassifier",
                objective="binary:logistic",
                target_definition=target_definition,
                run_status="fitted_published",
                training_years=training_years,
                validation_year=np.nan,
                training_keys=training_keys,
                binary_target=target,
                model_target=target,
                training_matrix=layer1_training,
                sample_weight=sample_weight,
                weight_audit=weight_audit,
                features=layer1_features,
                parameters=classifier_parameters,
                validation_keys=forecast_test,
                validation_matrix=layer1_test,
                validation_score=nowcast_layer1_score,
                model=nowcast_fit["layer1_model"],
                model_hash_path=None,
                published_model_path=nowcast_layer1_path,
                layer1_training_score=nowcast_fit["layer1_training_score"],
            )
        )
        residual_target = pd.Series(
            nowcast_fit["residual_target"], name="residual_target"
        )
        audit_rows.append(
            _model_audit_row(
                task="Nowcasting",
                candidate_id=candidate,
                fit_scope="final_refit",
                fold_id="final_pre2022",
                model_component="direct_nowcasting_layer2_regressor",
                estimator_class="XGBRegressor",
                objective="reg:squarederror",
                target_definition=(
                    "severe_rescue_target - direct Nowcasting Layer-1 "
                    "training score"
                ),
                run_status="fitted_published",
                training_years=training_years,
                validation_year=np.nan,
                training_keys=training_keys,
                binary_target=target,
                model_target=residual_target,
                training_matrix=layer2_training,
                sample_weight=sample_weight,
                weight_audit=weight_audit,
                features=NOWCAST_FEATURES,
                parameters=regressor_parameters,
                validation_keys=now_test,
                validation_matrix=layer2_test,
                validation_score=nowcast_residual_score,
                model=nowcast_fit["layer2_model"],
                model_hash_path=None,
                published_model_path=nowcast_layer2_path,
                layer1_training_score=nowcast_fit["layer1_training_score"],
                residual_target=nowcast_fit["residual_target"],
            )
        )
    final_audit = pd.DataFrame(audit_rows, columns=MODEL_AUDIT_COLUMNS)
    if len(final_audit) != 9:
        raise ValueError("Final model audit must contain nine rows.")
    return bundles, final_audit, policy_output.loc[:, list(SELECTED_POLICY_COLUMNS)]


def _method_contract(
    method: str, policies: pd.DataFrame
) -> tuple[object, str, str, bool]:
    if method == "frozen_base":
        return np.nan, "frozen_reference", "not_applicable", False
    if method == "legacy_direct_exact_phase4_050":
        return np.nan, "legacy_reporting_only", "not_applicable", False
    candidate = METHOD_CANDIDATES[method]
    policy = policies.loc[policies["candidate_id"].eq(candidate)].iloc[0]
    return (
        candidate,
        "candidate_sensitivity",
        str(policy["selection_status"]),
        bool(policy["primary_selected"]),
    )


def build_benchmark_predictions(
    benchmark_evidence: pd.DataFrame,
    bundles: Mapping[str, Mapping[str, Mapping[str, object]]],
    policies: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    policy_index = policies.set_index(["task", "candidate_id"])
    for task in TASK_ORDER:
        base = benchmark_evidence.loc[
            benchmark_evidence["task"].eq(task)
        ].copy()
        base = base.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
            drop=True
        )
        base_phase = base["base_overall_phase_pred"].to_numpy(dtype=int)
        for method in METHOD_ORDER:
            candidate_id, role, status, primary = _method_contract(
                method,
                policies.loc[policies["task"].eq(task)],
            )
            if method == "frozen_base":
                layer1_score = np.full(len(base), np.nan)
                residual_score = np.full(len(base), np.nan)
                score = np.full(len(base), np.nan)
                threshold = math.nan
                triggered = np.zeros(len(base), dtype=bool)
                final_phase = base_phase.copy()
            elif method == "legacy_direct_exact_phase4_050":
                layer1_score = base["direct_layer1_phase4_score"].to_numpy(
                    dtype=float
                )
                residual_score = base[
                    "direct_layer2_residual_score"
                ].to_numpy(dtype=float)
                score = base["direct_phase4_score"].to_numpy(dtype=float)
                threshold = 0.5
                triggered = base["direct_triggered"].to_numpy(dtype=bool)
                final_phase = base["direct_overall_phase_pred"].to_numpy(dtype=int)
                predecessor.assert_postclassification_invariants(
                    base_phase, final_phase
                )
            else:
                candidate = str(candidate_id)
                contract = bundles[candidate][task]
                layer1_score = np.asarray(contract["layer1_score"], dtype=float)
                residual_score = np.asarray(contract["residual_score"], dtype=float)
                score = np.asarray(contract["score"], dtype=float)
                threshold = float(policy_index.loc[(task, candidate), "threshold"])
                final_phase = apply_phase45_rescue(base_phase, score, threshold)
                triggered = base_phase != final_phase
            output = pd.DataFrame(
                {
                    "task": task,
                    "method": method,
                    "candidate_id": candidate_id,
                    "method_role": role,
                    "selection_status": status,
                    "primary_selected": primary,
                    "area_id": base["area_id"].astype(int),
                    "date": base["date"],
                    "country_code_3": base["country_code_3"],
                    "source_overall_phase": base["source_overall_phase"].astype(int),
                    "reconstructed_overall_phase": base[
                        "reconstructed_overall_phase"
                    ].astype(int),
                    "severe_rescue_target": base["severe_rescue_target"].astype(
                        np.uint8
                    ),
                    "phase2_test": base["phase2_test"],
                    "phase3_test": base["phase3_test"],
                    "phase4_test": base["phase4_test"],
                    "phase5_test": base["phase5_test"],
                    "phase2_pred_rounded": base["phase2_pred_rounded"],
                    "phase3_pred_rounded": base["phase3_pred_rounded"],
                    "phase4_pred_rounded": base["phase4_pred_rounded"],
                    "phase5_pred_rounded": base["phase5_pred_rounded"],
                    "base_overall_phase_pred": base_phase,
                    "in_base_phase3_gate": base["in_base_phase3_gate"].astype(bool),
                    "direct_layer1_score": layer1_score,
                    "direct_layer2_residual_score": residual_score,
                    "direct_phase45_score": score,
                    "threshold": threshold,
                    "triggered": triggered,
                    "final_overall_phase_pred": final_phase,
                }
            )
            rows.append(output.loc[:, list(BENCHMARK_COLUMNS)])
    result = pd.concat(rows, ignore_index=True)
    result["method"] = pd.Categorical(
        result["method"], categories=list(METHOD_ORDER), ordered=True
    )
    result = result.sort_values(
        ["task", "method", *KEY_COLUMNS], kind="mergesort"
    ).reset_index(drop=True)
    result["method"] = result["method"].astype(str)
    if len(result) != EXPECTED_BENCHMARK_ROWS_LONG or result.duplicated(
        ["task", "method", *KEY_COLUMNS]
    ).any():
        raise ValueError("Benchmark predictions violate the 11,700-row contract.")
    for (task, method), group in result.groupby(
        ["task", "method"], sort=False, observed=True
    ):
        if len(group) != EXPECTED_BENCHMARK_ROWS:
            raise ValueError(f"{task}/{method} does not contain 1,170 rows.")
        if canonical_key_sha256(group) != EXPECTED_BENCHMARK_KEY_SHA256:
            raise ValueError(f"{task}/{method} benchmark key hash drifted.")
        predecessor.assert_postclassification_invariants(
            group["base_overall_phase_pred"],
            group["final_overall_phase_pred"],
        )
    return result


def _balanced_accuracy(
    actual_phase: Sequence[int], predicted_phase: Sequence[int]
) -> float:
    actual = np.asarray(actual_phase, dtype=int)
    predicted = np.asarray(predicted_phase, dtype=int)
    recalls = [
        float(np.mean(predicted[actual == phase] == phase))
        for phase in sorted(np.unique(actual))
    ]
    return float(np.mean(recalls)) if recalls else math.nan


def _macro_f1(actual_phase: Sequence[int], predicted_phase: Sequence[int]) -> float:
    actual = np.asarray(actual_phase, dtype=int)
    predicted = np.asarray(predicted_phase, dtype=int)
    values: list[float] = []
    for phase in ALL_PHASES:
        tp = int(np.sum((actual == phase) & (predicted == phase)))
        fp = int(np.sum((actual != phase) & (predicted == phase)))
        fn = int(np.sum((actual == phase) & (predicted != phase)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return float(np.mean(values))


def _class_metric_bundle(
    actual_phase: Sequence[int], predicted_phase: Sequence[int], phase: int
) -> dict[str, float | int]:
    actual = np.asarray(actual_phase, dtype=int)
    predicted = np.asarray(predicted_phase, dtype=int)
    tp = int(np.sum((actual == phase) & (predicted == phase)))
    fp = int(np.sum((actual != phase) & (predicted == phase)))
    fn = int(np.sum((actual == phase) & (predicted != phase)))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
    }


def _benchmark_metric_core(frame: pd.DataFrame) -> dict[str, object]:
    actual = frame["reconstructed_overall_phase"].to_numpy(dtype=int)
    predicted = frame["final_overall_phase_pred"].to_numpy(dtype=int)
    base = frame["base_overall_phase_pred"].to_numpy(dtype=int)
    predecessor.assert_postclassification_invariants(base, predicted)
    severe = np.isin(actual, [4, 5])
    predicted_severe = np.isin(predicted, [4, 5])
    severe_metrics = binary_metric_bundle(
        severe.astype(int), predicted_severe.astype(int)
    )
    phase3 = _class_metric_bundle(actual, predicted, 3)
    phase4 = _class_metric_bundle(actual, predicted, 4)
    phase5 = _class_metric_bundle(actual, predicted, 5)
    actual_phase3plus = actual >= 3
    predicted_phase3plus = predicted >= 3
    phase3plus = binary_metric_bundle(
        actual_phase3plus.astype(int), predicted_phase3plus.astype(int)
    )
    if len(frame) < 2 or frame["phase3_test"].nunique(dropna=False) < 2:
        phase3plus_r2 = math.nan
    else:
        phase3plus_r2 = float(
            r2_score(frame["phase3_test"], frame["phase3_pred_rounded"])
        )
    changed = base != predicted
    return {
        "n_rows": len(frame),
        "n_gate": int(np.sum(base == 3)),
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": _balanced_accuracy(actual, predicted),
        "macro_f1": _macro_f1(actual, predicted),
        "ordinal_mae": float(np.mean(np.abs(actual - predicted))),
        "phase3_precision": phase3["precision"],
        "phase3_recall": phase3["recall"],
        "phase3_f1": phase3["f1"],
        "phase4_precision": phase4["precision"],
        "phase4_recall": phase4["recall"],
        "phase4_f1": phase4["f1"],
        "phase5_precision": phase5["precision"],
        "phase5_recall": phase5["recall"],
        "phase5_f1": phase5["f1"],
        "phase45_precision": severe_metrics["precision"],
        "phase45_recall": severe_metrics["recall"],
        "phase45_f1": severe_metrics["f1"],
        "phase45_f2": severe_metrics["f2"],
        "phase3plus_precision": phase3plus["precision"],
        "phase3plus_recall": phase3plus["recall"],
        "phase3plus_r2": phase3plus_r2,
        "changed_3_to_4": int(changed.sum()),
        "exact_phase4_rescues": int(np.sum(changed & (actual == 4))),
        "actual_phase5_promoted_to_phase4": int(
            np.sum(changed & (actual == 5) & (predicted == 4))
        ),
        "false_promotion_actual_phase1": int(
            np.sum(changed & (actual == 1))
        ),
        "false_promotion_actual_phase2": int(
            np.sum(changed & (actual == 2))
        ),
        "false_promotion_actual_phase3": int(
            np.sum(changed & (actual == 3))
        ),
    }


def _gate_metric_record(frame: pd.DataFrame) -> dict[str, object]:
    gate = frame.loc[frame["in_base_phase3_gate"].astype(bool)].copy()
    actual = gate["reconstructed_overall_phase"].to_numpy(dtype=int)
    final_phase = gate["final_overall_phase_pred"].to_numpy(dtype=int)
    severe = np.isin(actual, [4, 5])
    predicted_severe = np.isin(final_phase, [4, 5])
    metrics = binary_metric_bundle(
        severe.astype(int), predicted_severe.astype(int)
    )
    promoted = gate["triggered"].to_numpy(dtype=bool)
    false_promotions = promoted & ~severe
    score = gate["direct_phase45_score"].to_numpy(dtype=float)
    return {
        "task": str(gate["task"].iloc[0]),
        "method": str(gate["method"].iloc[0]),
        "method_role": str(gate["method_role"].iloc[0]),
        "selection_status": str(gate["selection_status"].iloc[0]),
        "primary_selected": bool(gate["primary_selected"].iloc[0]),
        "n_rows": len(gate),
        "positive_support": int(severe.sum()),
        "negative_support": int((~severe).sum()),
        "actual_phase4_support": int(np.sum(actual == 4)),
        "actual_phase5_support": int(np.sum(actual == 5)),
        "predicted_positive_count": int(predicted_severe.sum()),
        "true_positive": metrics["true_positive"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_negative": metrics["true_negative"],
        "phase45_precision": metrics["precision"],
        "phase45_recall": metrics["recall"],
        "phase45_f1": metrics["f1"],
        "phase45_f2": metrics["f2"],
        "phase45_pr_auc": _safe_pr_auc(severe.astype(int), score),
        "true_rescue_actual_phase4": int(np.sum(promoted & (actual == 4))),
        "true_rescue_actual_phase5": int(np.sum(promoted & (actual == 5))),
        "false_promotion_actual_phase1": int(
            np.sum(false_promotions & (actual == 1))
        ),
        "false_promotion_actual_phase2": int(
            np.sum(false_promotions & (actual == 2))
        ),
        "false_promotion_actual_phase3": int(
            np.sum(false_promotions & (actual == 3))
        ),
        "false_promotion_actual_phase12": int(
            np.sum(false_promotions & np.isin(actual, [1, 2]))
        ),
        "total_promotions": int(promoted.sum()),
        "false_promotions_per_100_gate_rows": (
            100.0 * int(false_promotions.sum()) / len(gate)
        ),
    }


def calculate_metric_tables(
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            frame = benchmark.loc[
                benchmark["task"].eq(task) & benchmark["method"].eq(method)
            ].copy()
            if len(frame) != EXPECTED_BENCHMARK_ROWS:
                raise ValueError(f"{task}/{method} benchmark support drifted.")
            gate_rows.append(_gate_metric_record(frame))
            metadata = {
                "task": task,
                "method": method,
                "method_role": str(frame["method_role"].iloc[0]),
                "selection_status": str(frame["selection_status"].iloc[0]),
                "primary_selected": bool(frame["primary_selected"].iloc[0]),
            }
            pooled_rows.append(
                {
                    **metadata,
                    **_benchmark_metric_core(frame),
                    "phase3plus_precision_delta_from_base": np.nan,
                    "phase3plus_recall_delta_from_base": np.nan,
                    "phase3plus_r2_delta_from_base": np.nan,
                }
            )
            for country, group in frame.groupby(
                "country_code_3", sort=True, observed=True
            ):
                country_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "country_code_3": country,
                        "method_role": metadata["method_role"],
                        "selection_status": metadata["selection_status"],
                        "primary_selected": metadata["primary_selected"],
                        **_benchmark_metric_core(group),
                    }
                )
    gate_metrics = pd.DataFrame(gate_rows, columns=GATE_METRIC_COLUMNS)
    pooled = pd.DataFrame(pooled_rows, columns=BENCHMARK_METRIC_COLUMNS)
    for task in TASK_ORDER:
        base = pooled.loc[
            pooled["task"].eq(task) & pooled["method"].eq("frozen_base")
        ].iloc[0]
        for index in pooled.index[pooled["task"].eq(task)]:
            for metric in (
                "phase3plus_precision",
                "phase3plus_recall",
                "phase3plus_r2",
            ):
                delta = float(pooled.loc[index, metric] - base[metric])
                if delta != 0.0:
                    raise ValueError(
                        f"{task}/{pooled.loc[index, 'method']} violates "
                        f"the {metric} invariant."
                    )
                pooled.loc[index, f"{metric}_delta_from_base"] = 0.0
    countries = pd.DataFrame(country_rows, columns=COUNTRY_METRIC_COLUMNS)
    if len(countries) != EXPECTED_COUNTRY_METRIC_ROWS or countries.duplicated(
        ["task", "method", "country_code_3"]
    ).any():
        raise ValueError("Country metrics violate the 270-row contract.")
    macro_rows: list[dict[str, object]] = []
    for (task, method), group in countries.groupby(
        ["task", "method"], sort=True, observed=True
    ):
        row: dict[str, object] = {
            "task": task,
            "method": method,
            "method_role": str(group["method_role"].iloc[0]),
            "selection_status": str(group["selection_status"].iloc[0]),
            "primary_selected": bool(group["primary_selected"].iloc[0]),
            "countries_total": int(group["country_code_3"].nunique()),
        }
        if row["countries_total"] != EXPECTED_BENCHMARK_COUNTRIES:
            raise ValueError(f"{task}/{method} country support drifted.")
        for metric in MACRO_METRICS:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[metric] = float(np.mean(finite)) if finite.size else math.nan
            row[f"{metric}_countries_defined"] = int(finite.size)
        macro_rows.append(row)
    country_macro = pd.DataFrame(macro_rows, columns=COUNTRY_MACRO_COLUMNS)
    if (
        len(gate_metrics) != 10
        or len(pooled) != 10
        or len(country_macro) != 10
        or gate_metrics.duplicated(["task", "method"]).any()
        or pooled.duplicated(["task", "method"]).any()
        or country_macro.duplicated(["task", "method"]).any()
    ):
        raise ValueError("Pooled or country-macro metric contract failed.")
    return gate_metrics, pooled, countries, country_macro


def build_confusion_tables(
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    binary_rows: list[pd.DataFrame] = []
    five_rows: list[pd.DataFrame] = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            frame = benchmark.loc[
                benchmark["task"].eq(task) & benchmark["method"].eq(method)
            ].copy()
            actual = frame["reconstructed_overall_phase"].to_numpy(dtype=int)
            predicted = frame["final_overall_phase_pred"].to_numpy(dtype=int)
            for scope, scoped in (
                ("gate", frame.loc[frame["in_base_phase3_gate"].astype(bool)]),
                ("benchmark", frame),
            ):
                scoped_actual = scoped[
                    "reconstructed_overall_phase"
                ].to_numpy(dtype=int)
                scoped_predicted = scoped[
                    "final_overall_phase_pred"
                ].to_numpy(dtype=int)
                cells = build_binary_confusion_cells(
                    np.isin(scoped_actual, [4, 5]).astype(int),
                    np.isin(scoped_predicted, [4, 5]).astype(int),
                )
                cells.insert(0, "method", method)
                cells.insert(0, "task", task)
                cells.insert(0, "population_scope", scope)
                binary_rows.append(cells.loc[:, list(BINARY_CONFUSION_COLUMNS)])
            cells5 = build_five_class_confusion_cells(actual, predicted)
            cells5.insert(0, "method", method)
            cells5.insert(0, "task", task)
            five_rows.append(cells5.loc[:, list(FIVE_CLASS_CONFUSION_COLUMNS)])
    binary = pd.concat(binary_rows, ignore_index=True)
    five = pd.concat(five_rows, ignore_index=True)
    if len(binary) != EXPECTED_BINARY_CONFUSION_ROWS or binary.duplicated(
        [
            "population_scope",
            "task",
            "method",
            "actual_binary",
            "predicted_binary",
        ]
    ).any():
        raise ValueError("Binary confusion matrices violate the 80-row contract.")
    if len(five) != EXPECTED_FIVE_CLASS_CONFUSION_ROWS or five.duplicated(
        ["task", "method", "actual_phase", "predicted_phase"]
    ).any():
        raise ValueError("Five-class confusion matrices violate the 250-row contract.")
    return binary, five


def _bootstrap_metric_bundle(
    actual_phase: np.ndarray, predicted_phase: np.ndarray
) -> dict[str, float]:
    severe = np.isin(actual_phase, [4, 5])
    predicted_severe = np.isin(predicted_phase, [4, 5])
    binary = binary_metric_bundle(severe.astype(int), predicted_severe.astype(int))
    return {
        "phase45_precision": float(binary["precision"]),
        "phase45_recall": float(binary["recall"]),
        "phase45_f1": float(binary["f1"]),
        "phase45_f2": float(binary["f2"]),
        "accuracy": float(np.mean(actual_phase == predicted_phase)),
        "balanced_accuracy": _balanced_accuracy(actual_phase, predicted_phase),
        "macro_f1": _macro_f1(actual_phase, predicted_phase),
        "ordinal_mae": float(np.mean(np.abs(actual_phase - predicted_phase))),
    }


def generate_bootstrap(
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_frames: dict[str, pd.DataFrame] = {}
    country_sets: list[tuple[str, ...]] = []
    for task in TASK_ORDER:
        base = benchmark.loc[
            benchmark["task"].eq(task) & benchmark["method"].eq("frozen_base")
        ].sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
        base_frames[task] = base
        country_sets.append(
            tuple(sorted(base["country_code_3"].unique().tolist()))
        )
    if (
        country_sets[0] != country_sets[1]
        or len(country_sets[0]) != EXPECTED_BENCHMARK_COUNTRIES
    ):
        raise ValueError("Bootstrap country universe drifted across tasks.")
    countries = np.asarray(country_sets[0], dtype=object)
    rng = np.random.RandomState(BOOTSTRAP_RANDOM_STATE)
    samples = [
        rng.choice(countries, size=len(countries), replace=True)
        for _ in range(BOOTSTRAP_REPETITIONS)
    ]
    draw_rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        base = base_frames[task]
        country_indices = {
            country: base.index[
                base["country_code_3"].eq(country)
            ].to_numpy(dtype=int)
            for country in countries
        }
        actual_all = base["reconstructed_overall_phase"].to_numpy(dtype=int)
        base_predicted_all = base["final_overall_phase_pred"].to_numpy(dtype=int)
        method_predictions: dict[str, np.ndarray] = {}
        for method in METHOD_ORDER[1:]:
            frame = benchmark.loc[
                benchmark["task"].eq(task) & benchmark["method"].eq(method)
            ].sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
            if not canonical_key_frame(frame).equals(canonical_key_frame(base)):
                raise ValueError(f"{task}/{method} bootstrap key alignment failed.")
            method_predictions[method] = frame[
                "final_overall_phase_pred"
            ].to_numpy(dtype=int)
        for bootstrap_id, sample in enumerate(samples, start=1):
            indices = np.concatenate([country_indices[country] for country in sample])
            actual = actual_all[indices]
            base_metrics = _bootstrap_metric_bundle(
                actual, base_predicted_all[indices]
            )
            sample_hash = json_sha256([str(country) for country in sample])
            for method in METHOD_ORDER[1:]:
                current = _bootstrap_metric_bundle(
                    actual, method_predictions[method][indices]
                )
                row: dict[str, object] = {
                    "task": task,
                    "method": method,
                    "bootstrap_id": bootstrap_id,
                    "sampled_country_count": len(sample),
                    "sample_row_count": len(indices),
                    "sampled_countries_sha256": sample_hash,
                }
                for metric in BOOTSTRAP_METRICS:
                    left = current[metric]
                    right = base_metrics[metric]
                    row[f"{metric}_delta"] = (
                        left - right
                        if math.isfinite(left) and math.isfinite(right)
                        else math.nan
                    )
                draw_rows.append(row)
    draws = pd.DataFrame(draw_rows, columns=BOOTSTRAP_DRAW_COLUMNS)
    if len(draws) != EXPECTED_BOOTSTRAP_DRAW_ROWS or draws.duplicated(
        ["task", "method", "bootstrap_id"]
    ).any():
        raise ValueError("Bootstrap draws violate the 16,000-row contract.")
    summary_rows: list[dict[str, object]] = []
    for (task, method), group in draws.groupby(
        ["task", "method"], sort=True, observed=True
    ):
        row: dict[str, object] = {
            "task": task,
            "method": method,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "ci_level": 0.95,
            "interval_method": "paired_country_cluster_percentile",
        }
        for metric in BOOTSTRAP_METRICS:
            values = group[f"{metric}_delta"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                lower, upper = np.percentile(finite, [2.5, 97.5])
            else:
                lower = upper = math.nan
            row[f"{metric}_delta_lower"] = lower
            row[f"{metric}_delta_upper"] = upper
            row[f"{metric}_finite_draws"] = int(finite.size)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows, columns=BOOTSTRAP_SUMMARY_COLUMNS)
    if len(summary) != EXPECTED_BOOTSTRAP_SUMMARY_ROWS or summary.duplicated(
        ["task", "method"]
    ).any():
        raise ValueError("Bootstrap summary violates the eight-row contract.")
    return draws, summary


METHOD_DISPLAY_NAMES = {
    "frozen_base": "Frozen base",
    "legacy_direct_exact_phase4_050": "Legacy exact-P4",
    "direct_phase45_unweighted": "P4/5 unweighted",
    "direct_phase45_sqrt_balance": "P4/5 sqrt-balance",
    "direct_phase45_full_balance": "P4/5 full-balance",
}
METHOD_COLORS = {
    "frozen_base": "#7A7A7A",
    "legacy_direct_exact_phase4_050": "#5B8FF9",
    "direct_phase45_unweighted": "#61DDAA",
    "direct_phase45_sqrt_balance": "#F6BD16",
    "direct_phase45_full_balance": "#E8684A",
}
PROMOTION_COLORS = {
    "true_rescue_actual_phase4": "#2E8B57",
    "true_rescue_actual_phase5": "#76B947",
    "false_promotion_actual_phase1": "#A9A9A9",
    "false_promotion_actual_phase2": "#D6A85F",
    "false_promotion_actual_phase3": "#C65D57",
}
PROMOTION_LABELS = {
    "true_rescue_actual_phase4": "True P4 rescue",
    "true_rescue_actual_phase5": "True P5 rescue",
    "false_promotion_actual_phase1": "False P1 promotion",
    "false_promotion_actual_phase2": "False P2 promotion",
    "false_promotion_actual_phase3": "False P3 promotion",
}
FIGURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.bbox": None,
}


def _panel_letter(axis: mpl.axes.Axes, letter: str) -> None:
    axis.text(
        -0.12,
        1.06,
        letter,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _assert_suptitle_legend_separation(figure: mpl.figure.Figure) -> None:
    if figure._suptitle is None:
        raise ValueError("Main comparison figure is missing its required suptitle.")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    title_box = figure._suptitle.get_window_extent(renderer=renderer)
    for axis in figure.axes:
        legend = axis.get_legend()
        if legend is None:
            continue
        if title_box.overlaps(legend.get_window_extent(renderer=renderer)):
            raise ValueError("Main comparison suptitle overlaps a panel legend.")


def _save_figure(
    figure: mpl.figure.Figure,
    pdf_path: Path,
    png_path: Path,
    title: str,
) -> None:
    pdf_metadata = {
        "Title": title,
        "Author": "",
        "Subject": (
            "Direct Phase-3 versus Phase-4/5 rescue sensitivity comparison"
        ),
        "Keywords": "",
        "Creator": "NCOMMS Direct Phase-3-vs-Phase-4/5 rescue generator",
        "CreationDate": None,
        "ModDate": None,
    }
    png_metadata = {
        "Software": "NCOMMS Direct Phase-3-vs-Phase-4/5 rescue generator"
    }
    figure.savefig(
        pdf_path,
        format="pdf",
        metadata=pdf_metadata,
        facecolor="white",
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=600,
        metadata=png_metadata,
        facecolor="white",
    )
    plt.close(figure)


def render_main_comparison_figure(
    gate_metrics: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
) -> None:
    metric_names = (
        "phase45_precision",
        "phase45_recall",
        "phase45_f1",
        "phase45_f2",
    )
    metric_labels = ("Precision", "Recall", "F1", "F2")
    with mpl.rc_context(FIGURE_RC):
        figure, axes = plt.subplots(3, 2, figsize=(12.0, 10.6), squeeze=False)
        figure.subplots_adjust(
            left=0.08,
            right=0.985,
            bottom=0.125,
            top=0.82,
            wspace=0.27,
            hspace=0.62,
        )
        for row_index, task in enumerate(FIGURE_TASK_ORDER):
            task_metrics = gate_metrics.loc[
                gate_metrics["task"].eq(task)
            ].set_index("method").loc[list(FIGURE_METHOD_ORDER)]
            population = FIGURE_POPULATIONS[task]
            axis = axes[row_index, 0]
            x = np.arange(len(metric_names), dtype=float)
            width = 0.22
            offsets = np.linspace(-width, width, len(FIGURE_METHOD_ORDER))
            for offset, method in zip(offsets, FIGURE_METHOD_ORDER):
                values = task_metrics.loc[method, list(metric_names)].to_numpy(
                    dtype=float
                )
                primary = bool(task_metrics.loc[method, "primary_selected"])
                axis.bar(
                    x + offset,
                    values,
                    width=width * 0.92,
                    color=METHOD_COLORS[method],
                    edgecolor="black" if primary else "white",
                    linewidth=1.2 if primary else 0.4,
                    label=METHOD_DISPLAY_NAMES[method],
                    zorder=3,
                )
            axis.set_xticks(x, metric_labels)
            axis.set_ylim(0.0, 1.02)
            axis.set_ylabel("P4/5 metric")
            axis.set_title(
                f"{task}: severe-outcome performance\n"
                f"{population['protocol']}; n = {population['n']:,}"
            )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8, zorder=0)
            _panel_letter(axis, chr(ord("a") + row_index * 2))

            axis = axes[row_index, 1]
            methods = list(FIGURE_METHOD_ORDER)
            y = np.arange(len(methods))
            left = np.zeros(len(methods), dtype=float)
            for category in PROMOTION_COLORS:
                values = task_metrics.loc[methods, category].to_numpy(dtype=float)
                axis.barh(
                    y,
                    values,
                    left=left,
                    height=0.66,
                    color=PROMOTION_COLORS[category],
                    edgecolor="white",
                    linewidth=0.4,
                    label=PROMOTION_LABELS[category],
                )
                left += values
            method_labels = [
                (
                    f"{METHOD_DISPLAY_NAMES[method]} (primary)"
                    if bool(task_metrics.loc[method, "primary_selected"])
                    else METHOD_DISPLAY_NAMES[method]
                )
                for method in methods
            ]
            axis.set_yticks(
                y, method_labels
            )
            axis.invert_yaxis()
            axis.set_xlabel("Promoted base-Phase-3 rows")
            axis.set_title(
                f"{task}: promotion composition\n"
                f"{population['protocol']}; n = {population['n']:,}"
            )
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.8)
            _panel_letter(axis, chr(ord("b") + row_index * 2))
        method_handles = [
            mpl_patches.Patch(
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                label=METHOD_DISPLAY_NAMES[method],
            )
            for method in FIGURE_METHOD_ORDER
        ]
        promotion_handles = [
            mpl_patches.Patch(
                facecolor=PROMOTION_COLORS[category],
                edgecolor="white",
                label=PROMOTION_LABELS[category],
            )
            for category in PROMOTION_COLORS
        ]
        figure.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.285, 0.91),
            ncol=3,
            fontsize=7,
        )
        figure.legend(
            handles=promotion_handles,
            loc="upper center",
            bbox_to_anchor=(0.735, 0.91),
            ncol=3,
            fontsize=7,
        )
        figure.suptitle(
            "Aggressive Phase-4/5 rescue trades severe-outcome recovery "
            "against visible false promotions",
            fontsize=11,
            fontweight="bold",
            y=0.975,
        )
        figure.text(
            0.08,
            0.022,
            (
                "Forecasting and Nowcasting use the fixed 2022 temporal holdout "
                "(n = 1,170 each). Contemporaneous uses seed-0 random five-fold "
                "row-level full-OOF predictions (n = 5,575) with thresholds selected "
                "and described on the same pooled OOF gate. Protocols and populations "
                "differ; values are not directly comparable. Rescue target = actual "
                "P4/5; allowed action = base P3 to predicted P4 only. Black outlines "
                "and '(primary)' labels mark the task-specific OOF-selected policy."
            ),
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#444444",
            wrap=True,
        )
        _assert_suptitle_legend_separation(figure)
        _save_figure(
            figure,
            output_pdf,
            output_png,
            "Direct Phase-3 versus Phase-4/5 rescue main comparison",
        )


def _blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    rgb = np.asarray(mpl_colors.to_rgb(color))
    alpha = min(max(float(strength), 0.0), 1.0)
    return tuple((1.0 - alpha) * np.ones(3) + alpha * rgb)


def render_binary_confusion_atlas(
    binary_confusion: pd.DataFrame,
    gate_metrics: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
) -> None:
    with mpl.rc_context(FIGURE_RC):
        figure, axes = plt.subplots(3, 3, figsize=(10.5, 8.8), squeeze=False)
        figure.subplots_adjust(
            left=0.205,
            right=0.985,
            bottom=0.145,
            top=0.84,
            wspace=0.30,
            hspace=0.54,
        )
        for row_index, task in enumerate(FIGURE_TASK_ORDER):
            population = FIGURE_POPULATIONS[task]
            task_primary = gate_metrics.loc[
                gate_metrics["task"].eq(task)
                & gate_metrics["primary_selected"].astype(bool),
                "method",
            ]
            if len(task_primary) != 1:
                raise ValueError(f"{task} requires one primary rescue method.")
            primary_method = str(task_primary.iloc[0])
            for column_index, method in enumerate(FIGURE_METHOD_ORDER):
                axis = axes[row_index, column_index]
                matrix = binary_confusion.loc[
                    binary_confusion["population_scope"].eq(population["scope"])
                    & binary_confusion["task"].eq(task)
                    & binary_confusion["method"].eq(method)
                ].set_index(["actual_binary", "predicted_binary"])
                if len(matrix) != 4:
                    raise ValueError(
                        f"{task}/{method} binary atlas matrix is incomplete."
                    )
                for actual_value in (0, 1):
                    for predicted_value in (0, 1):
                        cell = matrix.loc[(actual_value, predicted_value)]
                        share = float(cell["actual_row_share"])
                        strength = 0.12 + 0.78 * (share if math.isfinite(share) else 0.0)
                        rectangle = mpl_patches.Rectangle(
                            (predicted_value, 1 - actual_value),
                            1,
                            1,
                            facecolor=_blend_with_white(
                                METHOD_COLORS[method], strength
                            ),
                            edgecolor="white",
                            linewidth=1.0,
                        )
                        axis.add_patch(rectangle)
                        label = (
                            f"{int(cell['count'])}\n"
                            f"{share:.1%}" if math.isfinite(share) else f"{int(cell['count'])}\nNA"
                        )
                        axis.text(
                            predicted_value + 0.5,
                            1 - actual_value + 0.5,
                            label,
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
                axis.set_xlim(0, 2)
                axis.set_ylim(0, 2)
                axis.set_aspect("equal")
                axis.set_xticks([0.5, 1.5], ["Pred 0", "Pred P4/5"])
                axis.set_yticks([0.5, 1.5], ["Actual P4/5", "Actual 0"])
                axis.tick_params(length=0)
                for spine in axis.spines.values():
                    spine.set_visible(method == primary_method)
                    spine.set_color("#111111")
                    spine.set_linewidth(1.35)
                if row_index == 0:
                    axis.set_title(
                        METHOD_DISPLAY_NAMES[method],
                        color=METHOD_COLORS[method],
                        fontweight="bold",
                        pad=8,
                    )
                if column_index == 0:
                    axis.set_ylabel("")
        row_label_protocols = {
            "Forecasting": "2022 temporal holdout",
            "Nowcasting": "2022 temporal holdout",
            "Contemporaneous": "random five-fold OOF",
        }
        plot_bottom = 0.145
        plot_top = 0.84
        for row_index, task in enumerate(FIGURE_TASK_ORDER):
            row_center = plot_top - (row_index + 0.5) * (
                (plot_top - plot_bottom) / len(FIGURE_TASK_ORDER)
            )
            figure.text(
                0.175,
                row_center,
                f"{task}\n{row_label_protocols[task]}\nn = {FIGURE_POPULATIONS[task]['n']:,}",
                ha="right",
                va="center",
                fontsize=7.2,
                fontweight="bold",
            )
        figure.suptitle(
            "Phase-4/5 confusion matrices for the three rescue weighting policies",
            fontsize=11,
            fontweight="bold",
            y=0.96,
        )
        figure.text(
            0.205,
            0.035,
            (
                "Cells show count and actual-class row percentage. Black frames mark "
                "the task-specific OOF-selected primary policy. Forecasting/Nowcasting "
                "use the fixed 2022 temporal holdout (n = 1,170); Contemporaneous uses "
                "seed-0 random five-fold row-level full OOF (n = 5,575) with pooled-OOF "
                "threshold selection. Protocols and populations differ and are not "
                "directly comparable."
            ),
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#444444",
            wrap=True,
        )
        _save_figure(
            figure,
            output_pdf,
            output_png,
            "Direct Phase-3 versus Phase-4/5 binary confusion atlas",
        )


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
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_png_contract(path: Path) -> tuple[int, int, float | None]:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG signature: {path.name}")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    offset = 8
    dpi: float | None = None
    while offset + 12 <= len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_data = payload[offset + 8 : offset + 8 + length]
        if chunk_type == b"pHYs" and len(chunk_data) == 9 and chunk_data[8] == 1:
            pixels_per_meter = int.from_bytes(chunk_data[:4], "big")
            dpi = pixels_per_meter * 0.0254
            break
        offset += 12 + length
    return width, height, dpi


def _configuration_payload(
    environment: Mapping[str, object],
    policies: pd.DataFrame,
    predecessor_manifest: Mapping[str, str],
    contemporaneous_manifest: Mapping[str, str],
    artifact_hashes: Mapping[str, str],
    layer1_features: Sequence[str],
) -> dict[str, object]:
    package_versions = environment["package_versions"]
    policy_records = policies.loc[
        :,
        [
            "task",
            "candidate_id",
            "method",
            "selection_status",
            "threshold_rank",
            "threshold",
            "is_above_max_reference",
            "primary_selected",
            "final_class_ratio",
            "final_positive_row_weight",
        ],
    ].to_dict("records")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "freeze_id": FREEZE_ID,
        "environment": {
            "environment_id": REFERENCE_ENVIRONMENT_ID,
            "platform_family": "Windows",
            "python_version": "3.11.3",
            "numpy_version": package_versions["numpy"],
            "pandas_version": package_versions["pandas"],
            "scipy_version": package_versions["scipy"],
            "scikit_learn_version": package_versions["scikit-learn"],
            "xgboost_version": package_versions["xgboost"],
            "matplotlib_version": package_versions["matplotlib"],
            "xgboost_dll_sha256": environment["xgboost_dll_sha256"],
        },
        "decisions": {
            "truth": "reconstructed_fixed_0.20_phase",
            "training_population": "pre2022_reconstructed_phase_3_4_5",
            "positive_class": "reconstructed_phase_4_or_5",
            "negative_training_class": "reconstructed_phase_3",
            "selection_population": "complete_pre2022_oof_base_phase3_gate",
            "allowed_action": "phase3_to_phase4_only",
            "selection_objective": "pooled_oof_phase45_f2_first",
            "benchmark_role": "fixed_reused_2022_benchmark",
            "deployment_aligned_rescue_included": False,
            "manuscript_adoption_authorized": False,
        },
        "folds": [
            {
                "fold_id": fold.fold_id,
                "training_years": list(fold.training_years),
                "validation_year": fold.validation_year,
            }
            for fold in BASE_FOLDS
        ],
        "candidates": {
            "order": list(CANDIDATE_ORDER),
            "weight_formulas": {
                "unweighted": "1",
                "sqrt_balance": "sqrt(n_phase3/n_phase45)",
                "full_balance": "n_phase3/n_phase45",
            },
            "negative_row_weight": 1.0,
            "scale_pos_weight": 1.0,
            "random_state": 0,
            "n_jobs": 1,
            "selected_policies": policy_records,
        },
        "methods": list(METHOD_ORDER),
        "figure_evidence": {
            "displayed_tasks": list(FIGURE_TASK_ORDER),
            "displayed_methods": list(FIGURE_METHOD_ORDER),
            "forecasting_nowcasting_protocol": "fixed_2022_temporal_holdout_1170",
            "contemporaneous_protocol": "seed0_random_5fold_row_cv_full_oof_5575",
            "contemporaneous_threshold_selection": (
                "pooled_oof_same_population_not_nested_or_independent"
            ),
            "direct_comparison_authorized": False,
            "contemporaneous_directory": relative_path(
                CONTEMPORANEOUS_RESCUE_DIR
            ),
            "contemporaneous_artifact_manifest": dict(
                sorted(contemporaneous_manifest.items())
            ),
            "contemporaneous_artifact_manifest_sha256": manifest_sha256(
                contemporaneous_manifest
            ),
        },
        "features": {
            "forecasting_count": len(layer1_features),
            "nowcasting_layer1_count": len(layer1_features),
            "nowcasting_layer2_count": len(NOWCAST_FEATURES),
            "forecasting_order_sha256": json_sha256(list(layer1_features)),
            "nowcasting_layer2_order_sha256": json_sha256(list(NOWCAST_FEATURES)),
        },
        "parameters": {
            "classifier": _direct_parameters("classifier"),
            "regressor": _direct_parameters("regressor"),
            "phase3_parameter_file_sha256": file_sha256(predecessor.PHASE3_PARAMS),
        },
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "random_state": BOOTSTRAP_RANDOM_STATE,
            "cluster_unit": "country_code_3",
            "ci_level": 0.95,
            "interval_endpoints": [2.5, 97.5],
            "paired_within_task": True,
            "diagnostic_only": True,
        },
        "figures": {
            "backend": "Python matplotlib Agg",
            "archetype": "publication_quantitative_grid",
            "core_conclusion": (
                "More aggressive Phase-4/5 rescue can recover additional severe "
                "outcomes at the cost of explicit false promotions."
            ),
            "main_comparison": {
                "layout": "3x2",
                "width_inches": 12.0,
                "height_inches": 10.6,
                "png_dpi": 600,
            },
            "binary_confusion_atlas": {
                "layout": "3x3",
                "width_inches": 10.5,
                "height_inches": 8.8,
                "png_dpi": 600,
            },
        },
        "row_contracts": {
            "oof_predictions": EXPECTED_OOF_ROWS,
            "threshold_frontier": "sum(distinct_finite_pooled_gate_scores + 1)",
            "selected_policies": 6,
            "oof_stability": EXPECTED_OOF_STABILITY_ROWS,
            "benchmark_predictions": EXPECTED_BENCHMARK_ROWS_LONG,
            "country_metrics": EXPECTED_COUNTRY_METRIC_ROWS,
            "binary_confusion_matrices": EXPECTED_BINARY_CONFUSION_ROWS,
            "five_class_confusion_matrices": EXPECTED_FIVE_CLASS_CONFUSION_ROWS,
            "feature_manifest": EXPECTED_FEATURE_MANIFEST_ROWS,
            "model_audit": EXPECTED_MODEL_AUDIT_ROWS,
            "bootstrap_draws": EXPECTED_BOOTSTRAP_DRAW_ROWS,
            "bootstrap_summary": EXPECTED_BOOTSTRAP_SUMMARY_ROWS,
        },
        "csv_schemas": {
            name: list(columns) for name, columns in sorted(CSV_SCHEMAS.items())
        },
        "predecessor": {
            "directory": relative_path(PREDECESSOR_DIR),
            "source_audit_sha256": EXPECTED_PREDECESSOR_SOURCE_AUDIT_SHA256,
            "artifact_manifest_sha256": manifest_sha256(predecessor_manifest),
        },
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
    }


def _source_audit_row(
    environment: Mapping[str, object],
    predecessor_manifest: Mapping[str, str],
    contemporaneous_manifest: Mapping[str, str],
    protected_manifest: str,
    policies: pd.DataFrame,
    direct_population: pd.DataFrame,
    direct_target: pd.Series,
    benchmark: pd.DataFrame,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    package_versions = environment["package_versions"]
    manifest_json = json.dumps(
        dict(sorted(artifact_hashes.items())),
        sort_keys=True,
        separators=(",", ":"),
    )
    predecessor_manifest_json = json.dumps(
        dict(sorted(predecessor_manifest.items())),
        sort_keys=True,
        separators=(",", ":"),
    )
    contemporaneous_manifest_json = json.dumps(
        dict(sorted(contemporaneous_manifest.items())),
        sort_keys=True,
        separators=(",", ":"),
    )
    policy_json = json.dumps(
        json_safe(policies.to_dict("records")),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    forecast_benchmark = benchmark.loc[
        benchmark["task"].eq("Forecasting")
        & benchmark["method"].eq("frozen_base")
    ]
    return {
        "run_status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "freeze_id": FREEZE_ID,
        "reference_environment_id": REFERENCE_ENVIRONMENT_ID,
        "evaluation_population_id": EVALUATION_POPULATION_ID,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "pre2022_rows": EXPECTED_PRE2022_ROWS,
        "direct_training_rows": len(direct_population),
        "direct_phase3_count": int(
            (direct_population["reconstructed_overall_phase"] == 3).sum()
        ),
        "direct_phase4_count": int(
            (direct_population["reconstructed_overall_phase"] == 4).sum()
        ),
        "direct_phase5_count": int(
            (direct_population["reconstructed_overall_phase"] == 5).sum()
        ),
        "direct_phase45_count": int(direct_target.sum()),
        "oof_rows": EXPECTED_OOF_ROWS,
        "benchmark_rows": len(benchmark),
        "benchmark_areas": int(forecast_benchmark["area_id"].nunique()),
        "benchmark_countries": int(
            forecast_benchmark["country_code_3"].nunique()
        ),
        "benchmark_key_sha256": canonical_key_sha256(forecast_benchmark),
        "direct_training_key_sha256": canonical_key_sha256(direct_population),
        "direct_target_sha256": canonical_series_sha256(
            direct_target, "severe_rescue_target"
        ),
        "predecessor_directory": relative_path(PREDECESSOR_DIR),
        "predecessor_source_audit_sha256": file_sha256(
            PREDECESSOR_SOURCE_AUDIT
        ),
        "predecessor_artifact_manifest_json": predecessor_manifest_json,
        "predecessor_artifact_manifest_sha256": manifest_sha256(
            predecessor_manifest
        ),
        "contemporaneous_rescue_directory": relative_path(
            CONTEMPORANEOUS_RESCUE_DIR
        ),
        "contemporaneous_rescue_artifact_manifest_json": (
            contemporaneous_manifest_json
        ),
        "contemporaneous_rescue_artifact_manifest_sha256": manifest_sha256(
            contemporaneous_manifest
        ),
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
        "direct_random_state": 0,
        "direct_n_jobs": 1,
        "outer_workers": 1,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_random_state": BOOTSTRAP_RANDOM_STATE,
        "selected_policies_json": policy_json,
        "protected_manifest_sha256_before": protected_manifest,
        "protected_manifest_sha256_after": protected_manifest,
        "protected_manifest_match": True,
        "artifact_manifest_json": manifest_json,
        "artifact_manifest_sha256": manifest_sha256(artifact_hashes),
    }


def write_formal_artifacts(
    staging_dir: Path,
    *,
    environment: Mapping[str, object],
    predecessor_manifest: Mapping[str, str],
    contemporaneous_manifest: Mapping[str, str],
    protected_manifest: str,
    layer1_features: Sequence[str],
    direct_population: pd.DataFrame,
    direct_target: pd.Series,
    policies: pd.DataFrame,
    oof: pd.DataFrame,
    frontier: pd.DataFrame,
    stability: pd.DataFrame,
    benchmark: pd.DataFrame,
    gate_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    country_metrics: pd.DataFrame,
    country_macro: pd.DataFrame,
    binary_confusion: pd.DataFrame,
    five_confusion: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    model_audit: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    contemporaneous_gate_metrics: pd.DataFrame,
    contemporaneous_binary_confusion: pd.DataFrame,
) -> None:
    payloads = {
        f"{PREFIX}oof_predictions.csv": oof,
        f"{PREFIX}threshold_frontier.csv": frontier,
        f"{PREFIX}selected_policies.csv": policies,
        f"{PREFIX}oof_stability.csv": stability,
        f"{PREFIX}benchmark_predictions.csv": benchmark,
        f"{PREFIX}gate_pooled_metrics.csv": gate_metrics,
        f"{PREFIX}benchmark_pooled_metrics.csv": pooled_metrics,
        f"{PREFIX}country_metrics.csv": country_metrics,
        f"{PREFIX}benchmark_country_macro_metrics.csv": country_macro,
        f"{PREFIX}binary_confusion_matrices.csv": binary_confusion,
        f"{PREFIX}five_class_confusion_matrices.csv": five_confusion,
        f"{PREFIX}feature_manifest.csv": feature_manifest,
        f"{PREFIX}model_audit.csv": model_audit,
        f"{PREFIX}bootstrap_draws.csv": bootstrap_draws,
        f"{PREFIX}bootstrap_summary.csv": bootstrap_summary,
    }
    for name, frame in payloads.items():
        _write_csv(frame, staging_dir / name, CSV_SCHEMAS[name])
    figure_gate_metrics = pd.concat(
        [
            gate_metrics.loc[gate_metrics["method"].isin(FIGURE_METHOD_ORDER)],
            contemporaneous_gate_metrics,
        ],
        ignore_index=True,
    )
    figure_binary_confusion = pd.concat(
        [
            binary_confusion.loc[
                binary_confusion["method"].isin(FIGURE_METHOD_ORDER)
                & binary_confusion["population_scope"].eq("benchmark")
            ],
            contemporaneous_binary_confusion.loc[
                contemporaneous_binary_confusion["population_scope"].eq(
                    "full_oof"
                )
            ],
        ],
        ignore_index=True,
    )
    if (
        len(figure_gate_metrics) != 9
        or figure_gate_metrics.duplicated(["task", "method"]).any()
        or len(figure_binary_confusion) != 36
        or figure_binary_confusion.duplicated(
            ["population_scope", "task", "method", "actual_binary", "predicted_binary"]
        ).any()
    ):
        raise ValueError("Combined three-task figure evidence contract failed.")
    render_main_comparison_figure(
        figure_gate_metrics,
        staging_dir / f"{PREFIX}main_comparison.pdf",
        staging_dir / f"{PREFIX}main_comparison.png",
    )
    render_binary_confusion_atlas(
        figure_binary_confusion,
        figure_gate_metrics,
        staging_dir / f"{PREFIX}binary_confusion_atlas.pdf",
        staging_dir / f"{PREFIX}binary_confusion_atlas.png",
    )
    payload_names = [
        name
        for name in EXPECTED_ARTIFACTS
        if name not in {CONFIGURATION_BASENAME, f"{PREFIX}source_audit.csv"}
    ]
    artifact_hashes_28 = {
        name: file_sha256(staging_dir / name) for name in payload_names
    }
    if len(artifact_hashes_28) != 28:
        raise ValueError("Configuration payload manifest must contain 28 files.")
    configuration = _configuration_payload(
        environment,
        policies,
        predecessor_manifest,
        contemporaneous_manifest,
        artifact_hashes_28,
        layer1_features,
    )
    _write_json(configuration, staging_dir / CONFIGURATION_BASENAME)
    artifact_hashes_29 = {
        name: file_sha256(staging_dir / name)
        for name in EXPECTED_ARTIFACTS
        if name != f"{PREFIX}source_audit.csv"
    }
    if len(artifact_hashes_29) != 29:
        raise ValueError("Source-audit manifest must contain 29 files.")
    source_audit = pd.DataFrame(
        [
            _source_audit_row(
                environment,
                predecessor_manifest,
                contemporaneous_manifest,
                protected_manifest,
                policies,
                direct_population,
                direct_target,
                benchmark,
                artifact_hashes_29,
            )
        ],
        columns=SOURCE_AUDIT_COLUMNS,
    )
    _write_csv(
        source_audit,
        staging_dir / f"{PREFIX}source_audit.csv",
        SOURCE_AUDIT_COLUMNS,
    )
    validate_artifact_contract(staging_dir)


def _read_contract_csv(directory: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(
        directory / name,
        na_values=["<NA>"],
        keep_default_na=True,
        float_precision="round_trip",
    )
    if tuple(frame.columns) != tuple(CSV_SCHEMAS[name]):
        raise ValueError(f"{name} schema drifted.")
    return frame


def validate_artifact_contract(directory: Path) -> None:
    directory = Path(directory)
    files = sorted(path.name for path in directory.iterdir() if path.is_file())
    directories = [path.name for path in directory.iterdir() if path.is_dir()]
    if directories or set(files) != set(EXPECTED_ARTIFACTS):
        raise ValueError(
            f"Formal basename contract failed: files={files}, directories={directories}"
        )
    fixed_checks = {
        f"{PREFIX}oof_predictions.csv": (
            EXPECTED_OOF_ROWS,
            ["task", "candidate_id", "base_oof_fold", "area_id", "date"],
        ),
        f"{PREFIX}selected_policies.csv": (
            6,
            ["task", "candidate_id"],
        ),
        f"{PREFIX}oof_stability.csv": (
            EXPECTED_OOF_STABILITY_ROWS,
            ["task", "candidate_id", "period_id"],
        ),
        f"{PREFIX}benchmark_predictions.csv": (
            EXPECTED_BENCHMARK_ROWS_LONG,
            ["task", "method", "area_id", "date"],
        ),
        f"{PREFIX}gate_pooled_metrics.csv": (10, ["task", "method"]),
        f"{PREFIX}benchmark_pooled_metrics.csv": (10, ["task", "method"]),
        f"{PREFIX}country_metrics.csv": (
            EXPECTED_COUNTRY_METRIC_ROWS,
            ["task", "method", "country_code_3"],
        ),
        f"{PREFIX}benchmark_country_macro_metrics.csv": (
            10,
            ["task", "method"],
        ),
        f"{PREFIX}binary_confusion_matrices.csv": (
            EXPECTED_BINARY_CONFUSION_ROWS,
            [
                "population_scope",
                "task",
                "method",
                "actual_binary",
                "predicted_binary",
            ],
        ),
        f"{PREFIX}five_class_confusion_matrices.csv": (
            EXPECTED_FIVE_CLASS_CONFUSION_ROWS,
            ["task", "method", "actual_phase", "predicted_phase"],
        ),
        f"{PREFIX}feature_manifest.csv": (
            EXPECTED_FEATURE_MANIFEST_ROWS,
            ["task", "model_component", "feature_order"],
        ),
        f"{PREFIX}model_audit.csv": (
            EXPECTED_MODEL_AUDIT_ROWS,
            ["task", "candidate_id", "fit_scope", "fold_id", "model_component"],
        ),
        f"{PREFIX}bootstrap_draws.csv": (
            EXPECTED_BOOTSTRAP_DRAW_ROWS,
            ["task", "method", "bootstrap_id"],
        ),
        f"{PREFIX}bootstrap_summary.csv": (
            EXPECTED_BOOTSTRAP_SUMMARY_ROWS,
            ["task", "method"],
        ),
        f"{PREFIX}source_audit.csv": (1, []),
    }
    loaded: dict[str, pd.DataFrame] = {}
    for name, (expected_rows, keys) in fixed_checks.items():
        frame = _read_contract_csv(directory, name)
        loaded[name] = frame
        if len(frame) != expected_rows:
            raise ValueError(
                f"{name} has {len(frame)} rows, expected {expected_rows}."
            )
        if keys and frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate contract keys.")
    frontier = _read_contract_csv(directory, f"{PREFIX}threshold_frontier.csv")
    loaded[f"{PREFIX}threshold_frontier.csv"] = frontier
    if len(frontier) > MAX_THRESHOLD_FRONTIER_ROWS or frontier.duplicated(
        ["task", "candidate_id", "threshold_rank"]
    ).any():
        raise ValueError("Threshold frontier key or maximum-row contract failed.")
    oof = loaded[f"{PREFIX}oof_predictions.csv"]
    for task in TASK_ORDER:
        for candidate in CANDIDATE_ORDER:
            scores = oof.loc[
                oof["task"].eq(task)
                & oof["candidate_id"].eq(candidate)
                & oof["in_base_phase3_gate"].astype(bool),
                "direct_phase45_score",
            ].to_numpy(dtype=float)
            expected_rows = len(np.unique(scores[np.isfinite(scores)])) + 1
            group = frontier.loc[
                frontier["task"].eq(task)
                & frontier["candidate_id"].eq(candidate)
            ]
            if len(group) != expected_rows:
                raise ValueError(
                    f"{task}/{candidate} threshold formula contract failed."
                )
            if int(group["within_candidate_selected"].astype(bool).sum()) != 1:
                raise ValueError("Each candidate requires one selected threshold.")
    policies = loaded[f"{PREFIX}selected_policies.csv"]
    for task in TASK_ORDER:
        task_policies = policies.loc[policies["task"].eq(task)]
        statuses = set(task_policies["selection_status"])
        if statuses == {"no_effective_rescue"}:
            if (
                task_policies["primary_selected"].astype(bool).any()
                or not task_policies["is_above_max_reference"].astype(bool).all()
            ):
                raise ValueError("No-effective-rescue marker contract failed.")
        elif statuses == {"selected"}:
            if int(task_policies["primary_selected"].astype(bool).sum()) != 1:
                raise ValueError("Selected task requires one primary policy.")
        else:
            raise ValueError("Unknown or mixed task selection status.")
    benchmark = loaded[f"{PREFIX}benchmark_predictions.csv"]
    for (task, method), group in benchmark.groupby(
        ["task", "method"], sort=True, observed=True
    ):
        if len(group) != EXPECTED_BENCHMARK_ROWS:
            raise ValueError(f"{task}/{method} benchmark support drifted.")
        if canonical_key_sha256(group) != EXPECTED_BENCHMARK_KEY_SHA256:
            raise ValueError(f"{task}/{method} benchmark key hash drifted.")
        predecessor.assert_postclassification_invariants(
            group["base_overall_phase_pred"],
            group["final_overall_phase_pred"],
        )
    pooled = loaded[f"{PREFIX}benchmark_pooled_metrics.csv"]
    invariant_columns = (
        "phase3plus_precision_delta_from_base",
        "phase3plus_recall_delta_from_base",
        "phase3plus_r2_delta_from_base",
    )
    if not (pooled.loc[:, list(invariant_columns)].to_numpy(dtype=float) == 0.0).all():
        raise ValueError("Phase-3+ invariant deltas are not exact zero.")
    audit = loaded[f"{PREFIX}model_audit.csv"]
    final = audit.loc[audit["fit_scope"].eq("final_refit")]
    folds = audit.loc[audit["fit_scope"].eq("oof_fold")]
    if (
        len(final) != 9
        or len(folds) != 36
        or set(final["model_path"].dropna()) != set(MODEL_BASENAMES)
        or folds["model_path"].notna().any()
    ):
        raise ValueError("Model-audit publication path contract failed.")
    for name in MODEL_BASENAMES:
        json.loads((directory / name).read_text(encoding="utf-8"))
    configuration = json.loads(
        (directory / CONFIGURATION_BASENAME).read_text(encoding="utf-8")
    )
    required_configuration_keys = {
        "schema_version",
        "experiment_id",
        "freeze_id",
        "environment",
        "decisions",
        "folds",
        "candidates",
        "methods",
        "figure_evidence",
        "features",
        "parameters",
        "bootstrap",
        "figures",
        "row_contracts",
        "csv_schemas",
        "predecessor",
        "artifact_hashes",
    }
    if (
        set(configuration) != required_configuration_keys
        or len(configuration["artifact_hashes"]) != 28
    ):
        raise ValueError("Configuration JSON contract failed.")
    for name, expected_hash in configuration["artifact_hashes"].items():
        if file_sha256(directory / name) != expected_hash:
            raise ValueError(f"Configuration artifact hash mismatch for {name}.")
    figure_evidence = configuration["figure_evidence"]
    contemporaneous_manifest = figure_evidence.get(
        "contemporaneous_artifact_manifest", {}
    )
    if (
        figure_evidence.get("displayed_tasks") != list(FIGURE_TASK_ORDER)
        or figure_evidence.get("displayed_methods") != list(FIGURE_METHOD_ORDER)
        or figure_evidence.get("direct_comparison_authorized") is not False
        or manifest_sha256(contemporaneous_manifest)
        != figure_evidence.get("contemporaneous_artifact_manifest_sha256")
    ):
        raise ValueError("Configuration figure-evidence contract failed.")
    for name, expected_hash in contemporaneous_manifest.items():
        if file_sha256(CONTEMPORANEOUS_RESCUE_DIR / name) != expected_hash:
            raise ValueError(
                f"Configuration Contemporaneous evidence hash mismatch for {name}."
            )
    source = loaded[f"{PREFIX}source_audit.csv"].iloc[0]
    manifest = json.loads(str(source["artifact_manifest_json"]))
    if (
        len(manifest) != 29
        or manifest_sha256(manifest) != str(source["artifact_manifest_sha256"])
        or not bool(source["protected_manifest_match"])
    ):
        raise ValueError("Source-audit manifest or protected contract failed.")
    for name, expected_hash in manifest.items():
        if file_sha256(directory / name) != expected_hash:
            raise ValueError(f"Source-audit artifact hash mismatch for {name}.")
    source_contemporaneous_manifest = json.loads(
        str(source["contemporaneous_rescue_artifact_manifest_json"])
    )
    if (
        source_contemporaneous_manifest != contemporaneous_manifest
        or manifest_sha256(source_contemporaneous_manifest)
        != str(source["contemporaneous_rescue_artifact_manifest_sha256"])
    ):
        raise ValueError("Source-audit Contemporaneous evidence contract failed.")
    pdf_names = (
        f"{PREFIX}main_comparison.pdf",
        f"{PREFIX}binary_confusion_atlas.pdf",
    )
    for name in pdf_names:
        payload = (directory / name).read_bytes()
        if (
            not payload.startswith(b"%PDF")
            or b"CreationDate" in payload
            or b"ModDate" in payload
            or b"/Subtype /Image" in payload
        ):
            raise ValueError(f"{name} is not a deterministic vector PDF.")
    png_contracts = {
        f"{PREFIX}main_comparison.png": (7200, 6360),
        f"{PREFIX}binary_confusion_atlas.png": (6300, 5280),
    }
    for name, expected_size in png_contracts.items():
        width, height, dpi = _read_png_contract(directory / name)
        if (
            (width, height) != expected_size
            or dpi is None
            or abs(dpi - 600.0) > 1.0
        ):
            raise ValueError(
                f"{name} PNG contract failed: width={width} height={height} dpi={dpi}"
            )


def _replace_with_retry(
    source: Path, destination: Path, attempts: int = 20
) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.25)


def _rmtree_with_retry(path: Path, attempts: int = 40) -> None:
    path = Path(path)
    if not path.exists():
        return
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            gc.collect()
            time.sleep(0.25)


def assert_byte_identical_artifacts(
    first_directory: Path,
    second_directory: Path,
    expected_names: Sequence[str] = EXPECTED_ARTIFACTS,
) -> dict[str, str]:
    first = Path(first_directory)
    second = Path(second_directory)
    first_files = sorted(path.name for path in first.iterdir() if path.is_file())
    second_files = sorted(path.name for path in second.iterdir() if path.is_file())
    expected = sorted(expected_names)
    if first_files != expected or second_files != expected:
        raise ValueError(
            "Independent regeneration basename mismatch: "
            f"first={first_files} second={second_files} expected={expected}"
        )
    hashes: dict[str, str] = {}
    for name in expected:
        first_bytes = (first / name).read_bytes()
        second_bytes = (second / name).read_bytes()
        if first_bytes != second_bytes:
            raise RuntimeError(
                f"Independent regeneration is not byte-identical for {name}."
            )
        hashes[name] = bytes_sha256(first_bytes)
    return hashes


def _generate_staging_bundle(
    staging_dir: Path,
    *,
    environment: Mapping[str, object],
    predecessor_manifest: Mapping[str, str],
    contemporaneous_evidence: Mapping[str, object],
    protected_manifest: str,
) -> dict[str, object]:
    forecasting, nowcasting, layer1_features = load_prepared_inputs()
    direct_population, direct_target = build_phase345_training_population(
        forecasting
    )
    frozen_oof = load_frozen_oof_gate_evidence()
    benchmark_evidence = load_frozen_benchmark_evidence()
    feature_manifest = build_feature_manifest(
        forecasting, nowcasting, layer1_features
    )
    model_hash_dir = staging_dir / ".fold-model-hashes"
    oof, fold_audit = generate_oof_predictions(
        forecasting,
        nowcasting,
        layer1_features,
        frozen_oof,
        model_hash_dir,
    )
    if model_hash_dir.exists():
        model_hash_dir.rmdir()
    frontier, policies = build_thresholds_and_policies(oof)
    stability = build_oof_stability(oof, policies)
    bundles, final_audit, policies = refit_all_candidates(
        forecasting,
        nowcasting,
        layer1_features,
        benchmark_evidence,
        policies,
        staging_dir,
    )
    model_audit = pd.concat(
        [fold_audit, final_audit], ignore_index=True
    ).loc[:, list(MODEL_AUDIT_COLUMNS)]
    model_audit = model_audit.sort_values(
        ["task", "candidate_id", "fit_scope", "fold_id", "model_component"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(model_audit) != EXPECTED_MODEL_AUDIT_ROWS or model_audit.duplicated(
        ["task", "candidate_id", "fit_scope", "fold_id", "model_component"]
    ).any():
        raise ValueError("Model audit violates the 45-row contract.")
    benchmark = build_benchmark_predictions(
        benchmark_evidence, bundles, policies
    )
    gate_metrics, pooled_metrics, country_metrics, country_macro = (
        calculate_metric_tables(benchmark)
    )
    binary_confusion, five_confusion = build_confusion_tables(benchmark)
    bootstrap_draws, bootstrap_summary = generate_bootstrap(benchmark)
    write_formal_artifacts(
        staging_dir,
        environment=environment,
        predecessor_manifest=predecessor_manifest,
        contemporaneous_manifest=contemporaneous_evidence["artifact_manifest"],
        protected_manifest=protected_manifest,
        layer1_features=layer1_features,
        direct_population=direct_population,
        direct_target=direct_target,
        policies=policies,
        oof=oof,
        frontier=frontier,
        stability=stability,
        benchmark=benchmark,
        gate_metrics=gate_metrics,
        pooled_metrics=pooled_metrics,
        country_metrics=country_metrics,
        country_macro=country_macro,
        binary_confusion=binary_confusion,
        five_confusion=five_confusion,
        feature_manifest=feature_manifest,
        model_audit=model_audit,
        bootstrap_draws=bootstrap_draws,
        bootstrap_summary=bootstrap_summary,
        contemporaneous_gate_metrics=contemporaneous_evidence["gate_metrics"],
        contemporaneous_binary_confusion=contemporaneous_evidence[
            "binary_confusion"
        ],
    )
    return {"policies": policies}


def run_generation(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_dir = Path(output_dir)
    validate_generation_target(output_dir)
    predecessor_manifest = verify_frozen_sources()
    contemporaneous_evidence = load_contemporaneous_figure_evidence()
    environment = assert_formal_environment()
    protected_before = protected_artifact_manifest_sha256(
        excluded_paths=(output_dir,)
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    first_staging = Path(
        tempfile.mkdtemp(
            prefix=".direct_phase3_vs_phase45_rescue-staging-a-",
            dir=output_dir.parent,
        )
    )
    second_staging = Path(
        tempfile.mkdtemp(
            prefix=".direct_phase3_vs_phase45_rescue-staging-b-",
            dir=output_dir.parent,
        )
    )
    published = False
    try:
        first_result = _generate_staging_bundle(
            first_staging,
            environment=environment,
            predecessor_manifest=predecessor_manifest,
            contemporaneous_evidence=contemporaneous_evidence,
            protected_manifest=protected_before,
        )
        _generate_staging_bundle(
            second_staging,
            environment=environment,
            predecessor_manifest=predecessor_manifest,
            contemporaneous_evidence=contemporaneous_evidence,
            protected_manifest=protected_before,
        )
        validate_artifact_contract(first_staging)
        validate_artifact_contract(second_staging)
        assert_byte_identical_artifacts(first_staging, second_staging)
        _rmtree_with_retry(second_staging)
        if output_dir.exists():
            output_dir.rmdir()
        _replace_with_retry(first_staging, output_dir)
        published = True
        protected_after = protected_artifact_manifest_sha256(
            excluded_paths=(output_dir,)
        )
        if protected_after != protected_before:
            _rmtree_with_retry(output_dir)
            raise RuntimeError(
                "Protected produced_graph artifacts changed during generation."
            )
        validate_artifact_contract(output_dir)
        policies = first_result["policies"]
        primary = policies.loc[policies["primary_selected"].astype(bool)]
        return {
            "output_dir": output_dir,
            "selection_status": {
                task: str(
                    policies.loc[policies["task"].eq(task), "selection_status"].iloc[
                        0
                    ]
                )
                for task in TASK_ORDER
            },
            "primary_selected": {
                task: (
                    str(
                        primary.loc[primary["task"].eq(task), "candidate_id"].iloc[0]
                    )
                    if not primary.loc[primary["task"].eq(task)].empty
                    else None
                )
                for task in TASK_ORDER
            },
            "selected_thresholds": {
                f"{row.task}/{row.candidate_id}": float(row.threshold)
                for row in policies.itertuples(index=False)
            },
        }
    except BaseException:
        for staging in (first_staging, second_staging):
            if staging.exists():
                try:
                    _rmtree_with_retry(staging)
                except PermissionError:
                    pass
        if published and output_dir.exists():
            try:
                _rmtree_with_retry(output_dir)
            except PermissionError:
                pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the isolated Direct Phase-3 versus Phase-4/5 rescue "
            "comparison."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Absent or empty write-once target directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    result = run_generation(arguments.output_dir)
    print(
        json.dumps(
            json_safe(
                {
                    "output_dir": str(result["output_dir"]),
                    "selection_status": result["selection_status"],
                    "primary_selected": result["primary_selected"],
                    "selected_thresholds": result["selected_thresholds"],
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
