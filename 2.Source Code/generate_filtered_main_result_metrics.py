"""Refit selected Figure 1 models after full-input repeated-area filtering.

This metrics-only runner preserves the selected ``main-result-figure1-v1``
Forecasting and Nowcasting pipelines. It calibrates the unfiltered pipeline
against the frozen result, applies the area-frequency filter before the
temporal split, requires two identical filtered runs, and writes one separate
metrics table. It does not modify the original simple-baseline metrics table
or create figures, models, or row-level prediction artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
from pathlib import Path
import sys
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import accuracy_score, r2_score
import xgboost

import generate_leave_one_country_out_robustness as loco
import generate_phase_cumulative_scatter_comparison as phase_scatter
import main_result_figure1_v1 as frozen_main_result


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
SOURCE_CODE_DIR = REPO_ROOT / "2.Source Code"
OUTPUT_DIR = SOURCE_CODE_DIR / "produced_graph"

DEFAULT_FORECASTING_INPUT = SOURCE_DATA_DIR / "Forecasting_Analysis_010825.csv"
DEFAULT_NOWCASTING_INPUT = SOURCE_DATA_DIR / "Nowcasting_Analysis_010825.csv"
DEFAULT_COUNTRY_LOOKUP = SOURCE_DATA_DIR / "area_country_lookup.csv"
DEFAULT_GENERAL_PARAMS = SOURCE_CODE_DIR / "forecasting_hyperparameters.json"
DEFAULT_PHASE3_PARAMS = SOURCE_CODE_DIR / "forecasting_hyperparameters_p3.json"
DEFAULT_BASE_METRICS = OUTPUT_DIR / "simple_baseline_comparison_metrics.csv"
DEFAULT_BASE_PREDICTIONS = OUTPUT_DIR / "simple_baseline_comparison_predictions.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "selected_figure1_repeated_area_refit_metrics.csv"

CUTOFF_DATE = "2022-01-01"
TASK_ORDER = ("Nowcasting", "Forecasting")
METHOD_NAME = "Main result (repeated-area refit)"
TARGET_DEFINITION = "highest cumulative IPC phase with population share >= 0.20"
CONTINUOUS_R2_COLUMN = "phase3plus_continuous_r2"
BASE_METRIC_COLUMNS = [
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
OUTPUT_COLUMNS = [
    *BASE_METRIC_COLUMNS[:8],
    CONTINUOUS_R2_COLUMN,
    *BASE_METRIC_COLUMNS[8:],
]

EXPECTED_FILE_SHA256 = {
    DEFAULT_FORECASTING_INPUT: "60f17f079ab569060a29fd8c7af0d3f6feaab558f7a29a104c9a0e702172dc0e",
    DEFAULT_NOWCASTING_INPUT: "d541f7c0c21c1878fcf8a74c99c953ddda200d0dfeda7365e7ff5934af8f550f",
    DEFAULT_COUNTRY_LOOKUP: "01b58c577b53cbadd1258766244c071d8ef44759da92015566c5507d894f85a7",
    DEFAULT_GENERAL_PARAMS: "3742300661466f22eba8198e5d4f9c2a277615ba59562fbf27f251ecc932dd76",
    DEFAULT_PHASE3_PARAMS: "cdc0e55aa15bdda932465088568b9ee717d226208ffb32076f438fb274e6317b",
    DEFAULT_BASE_METRICS: "d7f6dc6eae93b1db5a52c7f624c91ea96af1d3f6dcba90a4666d9354b35ddcdb",
    DEFAULT_BASE_PREDICTIONS: "05cb67ab4f39a7e75751926c30cb6181301e6232e29b424bb9eaa740eb36ae16",
}
EXPECTED_XGBOOST_DLL_SHA256 = (
    "ca6f7a13af14d3ed08da0a12164ca042302151999191bb81062768cffbc95ce1"
)
EXPECTED_ENVIRONMENT = {
    "python": "3.11.3",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.17.1",
    "scikit-learn": "1.5.2",
    "xgboost": "2.0.3",
}

EXPECTED_FULL_COUNTS = {
    "source_rows": 5575,
    "source_areas": 1198,
    "train_rows": 4405,
    "test_rows": 1170,
    "test_areas": 646,
    "test_key_sha256": "288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2",
}
EXPECTED_FILTERED_COUNTS = {
    "source_rows": 5253,
    "source_areas": 876,
    "train_rows": 4150,
    "train_areas": 866,
    "test_rows": 1103,
    "test_areas": 579,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_environment() -> None:
    """Require the exact selected Figure 1 Windows execution environment."""
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    if platform.system() != "Windows":
        raise RuntimeError("Selected Figure 1 refit requires Windows.")
    if observed != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            f"Selected Figure 1 environment mismatch: expected={EXPECTED_ENVIRONMENT}, "
            f"observed={observed}."
        )
    dll_path = Path(xgboost.__file__).resolve().parent / "lib" / "xgboost.dll"
    dll_sha256 = file_sha256(dll_path)
    if dll_sha256 != EXPECTED_XGBOOST_DLL_SHA256:
        raise RuntimeError(
            "XGBoost DLL does not match the frozen selected Figure 1 binary: "
            f"expected={EXPECTED_XGBOOST_DLL_SHA256}, observed={dll_sha256}."
        )


def validate_default_inputs(paths: Mapping[Path, str]) -> None:
    """Reject silent source, parameter, or base-artifact drift."""
    for path, expected_sha256 in paths.items():
        observed_sha256 = file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"Frozen input hash mismatch for {path}: "
                f"expected={expected_sha256}, observed={observed_sha256}."
            )


def canonical_key_sha256(data: pd.DataFrame) -> str:
    """Hash sorted, normalized ``area_id,date`` observation keys."""
    keys = data.loc[:, ["area_id", "date"]].copy()
    if keys.isna().any().any() or keys.duplicated().any():
        raise ValueError("Canonical observation keys must be complete and unique.")
    parsed_dates = pd.to_datetime(keys["date"], errors="raise")
    if not parsed_dates.eq(parsed_dates.dt.normalize()).all():
        raise ValueError("Canonical dates must contain midnight values.")
    keys["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    keys = keys.sort_values(["area_id", "date"], kind="mergesort")
    payload = keys.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_key_match(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    label: str,
) -> None:
    forecast_keys = pd.MultiIndex.from_frame(forecasting[["area_id", "date"]])
    nowcast_keys = pd.MultiIndex.from_frame(nowcasting[["area_id", "date"]])
    if not forecast_keys.is_unique or not nowcast_keys.is_unique:
        raise ValueError(f"{label} inputs contain duplicate observation keys.")
    if set(forecast_keys) != set(nowcast_keys):
        raise ValueError(f"{label} Forecasting and Nowcasting key sets differ.")


def load_model_inputs(
    forecasting_path: Path,
    nowcasting_path: Path,
    *,
    repeated_area_filter: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load notebook inputs and optionally filter full-input singleton areas."""
    forecasting = pd.read_csv(forecasting_path)
    nowcasting = pd.read_csv(nowcasting_path)
    forecasting = forecasting.loc[forecasting["phase1_percent"].notna()].copy()
    nowcasting = nowcasting.loc[nowcasting["phase1_percent"].notna()].copy()
    _validate_key_match(forecasting, nowcasting, "Effective")

    if repeated_area_filter:
        area_frequency = forecasting["area_id"].value_counts()
        eligible_areas = area_frequency.index[area_frequency.gt(1)]
        forecasting = forecasting.loc[
            forecasting["area_id"].isin(eligible_areas)
        ].copy()
        nowcasting = nowcasting.loc[nowcasting["area_id"].isin(eligible_areas)].copy()
        _validate_key_match(forecasting, nowcasting, "Filtered")
        if (
            len(forecasting) != EXPECTED_FILTERED_COUNTS["source_rows"]
            or forecasting["area_id"].nunique()
            != EXPECTED_FILTERED_COUNTS["source_areas"]
        ):
            raise ValueError("Full-input repeated-area filter produced unexpected counts.")
    else:
        if (
            len(forecasting) != EXPECTED_FULL_COUNTS["source_rows"]
            or forecasting["area_id"].nunique()
            != EXPECTED_FULL_COUNTS["source_areas"]
        ):
            raise ValueError("Unfiltered selected Figure 1 population is unexpected.")
    return forecasting, nowcasting


def run_selected_models(
    *,
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    repeated_area_filter: bool,
) -> dict[str, pd.DataFrame]:
    """Run the selected Forecasting and cascading Nowcasting pipelines."""
    forecasting_raw, nowcasting_raw = load_model_inputs(
        forecasting_path,
        nowcasting_path,
        repeated_area_filter=repeated_area_filter,
    )
    lookup = loco.load_country_lookup(country_lookup_path)
    forecasting, nowcasting = loco.prepare_model_inputs(
        forecasting_raw,
        nowcasting_raw,
        lookup,
    )
    forecasting = loco.add_cumulative_targets(forecasting)
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path,
        phase3_params_path,
        random_state=None,
        estimator_n_jobs=None,
    )
    train_mask, test_mask, now_train_mask, now_test_mask = (
        phase_scatter.temporal_split_masks(
            forecasting,
            nowcasting,
            cutoff=CUTOFF_DATE,
        )
    )

    if repeated_area_filter:
        expected = EXPECTED_FILTERED_COUNTS
        train_areas = forecasting.loc[train_mask, "area_id"].nunique()
        test_areas = forecasting.loc[test_mask, "area_id"].nunique()
        if (
            int(train_mask.sum()) != expected["train_rows"]
            or int(test_mask.sum()) != expected["test_rows"]
            or train_areas != expected["train_areas"]
            or test_areas != expected["test_areas"]
        ):
            raise ValueError("Filtered temporal split produced unexpected counts.")
    elif (
        int(train_mask.sum()) != EXPECTED_FULL_COUNTS["train_rows"]
        or int(test_mask.sum()) != EXPECTED_FULL_COUNTS["test_rows"]
        or forecasting.loc[test_mask, "area_id"].nunique()
        != EXPECTED_FULL_COUNTS["test_areas"]
    ):
        raise ValueError("Unfiltered temporal split produced unexpected counts.")

    forecasting_predictions = loco.fit_forecasting_split(
        forecasting,
        train_mask,
        test_mask,
        "temporal_2022",
        general_params,
        phase3_params,
        fold_column="split_id",
    )
    nowcasting_predictions = loco.fit_nowcasting_split(
        forecasting,
        nowcasting,
        train_mask,
        test_mask,
        now_train_mask,
        now_test_mask,
        "temporal_2022",
        general_params,
        phase3_params,
        fold_column="split_id",
    )
    return {
        "Forecasting": forecasting_predictions.sort_values(
            ["area_id", "date"], kind="mergesort"
        ).reset_index(drop=True),
        "Nowcasting": nowcasting_predictions.sort_values(
            ["area_id", "date"], kind="mergesort"
        ).reset_index(drop=True),
    }


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    """Calculate classification, binary R2, and continuous Phase 3+ R2."""
    cumulative_test = [f"phase{phase}_test" for phase in range(2, 6)]
    cumulative_pred = [f"phase{phase}_pred" for phase in range(2, 6)]
    invalid = predictions[cumulative_test].sum(axis=1).le(0) | predictions[
        cumulative_pred
    ].sum(axis=1).le(0)
    if invalid.any():
        raise ValueError(
            "Selected pipeline produced nonpositive cumulative rows that the notebook "
            "would remove; the approved evaluation counts no longer apply."
        )

    actual = predictions["overall_phase"].to_numpy(dtype=int)
    predicted = predictions["overall_phase_pred"].to_numpy(dtype=int)
    actual_positive = actual >= 3
    predicted_positive = predicted >= 3
    true_positive = int((actual_positive & predicted_positive).sum())
    false_positive = int((~actual_positive & predicted_positive).sum())
    false_negative = int((actual_positive & ~predicted_positive).sum())
    true_negative = int((~actual_positive & ~predicted_positive).sum())
    return {
        "overall_accuracy": float(accuracy_score(actual, predicted)),
        "phase3plus_accuracy": float(
            accuracy_score(actual_positive, predicted_positive)
        ),
        "phase3plus_precision": float(
            true_positive / (true_positive + false_positive)
        ),
        "phase3plus_recall": float(
            true_positive / (true_positive + false_negative)
        ),
        "phase3above_r2": float(
            r2_score(actual_positive.astype(int), predicted_positive.astype(int))
        ),
        CONTINUOUS_R2_COLUMN: float(
            r2_score(predictions["phase3_test"], predictions["phase3_pred"])
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "correct_rows": int((actual == predicted).sum()),
    }


def validate_unfiltered_calibration(
    predictions_by_task: Mapping[str, pd.DataFrame],
) -> None:
    """Require exact frozen classification counts and calibrated continuous R2."""
    for task in TASK_ORDER:
        predictions = predictions_by_task[task]
        metrics = calculate_metrics(predictions)
        reference = frozen_main_result.RESULTS[task]
        expected_counts = {
            "correct_rows": int(reference["correct_rows"]),
            "true_positive": int(reference["true_positive"]),
            "false_positive": int(reference["false_positive"]),
            "false_negative": int(reference["false_negative"]),
            "true_negative": int(reference["true_negative"]),
        }
        observed_counts = {name: int(metrics[name]) for name in expected_counts}
        if observed_counts != expected_counts:
            raise RuntimeError(
                f"{task} unfiltered calibration counts failed: "
                f"expected={expected_counts}, observed={observed_counts}."
            )
        if not np.isclose(
            float(metrics[CONTINUOUS_R2_COLUMN]),
            float(reference["phase3plus_r2"]),
            rtol=0.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"{task} unfiltered continuous R2 calibration failed: "
                f"expected={reference['phase3plus_r2']}, "
                f"observed={metrics[CONTINUOUS_R2_COLUMN]}."
            )
        if len(predictions) != EXPECTED_FULL_COUNTS["test_rows"]:
            raise RuntimeError(f"{task} unfiltered calibration row count failed.")
        key_sha256 = canonical_key_sha256(predictions)
        if key_sha256 != EXPECTED_FULL_COUNTS["test_key_sha256"]:
            raise RuntimeError(f"{task} unfiltered test-key calibration failed.")


def prediction_sha256(predictions: pd.DataFrame) -> str:
    """Hash all selected row-level test values used by the metrics."""
    columns = [
        "area_id",
        "date",
        *[
            f"phase{phase}_{suffix}"
            for phase in range(2, 6)
            for suffix in ("test", "pred")
        ],
        "overall_phase",
        "overall_phase_pred",
    ]
    data = predictions.loc[:, columns].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    data = data.sort_values(["area_id", "date"], kind="mergesort")
    payload = data.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_identical_filtered_runs(
    first: Mapping[str, pd.DataFrame],
    second: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    """Require exact row-level equality across two independent filtered refits."""
    digests = {}
    for task in TASK_ORDER:
        first_digest = prediction_sha256(first[task])
        second_digest = prediction_sha256(second[task])
        if first_digest != second_digest:
            raise RuntimeError(
                f"{task} filtered refits are not identical: "
                f"first={first_digest}, second={second_digest}."
            )
        digests[task] = first_digest
    return digests


def build_output_table(
    *,
    base_metrics_path: Path,
    base_predictions_path: Path,
    filtered_predictions: Mapping[str, pd.DataFrame],
    filtered_prediction_hashes: Mapping[str, str],
) -> pd.DataFrame:
    """Copy the ten-row base table, add continuous R2, and append two refits."""
    base = pd.read_csv(base_metrics_path, float_precision="round_trip")
    if base.columns.tolist() != BASE_METRIC_COLUMNS or base.shape != (10, 19):
        raise ValueError("Base simple-baseline metrics must retain the formal 10x19 schema.")
    if base.duplicated(["task", "method"]).any():
        raise ValueError("Base simple-baseline metrics contain duplicate task-method rows.")
    base.insert(8, CONTINUOUS_R2_COLUMN, np.nan)

    for task in TASK_ORDER:
        main_mask = base["task"].eq(task) & base["method"].eq("Main result")
        if int(main_mask.sum()) != 1:
            raise ValueError(f"Base metrics contain an unexpected {task} Main result row.")
        base.loc[main_mask, CONTINUOUS_R2_COLUMN] = float(
            frozen_main_result.RESULTS[task]["phase3plus_r2"]
        )

    base_predictions = pd.read_csv(
        base_predictions_path,
        float_precision="round_trip",
    )
    for task in TASK_ORDER:
        group = base_predictions.loc[
            base_predictions["task"].eq(task)
            & base_predictions["method"].eq("Ensemble OLS")
        ]
        required = ["phase3_actual_cumulative", "phase3_pred_rounded"]
        if len(group) != EXPECTED_FULL_COUNTS["test_rows"] or group[
            required
        ].isna().any().any():
            raise ValueError(f"{task} Ensemble OLS continuous predictions are incomplete.")
        continuous_r2 = float(
            r2_score(
                group["phase3_actual_cumulative"],
                group["phase3_pred_rounded"],
            )
        )
        ols_mask = base["task"].eq(task) & base["method"].eq("Ensemble OLS")
        if int(ols_mask.sum()) != 1:
            raise ValueError(f"Base metrics contain an unexpected {task} Ensemble OLS row.")
        base.loc[ols_mask, CONTINUOUS_R2_COLUMN] = continuous_r2

    test_key_sha256 = canonical_key_sha256(filtered_predictions["Forecasting"])
    if test_key_sha256 != canonical_key_sha256(filtered_predictions["Nowcasting"]):
        raise ValueError("Filtered Forecasting and Nowcasting test-key hashes differ.")

    additions = []
    for task in TASK_ORDER:
        metrics = calculate_metrics(filtered_predictions[task])
        metric_source = (
            "main-result-figure1-v1 selected lineage; full 5,575-row input "
            "area_id frequency > 1 before 2022-01-01 temporal split; "
            "windows_py3113_xgb203_defaultthreads_no_explicit_seed; "
            f"double-run prediction sha256={filtered_prediction_hashes[task]}"
        )
        additions.append(
            {
                "task": task,
                "method": METHOD_NAME,
                "method_role": "main_result_filtered_refit",
                "overall_accuracy": metrics["overall_accuracy"],
                "phase3plus_accuracy": metrics["phase3plus_accuracy"],
                "phase3plus_precision": metrics["phase3plus_precision"],
                "phase3plus_recall": metrics["phase3plus_recall"],
                "phase3above_r2": metrics["phase3above_r2"],
                CONTINUOUS_R2_COLUMN: metrics[CONTINUOUS_R2_COLUMN],
                "true_positive": metrics["true_positive"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_negative": metrics["true_negative"],
                "n_train": EXPECTED_FILTERED_COUNTS["train_rows"],
                "n_test": EXPECTED_FILTERED_COUNTS["test_rows"],
                "test_key_sha256": test_key_sha256,
                "target_definition": TARGET_DEFINITION,
                "metric_source": metric_source,
                "fit_status": "refit_from_filtered_authoritative_inputs",
                "converged": np.nan,
            }
        )
    result = pd.concat(
        [base, pd.DataFrame(additions, columns=OUTPUT_COLUMNS)],
        ignore_index=True,
    ).loc[:, OUTPUT_COLUMNS]
    if result.shape != (12, 20) or result.duplicated(["task", "method"]).any():
        raise ValueError("Extended selected Figure 1 metrics table is invalid.")
    return result


def replace_with_retry(source: Path, destination: Path) -> None:
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


def write_output(data: pd.DataFrame, output_path: Path) -> None:
    """Atomically write and reload the sole generated artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".selected_figure1_refit_metrics_",
        dir=output_path.parent,
    ) as staging_name:
        staged_path = Path(staging_name) / output_path.name
        data.to_csv(
            staged_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
            na_rep="",
        )
        reloaded = pd.read_csv(staged_path, float_precision="round_trip")
        if reloaded.columns.tolist() != OUTPUT_COLUMNS or reloaded.shape != (12, 20):
            raise ValueError("Staged selected Figure 1 metrics table failed validation.")
        replace_with_retry(staged_path, output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-input", type=Path, default=DEFAULT_FORECASTING_INPUT)
    parser.add_argument("--nowcasting-input", type=Path, default=DEFAULT_NOWCASTING_INPUT)
    parser.add_argument("--country-lookup", type=Path, default=DEFAULT_COUNTRY_LOOKUP)
    parser.add_argument("--general-params", type=Path, default=DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=DEFAULT_PHASE3_PARAMS)
    parser.add_argument("--base-metrics", type=Path, default=DEFAULT_BASE_METRICS)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=DEFAULT_BASE_PREDICTIONS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    if arguments.output.resolve() == arguments.base_metrics.resolve():
        raise ValueError("The separate output must not overwrite the base metrics CSV.")
    validate_environment()
    default_paths = {
        DEFAULT_FORECASTING_INPUT: EXPECTED_FILE_SHA256[DEFAULT_FORECASTING_INPUT],
        DEFAULT_NOWCASTING_INPUT: EXPECTED_FILE_SHA256[DEFAULT_NOWCASTING_INPUT],
        DEFAULT_COUNTRY_LOOKUP: EXPECTED_FILE_SHA256[DEFAULT_COUNTRY_LOOKUP],
        DEFAULT_GENERAL_PARAMS: EXPECTED_FILE_SHA256[DEFAULT_GENERAL_PARAMS],
        DEFAULT_PHASE3_PARAMS: EXPECTED_FILE_SHA256[DEFAULT_PHASE3_PARAMS],
        DEFAULT_BASE_METRICS: EXPECTED_FILE_SHA256[DEFAULT_BASE_METRICS],
        DEFAULT_BASE_PREDICTIONS: EXPECTED_FILE_SHA256[DEFAULT_BASE_PREDICTIONS],
    }
    supplied_paths = {
        arguments.forecasting_input: default_paths[DEFAULT_FORECASTING_INPUT],
        arguments.nowcasting_input: default_paths[DEFAULT_NOWCASTING_INPUT],
        arguments.country_lookup: default_paths[DEFAULT_COUNTRY_LOOKUP],
        arguments.general_params: default_paths[DEFAULT_GENERAL_PARAMS],
        arguments.phase3_params: default_paths[DEFAULT_PHASE3_PARAMS],
        arguments.base_metrics: default_paths[DEFAULT_BASE_METRICS],
        arguments.base_predictions: default_paths[DEFAULT_BASE_PREDICTIONS],
    }
    validate_default_inputs(supplied_paths)
    original_metrics_sha256 = file_sha256(arguments.base_metrics)

    print("Running unfiltered selected-lineage calibration...", flush=True)
    calibration = run_selected_models(
        forecasting_path=arguments.forecasting_input,
        nowcasting_path=arguments.nowcasting_input,
        country_lookup_path=arguments.country_lookup,
        general_params_path=arguments.general_params,
        phase3_params_path=arguments.phase3_params,
        repeated_area_filter=False,
    )
    validate_unfiltered_calibration(calibration)

    print("Running filtered selected-lineage refit 1/2...", flush=True)
    filtered_first = run_selected_models(
        forecasting_path=arguments.forecasting_input,
        nowcasting_path=arguments.nowcasting_input,
        country_lookup_path=arguments.country_lookup,
        general_params_path=arguments.general_params,
        phase3_params_path=arguments.phase3_params,
        repeated_area_filter=True,
    )
    print("Running filtered selected-lineage refit 2/2...", flush=True)
    filtered_second = run_selected_models(
        forecasting_path=arguments.forecasting_input,
        nowcasting_path=arguments.nowcasting_input,
        country_lookup_path=arguments.country_lookup,
        general_params_path=arguments.general_params,
        phase3_params_path=arguments.phase3_params,
        repeated_area_filter=True,
    )
    filtered_hashes = validate_identical_filtered_runs(
        filtered_first,
        filtered_second,
    )
    output = build_output_table(
        base_metrics_path=arguments.base_metrics,
        base_predictions_path=arguments.base_predictions,
        filtered_predictions=filtered_first,
        filtered_prediction_hashes=filtered_hashes,
    )
    write_output(output, arguments.output)
    if file_sha256(arguments.base_metrics) != original_metrics_sha256:
        raise RuntimeError("Original simple-baseline metrics CSV changed unexpectedly.")

    print(f"output: {arguments.output}")
    print(f"output_sha256: {file_sha256(arguments.output)}")
    print(
        output.loc[output["method"].eq(METHOD_NAME)].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
