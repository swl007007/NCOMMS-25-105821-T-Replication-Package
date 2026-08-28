"""Generate cumulative-phase actual-vs-predicted scatter figures.

The legacy outputs retain separate Forecasting/Nowcasting figures for Phase 2+,
Phase 4+, and Phase 5. A combined 3 x 3 figure adds Contemporaneous predictions
from random five-fold row-level cross-validation. The figure explicitly separates
the 2022 temporal-holdout columns from the full-OOF random-CV column because the
validation protocols and populations are not directly comparable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
import platform
from pathlib import Path
from typing import Mapping

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache-phase-scatter")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import r2_score

import generate_leave_one_country_out_robustness as loco
import main_result_figure1_v1 as frozen_main_result


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
DEFAULT_FORECASTING_INPUT = loco.DEFAULT_FORECASTING_INPUT
DEFAULT_NOWCASTING_INPUT = loco.DEFAULT_NOWCASTING_INPUT
DEFAULT_COUNTRY_LOOKUP = loco.DEFAULT_COUNTRY_LOOKUP
DEFAULT_GENERAL_PARAMS = loco.DEFAULT_GENERAL_PARAMS
DEFAULT_PHASE3_PARAMS = loco.DEFAULT_PHASE3_PARAMS
DEFAULT_FORECASTING_ANCHOR = SOURCE_DATA_DIR / "r2_frame_forecasting.csv"
DEFAULT_NOWCASTING_ANCHOR = SOURCE_DATA_DIR / "r2_frame_nowcasting.csv"
DEFAULT_CONTEMPORANEOUS_PREDICTIONS = (
    DEFAULT_OUTPUT_DIR / "all_prediction_contemporaneous_random_cv_predictions.csv"
)
DEFAULT_CONTEMPORANEOUS_AUDIT = (
    DEFAULT_OUTPUT_DIR / "all_prediction_contemporaneous_random_cv_source_audit.csv"
)
EXPECTED_CONTEMPORANEOUS_PREDICTIONS_SHA256 = (
    "4eb561baedaf22a51d534177b768afb82865f44f7b781cc334a481d40759cde3"
)
EXPECTED_CONTEMPORANEOUS_AUDIT_SHA256 = (
    "1502cbf77e99d32ef9946d3df2a9655f9596dee600743b9139faea2a93ef75d3"
)
EXPECTED_CONTEMPORANEOUS_POPULATION_KEY_SHA256 = (
    "f540e216e9ee286c7502b3aa465fe222a66a1a87eec014158d0db9ace13f6651"
)
EXPECTED_CONTEMPORANEOUS_FOLD_ASSIGNMENT_SHA256 = (
    "bc2b0a13def3c2979b562236cedbeff0cc030bf67d4b5de9160f8b916937b099"
)
EXPECTED_CANONICAL_KEY_SHA256 = (
    "288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2"
)
DEFAULT_CUTOFF = "2022-01-01"
DEFAULT_EXPECTED_TEST_ROWS = 1170
DEFAULT_EXPECTED_CONTEMPORANEOUS_ROWS = 5575
DEFAULT_CONTEMPORANEOUS_SHUFFLE_SEED = 0
DEFAULT_RANDOM_STATE: int | None = None
DEFAULT_WORKERS = 1
DEFAULT_ESTIMATOR_N_JOBS: int | None = None
MODEL_ORDER = ("Forecasting", "Nowcasting")
THREE_MODEL_ORDER = (*MODEL_ORDER, "Contemporaneous")
THREE_MODEL_COLUMN_TITLES = {
    "Forecasting": "Forecasting\n(2022 temporal holdout)",
    "Nowcasting": "Nowcasting\n(2022 temporal holdout)",
    "Contemporaneous": "Contemporaneous\n(random 5-fold CV)",
}
PHASE_SPECS = {
    2: {"label": "Phase 2+", "stem": "phase2plus"},
    4: {"label": "Phase 4+", "stem": "phase4plus"},
    5: {"label": "Phase 5", "stem": "phase5"},
}
THREE_MODEL_PHASE_SPECS = {
    2: {"label": "Phase 2+", "stem": "phase2plus"},
    3: {"label": "Phase 3+", "stem": "phase3plus"},
    4: {"label": "Phase 4+", "stem": "phase4plus"},
}
THREE_MODEL_FIGURE_STEM = (
    "phase_cumulative_actual_vs_predicted_"
    "forecasting_nowcasting_contemporaneous"
)
THREE_MODEL_TABLE_STEM = "phase_cumulative_three_model_scatter"
KEY_COLUMNS = ["area_id", "date", "country_code_3", "source_row_index"]
CONTEMPORANEOUS_KEY_COLUMNS = ["source_row_index", "area_id", "date"]
EVALUATION_METADATA_COLUMNS = [
    "fold",
    "evaluation_protocol",
    "evaluation_population",
]


def temporal_split_masks(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    cutoff: str = DEFAULT_CUTOFF,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return notebook-faithful pre-2022 train and 2022 test masks."""
    for name, data in (("Forecasting", forecasting), ("Nowcasting", nowcasting)):
        if "date" not in data:
            raise ValueError(f"{name} input is missing the date column.")
    cutoff_date = pd.Timestamp(cutoff)
    forecasting_dates = pd.to_datetime(forecasting["date"], errors="coerce")
    nowcasting_dates = pd.to_datetime(nowcasting["date"], errors="coerce")
    if forecasting_dates.isna().any() or nowcasting_dates.isna().any():
        raise ValueError("Model inputs contain unparseable dates.")
    train_mask = forecasting_dates.lt(cutoff_date)
    test_mask = forecasting_dates.ge(cutoff_date)
    now_train_mask = nowcasting_dates.lt(cutoff_date)
    now_test_mask = nowcasting_dates.ge(cutoff_date)
    if not bool(train_mask.any()) or not bool(test_mask.any()):
        raise ValueError("Forecasting temporal split requires non-empty train and test rows.")
    if not bool(now_train_mask.any()) or not bool(now_test_mask.any()):
        raise ValueError("Nowcasting temporal split requires non-empty train and test rows.")
    return train_mask, test_mask, now_train_mask, now_test_mask


def _fit_model_from_paths(
    model_name: str,
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    cutoff: str,
    random_state: int | None,
    estimator_n_jobs: int | None,
) -> tuple[str, pd.DataFrame]:
    """Load inputs and fit one model in an isolated worker process."""
    forecasting_raw = pd.read_csv(forecasting_path)
    nowcasting_raw = pd.read_csv(nowcasting_path)
    lookup = loco.load_country_lookup(country_lookup_path)
    forecasting, nowcasting = loco.prepare_model_inputs(
        forecasting_raw, nowcasting_raw, lookup
    )
    forecasting = loco.add_cumulative_targets(forecasting)
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path,
        phase3_params_path,
        random_state,
        estimator_n_jobs=estimator_n_jobs,
    )
    train_mask, test_mask, now_train_mask, now_test_mask = temporal_split_masks(
        forecasting, nowcasting, cutoff
    )

    if model_name == "Forecasting":
        predictions = loco.fit_forecasting_split(
            forecasting,
            train_mask,
            test_mask,
            "temporal_2022",
            general_params,
            phase3_params,
            fold_column="split_id",
        )
    elif model_name == "Nowcasting":
        predictions = loco.fit_nowcasting_split(
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model_name, predictions.sort_values(
        ["area_id", "date"], kind="mergesort"
    ).reset_index(drop=True)


def run_temporal_predictions(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = DEFAULT_PHASE3_PARAMS,
    cutoff: str = DEFAULT_CUTOFF,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    workers: int = DEFAULT_WORKERS,
    estimator_n_jobs: int | None = DEFAULT_ESTIMATOR_N_JOBS,
) -> dict[str, pd.DataFrame]:
    """Fit Forecasting and two-layer Nowcasting, in parallel when requested."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    arguments = (
        Path(forecasting_path),
        Path(nowcasting_path),
        Path(country_lookup_path),
        Path(general_params_path),
        Path(phase3_params_path),
        cutoff,
        random_state,
        estimator_n_jobs,
    )
    results: dict[str, pd.DataFrame] = {}
    if workers == 1:
        for model_name in MODEL_ORDER:
            name, predictions = _fit_model_from_paths(model_name, *arguments)
            results[name] = predictions
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(MODEL_ORDER))) as pool:
            futures = {
                pool.submit(_fit_model_from_paths, model_name, *arguments): model_name
                for model_name in MODEL_ORDER
            }
            for future in as_completed(futures):
                name, predictions = future.result()
                results[name] = predictions
    return {model_name: results[model_name] for model_name in MODEL_ORDER}


def build_long_predictions(
    predictions_by_model: Mapping[str, pd.DataFrame],
    expected_test_rows: int = DEFAULT_EXPECTED_TEST_ROWS,
    *,
    model_order: tuple[str, ...] = MODEL_ORDER,
    phase_specs: Mapping[int, Mapping[str, str]] = PHASE_SPECS,
    expected_rows_by_model: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Convert standardized wide model outputs to an auditable long table."""
    frames: list[pd.DataFrame] = []
    for model_name in model_order:
        if model_name not in predictions_by_model:
            raise ValueError(f"Missing predictions for {model_name}.")
        predictions = predictions_by_model[model_name]
        expected_model_rows = (
            int(expected_rows_by_model[model_name])
            if expected_rows_by_model is not None
            else expected_test_rows
        )
        if len(predictions) != expected_model_rows:
            raise ValueError(
                f"Expected {expected_model_rows} test rows for {model_name}, "
                f"found {len(predictions)}."
            )
        required = [
            *KEY_COLUMNS,
            *[
                f"phase{phase}_{suffix}"
                for phase in phase_specs
                for suffix in ("test", "pred")
            ],
        ]
        missing = sorted(set(required).difference(predictions.columns))
        if missing:
            raise ValueError(f"{model_name} predictions are missing columns: {missing}")
        if predictions.duplicated(["area_id", "date"]).any():
            raise ValueError(f"{model_name} predictions contain duplicate area-date keys.")
        if not predictions["source_row_index"].is_unique:
            raise ValueError(f"{model_name} predictions contain duplicate source rows.")
        for phase, phase_spec in phase_specs.items():
            frame = predictions[KEY_COLUMNS].copy()
            for column in EVALUATION_METADATA_COLUMNS:
                if column in predictions:
                    frame[column] = predictions[column].to_numpy()
            frame["model"] = model_name
            frame["phase"] = phase
            frame["phase_label"] = phase_spec["label"]
            frame["actual"] = pd.to_numeric(
                predictions[f"phase{phase}_test"], errors="coerce"
            )
            frame["predicted"] = pd.to_numeric(
                predictions[f"phase{phase}_pred"], errors="coerce"
            )
            if frame[["actual", "predicted"]].isna().any().any():
                raise ValueError(
                    f"{model_name} {phase_spec['label']} contains missing values."
                )
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    model_rank = pd.Categorical(result["model"], model_order, ordered=True)
    result = (
        result.assign(_model_rank=model_rank)
        .sort_values(
            ["phase", "_model_rank", "area_id", "date"], kind="mergesort"
        )
        .drop(columns="_model_rank")
        .reset_index(drop=True)
    )
    if expected_rows_by_model is None:
        expected_total = expected_test_rows * len(model_order) * len(phase_specs)
    else:
        expected_total = sum(
            int(expected_rows_by_model[model_name]) for model_name in model_order
        ) * len(phase_specs)
    if len(result) != expected_total:
        raise ValueError("Long prediction assembly changed the evaluation population.")
    return result


def _normalize_area_date_keys(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize canonical scatter keys without changing row identity."""
    missing = sorted({"area_id", "date"}.difference(data.columns))
    if missing:
        raise ValueError(f"{name} is missing canonical keys: {missing}")
    normalized = data.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if normalized[["area_id", "date"]].isna().any().any():
        raise ValueError(f"{name} contains missing canonical keys.")
    if normalized.duplicated(["area_id", "date"]).any():
        raise ValueError(f"{name} contains duplicate area-date keys.")
    return normalized


def _canonical_key_sha256(data: pd.DataFrame) -> str:
    """Return the stable SHA-256 used for the canonical area-date population."""
    normalized = _normalize_area_date_keys(data, "Canonical hash input")
    ordered = normalized[["area_id", "date"]].sort_values(
        ["area_id", "date"], kind="mergesort"
    )
    try:
        serialized = ordered.to_csv(
            index=False,
            float_format="%.17g",
            na_rep="<NA>",
            lineterminator="\n",
        )
    except TypeError as error:
        if "lineterminator" not in str(error):
            raise
        serialized = ordered.to_csv(
            index=False,
            float_format="%.17g",
            na_rep="<NA>",
            line_terminator="\n",
        )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _fold_assignment_sha256(data: pd.DataFrame) -> str:
    required = {"source_row_index", "fold"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Fold hash input is missing columns: {missing}")
    source_row_index = pd.to_numeric(data["source_row_index"], errors="coerce")
    fold = pd.to_numeric(data["fold"], errors="coerce")
    if (
        source_row_index.isna().any()
        or fold.isna().any()
        or not np.allclose(source_row_index, source_row_index.round())
        or not np.allclose(fold, fold.round())
    ):
        raise ValueError("Fold hash input must contain complete integer values.")
    ordered = pd.DataFrame(
        {
            "source_row_index": source_row_index.astype(int),
            "fold": fold.astype(int),
        }
    ).sort_values("source_row_index", kind="mergesort")
    if not ordered["source_row_index"].is_unique:
        raise ValueError("Fold hash input contains duplicate source rows.")
    payload = ordered.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_contemporaneous_artifacts(
    predictions_path: Path,
    audit_path: Path,
    *,
    expected_rows: int = DEFAULT_EXPECTED_CONTEMPORANEOUS_ROWS,
    production_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the full-OOF random-CV Contemporaneous bundle."""
    predictions_path = Path(predictions_path)
    audit_path = Path(audit_path)
    predictions = _normalize_area_date_keys(
        pd.read_csv(predictions_path), "Contemporaneous predictions"
    )
    required = [
        *CONTEMPORANEOUS_KEY_COLUMNS,
        "fold",
        "source_overall_phase",
        "overall_phase",
        *[
            column
            for phase in range(2, 6)
            for column in (
                f"phase{phase}_actual",
                f"phase{phase}_contemporaneous",
            )
        ],
        "contemporaneous_predict",
        "evaluation_protocol",
        "evaluation_population",
        "shuffle_seed",
    ]
    missing = sorted(set(required).difference(predictions.columns))
    if missing:
        raise ValueError(f"Contemporaneous predictions are missing columns: {missing}")
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} contemporaneous rows, "
            f"found {len(predictions)}."
        )
    source_row_index = pd.to_numeric(
        predictions["source_row_index"], errors="coerce"
    )
    if source_row_index.isna().any() or not np.allclose(
        source_row_index, source_row_index.round()
    ):
        raise ValueError("Contemporaneous source_row_index must contain integers.")
    predictions["source_row_index"] = source_row_index.astype(int)
    if not predictions["source_row_index"].is_unique:
        raise ValueError("Contemporaneous source_row_index values are not unique.")
    fold = pd.to_numeric(predictions["fold"], errors="coerce")
    if fold.isna().any() or not np.allclose(fold, fold.round()):
        raise ValueError("Contemporaneous fold values must contain integers.")
    predictions["fold"] = fold.astype(int)
    if set(predictions["fold"]) != set(range(5)):
        raise ValueError("Contemporaneous predictions must contain folds 0--4.")
    if not predictions["fold"].value_counts().eq(1115).all():
        raise ValueError("Contemporaneous folds must each contain 1,115 rows.")
    numeric_columns = [
        column
        for phase in range(2, 6)
        for column in (f"phase{phase}_actual", f"phase{phase}_contemporaneous")
    ]
    numeric = predictions[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Contemporaneous cumulative predictions must be finite.")
    predictions[numeric_columns] = numeric

    audit = pd.read_csv(audit_path)
    if len(audit) != 1 or "model" not in audit or audit.loc[0, "model"] != "Contemporaneous":
        raise ValueError("Contemporaneous source audit must contain one model row.")
    required_audit = [
        "evaluation_protocol",
        "validation_design",
        "source_rows",
        "oof_rows",
        "test_rows",
        "evaluation_population",
        "n_splits",
        "fold_rows",
        "feature_count",
        "kfolds_predictor_included",
        "shuffle_seed",
        "fold_assignment_sha256",
        "population_key_sha256",
        "target_contract",
        "parameter_contract",
        "rerun_interpretation",
        "phase_round_decimals",
        "phase_threshold",
        "predictions_sha256",
        "estimator_n_jobs",
    ]
    missing_audit = sorted(set(required_audit).difference(audit.columns))
    if missing_audit:
        raise ValueError(f"Contemporaneous audit is missing columns: {missing_audit}")
    if int(audit.loc[0, "test_rows"]) != expected_rows:
        raise ValueError("Contemporaneous audit test-row count differs from the sidecar.")
    if int(audit.loc[0, "oof_rows"]) != expected_rows:
        raise ValueError("Contemporaneous audit OOF-row count differs from the sidecar.")
    key_sha256 = _canonical_key_sha256(predictions)
    if str(audit.loc[0, "population_key_sha256"]) != key_sha256:
        raise ValueError("Contemporaneous sidecar key hash differs from its audit.")
    fold_sha256 = _fold_assignment_sha256(predictions)
    if str(audit.loc[0, "fold_assignment_sha256"]) != fold_sha256:
        raise ValueError("Contemporaneous fold assignment differs from its audit.")
    predictions_sha256 = _sha256_file(predictions_path)
    if str(audit.loc[0, "predictions_sha256"]) != predictions_sha256:
        raise ValueError("Contemporaneous prediction-file hash differs from its audit.")
    expected_contract = {
        "evaluation_protocol": "random_5fold_row_cv",
        "validation_design": "random_row_cv",
        "evaluation_population": "random_5fold_full_oof_5575",
        "target_contract": "phase2plus_phase3plus_phase4plus_phase5",
        "parameter_contract": "notebook_effective_general_params_all_targets",
        "rerun_interpretation": (
            "reproducible_random_cv_rerun_not_exact_historical_last_fold"
        ),
    }
    for column, expected in expected_contract.items():
        if str(audit.loc[0, column]) != expected:
            raise ValueError(f"Contemporaneous audit {column} differs from the contract.")
    if int(audit.loc[0, "phase_round_decimals"]) != 2:
        raise ValueError("Contemporaneous audit must specify two-decimal rounding.")
    if not np.isclose(float(audit.loc[0, "phase_threshold"]), 0.20):
        raise ValueError("Contemporaneous audit phase threshold differs from 0.20.")
    if int(audit.loc[0, "n_splits"]) != 5:
        raise ValueError("Contemporaneous audit must specify five folds.")
    if str(audit.loc[0, "fold_rows"]) != "1115|1115|1115|1115|1115":
        raise ValueError("Contemporaneous audit fold sizes differ from the contract.")
    if int(audit.loc[0, "feature_count"]) != 174:
        raise ValueError("Contemporaneous audit feature count differs from 174.")
    if str(audit.loc[0, "kfolds_predictor_included"]).lower() != "true":
        raise ValueError("Contemporaneous audit must retain kfolds as a predictor.")
    if int(audit.loc[0, "shuffle_seed"]) != DEFAULT_CONTEMPORANEOUS_SHUFFLE_SEED:
        raise ValueError("Contemporaneous audit shuffle seed differs from zero.")
    if not pd.isna(audit.loc[0, "estimator_n_jobs"]):
        raise ValueError("Contemporaneous audit must retain default estimator threads.")
    if production_run:
        if key_sha256 != EXPECTED_CONTEMPORANEOUS_POPULATION_KEY_SHA256:
            raise ValueError("Formal contemporaneous population key hash changed.")
        if fold_sha256 != EXPECTED_CONTEMPORANEOUS_FOLD_ASSIGNMENT_SHA256:
            raise ValueError("Formal contemporaneous fold assignment hash changed.")
        if predictions_sha256 != EXPECTED_CONTEMPORANEOUS_PREDICTIONS_SHA256:
            raise ValueError("Formal contemporaneous prediction sidecar hash changed.")
        if _sha256_file(audit_path) != EXPECTED_CONTEMPORANEOUS_AUDIT_SHA256:
            raise ValueError("Formal contemporaneous source-audit hash changed.")
    return predictions, audit


def build_three_model_prediction_frames(
    predictions_by_model: Mapping[str, pd.DataFrame],
    contemporaneous_predictions: pd.DataFrame,
    expected_test_rows: int = DEFAULT_EXPECTED_TEST_ROWS,
    expected_contemporaneous_rows: int = DEFAULT_EXPECTED_CONTEMPORANEOUS_ROWS,
) -> dict[str, pd.DataFrame]:
    """Standardize panels while retaining each model's evaluation population."""
    standardized: dict[str, pd.DataFrame] = {}
    required_phase_columns = [
        f"phase{phase}_{suffix}"
        for phase in THREE_MODEL_PHASE_SPECS
        for suffix in ("test", "pred")
    ]
    for model_name in MODEL_ORDER:
        if model_name not in predictions_by_model:
            raise ValueError(f"Missing predictions for {model_name}.")
        frame = _normalize_area_date_keys(
            predictions_by_model[model_name], f"{model_name} predictions"
        )
        required = [*KEY_COLUMNS, *required_phase_columns]
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(f"{model_name} predictions are missing columns: {missing}")
        if len(frame) != expected_test_rows:
            raise ValueError(
                f"Expected {expected_test_rows} test rows for {model_name}, "
                f"found {len(frame)}."
            )
        frame = frame.copy()
        frame["fold"] = pd.NA
        frame["evaluation_protocol"] = "fixed_2022_temporal_holdout"
        frame["evaluation_population"] = "canonical_1170_temporal_test"
        standardized[model_name] = frame

    reference = standardized["Forecasting"]
    reference_columns = [
        *KEY_COLUMNS,
        *[f"phase{phase}_test" for phase in THREE_MODEL_PHASE_SPECS],
    ]
    reference_panel = reference[reference_columns].copy()
    nowcasting_check = reference_panel.merge(
        standardized["Nowcasting"][reference_columns],
        on=["area_id", "date"],
        how="outer",
        validate="one_to_one",
        suffixes=("_forecasting", "_nowcasting"),
        indicator=True,
    )
    if len(nowcasting_check) != expected_test_rows or not nowcasting_check[
        "_merge"
    ].eq("both").all():
        raise ValueError("Forecasting and Nowcasting grid key sets differ.")
    for column in ("country_code_3", "source_row_index"):
        if not nowcasting_check[f"{column}_forecasting"].equals(
            nowcasting_check[f"{column}_nowcasting"]
        ):
            raise ValueError(f"Forecasting and Nowcasting {column} values differ.")
    for phase in THREE_MODEL_PHASE_SPECS:
        forecasting_actual = pd.to_numeric(
            nowcasting_check[f"phase{phase}_test_forecasting"], errors="raise"
        )
        nowcasting_actual = pd.to_numeric(
            nowcasting_check[f"phase{phase}_test_nowcasting"], errors="raise"
        )
        if not np.allclose(forecasting_actual, nowcasting_actual, rtol=0.0, atol=0.0):
            raise ValueError(f"Forecasting and Nowcasting Phase {phase}+ truth differs.")

    sidecar = _normalize_area_date_keys(
        contemporaneous_predictions, "Contemporaneous predictions"
    )
    required_sidecar = [
        *CONTEMPORANEOUS_KEY_COLUMNS,
        "fold",
        "evaluation_protocol",
        "evaluation_population",
        *[
            column
            for phase in THREE_MODEL_PHASE_SPECS
            for column in (
                f"phase{phase}_actual",
                f"phase{phase}_contemporaneous",
            )
        ],
    ]
    missing_sidecar = sorted(set(required_sidecar).difference(sidecar.columns))
    if missing_sidecar:
        raise ValueError(
            f"Contemporaneous predictions are missing columns: {missing_sidecar}"
        )
    if len(sidecar) != expected_contemporaneous_rows:
        raise ValueError(
            f"Expected {expected_contemporaneous_rows} contemporaneous rows, "
            f"found {len(sidecar)}."
        )
    if not sidecar["source_row_index"].is_unique:
        raise ValueError("Contemporaneous predictions contain duplicate source rows.")
    if set(sidecar["evaluation_protocol"].astype(str)) != {"random_5fold_row_cv"}:
        raise ValueError("Contemporaneous panels must use random five-fold row CV.")
    if set(sidecar["evaluation_population"].astype(str)) != {
        "random_5fold_full_oof_5575"
    }:
        raise ValueError("Contemporaneous panels must use the full OOF population.")

    contemporaneous = pd.DataFrame(
        {
            "area_id": sidecar["area_id"].to_numpy(),
            "date": sidecar["date"].to_numpy(),
            "country_code_3": pd.Series([pd.NA] * len(sidecar), dtype="string"),
            "source_row_index": pd.to_numeric(
                sidecar["source_row_index"], errors="raise"
            ).astype(int),
            "fold": pd.to_numeric(sidecar["fold"], errors="raise").astype(int),
            "evaluation_protocol": sidecar["evaluation_protocol"].astype(str),
            "evaluation_population": sidecar["evaluation_population"].astype(str),
        }
    )
    for phase in THREE_MODEL_PHASE_SPECS:
        contemporaneous[f"phase{phase}_test"] = pd.to_numeric(
            sidecar[f"phase{phase}_actual"], errors="raise"
        )
        contemporaneous[f"phase{phase}_pred"] = pd.to_numeric(
            sidecar[f"phase{phase}_contemporaneous"], errors="raise"
        ).round(2)
    standardized["Contemporaneous"] = contemporaneous
    return {model_name: standardized[model_name] for model_name in THREE_MODEL_ORDER}


def calculate_r2_summary(long_predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate one R-squared value per model and cumulative phase."""
    records: list[dict[str, object]] = []
    for phase, phase_spec in PHASE_SPECS.items():
        for model_name in MODEL_ORDER:
            panel = long_predictions.loc[
                long_predictions["phase"].eq(phase)
                & long_predictions["model"].eq(model_name)
            ]
            if panel.empty:
                raise ValueError(f"No rows for {model_name} {phase_spec['label']}.")
            records.append(
                {
                    "model": model_name,
                    "phase": phase,
                    "phase_label": phase_spec["label"],
                    "n_test": int(len(panel)),
                    "n_areas": int(panel["area_id"].nunique()),
                    "r2": float(r2_score(panel["actual"], panel["predicted"])),
                    "actual_min": float(panel["actual"].min()),
                    "actual_max": float(panel["actual"].max()),
                    "predicted_min": float(panel["predicted"].min()),
                    "predicted_max": float(panel["predicted"].max()),
                }
            )
    return pd.DataFrame(records)


def _panel_statistics(panel: pd.DataFrame) -> dict[str, float | bool]:
    """Return R-squared and the actual-on-predicted linear-fit coefficients."""
    x_values = panel["predicted"].to_numpy(dtype=float)
    y_values = panel["actual"].to_numpy(dtype=float)
    fit_estimable = np.unique(x_values).size >= 2
    if fit_estimable:
        slope, intercept = np.polyfit(x_values, y_values, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        "r2": float(r2_score(y_values, x_values)),
        "intercept": float(intercept),
        "slope": float(slope),
        "fit_estimable": bool(fit_estimable),
    }


def calculate_three_model_grid_summary(
    long_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the displayed statistics for all nine grid panels."""
    records: list[dict[str, object]] = []
    for phase, phase_spec in THREE_MODEL_PHASE_SPECS.items():
        for model_name in THREE_MODEL_ORDER:
            panel = long_predictions.loc[
                long_predictions["phase"].eq(phase)
                & long_predictions["model"].eq(model_name)
            ]
            if panel.empty:
                raise ValueError(f"No rows for {model_name} {phase_spec['label']}.")
            protocols = panel["evaluation_protocol"].dropna().astype(str).unique()
            populations = panel["evaluation_population"].dropna().astype(str).unique()
            if len(protocols) != 1 or len(populations) != 1:
                raise ValueError(
                    f"{model_name} {phase_spec['label']} has ambiguous evaluation metadata."
                )
            records.append(
                {
                    "model": model_name,
                    "phase": phase,
                    "phase_label": phase_spec["label"],
                    "evaluation_protocol": protocols[0],
                    "evaluation_population": populations[0],
                    "n_test": int(len(panel)),
                    "n_areas": int(panel["area_id"].nunique()),
                    "n_folds": int(panel["fold"].nunique(dropna=True)),
                    **_panel_statistics(panel),
                    "actual_min": float(panel["actual"].min()),
                    "actual_max": float(panel["actual"].max()),
                    "predicted_min": float(panel["predicted"].min()),
                    "predicted_max": float(panel["predicted"].max()),
                }
            )
    return pd.DataFrame(records)


def create_phase_scatter_figure(
    long_predictions: pd.DataFrame, phase: int
) -> plt.Figure:
    """Create one reference-matched 1 x 2 actual-vs-predicted scatter figure."""
    if phase not in PHASE_SPECS:
        raise ValueError(f"Unsupported phase: {phase}")
    phase_spec = PHASE_SPECS[phase]
    plotting = long_predictions.loc[long_predictions["phase"].eq(phase)].copy()
    if plotting.empty:
        raise ValueError(f"No plotting rows for {phase_spec['label']}.")
    loco.apply_figure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.65),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    values = plotting[["actual", "predicted"]].to_numpy(dtype=float)
    lower = min(0.0, float(np.nanmin(values)))
    upper = max(0.0, float(np.nanmax(values)))
    if upper <= lower:
        upper = lower + 0.1
    padding = max(0.01, (upper - lower) * 0.04)
    limits = (lower - padding, upper + padding)
    letters = ("a", "b")

    for axis, model_name, letter in zip(axes, MODEL_ORDER, letters):
        panel = plotting.loc[plotting["model"].eq(model_name)].copy()
        if panel.empty:
            raise ValueError(f"No plotting rows for {model_name}.")
        axis.scatter(
            panel["predicted"],
            panel["actual"],
            s=12,
            color="#E69F00",
            edgecolor="none",
            alpha=0.38,
            label="Data points",
            zorder=3,
        )
        x_values = panel["predicted"].to_numpy(dtype=float)
        y_values = panel["actual"].to_numpy(dtype=float)
        if np.unique(x_values).size >= 2:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            fit_x = np.linspace(float(x_values.min()), float(x_values.max()), 200)
            axis.plot(
                fit_x,
                intercept + slope * fit_x,
                color="#D55E00",
                linewidth=1.5,
                label="Linear fit",
                zorder=4,
            )
        else:
            axis.text(
                0.045,
                0.865,
                "Linear fit not estimable\n(constant predictions)",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="#555555",
            )
        axis.plot(
            limits,
            limits,
            color="#777777",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            label="Perfect prediction (y=x)",
            zorder=2,
        )
        r2 = float(r2_score(y_values, x_values))
        axis.text(
            0.045,
            0.94,
            f"R² = {r2:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#222222",
        )
        axis.text(
            -0.12,
            1.04,
            letter,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )
        axis.set_title(model_name, loc="left", fontweight="normal", pad=5)
        axis.set_xlabel("Predicted population share")
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(
            True,
            color="#D9D9D9",
            linewidth=0.5,
            linestyle=(0, (2, 3)),
            alpha=0.8,
            zorder=0,
        )
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(0.8)
    axes[0].set_ylabel("Actual population share")
    handles_by_label: dict[str, object] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    legend_order = [
        "Data points",
        "Linear fit",
        "Perfect prediction (y=x)",
    ]
    legend_labels = [label for label in legend_order if label in handles_by_label]
    fig.legend(
        [handles_by_label[label] for label in legend_labels],
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.suptitle(
        f"Actual vs predicted population share in {phase_spec['label']}",
        x=0.08,
        ha="left",
        fontsize=8,
        fontweight="normal",
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.86, wspace=0.08)
    fig.canvas.draw()
    return fig


def create_three_model_phase_grid(
    long_predictions: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
) -> plt.Figure:
    """Create the Phase 2+/3+/4+ by three-model scatter grid."""
    expected_rows = len(THREE_MODEL_ORDER) * len(THREE_MODEL_PHASE_SPECS)
    if metrics is None:
        metrics = calculate_three_model_grid_summary(long_predictions)
    if len(metrics) != expected_rows:
        raise ValueError(f"Expected {expected_rows} grid metric rows, found {len(metrics)}.")
    metric_lookup = metrics.set_index(["phase", "model"])

    loco.apply_figure_style()
    fig, axes = plt.subplots(
        len(THREE_MODEL_PHASE_SPECS),
        len(THREE_MODEL_ORDER),
        figsize=(9.4, 9.2),
        gridspec_kw={"wspace": 0.10, "hspace": 0.13},
    )
    letters = iter("abcdefghi")
    handles_by_label: dict[str, object] = {}

    for row_index, (phase, phase_spec) in enumerate(THREE_MODEL_PHASE_SPECS.items()):
        row_plotting = long_predictions.loc[long_predictions["phase"].eq(phase)]
        if row_plotting.empty:
            raise ValueError(f"No plotting rows for {phase_spec['label']}.")
        values = row_plotting[["actual", "predicted"]].to_numpy(dtype=float)
        lower = min(0.0, float(np.nanmin(values)))
        upper = max(0.0, float(np.nanmax(values)))
        if upper <= lower:
            upper = lower + 0.1
        padding = max(0.01, (upper - lower) * 0.04)
        limits = (lower - padding, upper + padding)

        for column_index, model_name in enumerate(THREE_MODEL_ORDER):
            axis = axes[row_index, column_index]
            panel = row_plotting.loc[row_plotting["model"].eq(model_name)]
            if panel.empty:
                raise ValueError(f"No plotting rows for {model_name} {phase_spec['label']}.")
            axis.scatter(
                panel["predicted"],
                panel["actual"],
                s=5 if model_name == "Contemporaneous" else 9,
                color="#E69F00",
                edgecolor="none",
                alpha=0.17 if model_name == "Contemporaneous" else 0.34,
                label="Data points",
                zorder=3,
            )

            statistics = metric_lookup.loc[(phase, model_name)]
            x_values = panel["predicted"].to_numpy(dtype=float)
            if bool(statistics["fit_estimable"]):
                fit_x = np.linspace(float(x_values.min()), float(x_values.max()), 200)
                axis.plot(
                    fit_x,
                    float(statistics["intercept"])
                    + float(statistics["slope"]) * fit_x,
                    color="#D55E00",
                    linewidth=1.35,
                    label="Linear fit",
                    zorder=4,
                )
                annotation = (
                    f"R² = {float(statistics['r2']):.3f}\n"
                    f"Intercept = {float(statistics['intercept']):.3f}\n"
                    f"Slope = {float(statistics['slope']):.3f}\n"
                    f"n = {int(statistics['n_test']):,}"
                )
            else:
                annotation = (
                    f"R² = {float(statistics['r2']):.3f}\n"
                    "Intercept = n.e.\nSlope = n.e.\n"
                    f"n = {int(statistics['n_test']):,}"
                )
            axis.plot(
                limits,
                limits,
                color="#777777",
                linewidth=0.9,
                linestyle=(0, (4, 3)),
                label="Perfect prediction (y=x)",
                zorder=2,
            )
            axis.text(
                0.045,
                0.955,
                annotation,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.4,
                linespacing=1.22,
                color="#222222",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.2,
                },
                zorder=5,
            )
            axis.text(
                -0.12,
                1.035,
                next(letters),
                transform=axis.transAxes,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="top",
            )
            if row_index == 0:
                axis.set_title(
                    THREE_MODEL_COLUMN_TITLES[model_name],
                    loc="center",
                    pad=6,
                    fontsize=8,
                )
            if column_index == 0:
                axis.set_ylabel(
                    f"{phase_spec['label']}\nActual population share",
                    labelpad=5,
                )
            else:
                axis.tick_params(labelleft=False)
            if row_index == len(THREE_MODEL_PHASE_SPECS) - 1:
                axis.set_xlabel("Predicted population share")
            else:
                axis.tick_params(labelbottom=False)
            axis.set_xlim(*limits)
            axis.set_ylim(*limits)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(
                True,
                color="#D9D9D9",
                linewidth=0.45,
                linestyle=(0, (2, 3)),
                alpha=0.75,
                zorder=0,
            )
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#222222")
                spine.set_linewidth(0.75)
            handles, labels = axis.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                handles_by_label.setdefault(label, handle)

    legend_order = ["Data points", "Linear fit", "Perfect prediction (y=x)"]
    legend_labels = [label for label in legend_order if label in handles_by_label]
    fig.legend(
        [handles_by_label[label] for label in legend_labels],
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.032),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Actual vs predicted cumulative population shares",
        x=0.085,
        y=0.993,
        ha="left",
        fontsize=9,
        fontweight="normal",
    )
    fig.text(
        0.085,
        0.966,
        "Forecasting and Nowcasting use the fixed 2022 temporal holdout "
        "(n = 1,170 per panel).\nContemporaneous uses random five-fold "
        "full-OOF row CV (n = 5,575 per panel).",
        ha="left",
        va="top",
        fontsize=6.8,
        color="#4D4D4D",
    )
    fig.text(
        0.5,
        0.009,
        "Validation protocols and populations differ; panel metrics are descriptive "
        "and not directly comparable.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#4D4D4D",
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.105, top=0.875)
    fig.canvas.draw()
    return fig


def _save_phase_figure(
    figure: plt.Figure, output_dir: Path, phase: int
) -> dict[str, Path]:
    stem = (
        f"{PHASE_SPECS[phase]['stem']}_actual_vs_predicted_"
        "forecasting_nowcasting"
    )
    paths = {suffix: output_dir / f"{stem}.{suffix}" for suffix in ("jpg", "png", "pdf")}
    figure.savefig(paths["jpg"], dpi=300, facecolor="white")
    figure.savefig(paths["png"], dpi=300, facecolor="white")
    figure.savefig(paths["pdf"], facecolor="white")
    plt.close(figure)
    return paths


def _save_three_model_grid_figure(
    figure: plt.Figure, output_dir: Path
) -> dict[str, Path]:
    paths = {
        suffix: output_dir / f"{THREE_MODEL_FIGURE_STEM}.{suffix}"
        for suffix in ("jpg", "png", "pdf")
    }
    figure.savefig(paths["jpg"], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths


def write_artifacts(
    predictions_by_model: Mapping[str, pd.DataFrame],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_test_rows: int = DEFAULT_EXPECTED_TEST_ROWS,
    expected_contemporaneous_rows: int = DEFAULT_EXPECTED_CONTEMPORANEOUS_ROWS,
    source_audit: pd.DataFrame | None = None,
    contemporaneous_predictions: pd.DataFrame | None = None,
    contemporaneous_audit: pd.DataFrame | None = None,
    contemporaneous_predictions_path: Path | None = None,
    contemporaneous_audit_path: Path | None = None,
) -> dict[str, Path]:
    """Write legacy outputs and, when supplied, the three-model 3 x 3 grid."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_predictions = build_long_predictions(
        predictions_by_model, expected_test_rows=expected_test_rows
    )
    metrics = calculate_r2_summary(long_predictions)
    if source_audit is None:
        source_audit = pd.DataFrame(
            {
                "model": list(MODEL_ORDER),
                "rows": [expected_test_rows] * len(MODEL_ORDER),
            }
        )
    else:
        source_audit = source_audit.copy()

    if (contemporaneous_predictions is None) != (contemporaneous_audit is None):
        raise ValueError(
            "Contemporaneous predictions and source audit must be supplied together."
        )
    three_model_long: pd.DataFrame | None = None
    three_model_metrics: pd.DataFrame | None = None
    if contemporaneous_predictions is not None:
        three_model_frames = build_three_model_prediction_frames(
            predictions_by_model,
            contemporaneous_predictions,
            expected_test_rows=expected_test_rows,
            expected_contemporaneous_rows=expected_contemporaneous_rows,
        )
        three_model_long = build_long_predictions(
            three_model_frames,
            expected_test_rows=expected_test_rows,
            model_order=THREE_MODEL_ORDER,
            phase_specs=THREE_MODEL_PHASE_SPECS,
            expected_rows_by_model={
                "Forecasting": expected_test_rows,
                "Nowcasting": expected_test_rows,
                "Contemporaneous": expected_contemporaneous_rows,
            },
        )
        three_model_metrics = calculate_three_model_grid_summary(three_model_long)
        source_audit = pd.concat(
            [source_audit, contemporaneous_audit.copy()],
            ignore_index=True,
            sort=False,
        )
        source_audit["three_model_grid_included"] = True
        source_audit["three_model_grid_figure_stem"] = THREE_MODEL_FIGURE_STEM
        source_audit["three_model_grid_phases"] = "2|3|4"
        source_audit["three_model_grid_rows_by_model"] = (
            f"Forecasting:{expected_test_rows}|Nowcasting:{expected_test_rows}|"
            f"Contemporaneous:{expected_contemporaneous_rows}"
        )
        source_audit["three_model_grid_panel_count"] = len(THREE_MODEL_ORDER) * len(
            THREE_MODEL_PHASE_SPECS
        )
        source_audit["forecasting_nowcasting_key_sha256"] = _canonical_key_sha256(
            three_model_long.loc[
                three_model_long["model"].eq("Forecasting")
                & three_model_long["phase"].eq(2)
            ]
        )
        source_audit["contemporaneous_population_key_sha256"] = (
            _canonical_key_sha256(
                three_model_long.loc[
                    three_model_long["model"].eq("Contemporaneous")
                    & three_model_long["phase"].eq(2)
                ]
            )
        )
        source_audit["contemporaneous_fold_assignment_sha256"] = (
            _fold_assignment_sha256(contemporaneous_predictions)
        )
        source_audit["three_model_grid_directly_comparable"] = False
        source_audit["three_model_grid_comparability_note"] = (
            "forecasting_nowcasting_temporal_holdout_vs_contemporaneous_random_cv"
        )
        source_audit["contemporaneous_prediction_round_decimals"] = 2
        source_audit["contemporaneous_predictions_sha256"] = (
            _sha256_file(Path(contemporaneous_predictions_path))
            if contemporaneous_predictions_path is not None
            else pd.NA
        )
        source_audit["contemporaneous_source_audit_sha256"] = (
            _sha256_file(Path(contemporaneous_audit_path))
            if contemporaneous_audit_path is not None
            else pd.NA
        )

    phase_figures: dict[int, plt.Figure] = {}
    three_model_figure: plt.Figure | None = None
    try:
        for phase in PHASE_SPECS:
            phase_figures[phase] = create_phase_scatter_figure(
                long_predictions, phase
            )
        if three_model_long is not None and three_model_metrics is not None:
            three_model_figure = create_three_model_phase_grid(
                three_model_long, three_model_metrics
            )
    except Exception:
        for figure in phase_figures.values():
            plt.close(figure)
        if three_model_figure is not None:
            plt.close(three_model_figure)
        raise

    artifacts: dict[str, Path] = {}
    for phase, figure in phase_figures.items():
        stem = (
            f"{PHASE_SPECS[phase]['stem']}_actual_vs_predicted_"
            "forecasting_nowcasting"
        )
        for suffix, path in _save_phase_figure(figure, output_dir, phase).items():
            artifacts[f"{stem}_{suffix}"] = path

    if (
        three_model_figure is not None
        and three_model_long is not None
        and three_model_metrics is not None
    ):
        for suffix, path in _save_three_model_grid_figure(
            three_model_figure, output_dir
        ).items():
            artifacts[f"three_model_grid_{suffix}"] = path
        three_model_predictions_path = (
            output_dir / f"{THREE_MODEL_TABLE_STEM}_predictions.csv"
        )
        three_model_metrics_path = output_dir / f"{THREE_MODEL_TABLE_STEM}_metrics.csv"
        three_model_long.to_csv(
            three_model_predictions_path, index=False, float_format="%.10g"
        )
        three_model_metrics.to_csv(
            three_model_metrics_path, index=False, float_format="%.10g"
        )
        artifacts["three_model_predictions_csv"] = three_model_predictions_path
        artifacts["three_model_metrics_csv"] = three_model_metrics_path

    predictions_path = output_dir / "phase_cumulative_scatter_predictions.csv"
    metrics_path = output_dir / "phase_cumulative_scatter_metrics.csv"
    audit_path = output_dir / "phase_cumulative_scatter_source_audit.csv"
    long_predictions.to_csv(predictions_path, index=False, float_format="%.10g")
    metrics.to_csv(metrics_path, index=False, float_format="%.10g")
    source_audit.to_csv(audit_path, index=False, float_format="%.10g")
    artifacts["predictions_csv"] = predictions_path
    artifacts["metrics_csv"] = metrics_path
    artifacts["source_audit_csv"] = audit_path
    return artifacts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_audit(
    predictions_by_model: Mapping[str, pd.DataFrame],
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    forecasting_anchor_path: Path,
    nowcasting_anchor_path: Path,
    cutoff: str,
    random_state: int | None,
    workers: int,
    estimator_n_jobs: int | None,
    production_run: bool,
) -> pd.DataFrame:
    """Record inputs, environment, temporal contract, and Phase 3+ calibration."""
    forecasting_raw = pd.read_csv(forecasting_path, usecols=["date"])
    raw_dates = pd.to_datetime(forecasting_raw["date"], errors="raise")
    cutoff_date = pd.Timestamp(cutoff)
    anchor_specs = {
        "Forecasting": (
            Path(forecasting_anchor_path),
            "phase3_pred_fc",
            "phase3_test_fc",
        ),
        "Nowcasting": (
            Path(nowcasting_anchor_path),
            "phase3_pred_nc",
            "phase3_test_nc",
        ),
    }
    shared = {
        "evaluation_protocol": "fixed_2022_temporal_holdout",
        "evaluation_population": "canonical_1170_temporal_test",
        "cutoff": cutoff,
        "source_rows": int(len(forecasting_raw)),
        "train_rows": int(raw_dates.lt(cutoff_date).sum()),
        "test_rows": int(raw_dates.ge(cutoff_date).sum()),
        "freeze_id": frozen_main_result.FREEZE_ID,
        "reference_environment_id": frozen_main_result.ENVIRONMENT["environment_id"],
        "run_label": "formal" if production_run else "diagnostic",
        "formal_environment_passed": bool(production_run),
        "xgboost_random_state_override": random_state,
        "xgboost_n_jobs_override": estimator_n_jobs,
        "xgboost_uses_default_threads": estimator_n_jobs is None,
        "outer_model_workers": workers,
        "max_parallel_models": min(max(workers, 1), len(MODEL_ORDER)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "forecasting_input_sha256": _sha256_file(forecasting_path),
        "nowcasting_input_sha256": _sha256_file(nowcasting_path),
        "country_lookup_sha256": _sha256_file(country_lookup_path),
        "general_params_sha256": _sha256_file(general_params_path),
        "phase3_params_sha256": _sha256_file(phase3_params_path),
        "generator_sha256": _sha256_file(Path(__file__)),
        "freeze_source_path": str(Path(frozen_main_result.__file__)),
        "freeze_source_sha256": _sha256_file(Path(frozen_main_result.__file__)),
        "phase2_parameter_source": Path(general_params_path).name,
        "phase3_phase4_phase5_parameter_source": Path(phase3_params_path).name,
        "nowcasting_layer2_target": "training_residual",
    }
    records: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        predictions = predictions_by_model[model_name]
        frozen_reference = frozen_main_result.RESULTS[model_name]
        anchor_path, anchor_predicted, anchor_actual = anchor_specs[model_name]
        anchor = pd.read_csv(anchor_path)
        generated_pairs = predictions[["phase3_pred", "phase3_test"]].to_numpy(float)
        anchor_pairs = anchor[[anchor_predicted, anchor_actual]].to_numpy(float)
        same_shape = generated_pairs.shape == anchor_pairs.shape
        records.append(
            {
                **shared,
                "model": model_name,
                "model_test_rows": int(len(predictions)),
                "model_test_areas": int(predictions["area_id"].nunique()),
                "phase3_generated_r2": float(
                    r2_score(predictions["phase3_test"], predictions["phase3_pred"])
                ),
                "frozen_figure1_phase3plus_r2": float(
                    frozen_reference["phase3plus_r2"]
                ),
                "serialized_anchor_recomputed_r2": float(
                    r2_score(anchor[anchor_actual], anchor[anchor_predicted])
                ),
                "generated_minus_frozen_figure1_r2": float(
                    r2_score(predictions["phase3_test"], predictions["phase3_pred"])
                    - float(frozen_reference["phase3plus_r2"])
                ),
                "phase3_serialized_anchor_rows": int(len(anchor)),
                "phase3_serialized_anchor_within_tolerance": bool(
                    same_shape
                    and np.allclose(
                        generated_pairs, anchor_pairs, rtol=0.0, atol=5e-8
                    )
                ),
                "phase3_serialized_anchor_tolerance": 5e-8,
                "phase3_serialized_anchor_max_abs_prediction_difference": (
                    float(
                        np.max(
                            np.abs(generated_pairs[:, 0] - anchor_pairs[:, 0])
                        )
                    )
                    if same_shape
                    else np.nan
                ),
                "phase3_serialized_anchor_path": str(anchor_path.relative_to(REPO_ROOT)),
                "phase3_serialized_anchor_sha256": _sha256_file(anchor_path),
                "calibration_note": (
                    "Main result uses the confirmed frozen Windows environment. "
                    "The serialized row anchor is retained separately because CSV "
                    "precision shifts recomputed R-squared by about 1e-9."
                ),
            }
        )
    audit = pd.DataFrame(records)
    if production_run:
        if not audit["phase3_serialized_anchor_within_tolerance"].all():
            raise ValueError(
                "Frozen Phase 3+ predictions exceed serialized-anchor tolerance."
            )
        if not np.allclose(
            audit["phase3_generated_r2"],
            audit["frozen_figure1_phase3plus_r2"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Frozen Phase 3+ R-squared values were not reproduced.")
    return audit


def validate_run_configuration(
    *,
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    forecasting_anchor_path: Path,
    nowcasting_anchor_path: Path,
    contemporaneous_predictions_path: Path,
    contemporaneous_audit_path: Path,
    output_dir: Path,
    cutoff: str,
    expected_test_rows: int,
    expected_contemporaneous_rows: int,
    random_state: int | None,
    workers: int,
    estimator_n_jobs: int | None,
) -> bool:
    """Protect formal outputs from environment or execution-control drift."""
    formal_output = Path(output_dir).resolve() == DEFAULT_OUTPUT_DIR.resolve()
    if not formal_output:
        return False
    expected_paths = {
        "forecasting input": (forecasting_path, DEFAULT_FORECASTING_INPUT),
        "nowcasting input": (nowcasting_path, DEFAULT_NOWCASTING_INPUT),
        "country lookup": (country_lookup_path, DEFAULT_COUNTRY_LOOKUP),
        "general parameters": (general_params_path, DEFAULT_GENERAL_PARAMS),
        "Phase 3 parameters": (phase3_params_path, DEFAULT_PHASE3_PARAMS),
        "Forecasting anchor": (forecasting_anchor_path, DEFAULT_FORECASTING_ANCHOR),
        "Nowcasting anchor": (nowcasting_anchor_path, DEFAULT_NOWCASTING_ANCHOR),
        "Contemporaneous predictions": (
            contemporaneous_predictions_path,
            DEFAULT_CONTEMPORANEOUS_PREDICTIONS,
        ),
        "Contemporaneous audit": (
            contemporaneous_audit_path,
            DEFAULT_CONTEMPORANEOUS_AUDIT,
        ),
    }
    mismatches = [
        name
        for name, (actual, expected) in expected_paths.items()
        if Path(actual).resolve() != Path(expected).resolve()
    ]
    if cutoff != DEFAULT_CUTOFF:
        mismatches.append("cutoff")
    if expected_test_rows != DEFAULT_EXPECTED_TEST_ROWS:
        mismatches.append("expected test rows")
    if expected_contemporaneous_rows != DEFAULT_EXPECTED_CONTEMPORANEOUS_ROWS:
        mismatches.append("expected contemporaneous rows")
    if random_state is not None:
        mismatches.append("explicit XGBoost random state")
    if estimator_n_jobs is not None:
        mismatches.append("explicit XGBoost n_jobs")
    if workers != 1:
        mismatches.append("outer model workers")
    if mismatches:
        raise ValueError(
            "Formal phase-cumulative output requires the frozen configuration; "
            f"mismatched: {', '.join(mismatches)}"
        )
    frozen_main_result.assert_frozen_environment(("matplotlib",))
    return True


def run_analysis(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = DEFAULT_PHASE3_PARAMS,
    forecasting_anchor_path: Path = DEFAULT_FORECASTING_ANCHOR,
    nowcasting_anchor_path: Path = DEFAULT_NOWCASTING_ANCHOR,
    contemporaneous_predictions_path: Path = DEFAULT_CONTEMPORANEOUS_PREDICTIONS,
    contemporaneous_audit_path: Path = DEFAULT_CONTEMPORANEOUS_AUDIT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cutoff: str = DEFAULT_CUTOFF,
    expected_test_rows: int = DEFAULT_EXPECTED_TEST_ROWS,
    expected_contemporaneous_rows: int = DEFAULT_EXPECTED_CONTEMPORANEOUS_ROWS,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    workers: int = DEFAULT_WORKERS,
    estimator_n_jobs: int | None = DEFAULT_ESTIMATOR_N_JOBS,
) -> dict[str, Path]:
    """Run both models and save all requested scatter-plot artifacts."""
    production_run = validate_run_configuration(
        forecasting_path=Path(forecasting_path),
        nowcasting_path=Path(nowcasting_path),
        country_lookup_path=Path(country_lookup_path),
        general_params_path=Path(general_params_path),
        phase3_params_path=Path(phase3_params_path),
        forecasting_anchor_path=Path(forecasting_anchor_path),
        nowcasting_anchor_path=Path(nowcasting_anchor_path),
        contemporaneous_predictions_path=Path(contemporaneous_predictions_path),
        contemporaneous_audit_path=Path(contemporaneous_audit_path),
        output_dir=Path(output_dir),
        cutoff=cutoff,
        expected_test_rows=expected_test_rows,
        expected_contemporaneous_rows=expected_contemporaneous_rows,
        random_state=random_state,
        workers=workers,
        estimator_n_jobs=estimator_n_jobs,
    )
    predictions = run_temporal_predictions(
        forecasting_path=Path(forecasting_path),
        nowcasting_path=Path(nowcasting_path),
        country_lookup_path=Path(country_lookup_path),
        general_params_path=Path(general_params_path),
        phase3_params_path=Path(phase3_params_path),
        cutoff=cutoff,
        random_state=random_state,
        workers=workers,
        estimator_n_jobs=estimator_n_jobs,
    )
    audit = build_source_audit(
        predictions,
        Path(forecasting_path),
        Path(nowcasting_path),
        Path(country_lookup_path),
        Path(general_params_path),
        Path(phase3_params_path),
        Path(forecasting_anchor_path),
        Path(nowcasting_anchor_path),
        cutoff,
        random_state,
        workers,
        estimator_n_jobs,
        production_run,
    )
    contemporaneous_predictions, contemporaneous_audit = (
        load_contemporaneous_artifacts(
            Path(contemporaneous_predictions_path),
            Path(contemporaneous_audit_path),
            expected_rows=expected_contemporaneous_rows,
            production_run=production_run,
        )
    )
    return write_artifacts(
        predictions,
        output_dir=Path(output_dir),
        expected_test_rows=expected_test_rows,
        expected_contemporaneous_rows=expected_contemporaneous_rows,
        source_audit=audit,
        contemporaneous_predictions=contemporaneous_predictions,
        contemporaneous_audit=contemporaneous_audit,
        contemporaneous_predictions_path=Path(contemporaneous_predictions_path),
        contemporaneous_audit_path=Path(contemporaneous_audit_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-input", type=Path, default=DEFAULT_FORECASTING_INPUT)
    parser.add_argument("--nowcasting-input", type=Path, default=DEFAULT_NOWCASTING_INPUT)
    parser.add_argument("--country-lookup", type=Path, default=DEFAULT_COUNTRY_LOOKUP)
    parser.add_argument("--general-params", type=Path, default=DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=DEFAULT_PHASE3_PARAMS)
    parser.add_argument("--forecasting-anchor", type=Path, default=DEFAULT_FORECASTING_ANCHOR)
    parser.add_argument("--nowcasting-anchor", type=Path, default=DEFAULT_NOWCASTING_ANCHOR)
    parser.add_argument(
        "--contemporaneous-predictions",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_PREDICTIONS,
    )
    parser.add_argument(
        "--contemporaneous-audit",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_AUDIT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--expected-test-rows", type=int, default=DEFAULT_EXPECTED_TEST_ROWS)
    parser.add_argument(
        "--expected-contemporaneous-rows",
        type=int,
        default=DEFAULT_EXPECTED_CONTEMPORANEOUS_ROWS,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Supplying a value creates a diagnostic, non-frozen run.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--estimator-n-jobs",
        type=int,
        default=DEFAULT_ESTIMATOR_N_JOBS,
        help="Supplying a value overrides frozen default XGBoost threads.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    artifacts = run_analysis(
        forecasting_path=arguments.forecasting_input,
        nowcasting_path=arguments.nowcasting_input,
        country_lookup_path=arguments.country_lookup,
        general_params_path=arguments.general_params,
        phase3_params_path=arguments.phase3_params,
        forecasting_anchor_path=arguments.forecasting_anchor,
        nowcasting_anchor_path=arguments.nowcasting_anchor,
        contemporaneous_predictions_path=arguments.contemporaneous_predictions,
        contemporaneous_audit_path=arguments.contemporaneous_audit,
        output_dir=arguments.output_dir,
        cutoff=arguments.cutoff,
        expected_test_rows=arguments.expected_test_rows,
        expected_contemporaneous_rows=arguments.expected_contemporaneous_rows,
        random_state=arguments.random_state,
        workers=arguments.workers,
        estimator_n_jobs=arguments.estimator_n_jobs,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
