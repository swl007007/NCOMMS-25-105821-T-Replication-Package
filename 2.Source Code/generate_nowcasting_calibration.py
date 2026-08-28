from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

import generate_leave_one_country_out_robustness as loco


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = (2, 3, 4, 5)
THRESHOLD = 0.20
PROJECTOR_X = np.arange(4, dtype=float)


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    train_rows: int
    calibration_rows: int
    test_rows: int
    evaluation_population_id: str
    sensitivity_analysis: bool
    default_output_dir: Path


DEFAULT_EXPERIMENT_VARIANT = "2021h2_2022"
EXPERIMENT_VARIANTS = {
    "2021h2_2022": ExperimentVariant(
        name="2021h2_2022",
        train_end=pd.Timestamp("2021-06-30"),
        calibration_start=pd.Timestamp("2021-07-01"),
        calibration_end=pd.Timestamp("2021-12-31"),
        test_start=pd.Timestamp("2022-01-01"),
        train_rows=3909,
        calibration_rows=496,
        test_rows=1170,
        evaluation_population_id="nowcasting_calibration_2021h2_2022",
        sensitivity_analysis=False,
        default_output_dir=loco.DEFAULT_OUTPUT_DIR / "nowcasting_calibration",
    ),
    "full2021_2022": ExperimentVariant(
        name="full2021_2022",
        train_end=pd.Timestamp("2020-12-31"),
        calibration_start=pd.Timestamp("2021-01-01"),
        calibration_end=pd.Timestamp("2021-12-31"),
        test_start=pd.Timestamp("2022-01-01"),
        train_rows=3446,
        calibration_rows=959,
        test_rows=1170,
        evaluation_population_id="nowcasting_calibration_full2021_2022_sensitivity",
        sensitivity_analysis=True,
        default_output_dir=loco.DEFAULT_OUTPUT_DIR / "nowcasting_calibration_full2021",
    ),
}


def get_experiment_variant(name: str) -> ExperimentVariant:
    try:
        return EXPERIMENT_VARIANTS[str(name)]
    except KeyError as error:
        raise ValueError(f"Unknown experiment variant: {name}") from error


RAW_SCORE_COLUMNS = tuple(f"phase{phase}_pred_raw" for phase in PHASES)
CALIBRATION_METRIC_COLUMNS = [
    "n_rows",
    "ordinal_mae",
    "balanced_accuracy",
    "phase4_support",
    "phase4_recall",
    "phase3plus_precision",
    "phase3plus_recall",
    "distribution_discrepancy",
    *[f"actual_phase{phase}_count" for phase in range(1, 6)],
    *[f"predicted_phase{phase}_count" for phase in range(1, 6)],
]
PREDICTION_COLUMNS = [
    "split",
    "area_id",
    "date",
    "country_code_3",
    "source_row_index",
    "source_overall_phase",
    "overall_phase",
    *[f"phase{phase}_test" for phase in PHASES],
    *RAW_SCORE_COLUMNS,
    *[
        column
        for phase in PHASES
        for column in (
            f"phase{phase}_layer1_pred",
            f"phase{phase}_residual_pred",
        )
    ],
    *[f"phase{phase}_identity_projected" for phase in PHASES],
    *[f"phase{phase}_identity_rounded" for phase in PHASES],
    *[f"phase{phase}_pred_adjusted" for phase in PHASES],
    *[f"phase{phase}_pred_projected" for phase in PHASES],
    *[f"phase{phase}_pred_rounded" for phase in PHASES],
    "identity_overall_phase_pred",
    "calibrated_overall_phase_pred",
    "selected_delta3",
    "selected_delta4",
]
METRICS_COLUMNS = ["split", "variant", *CALIBRATION_METRIC_COLUMNS]
GRID_COLUMNS = [
    "delta3",
    "delta4",
    *CALIBRATION_METRIC_COLUMNS,
    "ordinal_mae_change",
    "balanced_accuracy_change",
    "phase4_recall_change",
    "phase3plus_precision_change",
    "phase3plus_recall_change",
    "distribution_discrepancy_change",
    "offset_l1",
    "mae_improved",
    "phase4_recall_improved",
    "balanced_accuracy_not_lower",
    "phase3plus_precision_within_tolerance",
    "phase3plus_recall_within_tolerance",
    "accepted",
    "selection_rank",
    "selected",
]
SOURCE_AUDIT_COLUMNS = [
    "evaluation_population_id",
    "source_rows",
    "train_rows",
    "calibration_rows",
    "test_rows",
    "train_end",
    "calibration_start",
    "calibration_end",
    "test_start",
    "calibration_key_sha256",
    "test_key_sha256",
    "delta3",
    "delta4",
    "identity_selected",
    "random_state",
    "estimator_n_jobs",
    "estimator_uses_default_n_jobs",
    "model_workers",
    "python_version",
    "numpy_version",
    "pandas_version",
    "scikit_learn_version",
    "xgboost_version",
    "platform",
    "protected_manifest_sha256_before",
    "protected_manifest_sha256_after",
    "protected_manifest_match",
    "generator_path",
    "generator_sha256",
    "forecasting_input_path",
    "forecasting_input_sha256",
    "nowcasting_input_path",
    "nowcasting_input_sha256",
    "country_lookup_path",
    "country_lookup_sha256",
    "general_params_path",
    "general_params_sha256",
    "phase3_params_path",
    "phase3_params_sha256",
    "grid_path",
    "grid_sha256",
    "predictions_path",
    "predictions_sha256",
    "metrics_path",
    "metrics_sha256",
]
FULL2021_SOURCE_AUDIT_EXTRA_COLUMNS = [
    "experiment_variant",
    "sensitivity_analysis",
    "h2_artifact_manifest_sha256_before",
    "h2_artifact_manifest_sha256_after",
]
PROTECTED_RELATIVE_PATHS = (
    Path("2.Source Code/Table1_Forecasting_main.ipynb"),
    Path("2.Source Code/Table1_Nowcasting_two_layer.ipynb"),
    Path("1.Source Data/Forecasting_Analysis_010825.csv"),
    Path("1.Source Data/Nowcasting_Analysis_010825.csv"),
    Path("2.Source Code/forecasting_hyperparameters.json"),
    Path("2.Source Code/forecasting_hyperparameters_p3.json"),
    Path("1.Source Data/All_prediction.csv"),
    Path("1.Source Data/df_vis_nowacast.csv"),
    Path("1.Source Data/All_prediction_truth_disagreements.csv"),
    Path("1.Source Data/r2_frame_forecasting.csv"),
    Path("1.Source Data/r2_frame_nowcasting.csv"),
)
H2_ARTIFACT_PATHS = tuple(
    loco.DEFAULT_OUTPUT_DIR / "nowcasting_calibration" / name
    for name in (
        "calibration_grid.csv",
        "calibration_predictions.csv",
        "calibration_metrics.csv",
        "calibration_source_audit.csv",
    )
)
FORMAL_ENVIRONMENT = {
    "python_version": "3.12.10",
    "numpy_version": "2.2.6",
    "pandas_version": "2.2.3",
    "scikit_learn_version": "1.6.1",
    "xgboost_version": "3.0.0",
}


def project_cumulative_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Cumulative scores must have shape (n_rows, 4).")
    if not np.isfinite(values).all():
        raise ValueError("Cumulative scores must be finite.")
    projector = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False)
    return np.vstack(
        [projector.fit_transform(PROJECTOR_X, row) for row in values]
    )


def phase_from_rounded_scores(scores: np.ndarray) -> np.ndarray:
    rounded = np.asarray(scores, dtype=float)
    if rounded.ndim != 2 or rounded.shape[1] != 4:
        raise ValueError("Rounded cumulative scores must have shape (n_rows, 4).")
    frame = pd.DataFrame(
        rounded,
        columns=[f"phase{phase}_candidate" for phase in PHASES],
    )
    return loco._phase_from_cumulative(frame, "candidate")


def transform_scores(
    raw_scores: np.ndarray, delta3: float, delta4: float
) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_scores, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("Raw cumulative scores must have shape (n_rows, 4).")
    adjusted = raw.copy()
    adjusted[:, 1] += float(delta3)
    adjusted[:, 2] += float(delta4)
    projected = project_cumulative_scores(adjusted)
    rounded = np.round(projected, 2)
    predicted_phase = phase_from_rounded_scores(rounded)
    return {
        "adjusted": adjusted,
        "projected": projected,
        "rounded": rounded,
        "predicted_phase": predicted_phase,
    }


def calculate_calibration_metrics(
    actual_phase: np.ndarray, predicted_phase: np.ndarray
) -> dict[str, float | int]:
    actual = np.asarray(actual_phase, dtype=int)
    predicted = np.asarray(predicted_phase, dtype=int)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("Actual and predicted phases must be aligned vectors.")
    if len(actual) == 0:
        raise ValueError("Actual and predicted phases must not be empty.")
    actual_positive = actual >= 3
    predicted_positive = predicted >= 3
    phase4_mask = actual == 4
    phase4_support = int(phase4_mask.sum())
    phase4_recall = (
        float(np.mean(predicted[phase4_mask] == 4))
        if phase4_support
        else float("nan")
    )
    actual_shares = np.bincount(actual, minlength=6)[1:6] / len(actual)
    predicted_shares = np.bincount(predicted, minlength=6)[1:6] / len(predicted)
    record: dict[str, float | int] = {
        "n_rows": int(len(actual)),
        "ordinal_mae": float(np.mean(np.abs(actual - predicted))),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "phase4_support": phase4_support,
        "phase4_recall": phase4_recall,
        "phase3plus_precision": float(
            precision_score(actual_positive, predicted_positive, zero_division=0)
        ),
        "phase3plus_recall": float(
            recall_score(actual_positive, predicted_positive, zero_division=0)
        ),
        "distribution_discrepancy": float(
            np.abs(actual_shares - predicted_shares).sum()
        ),
    }
    for phase in range(1, 6):
        record[f"actual_phase{phase}_count"] = int(np.sum(actual == phase))
    for phase in range(1, 6):
        record[f"predicted_phase{phase}_count"] = int(np.sum(predicted == phase))
    return record


def build_temporal_masks(
    data: pd.DataFrame,
    experiment_variant: str = DEFAULT_EXPERIMENT_VARIANT,
) -> dict[str, pd.Series]:
    variant = get_experiment_variant(experiment_variant)
    if "date" not in data:
        raise ValueError("Model input is missing the date column.")
    dates = pd.to_datetime(data["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Model input contains an unparseable date.")
    masks = {
        "train": dates.le(variant.train_end),
        "calibration": dates.between(
            variant.calibration_start, variant.calibration_end
        ),
        "test": dates.ge(variant.test_start),
    }
    coverage = sum(mask.astype(int) for mask in masks.values())
    if not coverage.eq(1).all() or any(
        not bool(mask.any()) for mask in masks.values()
    ):
        raise ValueError(
            "Train, calibration, and test masks must be complete and disjoint."
        )
    return masks


def validate_split_key_alignment(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    forecasting_masks: dict[str, pd.Series],
    nowcasting_masks: dict[str, pd.Series],
) -> None:
    for split in ("train", "calibration", "test"):
        left = pd.MultiIndex.from_frame(
            forecasting.loc[forecasting_masks[split], loco.KEY_COLUMNS]
        )
        right = pd.MultiIndex.from_frame(
            nowcasting.loc[nowcasting_masks[split], loco.KEY_COLUMNS]
        )
        if left.has_duplicates or right.has_duplicates:
            raise ValueError(f"Forecasting and Nowcasting {split} keys must be unique.")
        if set(left) != set(right):
            raise ValueError(f"Forecasting and Nowcasting {split} keys differ.")


def validate_production_counts(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    experiment_variant: str = DEFAULT_EXPERIMENT_VARIANT,
) -> None:
    variant = get_experiment_variant(experiment_variant)
    expected = {
        "train": variant.train_rows,
        "calibration": variant.calibration_rows,
        "test": variant.test_rows,
    }
    if len(data) != 5575:
        raise ValueError("Prepared model input must contain 5,575 rows.")
    observed = {name: int(mask.sum()) for name, mask in masks.items()}
    if observed != expected:
        raise ValueError(
            f"Temporal split row counts differ: expected {expected}, observed {observed}."
        )


def load_prepared_inputs(
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasting_raw = pd.read_csv(forecasting_path)
    nowcasting_raw = pd.read_csv(nowcasting_path)
    lookup = loco.load_country_lookup(country_lookup_path)
    forecasting, nowcasting = loco.prepare_model_inputs(
        forecasting_raw, nowcasting_raw, lookup
    )
    forecasting = loco.add_cumulative_targets(forecasting)
    if len(forecasting) != 5575 or len(nowcasting) != 5575:
        raise ValueError("Prepared model inputs must each contain 5,575 rows.")
    return forecasting, nowcasting


def _validate_masks(
    data: pd.DataFrame, masks: dict[str, pd.Series], label: str
) -> None:
    if set(masks) != {"train", "calibration", "test"}:
        raise ValueError(f"{label} masks must contain train, calibration, and test.")
    for split, mask in masks.items():
        if not isinstance(mask, pd.Series) or not mask.index.equals(data.index):
            raise ValueError(f"{label} {split} mask index differs from its data.")
        if not pd.api.types.is_bool_dtype(mask.dtype):
            raise ValueError(f"{label} {split} mask must be boolean.")


def fit_calibration_branch(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    forecasting_masks: dict[str, pd.Series],
    nowcasting_masks: dict[str, pd.Series],
    layer1_features: tuple[str, ...],
    layer2_features: tuple[str, ...],
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    estimator_factory=xgb.XGBRegressor,
) -> pd.DataFrame:
    _validate_masks(forecasting, forecasting_masks, "Forecasting")
    _validate_masks(nowcasting, nowcasting_masks, "Nowcasting")
    validate_split_key_alignment(
        forecasting, nowcasting, forecasting_masks, nowcasting_masks
    )
    keys = [*loco.KEY_COLUMNS, "country_code_3"]
    required_forecasting = {
        *keys,
        "overall_phase",
        *loco.CUMULATIVE_TARGETS.values(),
        *layer1_features,
    }
    required_nowcasting = {*keys, *layer2_features}
    missing_forecasting = sorted(required_forecasting.difference(forecasting.columns))
    missing_nowcasting = sorted(required_nowcasting.difference(nowcasting.columns))
    if missing_forecasting:
        raise ValueError(f"Forecasting input is missing columns: {missing_forecasting}")
    if missing_nowcasting:
        raise ValueError(f"Nowcasting input is missing columns: {missing_nowcasting}")

    split_frames: dict[str, pd.DataFrame] = {}
    for split in ("calibration", "test"):
        source = forecasting.loc[forecasting_masks[split]]
        frame = source.loc[:, keys].copy()
        frame["source_row_index"] = source.index.to_numpy()
        frame["source_overall_phase"] = source["overall_phase"].to_numpy()
        frame["split"] = split
        split_frames[split] = frame

    for phase, target_column in loco.CUMULATIVE_TARGETS.items():
        params = general_params if phase == 2 else phase3_params
        train_forecasting = forecasting.loc[forecasting_masks["train"]]
        train_target = train_forecasting[target_column]
        layer1 = estimator_factory(**dict(params))
        layer1.fit(train_forecasting.loc[:, list(layer1_features)], train_target)
        layer1_train = np.asarray(
            layer1.predict(train_forecasting.loc[:, list(layer1_features)]),
            dtype=float,
        )
        residual_train = train_forecasting.loc[:, keys].copy()
        residual_train["layer1_residual"] = train_target.to_numpy() - layer1_train
        keyed_train = nowcasting.loc[
            nowcasting_masks["train"], [*keys, *layer2_features]
        ].merge(residual_train, on=keys, how="inner", validate="one_to_one")
        if len(keyed_train) != int(forecasting_masks["train"].sum()):
            raise ValueError("Layer 2 training merge lost rows.")
        layer2 = estimator_factory(**dict(params))
        layer2.fit(
            keyed_train.loc[:, list(layer2_features)], keyed_train["layer1_residual"]
        )

        for split in ("calibration", "test"):
            forecasting_split = forecasting.loc[forecasting_masks[split]]
            nowcasting_split = nowcasting.loc[nowcasting_masks[split]]
            layer1_prediction = np.asarray(
                layer1.predict(forecasting_split.loc[:, list(layer1_features)]),
                dtype=float,
            )
            residual_prediction = np.asarray(
                layer2.predict(nowcasting_split.loc[:, list(layer2_features)]),
                dtype=float,
            )
            phase_frame = forecasting_split.loc[:, keys].copy()
            phase_frame[f"phase{phase}_test"] = forecasting_split[
                target_column
            ].to_numpy()
            phase_frame[f"phase{phase}_layer1_pred"] = layer1_prediction
            residual_frame = nowcasting_split.loc[:, keys].copy()
            residual_frame[f"phase{phase}_residual_pred"] = residual_prediction
            phase_frame = phase_frame.merge(
                residual_frame, on=keys, how="inner", validate="one_to_one"
            )
            phase_frame[f"phase{phase}_pred_raw"] = (
                phase_frame[f"phase{phase}_layer1_pred"]
                + phase_frame[f"phase{phase}_residual_pred"]
            )
            split_frames[split] = split_frames[split].merge(
                phase_frame, on=keys, how="inner", validate="one_to_one"
            )

    combined = pd.concat(
        [split_frames["calibration"], split_frames["test"]], ignore_index=True
    )
    expected_rows = int(forecasting_masks["calibration"].sum()) + int(
        forecasting_masks["test"].sum()
    )
    if len(combined) != expected_rows:
        raise ValueError("Prediction assembly lost calibration or test rows.")
    prediction_columns = [
        column
        for phase in PHASES
        for column in (
            f"phase{phase}_layer1_pred",
            f"phase{phase}_residual_pred",
            f"phase{phase}_pred_raw",
        )
    ]
    if not np.isfinite(combined[prediction_columns].to_numpy(dtype=float)).all():
        raise ValueError("Model produced a non-finite cumulative prediction.")
    combined["overall_phase"] = loco._phase_from_cumulative(combined, "test")
    return combined.sort_values(
        ["split", "area_id", "date"], kind="mergesort"
    ).reset_index(drop=True)


def build_offset_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"delta3": delta3_int / 100.0, "delta4": delta4_int / 100.0}
            for delta3_int in range(0, -21, -1)
            for delta4_int in range(0, 21)
        ]
    )


def evaluate_candidate(
    calibration_rows: pd.DataFrame, delta3: float, delta4: float
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    missing = sorted(
        {"overall_phase", *RAW_SCORE_COLUMNS}.difference(calibration_rows.columns)
    )
    if missing:
        raise ValueError(f"Calibration rows are missing columns: {missing}")
    transformed = transform_scores(
        calibration_rows.loc[:, RAW_SCORE_COLUMNS].to_numpy(dtype=float),
        delta3,
        delta4,
    )
    metrics = calculate_calibration_metrics(
        calibration_rows["overall_phase"].to_numpy(dtype=int),
        transformed["predicted_phase"],
    )
    return {"delta3": delta3, "delta4": delta4, **metrics}, transformed


def _candidate_sort_columns() -> tuple[list[str], list[bool]]:
    return (
        [
            "ordinal_mae",
            "balanced_accuracy",
            "phase4_recall",
            "distribution_discrepancy",
            "offset_l1",
            "delta3",
            "delta4",
        ],
        [True, False, False, True, True, False, True],
    )


def select_candidate_index(candidates: pd.DataFrame) -> object:
    accepted = candidates.loc[candidates["accepted"].astype(bool)]
    if accepted.empty:
        identity = candidates.loc[
            candidates["delta3"].eq(0.0) & candidates["delta4"].eq(0.0)
        ]
        if len(identity) != 1:
            raise ValueError("Candidate table must contain exactly one identity row.")
        return identity.index[0]
    columns, ascending = _candidate_sort_columns()
    return accepted.sort_values(
        columns, ascending=ascending, kind="mergesort"
    ).index[0]


def search_offsets(
    calibration_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if "split" not in calibration_rows:
        raise ValueError("Offset search requires labelled calibration rows.")
    if calibration_rows.empty or set(calibration_rows["split"].astype(str)) != {
        "calibration"
    }:
        raise ValueError("Offset search accepts only calibration rows.")
    records = [
        evaluate_candidate(calibration_rows, row.delta3, row.delta4)[0]
        for row in build_offset_grid().itertuples(index=False)
    ]
    grid = pd.DataFrame.from_records(records)
    identity = grid.loc[grid["delta3"].eq(0.0) & grid["delta4"].eq(0.0)]
    if len(identity) != 1:
        raise ValueError("Offset grid must contain exactly one identity row.")
    identity_row = identity.iloc[0]
    for metric in (
        "ordinal_mae",
        "balanced_accuracy",
        "phase4_recall",
        "phase3plus_precision",
        "phase3plus_recall",
        "distribution_discrepancy",
    ):
        grid[f"{metric}_change"] = grid[metric] - float(identity_row[metric])
    grid["offset_l1"] = grid["delta3"].abs() + grid["delta4"].abs()
    nonidentity = ~(grid["delta3"].eq(0.0) & grid["delta4"].eq(0.0))
    grid["mae_improved"] = grid["ordinal_mae"] < float(
        identity_row["ordinal_mae"]
    )
    grid["phase4_recall_improved"] = grid["phase4_recall"] > float(
        identity_row["phase4_recall"]
    )
    grid["balanced_accuracy_not_lower"] = grid["balanced_accuracy"] >= float(
        identity_row["balanced_accuracy"]
    )
    grid["phase3plus_precision_within_tolerance"] = grid[
        "phase3plus_precision"
    ] >= float(identity_row["phase3plus_precision"]) - 0.02
    grid["phase3plus_recall_within_tolerance"] = grid[
        "phase3plus_recall"
    ] >= float(identity_row["phase3plus_recall"]) - 0.02
    grid["accepted"] = (
        nonidentity
        & grid["mae_improved"]
        & grid["phase4_recall_improved"]
        & grid["balanced_accuracy_not_lower"]
        & grid["phase3plus_precision_within_tolerance"]
        & grid["phase3plus_recall_within_tolerance"]
    )
    grid["selection_rank"] = pd.Series(pd.NA, index=grid.index, dtype="Int64")
    columns, ascending = _candidate_sort_columns()
    accepted_order = grid.loc[grid["accepted"]].sort_values(
        columns, ascending=ascending, kind="mergesort"
    )
    if not accepted_order.empty:
        grid.loc[accepted_order.index, "selection_rank"] = np.arange(
            1, len(accepted_order) + 1
        )
    selected_index = select_candidate_index(grid)
    if accepted_order.empty:
        grid.loc[selected_index, "selection_rank"] = 1
    grid["selected"] = False
    grid.loc[selected_index, "selected"] = True
    return grid.loc[:, GRID_COLUMNS], grid.loc[selected_index, GRID_COLUMNS].copy()


def build_prediction_table(
    raw_predictions: pd.DataFrame, selected_delta3: float, selected_delta4: float
) -> pd.DataFrame:
    required = set(PREDICTION_COLUMNS[:7])
    required.update(f"phase{phase}_test" for phase in PHASES)
    required.update(RAW_SCORE_COLUMNS)
    required.update(
        column
        for phase in PHASES
        for column in (
            f"phase{phase}_layer1_pred",
            f"phase{phase}_residual_pred",
        )
    )
    missing = sorted(required.difference(raw_predictions.columns))
    if missing:
        raise ValueError(f"Raw predictions are missing columns: {missing}")
    predictions = raw_predictions.loc[:, sorted(required, key=lambda x: list(raw_predictions.columns).index(x))].copy()
    if predictions.duplicated(["split", "area_id", "date"]).any():
        raise ValueError("Raw predictions contain duplicate split observation keys.")
    if set(predictions["split"].astype(str)) != {"calibration", "test"}:
        raise ValueError("Raw predictions must contain calibration and test rows.")
    raw_scores = predictions.loc[:, RAW_SCORE_COLUMNS].to_numpy(dtype=float)
    identity = transform_scores(raw_scores, 0.0, 0.0)
    selected = transform_scores(raw_scores, selected_delta3, selected_delta4)
    for index, phase in enumerate(PHASES):
        predictions[f"phase{phase}_identity_projected"] = identity["projected"][
            :, index
        ]
        predictions[f"phase{phase}_identity_rounded"] = identity["rounded"][:, index]
        predictions[f"phase{phase}_pred_adjusted"] = selected["adjusted"][:, index]
        predictions[f"phase{phase}_pred_projected"] = selected["projected"][:, index]
        predictions[f"phase{phase}_pred_rounded"] = selected["rounded"][:, index]
    predictions["identity_overall_phase_pred"] = identity["predicted_phase"]
    predictions["calibrated_overall_phase_pred"] = selected["predicted_phase"]
    predictions["selected_delta3"] = float(selected_delta3)
    predictions["selected_delta4"] = float(selected_delta4)
    predictions["split"] = pd.Categorical(
        predictions["split"], categories=["calibration", "test"], ordered=True
    )
    predictions = predictions.sort_values(
        ["split", "area_id", "date"], kind="mergesort"
    ).reset_index(drop=True)
    predictions["split"] = predictions["split"].astype("string")
    return predictions.loc[:, PREDICTION_COLUMNS]


def build_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("calibration", "test"):
        subset = predictions.loc[predictions["split"].eq(split)]
        if subset.empty:
            raise ValueError(f"Predictions are missing {split} rows.")
        for variant, predicted_column in (
            ("identity", "identity_overall_phase_pred"),
            ("selected", "calibrated_overall_phase_pred"),
        ):
            metrics = calculate_calibration_metrics(
                subset["overall_phase"].to_numpy(dtype=int),
                subset[predicted_column].to_numpy(dtype=int),
            )
            rows.append({"split": split, "variant": variant, **metrics})
    return pd.DataFrame.from_records(rows).loc[:, METRICS_COLUMNS]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def h2_artifact_manifest_sha256() -> str:
    missing = [path for path in H2_ARTIFACT_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Completed H2 artifacts are missing: {missing}")
    payload = "".join(
        f"{path.name}\t{file_sha256(path)}\n"
        for path in sorted(H2_ARTIFACT_PATHS, key=lambda item: item.name)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_output_dir(
    experiment_variant: str,
    output_dir: Path | None,
) -> Path:
    variant = get_experiment_variant(experiment_variant)
    return variant.default_output_dir if output_dir is None else Path(output_dir)


def source_audit_columns(experiment_variant: str) -> list[str]:
    variant = get_experiment_variant(experiment_variant)
    if variant.sensitivity_analysis:
        return [*SOURCE_AUDIT_COLUMNS, *FULL2021_SOURCE_AUDIT_EXTRA_COLUMNS]
    return list(SOURCE_AUDIT_COLUMNS)


def validate_generation_target(
    experiment_variant: str,
    output_dir: Path,
) -> None:
    variant = get_experiment_variant(experiment_variant)
    target = Path(output_dir).resolve()
    h2_target = get_experiment_variant("2021h2_2022").default_output_dir.resolve()
    full_target = get_experiment_variant(
        "full2021_2022"
    ).default_output_dir.resolve()
    if variant.name == "full2021_2022" and target == h2_target:
        raise ValueError("The full-year variant cannot use the H2 output directory.")
    if variant.name == "2021h2_2022" and target == full_target:
        raise ValueError("The H2 variant cannot use the full-year output directory.")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Generation target must be empty before writing.")


def canonical_key_sha256(data: pd.DataFrame) -> str:
    missing = sorted(set(loco.KEY_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Key hash input is missing columns: {missing}")
    keys = data.loc[:, loco.KEY_COLUMNS].copy()
    if keys["area_id"].isna().any():
        raise ValueError("Key hash input contains a missing area_id.")
    dates = pd.to_datetime(keys["date"], errors="coerce")
    if dates.isna().any() or not dates.dt.normalize().eq(dates).all():
        raise ValueError("Key hash dates must be valid timezone-naive midnight values.")
    keys["date"] = dates.dt.strftime("%Y-%m-%d")
    if keys.duplicated(loco.KEY_COLUMNS).any():
        raise ValueError("Key hash input contains duplicate normalized keys.")
    keys = keys.sort_values(loco.KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    payload = keys.to_csv(
        index=False,
        float_format="%.17g",
        na_rep="<NA>",
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protected_artifact_paths() -> list[Path]:
    paths = [REPO_ROOT / relative for relative in PROTECTED_RELATIVE_PATHS]
    graph_dir = REPO_ROOT / "2.Source Code" / "produced_graph"
    paths.extend(
        path
        for path in graph_dir.iterdir()
        if path.is_file() and path.name.startswith("all_prediction_temporal_test_")
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected baseline artifacts are missing: {missing}")
    return sorted(paths, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def protected_artifact_manifest_sha256(output_dir: Path) -> str:
    del output_dir
    payload = "".join(
        f"{path.relative_to(REPO_ROOT).as_posix()}\t{file_sha256(path)}\n"
        for path in protected_artifact_paths()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def formal_environment_record() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "platform": platform.platform(),
    }


def assert_formal_environment(observed: dict[str, str] | None = None) -> None:
    environment = formal_environment_record() if observed is None else dict(observed)
    mismatches = {
        key: (FORMAL_ENVIRONMENT[key], environment.get(key))
        for key in FORMAL_ENVIRONMENT
        if environment.get(key) != FORMAL_ENVIRONMENT[key]
    }
    if not str(environment.get("platform", "")).startswith("Windows-11"):
        mismatches["platform"] = ("Windows-11*", environment.get("platform"))
    if mismatches:
        raise RuntimeError(
            f"Formal calibration requires the approved formal Windows lineage: {mismatches}"
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="<NA>",
        lineterminator="\n",
    )


def _sorted_grid(grid: pd.DataFrame) -> pd.DataFrame:
    columns, ascending = _candidate_sort_columns()
    return grid.sort_values(
        ["selected", *columns],
        ascending=[False, *ascending],
        kind="mergesort",
    ).reset_index(drop=True)


def _sorted_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["split"] = pd.Categorical(
        frame["split"], categories=["calibration", "test"], ordered=True
    )
    frame = frame.sort_values(
        ["split", "area_id", "date"], kind="mergesort"
    ).reset_index(drop=True)
    frame["split"] = frame["split"].astype("string")
    return frame


def write_model_outputs(
    grid: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "grid": output_dir / "calibration_grid.csv",
        "predictions": output_dir / "calibration_predictions.csv",
        "metrics": output_dir / "calibration_metrics.csv",
    }
    _write_csv(_sorted_grid(grid).loc[:, GRID_COLUMNS], paths["grid"])
    _write_csv(
        _sorted_predictions(predictions).loc[:, PREDICTION_COLUMNS],
        paths["predictions"],
    )
    _write_csv(metrics.loc[:, METRICS_COLUMNS], paths["metrics"])
    return paths


def write_source_audit(
    audit: pd.DataFrame,
    output_dir: Path,
    experiment_variant: str = DEFAULT_EXPERIMENT_VARIANT,
) -> Path:
    expected_columns = source_audit_columns(experiment_variant)
    if audit.columns.tolist() != expected_columns or len(audit) != 1:
        raise ValueError("Source audit must contain the exact one-row schema.")
    path = Path(output_dir) / "calibration_source_audit.csv"
    _write_csv(audit, path)
    return path


def _assert_close(actual: object, expected: object, label: str) -> None:
    if isinstance(expected, (int, np.integer)):
        if int(actual) != int(expected):
            raise ValueError(f"Saved {label} does not recompute.")
        return
    if not np.isclose(
        float(actual), float(expected), rtol=0.0, atol=1e-12, equal_nan=True
    ):
        raise ValueError(f"Saved {label} does not recompute.")


def validate_written_model_outputs(paths: dict[str, Path]) -> None:
    required = {"grid", "predictions", "metrics"}
    if not required.issubset(paths):
        raise ValueError("Model output paths are incomplete.")
    grid = pd.read_csv(paths["grid"])
    predictions = pd.read_csv(paths["predictions"])
    metrics = pd.read_csv(paths["metrics"])
    if grid.columns.tolist() != GRID_COLUMNS or len(grid) != 441:
        raise ValueError("Saved calibration grid has the wrong schema or row count.")
    if grid[["delta3", "delta4"]].duplicated().any():
        raise ValueError("Saved calibration grid contains duplicate candidates.")
    if int(grid["selected"].sum()) != 1:
        raise ValueError("Saved calibration grid must contain one selected row.")
    if predictions.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Saved calibration predictions have the wrong schema.")
    if predictions.duplicated(["split", "area_id", "date"]).any():
        raise ValueError("Saved calibration predictions contain duplicate keys.")
    if set(predictions["split"].astype(str)) != {"calibration", "test"}:
        raise ValueError("Saved calibration predictions have the wrong splits.")
    for prefix in ("identity", "pred"):
        projected = predictions.loc[
            :, [f"phase{phase}_{prefix}_projected" for phase in PHASES]
        ].to_numpy(dtype=float)
        if not np.isfinite(projected).all():
            raise ValueError("Saved projected scores must be finite.")
        if not ((projected >= 0.0) & (projected <= 1.0)).all():
            raise ValueError("Saved projected scores must be bounded.")
        if not (projected[:, :-1] >= projected[:, 1:]).all():
            raise ValueError("Saved projected scores must be monotone.")
    selected = grid.loc[grid["selected"]].iloc[0]
    if not predictions["selected_delta3"].eq(float(selected["delta3"])).all():
        raise ValueError("Saved selected delta3 differs from the grid.")
    if not predictions["selected_delta4"].eq(float(selected["delta4"])).all():
        raise ValueError("Saved selected delta4 differs from the grid.")
    is_identity = float(selected["delta3"]) == 0.0 and float(selected["delta4"]) == 0.0
    if not is_identity and not bool(selected["accepted"]):
        raise ValueError("Saved non-identity selection does not pass the gate.")
    if is_identity and bool(grid["accepted"].any()):
        raise ValueError("Identity was selected even though an accepted candidate exists.")
    if metrics.columns.tolist() != METRICS_COLUMNS or len(metrics) != 4:
        raise ValueError("Saved calibration metrics have the wrong schema or row count.")
    expected_order = [
        ("calibration", "identity"),
        ("calibration", "selected"),
        ("test", "identity"),
        ("test", "selected"),
    ]
    if list(metrics[["split", "variant"]].itertuples(index=False, name=None)) != expected_order:
        raise ValueError("Saved calibration metrics have the wrong row order.")
    for row in metrics.itertuples(index=False):
        subset = predictions.loc[predictions["split"].eq(row.split)]
        predicted_column = (
            "identity_overall_phase_pred"
            if row.variant == "identity"
            else "calibrated_overall_phase_pred"
        )
        recomputed = calculate_calibration_metrics(
            subset["overall_phase"].to_numpy(dtype=int),
            subset[predicted_column].to_numpy(dtype=int),
        )
        for name, expected in recomputed.items():
            _assert_close(getattr(row, name), expected, f"{row.split}/{row.variant}/{name}")


def build_source_audit(
    forecasting: pd.DataFrame,
    forecasting_masks: dict[str, pd.Series],
    predictions: pd.DataFrame,
    selected: pd.Series,
    model_output_paths: dict[str, Path],
    input_paths: dict[str, Path],
    protected_before: str,
    protected_after: str,
    experiment_variant: str = DEFAULT_EXPERIMENT_VARIANT,
    h2_before: str | None = None,
    h2_after: str | None = None,
) -> pd.DataFrame:
    variant = get_experiment_variant(experiment_variant)
    environment = formal_environment_record()
    calibration = predictions.loc[predictions["split"].eq("calibration")]
    test = predictions.loc[predictions["split"].eq("test")]
    record: dict[str, object] = {
        "evaluation_population_id": variant.evaluation_population_id,
        "source_rows": int(len(forecasting)),
        "train_rows": int(forecasting_masks["train"].sum()),
        "calibration_rows": int(forecasting_masks["calibration"].sum()),
        "test_rows": int(forecasting_masks["test"].sum()),
        "train_end": variant.train_end.strftime("%Y-%m-%d"),
        "calibration_start": variant.calibration_start.strftime("%Y-%m-%d"),
        "calibration_end": variant.calibration_end.strftime("%Y-%m-%d"),
        "test_start": variant.test_start.strftime("%Y-%m-%d"),
        "calibration_key_sha256": canonical_key_sha256(calibration),
        "test_key_sha256": canonical_key_sha256(test),
        "delta3": float(selected["delta3"]),
        "delta4": float(selected["delta4"]),
        "identity_selected": bool(
            float(selected["delta3"]) == 0.0 and float(selected["delta4"]) == 0.0
        ),
        "random_state": 0,
        "estimator_n_jobs": "<default>",
        "estimator_uses_default_n_jobs": True,
        "model_workers": 1,
        **environment,
        "protected_manifest_sha256_before": protected_before,
        "protected_manifest_sha256_after": protected_after,
        "protected_manifest_match": protected_before == protected_after,
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": file_sha256(Path(__file__).resolve()),
    }
    for name in (
        "forecasting",
        "nowcasting",
        "country_lookup",
        "general_params",
        "phase3_params",
    ):
        path = Path(input_paths[name])
        record[f"{name}_input_path" if name in {"forecasting", "nowcasting"} else f"{name}_path"] = str(
            path.resolve()
        )
        record[
            f"{name}_input_sha256" if name in {"forecasting", "nowcasting"} else f"{name}_sha256"
        ] = file_sha256(path)
    for name in ("grid", "predictions", "metrics"):
        path = Path(model_output_paths[name])
        record[f"{name}_path"] = str(path.resolve())
        record[f"{name}_sha256"] = file_sha256(path)
    if variant.sensitivity_analysis:
        if h2_before is None or h2_after is None:
            raise ValueError("Full-year audit requires H2 artifact hashes.")
        record.update(
            {
                "experiment_variant": variant.name,
                "sensitivity_analysis": True,
                "h2_artifact_manifest_sha256_before": h2_before,
                "h2_artifact_manifest_sha256_after": h2_after,
            }
        )
    columns = source_audit_columns(variant.name)
    return pd.DataFrame([{column: record[column] for column in columns}])


def validate_written_artifacts(
    paths: dict[str, Path],
    experiment_variant: str = DEFAULT_EXPERIMENT_VARIANT,
) -> None:
    variant = get_experiment_variant(experiment_variant)
    validate_written_model_outputs(paths)
    if "source_audit" not in paths:
        raise ValueError("Source-audit path is missing.")
    audit = pd.read_csv(paths["source_audit"], keep_default_na=False)
    expected_columns = source_audit_columns(variant.name)
    if audit.columns.tolist() != expected_columns or len(audit) != 1:
        raise ValueError("Saved source audit has the wrong schema or row count.")
    row = audit.iloc[0]
    if str(row["evaluation_population_id"]) != variant.evaluation_population_id:
        raise ValueError("Saved source audit has the wrong evaluation population.")
    if str(row["protected_manifest_sha256_before"]) != str(
        row["protected_manifest_sha256_after"]
    ) or str(row["protected_manifest_match"]).lower() != "true":
        raise ValueError("Saved source audit reports protected-artifact drift.")
    current_manifest = protected_artifact_manifest_sha256(paths["source_audit"].parent)
    if str(row["protected_manifest_sha256_after"]) != current_manifest:
        raise ValueError("Saved protected manifest hash differs from current files.")
    for name in ("grid", "predictions", "metrics"):
        if str(row[f"{name}_sha256"]) != file_sha256(paths[name]):
            raise ValueError(f"Saved source audit {name} hash is incorrect.")
    if variant.sensitivity_analysis:
        if str(row["experiment_variant"]) != variant.name:
            raise ValueError("Saved source audit has the wrong experiment variant.")
        if str(row["sensitivity_analysis"]).lower() != "true":
            raise ValueError(
                "Saved source audit is not labelled as a sensitivity analysis."
            )
        if str(row["h2_artifact_manifest_sha256_before"]) != str(
            row["h2_artifact_manifest_sha256_after"]
        ):
            raise ValueError("Saved source audit reports H2 artifact drift.")
        if str(
            row["h2_artifact_manifest_sha256_after"]
        ) != h2_artifact_manifest_sha256():
            raise ValueError("Saved H2 artifact hash differs from current files.")


def run_generation(
    forecasting_path: Path = loco.DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = loco.DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = loco.DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = loco.DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = loco.DEFAULT_PHASE3_PARAMS,
    output_dir: Path | None = None,
    experiment_variant: str = DEFAULT_EXPERIMENT_VARIANT,
) -> dict[str, Path]:
    variant = get_experiment_variant(experiment_variant)
    output_dir = resolve_output_dir(variant.name, output_dir)
    validate_generation_target(variant.name, output_dir)
    assert_formal_environment()
    protected_before = protected_artifact_manifest_sha256(output_dir)
    h2_before = h2_artifact_manifest_sha256() if variant.sensitivity_analysis else None
    forecasting, nowcasting = load_prepared_inputs(
        forecasting_path, nowcasting_path, country_lookup_path
    )
    forecasting_masks = build_temporal_masks(forecasting, variant.name)
    nowcasting_masks = build_temporal_masks(nowcasting, variant.name)
    validate_production_counts(forecasting, forecasting_masks, variant.name)
    validate_production_counts(nowcasting, nowcasting_masks, variant.name)
    validate_split_key_alignment(
        forecasting, nowcasting, forecasting_masks, nowcasting_masks
    )
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path,
        phase3_params_path,
        random_state=0,
        estimator_n_jobs=None,
    )
    raw = fit_calibration_branch(
        forecasting,
        nowcasting,
        forecasting_masks,
        nowcasting_masks,
        tuple(loco.select_layer1_features(forecasting)),
        tuple(loco.NOWCAST_FEATURES),
        general_params,
        phase3_params,
    )
    grid, selected = search_offsets(raw.loc[raw["split"].eq("calibration")])
    predictions = build_prediction_table(
        raw, float(selected["delta3"]), float(selected["delta4"])
    )
    metrics = build_metrics_table(predictions)
    paths = write_model_outputs(grid, predictions, metrics, output_dir)
    validate_written_model_outputs(paths)
    protected_after = protected_artifact_manifest_sha256(output_dir)
    if protected_before != protected_after:
        raise RuntimeError("A protected baseline artifact changed during generation.")
    h2_after = h2_artifact_manifest_sha256() if variant.sensitivity_analysis else None
    if variant.sensitivity_analysis and h2_before != h2_after:
        raise RuntimeError(
            "Completed H2 calibration artifacts changed during generation."
        )
    audit = build_source_audit(
        forecasting=forecasting,
        forecasting_masks=forecasting_masks,
        predictions=predictions,
        selected=selected,
        model_output_paths=paths,
        input_paths={
            "forecasting": forecasting_path,
            "nowcasting": nowcasting_path,
            "country_lookup": country_lookup_path,
            "general_params": general_params_path,
            "phase3_params": phase3_params_path,
        },
        protected_before=protected_before,
        protected_after=protected_after,
        experiment_variant=variant.name,
        h2_before=h2_before,
        h2_after=h2_after,
    )
    paths["source_audit"] = write_source_audit(
        audit, output_dir, variant.name
    )
    validate_written_artifacts(paths, variant.name)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the isolated cascading Nowcasting calibration artifacts."
    )
    parser.add_argument("--forecasting-input", type=Path, default=loco.DEFAULT_FORECASTING_INPUT)
    parser.add_argument("--nowcasting-input", type=Path, default=loco.DEFAULT_NOWCASTING_INPUT)
    parser.add_argument("--country-lookup", type=Path, default=loco.DEFAULT_COUNTRY_LOOKUP)
    parser.add_argument("--general-params", type=Path, default=loco.DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=loco.DEFAULT_PHASE3_PARAMS)
    parser.add_argument(
        "--experiment-variant",
        choices=tuple(EXPERIMENT_VARIANTS),
        default=DEFAULT_EXPERIMENT_VARIANT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    args.output_dir = resolve_output_dir(args.experiment_variant, args.output_dir)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = run_generation(
        forecasting_path=args.forecasting_input,
        nowcasting_path=args.nowcasting_input,
        country_lookup_path=args.country_lookup,
        general_params_path=args.general_params,
        phase3_params_path=args.phase3_params,
        output_dir=args.output_dir,
        experiment_variant=args.experiment_variant,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
