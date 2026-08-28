"""Generate fixed-hyperparameter leave-one-country-out robustness results."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Callable, Iterable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache-loco")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
CODESPACE_ROOT = REPO_ROOT.parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
DEFAULT_COUNTRY_LOOKUP = SOURCE_DATA_DIR / "area_country_lookup.csv"
DEFAULT_UPSTREAM_COUNTRY_SOURCE = (
    CODESPACE_ROOT / "0.Archived" / "new_merge_0108_with_country_code.csv"
)
DEFAULT_FORECASTING_INPUT = SOURCE_DATA_DIR / "Forecasting_Analysis_010825.csv"
DEFAULT_NOWCASTING_INPUT = SOURCE_DATA_DIR / "Nowcasting_Analysis_010825.csv"
DEFAULT_GENERAL_PARAMS = (
    REPO_ROOT / "2.Source Code" / "forecasting_hyperparameters.json"
)
DEFAULT_PHASE3_PARAMS = (
    REPO_ROOT / "2.Source Code" / "forecasting_hyperparameters_p3.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
DEFAULT_RANDOM_STATE = 0
DEFAULT_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
KEY_COLUMNS = ["area_id", "date"]
PHASE_SHARE_COLUMNS = [f"phase{i}_percent" for i in range(1, 6)]
OUTCOME_COLUMNS = ["overall_phase", *PHASE_SHARE_COLUMNS]
CUMULATIVE_TARGETS = {
    2: "phase2_worse",
    3: "phase3_worse",
    4: "phase4_worse",
    5: "phase5_worse",
}
LAYER1_EXCLUDED_COLUMNS = {
    "country_code_3",
    "area_id",
    "date",
    "overall_phase",
    *PHASE_SHARE_COLUMNS,
    *CUMULATIVE_TARGETS.values(),
}
NOWCAST_FEATURES = (
    "CPI",
    "fatalities_explosions_w5_m12",
    "GDP",
    "soil_moisture_mean_m12",
    "fatalities_explosions",
    "chirps_z_score",
    "EVI_m12",
    "event_count_battles_w10_m12",
    "event_count_battles_w5_m12",
    "price_index_s12",
    "event_count_explosions",
    "rainfall_chirps_s12",
    "event_count_violence_w10_m12",
    "WFP_Price_Change_s12",
    "WFP_Price_Volatility_m12",
    "WFP_Price_Change_m12",
    "infra_index_m12",
    "temperature_2m_mean_s12",
    "event_count_battles_m12",
    "fatalities_battles",
    "event_count_violence_m12",
    "event_count_explosions_m12",
    "fatalities_battles_w5_m12",
    "event_count_violence_w5_m12",
    "soil_moisture_mean_s12",
    "event_count_explosions_w5_m12",
    "WFP_Price_Volatility_s12",
    "event_count_violence",
    "precipitation_sum_m12",
    "event_count_violence_w10",
    "fatalities_violence_w10_m12",
    "event_count_battles",
    "event_count_explosions_w5",
    "fatalities_explosions_w10_m12",
    "event_count_battles_w10",
    "temperature_2m_mean_m12",
    "rainfall_chirps_m12",
    "fatalities_violence_m12",
    "WFP_Price_Change",
    "event_count_explosions_w10",
    "gini",
    "cpi",
    "fatalities_violence_w5_m12",
    "infra_index_s12",
    "shortwave_radiation_sum_m12",
    "price_index_m12",
    "fatalities_battles_w10_m12",
    "event_count_violence_w5",
    "fatalities_explosions_w10",
    "CC.PER.RNK",
    "fatalities_violence_w5",
    "shortwave_radiation_sum_s12",
    "fatalities_battles_w10",
    "fatalities_battles_m12",
    "nightlight_mean_m12",
    "fatalities_battles_w5",
    "event_count_battles_w5",
    "fatalities_violence_w10",
    "fatalities_violence",
    "infra_index",
    "fatalities_explosions_m12",
    "temperature_z_score",
    "event_count_explosions_w10_m12",
    "NY.GDP.PCAP.KD",
    "fatalities_explosions_w5",
    "precipitation_sum_s12",
    "WFP_Price_Volatility",
    "nightlight_mean_s12",
    "GOSIF_GPP_m12",
)
assert len(NOWCAST_FEATURES) == 69
assert len(set(NOWCAST_FEATURES)) == 69

_WORKER_FORECASTING: pd.DataFrame | None = None
_WORKER_NOWCASTING: pd.DataFrame | None = None
_WORKER_GENERAL_PARAMS: dict[str, object] | None = None
_WORKER_PHASE3_PARAMS: dict[str, object] | None = None


def _require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"Input data are missing required columns: {missing}")


def normalize_country_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, unique, deterministically sorted area-country mapping."""
    _require_columns(data, ["area_id", "country_code_3"])
    lookup = data[["area_id", "country_code_3"]].copy()
    if lookup.isna().any().any():
        raise ValueError(
            "Country lookup contains missing area_id or country_code_3 values."
        )
    lookup["area_id"] = lookup["area_id"].astype(int)
    country_counts = lookup.groupby("area_id")["country_code_3"].nunique()
    ambiguous = country_counts[country_counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            f"Some area_id values map to multiple countries: {ambiguous.index.tolist()}"
        )
    return (
        lookup.drop_duplicates(["area_id", "country_code_3"])
        .sort_values(["area_id", "country_code_3"], kind="mergesort")
        .reset_index(drop=True)
    )


def export_country_lookup(source_path: Path, output_path: Path) -> pd.DataFrame:
    """Create the package-local area-country lookup from its upstream source."""
    lookup = normalize_country_lookup(
        pd.read_csv(source_path, usecols=["area_id", "country_code_3"])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(output_path, index=False)
    return lookup


def load_country_lookup(path: Path = DEFAULT_COUNTRY_LOOKUP) -> pd.DataFrame:
    """Load and validate the package-local area-country lookup."""
    return normalize_country_lookup(pd.read_csv(path))


def _validate_unique_keys(data: pd.DataFrame, name: str) -> None:
    _require_columns(data, KEY_COLUMNS)
    if data[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{name} contains missing area_id or date values.")
    duplicates = data.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        raise ValueError(
            f"{name} contains duplicate (area_id, date) keys: "
            f"{data.loc[duplicates, KEY_COLUMNS].head().to_dict('records')}"
        )


def _attach_country_codes(
    data: pd.DataFrame, lookup: pd.DataFrame, name: str
) -> pd.DataFrame:
    merged = data.merge(
        lookup,
        on="area_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        missing_areas = sorted(merged.loc[unmatched, "area_id"].unique().tolist())
        raise ValueError(
            f"{name} contains area_id values missing from country lookup: "
            f"{missing_areas[:20]}"
        )
    return merged.drop(columns="_merge")


def prepare_model_inputs(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate, key-align, country-map, and normalize the two model tables."""
    _require_columns(forecasting, [*KEY_COLUMNS, *OUTCOME_COLUMNS])
    _require_columns(nowcasting, [*KEY_COLUMNS, *OUTCOME_COLUMNS])
    _validate_unique_keys(forecasting, "Forecasting input")
    _validate_unique_keys(nowcasting, "Nowcasting input")
    lookup = normalize_country_lookup(lookup)

    forecasting_indexed = forecasting.set_index(KEY_COLUMNS).sort_index()
    nowcasting_indexed = nowcasting.set_index(KEY_COLUMNS).sort_index()
    if not forecasting_indexed.index.equals(nowcasting_indexed.index):
        raise ValueError("Forecasting and nowcasting inputs have different key sets.")
    for column in OUTCOME_COLUMNS:
        left = forecasting_indexed[column]
        right = nowcasting_indexed[column]
        equal = left.eq(right) | (left.isna() & right.isna())
        if not bool(equal.all()):
            bad_keys = equal.index[~equal][:5].tolist()
            raise ValueError(
                f"Forecasting and nowcasting outcomes differ for {column}: {bad_keys}"
            )

    forecasting_prepared = _attach_country_codes(
        forecasting.copy(), lookup, "Forecasting input"
    )
    nowcasting_prepared = _attach_country_codes(
        nowcasting.copy(), lookup, "Nowcasting input"
    )

    rename_map = {
        "infra_index_m12": "infra_index_m12_l12",
        "infra_index_s12": "infra_index_s12_l12",
    }
    for source, destination in rename_map.items():
        if source in forecasting_prepared and destination in forecasting_prepared:
            raise ValueError(
                f"Forecasting input contains both {source} and {destination}."
            )
    forecasting_prepared = forecasting_prepared.rename(columns=rename_map)

    forecasting_prepared = forecasting_prepared.sort_values(
        KEY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    nowcasting_prepared = nowcasting_prepared.sort_values(
        KEY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    return forecasting_prepared, nowcasting_prepared


def add_cumulative_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Add the four cumulative phase-share regression targets."""
    _require_columns(data, PHASE_SHARE_COLUMNS)
    result = data.copy()
    result["phase2_worse"] = result[
        [f"phase{i}_percent" for i in range(2, 6)]
    ].sum(axis=1)
    result["phase3_worse"] = result[
        [f"phase{i}_percent" for i in range(3, 6)]
    ].sum(axis=1)
    result["phase4_worse"] = result[["phase4_percent", "phase5_percent"]].sum(
        axis=1
    )
    result["phase5_worse"] = result["phase5_percent"]
    return result


def country_masks(
    country_codes: pd.Series, held_out_country: str
) -> tuple[pd.Series, pd.Series]:
    """Return mutually exclusive train and held-country test masks."""
    test_mask = country_codes.eq(held_out_country)
    train_mask = ~test_mask
    if not bool(test_mask.any()):
        raise ValueError(f"Held-out country has no rows: {held_out_country}")
    if not bool(train_mask.any()):
        raise ValueError(
            f"Training set is empty for held-out country: {held_out_country}"
        )
    return train_mask, test_mask


def _phase_from_cumulative(data: pd.DataFrame, suffix: str) -> np.ndarray:
    conditions = [
        data[f"phase5_{suffix}"].ge(0.20),
        data[f"phase4_{suffix}"].ge(0.20),
        data[f"phase3_{suffix}"].ge(0.20),
        data[f"phase2_{suffix}"].ge(0.20),
    ]
    return np.select(conditions, [5, 4, 3, 2], default=1).astype(int)


def wide_predictions_to_phases(data: pd.DataFrame) -> pd.DataFrame:
    """Convert keyed cumulative predictions to actual and predicted phases."""
    required = [
        f"phase{phase}_{suffix}"
        for phase in range(2, 6)
        for suffix in ("test", "pred")
    ]
    _require_columns(data, required)
    result = data.copy().reset_index(drop=True)
    prediction_columns = [f"phase{phase}_pred" for phase in range(2, 6)]
    result[prediction_columns] = result[prediction_columns].round(2)
    result["nonpositive_cumulative_prediction_sum"] = (
        result[prediction_columns].sum(axis=1) <= 0
    )
    result["overall_phase"] = _phase_from_cumulative(result, "test")
    result["overall_phase_pred"] = _phase_from_cumulative(result, "pred")
    return result


def calculate_country_metrics(
    predictions: pd.DataFrame,
    model_name: str,
    country_code_3: str,
) -> dict[str, object]:
    """Calculate per-country five-phase and Phase 3+ metrics."""
    _require_columns(
        predictions,
        ["overall_phase", "overall_phase_pred", "phase3_test", "phase3_pred"],
    )
    actual_positive = predictions["overall_phase"].ge(3)
    predicted_positive = predictions["overall_phase_pred"].ge(3)
    true_positive = int((actual_positive & predicted_positive).sum())
    false_positive = int((~actual_positive & predicted_positive).sum())
    false_negative = int((actual_positive & ~predicted_positive).sum())
    true_negative = int((~actual_positive & ~predicted_positive).sum())
    predicted_positive_count = true_positive + false_positive
    actual_positive_count = true_positive + false_negative

    if predicted_positive_count:
        precision = true_positive / predicted_positive_count
        precision_reason = None
    else:
        precision = np.nan
        precision_reason = "no_predicted_phase3plus"
    if actual_positive_count:
        recall = true_positive / actual_positive_count
        recall_reason = None
    else:
        recall = np.nan
        recall_reason = "no_actual_phase3plus"

    actual_share = predictions["phase3_test"]
    predicted_share = predictions["phase3_pred"]
    if len(predictions) < 2:
        phase3plus_r2 = np.nan
        r2_reason = "insufficient_observations"
    elif actual_share.nunique(dropna=False) < 2:
        phase3plus_r2 = np.nan
        r2_reason = "constant_actual_phase3plus_share"
    else:
        phase3plus_r2 = float(r2_score(actual_share, predicted_share))
        r2_reason = None

    return {
        "model": model_name,
        "country_code_3": country_code_3,
        "n_test": int(len(predictions)),
        "actual_phase3plus_count": actual_positive_count,
        "predicted_phase3plus_count": predicted_positive_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "phase3plus_precision": precision,
        "phase3plus_recall": recall,
        "overall_accuracy": float(
            accuracy_score(
                predictions["overall_phase"], predictions["overall_phase_pred"]
            )
        ),
        "phase3plus_r2": phase3plus_r2,
        "precision_undefined_reason": precision_reason,
        "recall_undefined_reason": recall_reason,
        "r2_undefined_reason": r2_reason,
        "nonpositive_cumulative_prediction_count": int(
            predictions.get(
                "nonpositive_cumulative_prediction_sum",
                pd.Series(False, index=predictions.index),
            ).sum()
        ),
    }


def calculate_area_macro_metrics(
    forecasting_predictions: pd.DataFrame,
    nowcasting_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Average per-area LOCO metrics with equal area weights.

    Each metric is averaged over the areas where that metric is defined. Missing
    per-area precision, recall, or R-squared values remain missing and are never
    replaced with zero.
    """
    records: list[dict[str, object]] = []
    for model_name, predictions in (
        ("Nowcasting", nowcasting_predictions),
        ("Forecasting", forecasting_predictions),
    ):
        _require_columns(predictions, ["area_id"])
        area_records = [
            calculate_country_metrics(area_predictions, model_name, str(area_id))
            for area_id, area_predictions in predictions.groupby(
                "area_id", sort=True, observed=True
            )
        ]
        if not area_records:
            raise ValueError(f"{model_name} predictions contain no areas.")
        area_metrics = pd.DataFrame(area_records)
        records.append(
            {
                "model": model_name,
                "accuracy": area_metrics["overall_accuracy"].mean(),
                "precision": area_metrics["phase3plus_precision"].mean(),
                "recall": area_metrics["phase3plus_recall"].mean(),
                "R2(p3)": area_metrics["phase3plus_r2"].mean(),
            }
        )
    return pd.DataFrame(records).set_index("model")


def calculate_micro_metrics(
    forecasting_predictions: pd.DataFrame,
    nowcasting_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate primary LOCO metrics after pooling every test observation."""
    for model_name, predictions in (
        ("Forecasting", forecasting_predictions),
        ("Nowcasting", nowcasting_predictions),
    ):
        _require_columns(
            predictions,
            [*KEY_COLUMNS, "overall_phase", "overall_phase_pred", "phase3_test", "phase3_pred"],
        )
        _validate_unique_keys(predictions, f"{model_name} LOCO predictions")
        if predictions.empty:
            raise ValueError(f"{model_name} LOCO predictions are empty.")
    forecasting_keys = pd.MultiIndex.from_frame(
        forecasting_predictions[KEY_COLUMNS]
    )
    nowcasting_keys = pd.MultiIndex.from_frame(nowcasting_predictions[KEY_COLUMNS])
    if set(forecasting_keys) != set(nowcasting_keys):
        raise ValueError("Forecasting and nowcasting LOCO prediction keys differ.")

    records: list[dict[str, object]] = []
    for model_name, predictions in (
        ("Nowcasting", nowcasting_predictions),
        ("Forecasting", forecasting_predictions),
    ):
        metric_predictions = predictions.copy()
        metric_predictions["phase3_test"] = metric_predictions["phase3_test"].round(2)
        metric_predictions["phase3_pred"] = metric_predictions["phase3_pred"].round(2)
        metric = calculate_country_metrics(
            metric_predictions,
            model_name,
            "ALL_LOCO_TEST_ROWS",
        )
        records.append(
            {
                "model": model_name,
                "accuracy": metric["overall_accuracy"],
                "precision": metric["phase3plus_precision"],
                "recall": metric["phase3plus_recall"],
                "R2(p3)": metric["phase3plus_r2"],
            }
        )
    return pd.DataFrame(records).set_index("model")[
        ["accuracy", "precision", "recall", "R2(p3)"]
    ]


def aggregate_existing_loco_predictions(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Replace the primary two-row table from saved LOCO predictions."""
    forecasting_path = (
        output_dir / "leave_one_country_out_forecasting_predictions.csv"
    )
    nowcasting_path = output_dir / "leave_one_country_out_nowcasting_predictions.csv"
    for path in (forecasting_path, nowcasting_path):
        if not path.is_file():
            raise FileNotFoundError(f"LOCO prediction artifact not found: {path}")
    metrics = calculate_micro_metrics(
        pd.read_csv(forecasting_path, float_precision="round_trip"),
        pd.read_csv(nowcasting_path, float_precision="round_trip"),
    )
    output_path = output_dir / "leave_one_country_out_micro_metrics.csv"
    metrics.to_csv(output_path, index=True, index_label="model", float_format="%.6f")
    return output_path


def load_hyperparameters(
    general_path: Path,
    phase3_path: Path,
    random_state: int | None = 0,
    estimator_n_jobs: int | None = 1,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load the two notebook parameter files with explicit execution controls."""
    with general_path.open("r", encoding="utf-8") as file:
        general_params = json.load(file)
    with phase3_path.open("r", encoding="utf-8") as file:
        phase3_params = json.load(file)
    for params in (general_params, phase3_params):
        if random_state is None:
            params.pop("random_state", None)
        else:
            params["random_state"] = int(random_state)
        if estimator_n_jobs is None:
            params.pop("n_jobs", None)
        else:
            params["n_jobs"] = estimator_n_jobs
    return general_params, phase3_params


def select_layer1_features(data: pd.DataFrame) -> list[str]:
    """Select the notebook's Layer 1 predictors while excluding all outcomes."""
    features = [
        column for column in data.columns if column not in LAYER1_EXCLUDED_COLUMNS
    ]
    if "fews_ipc_ha" not in features:
        raise ValueError("Layer 1 features must retain fews_ipc_ha.")
    non_numeric = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        raise ValueError(f"Layer 1 contains non-numeric features: {non_numeric}")
    return features


def _validate_split_masks(
    data: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    split_id: object,
    source_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Validate complete, disjoint, index-aligned Boolean split masks."""
    for name, mask in (("train", train_mask), ("test", test_mask)):
        if not isinstance(mask, pd.Series):
            raise TypeError(f"{source_name} {name} mask for {split_id} must be a Series.")
        if not mask.index.equals(data.index):
            raise ValueError(
                f"{source_name} {name} mask index does not match data for {split_id}."
            )
        if not pd.api.types.is_bool_dtype(mask):
            raise TypeError(
                f"{source_name} {name} mask for {split_id} must be Boolean."
            )
    if bool((train_mask & test_mask).any()):
        raise ValueError(f"{source_name} train/test masks overlap for {split_id}.")
    if not bool((train_mask | test_mask).all()):
        raise ValueError(f"{source_name} split leaves rows unassigned for {split_id}.")
    if not bool(train_mask.any()) or not bool(test_mask.any()):
        raise ValueError(f"{source_name} split requires non-empty train and test rows.")
    return train_mask, test_mask


def fit_forecasting_split(
    data: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    split_id: object,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    fold_column: str = "fold_id",
    estimator_factory: Callable[..., object] = xgb.XGBRegressor,
) -> pd.DataFrame:
    """Fit four cumulative forecasting regressors for one explicit split."""
    _require_columns(
        data,
        [*KEY_COLUMNS, "country_code_3", "overall_phase", *CUMULATIVE_TARGETS.values()],
    )
    train_mask, test_mask = _validate_split_masks(
        data, train_mask, test_mask, split_id, "Forecasting"
    )
    if not fold_column or fold_column in [*KEY_COLUMNS, "country_code_3"]:
        raise ValueError("fold_column must be a non-key metadata column.")
    feature_columns = select_layer1_features(data)
    keys = [*KEY_COLUMNS, "country_code_3"]
    test_rows = data.loc[test_mask]
    base = test_rows[keys].copy()
    base["source_row_index"] = test_rows.index.to_numpy()
    base[fold_column] = split_id
    base["source_overall_phase"] = test_rows["overall_phase"].to_numpy()

    wide = base
    for phase, target_column in CUMULATIVE_TARGETS.items():
        y_train = data.loc[train_mask, target_column]
        y_test = data.loc[test_mask, target_column]
        if y_train.isna().any() or y_test.isna().any():
            raise ValueError(
                f"Missing {target_column} values for forecasting split {split_id}."
            )
        params = general_params if phase == 2 else phase3_params
        model = estimator_factory(**dict(params))
        model.fit(data.loc[train_mask, feature_columns], y_train)
        predictions = model.predict(data.loc[test_mask, feature_columns])
        phase_frame = test_rows[keys].copy()
        phase_frame[f"phase{phase}_test"] = y_test.to_numpy()
        phase_frame[f"phase{phase}_pred"] = np.asarray(predictions)
        wide = wide.merge(phase_frame, on=keys, how="inner", validate="one_to_one")

    if len(wide) != int(test_mask.sum()):
        raise ValueError(
            f"Forecasting split {split_id} lost rows during keyed assembly."
        )
    return wide_predictions_to_phases(wide)


def fit_forecasting_fold(
    data: pd.DataFrame,
    held_out_country: str,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    estimator_factory: Callable[..., object] = xgb.XGBRegressor,
) -> pd.DataFrame:
    """Fit the four cumulative forecasting regressors for one held-out country."""
    train_mask, test_mask = country_masks(data["country_code_3"], held_out_country)
    result = fit_forecasting_split(
        data,
        train_mask,
        test_mask,
        held_out_country,
        general_params,
        phase3_params,
        fold_column="fold_country",
        estimator_factory=estimator_factory,
    )
    if not result["country_code_3"].eq(held_out_country).all():
        raise ValueError(
            f"Forecasting fold {held_out_country} contains another country."
        )
    return result


def fit_nowcasting_split(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    now_train_mask: pd.Series,
    now_test_mask: pd.Series,
    split_id: object,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    fold_column: str = "fold_id",
    estimator_factory: Callable[..., object] = xgb.XGBRegressor,
) -> pd.DataFrame:
    """Fit the notebook-faithful cascading two-layer model for one split."""
    _require_columns(
        forecasting,
        [*KEY_COLUMNS, "country_code_3", "overall_phase", *CUMULATIVE_TARGETS.values()],
    )
    _require_columns(nowcasting, [*KEY_COLUMNS, "country_code_3", *NOWCAST_FEATURES])
    _validate_unique_keys(forecasting, "Prepared forecasting input")
    _validate_unique_keys(nowcasting, "Prepared nowcasting input")
    forecasting_keys = pd.MultiIndex.from_frame(forecasting[KEY_COLUMNS])
    nowcasting_keys = pd.MultiIndex.from_frame(nowcasting[KEY_COLUMNS])
    if set(forecasting_keys) != set(nowcasting_keys):
        raise ValueError("Prepared forecasting and nowcasting key sets differ.")

    train_mask, test_mask = _validate_split_masks(
        forecasting, train_mask, test_mask, split_id, "Forecasting"
    )
    now_train_mask, now_test_mask = _validate_split_masks(
        nowcasting, now_train_mask, now_test_mask, split_id, "Nowcasting"
    )
    for label, forecast_mask, nowcast_mask in (
        ("train", train_mask, now_train_mask),
        ("test", test_mask, now_test_mask),
    ):
        forecast_partition_keys = pd.MultiIndex.from_frame(
            forecasting.loc[forecast_mask, KEY_COLUMNS]
        )
        nowcast_partition_keys = pd.MultiIndex.from_frame(
            nowcasting.loc[nowcast_mask, KEY_COLUMNS]
        )
        if set(forecast_partition_keys) != set(nowcast_partition_keys):
            raise ValueError(
                f"Forecasting and nowcasting {label} keys differ for split {split_id}."
            )
    if not fold_column or fold_column in [*KEY_COLUMNS, "country_code_3"]:
        raise ValueError("fold_column must be a non-key metadata column.")
    feature_columns = select_layer1_features(forecasting)
    non_numeric_layer2 = [
        column
        for column in NOWCAST_FEATURES
        if not pd.api.types.is_numeric_dtype(nowcasting[column])
    ]
    if non_numeric_layer2:
        raise ValueError(
            f"Layer 2 contains non-numeric features: {non_numeric_layer2}"
        )

    keys = [*KEY_COLUMNS, "country_code_3"]
    test_rows = forecasting.loc[test_mask]
    base = test_rows[keys].copy()
    base["source_row_index"] = test_rows.index.to_numpy()
    base[fold_column] = split_id
    base["source_overall_phase"] = test_rows["overall_phase"].to_numpy()
    wide = base

    for phase, target_column in CUMULATIVE_TARGETS.items():
        y_train = forecasting.loc[train_mask, target_column]
        y_test = forecasting.loc[test_mask, target_column]
        if y_train.isna().any() or y_test.isna().any():
            raise ValueError(
                f"Missing {target_column} values for nowcasting split {split_id}."
            )
        params = general_params if phase == 2 else phase3_params

        layer1 = estimator_factory(**dict(params))
        layer1.fit(forecasting.loc[train_mask, feature_columns], y_train)
        layer1_train_predictions = np.asarray(
            layer1.predict(forecasting.loc[train_mask, feature_columns])
        )
        layer1_test_predictions = np.asarray(
            layer1.predict(forecasting.loc[test_mask, feature_columns])
        )

        residual_frame = forecasting.loc[train_mask, keys].copy()
        residual_frame["layer1_residual"] = (
            y_train.to_numpy() - layer1_train_predictions
        )
        now_train = nowcasting.loc[now_train_mask, [*keys, *NOWCAST_FEATURES]]
        keyed_train = now_train.merge(
            residual_frame,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(keyed_train) != int(train_mask.sum()):
            raise ValueError(
                f"Nowcasting split {split_id} lost training residual rows."
            )

        layer2 = estimator_factory(**dict(params))
        layer2.fit(keyed_train[list(NOWCAST_FEATURES)], keyed_train["layer1_residual"])
        now_test = nowcasting.loc[now_test_mask, [*keys, *NOWCAST_FEATURES]]
        residual_predictions = np.asarray(
            layer2.predict(now_test[list(NOWCAST_FEATURES)])
        )
        residual_prediction_frame = now_test[keys].copy()
        residual_prediction_frame[f"phase{phase}_residual_pred"] = (
            residual_predictions
        )

        phase_frame = test_rows[keys].copy()
        phase_frame[f"phase{phase}_test"] = y_test.to_numpy()
        phase_frame[f"phase{phase}_layer1_pred"] = layer1_test_predictions
        phase_frame = phase_frame.merge(
            residual_prediction_frame,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        phase_frame[f"phase{phase}_pred"] = (
            phase_frame[f"phase{phase}_layer1_pred"]
            + phase_frame[f"phase{phase}_residual_pred"]
        )
        wide = wide.merge(phase_frame, on=keys, how="inner", validate="one_to_one")

    if len(wide) != int(test_mask.sum()):
        raise ValueError(
            f"Nowcasting split {split_id} lost rows during keyed assembly."
        )
    return wide_predictions_to_phases(wide)


def fit_nowcasting_fold(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    held_out_country: str,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    estimator_factory: Callable[..., object] = xgb.XGBRegressor,
) -> pd.DataFrame:
    """Fit the notebook-faithful cascading two-layer model for one country."""
    train_mask, test_mask = country_masks(
        forecasting["country_code_3"], held_out_country
    )
    now_train_mask, now_test_mask = country_masks(
        nowcasting["country_code_3"], held_out_country
    )
    result = fit_nowcasting_split(
        forecasting,
        nowcasting,
        train_mask,
        test_mask,
        now_train_mask,
        now_test_mask,
        held_out_country,
        general_params,
        phase3_params,
        fold_column="fold_country",
        estimator_factory=estimator_factory,
    )
    if not result["country_code_3"].eq(held_out_country).all():
        raise ValueError(f"Nowcasting fold {held_out_country} contains another country.")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    script_path: Path,
    random_state: int,
) -> dict[str, object]:
    """Build the immutable portion of a restart-checkpoint manifest."""
    return {
        "forecasting_input_sha256": _sha256_file(forecasting_path),
        "nowcasting_input_sha256": _sha256_file(nowcasting_path),
        "country_lookup_sha256": _sha256_file(country_lookup_path),
        "general_params_sha256": _sha256_file(general_params_path),
        "phase3_params_sha256": _sha256_file(phase3_params_path),
        "script_sha256": _sha256_file(script_path),
        "random_state": int(random_state),
        "evaluation_protocol": "fixed_hyperparameter_loco",
        "temporal_interpretation": (
            "all_dates_cross_country_transfer_not_out_of_time"
        ),
    }


def _checkpoint_paths(checkpoint_dir: Path, country: str) -> dict[str, Path]:
    country_dir = checkpoint_dir / country
    return {
        "country_dir": country_dir,
        "forecasting": country_dir / "forecasting_predictions.csv",
        "nowcasting": country_dir / "nowcasting_predictions.csv",
        "manifest": country_dir / "manifest.json",
    }


def save_country_checkpoint(
    checkpoint_dir: Path,
    country: str,
    manifest: dict[str, object],
    forecasting_predictions: pd.DataFrame,
    nowcasting_predictions: pd.DataFrame,
) -> None:
    """Atomically persist one country's completed model predictions."""
    paths = _checkpoint_paths(checkpoint_dir, country)
    paths["country_dir"].mkdir(parents=True, exist_ok=True)
    forecast_tmp = paths["forecasting"].with_suffix(".csv.tmp")
    nowcast_tmp = paths["nowcasting"].with_suffix(".csv.tmp")
    manifest_tmp = paths["manifest"].with_suffix(".json.tmp")
    forecasting_predictions.to_csv(forecast_tmp, index=False, float_format="%.10g")
    nowcasting_predictions.to_csv(nowcast_tmp, index=False, float_format="%.10g")
    with manifest_tmp.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(forecast_tmp, paths["forecasting"])
    os.replace(nowcast_tmp, paths["nowcasting"])
    os.replace(manifest_tmp, paths["manifest"])


def load_country_checkpoint(
    checkpoint_dir: Path,
    country: str,
    expected_manifest: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load a country checkpoint only when its manifest matches exactly."""
    paths = _checkpoint_paths(checkpoint_dir, country)
    if not all(paths[name].exists() for name in ("forecasting", "nowcasting", "manifest")):
        return None
    try:
        with paths["manifest"].open("r", encoding="utf-8") as file:
            observed_manifest = json.load(file)
        if observed_manifest != expected_manifest:
            return None
        return pd.read_csv(paths["forecasting"]), pd.read_csv(paths["nowcasting"])
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return None


def _initialize_worker(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
) -> None:
    global _WORKER_FORECASTING
    global _WORKER_NOWCASTING
    global _WORKER_GENERAL_PARAMS
    global _WORKER_PHASE3_PARAMS
    _WORKER_FORECASTING = forecasting
    _WORKER_NOWCASTING = nowcasting
    _WORKER_GENERAL_PARAMS = general_params
    _WORKER_PHASE3_PARAMS = phase3_params


def _run_country_in_worker(
    country: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    if any(
        item is None
        for item in (
            _WORKER_FORECASTING,
            _WORKER_NOWCASTING,
            _WORKER_GENERAL_PARAMS,
            _WORKER_PHASE3_PARAMS,
        )
    ):
        raise RuntimeError("LOCO worker was not initialized.")
    forecast = fit_forecasting_fold(
        _WORKER_FORECASTING,
        country,
        _WORKER_GENERAL_PARAMS,
        _WORKER_PHASE3_PARAMS,
    )
    nowcast = fit_nowcasting_fold(
        _WORKER_FORECASTING,
        _WORKER_NOWCASTING,
        country,
        _WORKER_GENERAL_PARAMS,
        _WORKER_PHASE3_PARAMS,
    )
    return country, forecast, nowcast


def _validate_prediction_coverage(
    source: pd.DataFrame,
    predictions: pd.DataFrame,
    countries: Sequence[str],
    model_name: str,
) -> None:
    expected = source.loc[source["country_code_3"].isin(countries), KEY_COLUMNS]
    if len(predictions) != len(expected):
        raise ValueError(
            f"{model_name} predictions have {len(predictions)} rows; "
            f"expected {len(expected)}."
        )
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{model_name} predictions contain duplicate keys.")
    observed_keys = pd.MultiIndex.from_frame(predictions[KEY_COLUMNS])
    expected_keys = pd.MultiIndex.from_frame(expected)
    if set(observed_keys) != set(expected_keys):
        raise ValueError(f"{model_name} prediction keys do not match the requested rows.")


def _audit_records_for_country(
    forecasting: pd.DataFrame,
    country: str,
    forecast_predictions: pd.DataFrame,
    nowcast_predictions: pd.DataFrame,
    random_state: int,
    checkpoint_reused: bool,
    manifest_base: dict[str, object] | None,
) -> list[dict[str, object]]:
    test_mask = forecasting["country_code_3"].eq(country)
    n_test = int(test_mask.sum())
    n_train = int((~test_mask).sum())
    common = {
        "evaluation_protocol": "fixed_hyperparameter_loco",
        "temporal_interpretation": (
            "all_dates_cross_country_transfer_not_out_of_time"
        ),
        "held_out_country": country,
        "n_train": n_train,
        "n_test": n_test,
        "train_country_count": int(
            forecasting.loc[~test_mask, "country_code_3"].nunique()
        ),
        "test_country_count": int(
            forecasting.loc[test_mask, "country_code_3"].nunique()
        ),
        "layer1_feature_count": len(select_layer1_features(forecasting)),
        "layer2_feature_count": len(NOWCAST_FEATURES),
        "fews_ipc_ha_in_layer1": True,
        "random_state": int(random_state),
        "checkpoint_reused": bool(checkpoint_reused),
        "train_excludes_held_country": True,
        "test_only_held_country": True,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
    }
    if manifest_base:
        common.update(manifest_base)
    return [
        {
            **common,
            "model": "Forecasting",
            "nonpositive_cumulative_prediction_count": int(
                forecast_predictions["nonpositive_cumulative_prediction_sum"].sum()
            ),
        },
        {
            **common,
            "model": "Nowcasting",
            "nonpositive_cumulative_prediction_count": int(
                nowcast_predictions["nonpositive_cumulative_prediction_sum"].sum()
            ),
        },
    ]


def run_loco_predictions(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    countries: Sequence[str] | None = None,
    workers: int = DEFAULT_WORKERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    manifest_base: dict[str, object] | None = None,
    forecasting_runner: Callable[..., pd.DataFrame] = fit_forecasting_fold,
    nowcasting_runner: Callable[..., pd.DataFrame] = fit_nowcasting_fold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run requested LOCO folds and assemble predictions, metrics, and audit."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    forecasting = add_cumulative_targets(forecasting)
    available_countries = sorted(forecasting["country_code_3"].unique().tolist())
    requested_countries = (
        available_countries if countries is None else sorted(set(countries))
    )
    if not requested_countries:
        raise ValueError("At least one country must be requested.")
    unknown = sorted(set(requested_countries).difference(available_countries))
    if unknown:
        raise ValueError(f"Unknown requested countries: {unknown}")
    if set(nowcasting["country_code_3"].unique()) != set(available_countries):
        raise ValueError("Forecasting and nowcasting country sets differ.")

    general_params = dict(general_params)
    phase3_params = dict(phase3_params)
    for params in (general_params, phase3_params):
        params["random_state"] = int(random_state)
        params["n_jobs"] = 1

    country_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    checkpoint_reused: dict[str, bool] = {}
    pending: list[str] = []
    for country in requested_countries:
        manifest = {**(manifest_base or {}), "country": country}
        loaded = (
            load_country_checkpoint(checkpoint_dir, country, manifest)
            if resume and checkpoint_dir is not None
            else None
        )
        if loaded is None:
            pending.append(country)
            checkpoint_reused[country] = False
        else:
            country_results[country] = loaded
            checkpoint_reused[country] = True
            print(f"[resume] {country}: checkpoint reused", flush=True)

    custom_runners = (
        forecasting_runner is not fit_forecasting_fold
        or nowcasting_runner is not fit_nowcasting_fold
    )
    effective_workers = min(workers, max(1, len(pending)))
    if pending and (effective_workers == 1 or custom_runners):
        for position, country in enumerate(pending, start=1):
            print(
                f"[{position}/{len(pending)}] {country}: forecasting",
                flush=True,
            )
            forecast = forecasting_runner(
                forecasting, country, general_params, phase3_params
            )
            print(f"[{position}/{len(pending)}] {country}: nowcasting", flush=True)
            nowcast = nowcasting_runner(
                forecasting, nowcasting, country, general_params, phase3_params
            )
            country_results[country] = (forecast, nowcast)
            if checkpoint_dir is not None:
                save_country_checkpoint(
                    checkpoint_dir,
                    country,
                    {**(manifest_base or {}), "country": country},
                    forecast,
                    nowcast,
                )
    elif pending:
        futures = {}
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_initialize_worker,
            initargs=(forecasting, nowcasting, general_params, phase3_params),
        ) as executor:
            for country in pending:
                futures[executor.submit(_run_country_in_worker, country)] = country
            completed = 0
            try:
                for future in as_completed(futures):
                    country = futures[future]
                    completed += 1
                    try:
                        returned_country, forecast, nowcast = future.result()
                    except Exception as error:
                        for other in futures:
                            other.cancel()
                        raise RuntimeError(f"LOCO country {country} failed") from error
                    if returned_country != country:
                        raise RuntimeError(
                            f"Worker country mismatch: expected {country}, got {returned_country}"
                        )
                    print(
                        f"[{completed}/{len(pending)}] {country}: both models complete",
                        flush=True,
                    )
                    country_results[country] = (forecast, nowcast)
                    if checkpoint_dir is not None:
                        save_country_checkpoint(
                            checkpoint_dir,
                            country,
                            {**(manifest_base or {}), "country": country},
                            forecast,
                            nowcast,
                        )
            finally:
                for future in futures:
                    future.cancel()

    forecast_predictions = pd.concat(
        [country_results[country][0] for country in requested_countries],
        ignore_index=True,
    ).sort_values(["country_code_3", *KEY_COLUMNS], kind="mergesort")
    nowcast_predictions = pd.concat(
        [country_results[country][1] for country in requested_countries],
        ignore_index=True,
    ).sort_values(["country_code_3", *KEY_COLUMNS], kind="mergesort")
    forecast_predictions = forecast_predictions.reset_index(drop=True)
    nowcast_predictions = nowcast_predictions.reset_index(drop=True)
    _validate_prediction_coverage(
        forecasting, forecast_predictions, requested_countries, "Forecasting"
    )
    _validate_prediction_coverage(
        nowcasting, nowcast_predictions, requested_countries, "Nowcasting"
    )

    metrics_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    for country in requested_countries:
        forecast = country_results[country][0]
        nowcast = country_results[country][1]
        metrics_records.append(
            calculate_country_metrics(forecast, "Forecasting", country)
        )
        metrics_records.append(
            calculate_country_metrics(nowcast, "Nowcasting", country)
        )
        audit_records.extend(
            _audit_records_for_country(
                forecasting,
                country,
                forecast,
                nowcast,
                random_state,
                checkpoint_reused[country],
                manifest_base,
            )
        )
    metrics = pd.DataFrame(metrics_records).sort_values(
        ["model", "country_code_3"], kind="mergesort"
    ).reset_index(drop=True)
    audit = pd.DataFrame(audit_records).sort_values(
        ["model", "held_out_country"], kind="mergesort"
    ).reset_index(drop=True)
    return forecast_predictions, nowcast_predictions, metrics, audit


def apply_figure_style() -> None:
    """Apply a compact publication typography and export contract."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _bboxes_overlap_with_padding(
    left: mpl.transforms.Bbox,
    right: mpl.transforms.Bbox,
    padding_pixels: float = 1.5,
) -> bool:
    """Return whether two display-coordinate boxes overlap after padding."""
    return not (
        left.x1 + padding_pixels <= right.x0
        or right.x1 + padding_pixels <= left.x0
        or left.y1 + padding_pixels <= right.y0
        or right.y1 + padding_pixels <= left.y0
    )


def _country_label_offset_candidates(x: float, y: float) -> list[tuple[int, int]]:
    """Return deterministic, edge-aware label offsets in points."""
    inward_x = -1 if x >= 0.75 else 1
    inward_y = -1 if y >= 0.80 else 1
    horizontal_order = [
        0,
        10 * inward_x,
        -10 * inward_x,
        20 * inward_x,
        -20 * inward_x,
        30 * inward_x,
        -30 * inward_x,
        40 * inward_x,
        -40 * inward_x,
        50 * inward_x,
        -50 * inward_x,
    ]
    candidates: list[tuple[int, int]] = []
    for vertical_distance in (9, 16, 23, 30, 37, 44):
        candidates.extend(
            (horizontal, inward_y * vertical_distance)
            for horizontal in horizontal_order
        )
    for horizontal_distance in (10, 18, 26, 34, 42, 50):
        horizontal = inward_x * horizontal_distance
        candidates.extend(
            (horizontal, vertical)
            for vertical in (0, 8, -8, 16, -16, 24, -24, 32, -32)
        )
    for vertical_distance in (9, 16, 23, 30):
        candidates.extend(
            (horizontal, -inward_y * vertical_distance)
            for horizontal in horizontal_order
        )
    return list(dict.fromkeys(candidates))


def _place_country_labels(
    axis: plt.Axes,
    included: pd.DataFrame,
    figure: plt.Figure,
) -> None:
    """Place ISO3 labels without overlaps or cross-panel spillover."""
    renderer = figure.canvas.get_renderer()
    safe_box = axis.bbox
    placed_boxes: list[mpl.transforms.Bbox] = []
    marker_centers = axis.transData.transform(
        included[["phase3plus_precision", "phase3plus_recall"]].to_numpy()
    )
    marker_boxes = [
        mpl.transforms.Bbox.from_bounds(x - 4, y - 4, 8, 8)
        for x, y in marker_centers
    ]
    ordered = included.sort_values(
        ["phase3plus_recall", "phase3plus_precision", "country_code_3"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    for row in ordered.itertuples(index=False):
        annotation = axis.annotate(
            str(row.country_code_3),
            (row.phase3plus_precision, row.phase3plus_recall),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=6,
            color="#222222",
            ha="center",
            va="center",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.35,
                "shrinkA": 1.5,
                "shrinkB": 3.0,
            },
            zorder=4,
        )
        selected_offset: tuple[int, int] | None = None
        for offset in _country_label_offset_candidates(
            float(row.phase3plus_precision), float(row.phase3plus_recall)
        ):
            annotation.set_position(offset)
            annotation.update_positions(renderer)
            bbox = mpl.text.Text.get_window_extent(annotation, renderer)
            stays_inside = (
                bbox.x0 >= safe_box.x0
                and bbox.x1 <= safe_box.x1
                and bbox.y0 >= safe_box.y0
                and bbox.y1 <= safe_box.y1
            )
            overlaps_label = any(
                _bboxes_overlap_with_padding(bbox, placed)
                for placed in placed_boxes
            )
            overlaps_marker = any(
                _bboxes_overlap_with_padding(bbox, marker, padding_pixels=0.5)
                for marker in marker_boxes
            )
            if stays_inside and not overlaps_label and not overlaps_marker:
                selected_offset = offset
                placed_boxes.append(bbox)
                break
        if selected_offset is None:
            raise RuntimeError(
                "Could not place a non-overlapping country label for "
                f"{row.country_code_3}."
            )
        annotation.set_position(selected_offset)


def create_precision_recall_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Create the approved per-country 1 x 2 precision-recall scatter plot."""
    _require_columns(
        metrics,
        [
            "model",
            "country_code_3",
            "phase3plus_precision",
            "phase3plus_recall",
        ],
    )
    apply_figure_style()
    plotting = metrics.copy()
    plotting["phase3plus_precision"] = pd.to_numeric(
        plotting["phase3plus_precision"], errors="coerce"
    )
    plotting["phase3plus_recall"] = pd.to_numeric(
        plotting["phase3plus_recall"], errors="coerce"
    )

    panel_specs = [
        ("Forecasting", "#0072B2", "a"),
        ("Nowcasting", "#E69F00", "b"),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.6),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    included_by_axis: list[tuple[plt.Axes, pd.DataFrame]] = []
    for axis, (model_name, color, letter) in zip(axes, panel_specs):
        panel = plotting.loc[plotting["model"].eq(model_name)].copy()
        included = panel.dropna(
            subset=["phase3plus_precision", "phase3plus_recall"]
        ).sort_values("country_code_3", kind="mergesort")
        axis.scatter(
            included["phase3plus_precision"],
            included["phase3plus_recall"],
            s=30,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )
        included_by_axis.append((axis, included))
        axis.set_title(model_name, loc="left", fontweight="normal", pad=5)
        axis.set_xlabel("Phase 3+ precision")
        axis.set_xlim(-0.025, 1.025)
        axis.set_ylim(-0.025, 1.025)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7, zorder=0)
        axis.set_axisbelow(True)
        axis.text(
            0.02,
            0.02,
            f"Plotted countries: {len(included)} / {len(panel)}",
            transform=axis.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            color="#333333",
        )
        axis.text(
            -0.13,
            1.05,
            letter,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Phase 3+ recall")
    fig.suptitle(
        "Leave-one-country-out precision and recall by held-out country",
        x=0.08,
        ha="left",
        fontsize=8,
        fontweight="normal",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.84, wspace=0.12)
    figure_canvas = fig.canvas
    figure_canvas.draw()
    for axis, included in included_by_axis:
        _place_country_labels(axis, included, fig)
    figure_canvas.draw()
    return fig


def _save_figure(fig: plt.Figure, output_dir: Path) -> dict[str, Path]:
    paths = {
        "jpg": output_dir / "precision_recall_scatter_leave_one_country_out.jpg",
        "png": output_dir / "precision_recall_scatter_leave_one_country_out.png",
        "pdf": output_dir / "precision_recall_scatter_leave_one_country_out.pdf",
    }
    fig.savefig(paths["jpg"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def run_analysis(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = DEFAULT_PHASE3_PARAMS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    countries: Sequence[str] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    workers: int = DEFAULT_WORKERS,
    prepare_country_lookup_from: Path | None = None,
    resume: bool = False,
) -> dict[str, Path]:
    """Run LOCO evaluation and save tabular artifacts."""
    if countries is not None and output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(
            "A partial country run requires a non-default output directory."
        )
    if prepare_country_lookup_from is not None:
        export_country_lookup(prepare_country_lookup_from, country_lookup_path)
    lookup = load_country_lookup(country_lookup_path)
    forecasting, nowcasting = prepare_model_inputs(
        pd.read_csv(forecasting_path), pd.read_csv(nowcasting_path), lookup
    )
    general_params, phase3_params = load_hyperparameters(
        general_params_path, phase3_params_path, random_state
    )
    manifest_base = build_run_manifest(
        forecasting_path,
        nowcasting_path,
        country_lookup_path,
        general_params_path,
        phase3_params_path,
        Path(__file__),
        random_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / ".leave_one_country_out_checkpoints"
    forecast, nowcast, metrics, audit = run_loco_predictions(
        forecasting,
        nowcasting,
        general_params,
        phase3_params,
        countries=countries,
        workers=workers,
        random_state=random_state,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        manifest_base=manifest_base,
    )
    paths = {
        "metrics": output_dir / "leave_one_country_out_country_metrics.csv",
        "micro_metrics": output_dir / "leave_one_country_out_micro_metrics.csv",
        "forecasting_predictions": output_dir
        / "leave_one_country_out_forecasting_predictions.csv",
        "nowcasting_predictions": output_dir
        / "leave_one_country_out_nowcasting_predictions.csv",
        "source_audit": output_dir / "leave_one_country_out_source_audit.csv",
    }
    metrics.to_csv(paths["metrics"], index=False, float_format="%.6f")
    forecast.to_csv(
        paths["forecasting_predictions"], index=False, float_format="%.6f"
    )
    nowcast.to_csv(
        paths["nowcasting_predictions"], index=False, float_format="%.6f"
    )
    calculate_micro_metrics(forecast, nowcast).to_csv(
        paths["micro_metrics"],
        index=True,
        index_label="model",
        float_format="%.6f",
    )
    audit.to_csv(paths["source_audit"], index=False, float_format="%.6f")
    figure = create_precision_recall_figure(metrics)
    paths.update(_save_figure(figure, output_dir))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasting-input", type=Path, default=DEFAULT_FORECASTING_INPUT
    )
    parser.add_argument(
        "--nowcasting-input", type=Path, default=DEFAULT_NOWCASTING_INPUT
    )
    parser.add_argument(
        "--country-lookup", type=Path, default=DEFAULT_COUNTRY_LOOKUP
    )
    parser.add_argument("--general-params", type=Path, default=DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=DEFAULT_PHASE3_PARAMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--prepare-country-lookup-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help="Replace only the pooled-micro table from saved predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate_existing:
        path = aggregate_existing_loco_predictions(args.output_dir)
        print(f"micro_metrics: {path}")
        return
    paths = run_analysis(
        forecasting_path=args.forecasting_input,
        nowcasting_path=args.nowcasting_input,
        country_lookup_path=args.country_lookup,
        general_params_path=args.general_params,
        phase3_params_path=args.phase3_params,
        output_dir=args.output_dir,
        countries=args.countries,
        random_state=args.random_state,
        workers=args.workers,
        prepare_country_lookup_from=args.prepare_country_lookup_from,
        resume=args.resume,
    )
    for artifact, path in paths.items():
        print(f"{artifact}: {path}")


if __name__ == "__main__":
    main()
