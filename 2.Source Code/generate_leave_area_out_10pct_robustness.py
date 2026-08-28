"""Generate a fixed 10% stratified leave-area-out robustness evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import platform
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache-area-holdout")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score

import generate_leave_one_country_out_robustness as loco


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
DEFAULT_COUNTRY_LOOKUP = SOURCE_DATA_DIR / "area_country_lookup.csv"
DEFAULT_FORECASTING_INPUT = loco.DEFAULT_FORECASTING_INPUT
DEFAULT_NOWCASTING_INPUT = loco.DEFAULT_NOWCASTING_INPUT
DEFAULT_GENERAL_PARAMS = loco.DEFAULT_GENERAL_PARAMS
DEFAULT_PHASE3_PARAMS = loco.DEFAULT_PHASE3_PARAMS
DEFAULT_OUTPUT_DIR = loco.DEFAULT_OUTPUT_DIR
DEFAULT_SAMPLE_FRACTION = 0.10
DEFAULT_RANDOM_STATE = 0
DEFAULT_WORKERS = 2
SPLIT_ID_TEMPLATE = "area_holdout_10pct_seed{random_state}"
EVALUATION_PROTOCOL = "fixed_hyperparameter_area_holdout_10pct"
SAMPLING_METHOD = (
    "country_stratified_minimum_one_hamilton_remaining_capacity_shared_rng"
)

_AREA_WORKER_FORECASTING: pd.DataFrame | None = None
_AREA_WORKER_NOWCASTING: pd.DataFrame | None = None
_AREA_WORKER_TRAIN_MASK: pd.Series | None = None
_AREA_WORKER_TEST_MASK: pd.Series | None = None
_AREA_WORKER_NOW_TRAIN_MASK: pd.Series | None = None
_AREA_WORKER_NOW_TEST_MASK: pd.Series | None = None
_AREA_WORKER_GENERAL_PARAMS: dict[str, object] | None = None
_AREA_WORKER_PHASE3_PARAMS: dict[str, object] | None = None
_AREA_WORKER_SPLIT_ID: str | None = None


def allocate_country_area_quotas(
    lookup: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """Allocate an exact sample with one area per country plus Hamilton extras."""
    normalized = loco.normalize_country_lookup(lookup)
    counts = (
        normalized.groupby("country_code_3", sort=True)["area_id"]
        .size()
        .rename("area_count")
        .reset_index()
    )
    country_count = len(counts)
    total_areas = int(counts["area_count"].sum())
    if sample_size < country_count:
        raise ValueError(
            f"sample_size must be at least the number of countries ({country_count})."
        )
    if sample_size > total_areas:
        raise ValueError("sample_size cannot exceed the number of available areas.")

    counts["sample_quota"] = 1
    remaining = sample_size - country_count
    capacity = counts["area_count"] - 1
    total_capacity = int(capacity.sum())
    if remaining > total_capacity:
        raise ValueError("Requested sample exceeds remaining country capacity.")
    if remaining:
        raw_extra = remaining * capacity / total_capacity
        floor_extra = np.floor(raw_extra).astype(int)
        counts["sample_quota"] += floor_extra
        leftover = remaining - int(floor_extra.sum())
        if leftover:
            ranking = pd.DataFrame(
                {
                    "country_code_3": counts["country_code_3"],
                    "fractional_remainder": raw_extra - floor_extra,
                }
            ).sort_values(
                ["fractional_remainder", "country_code_3"],
                ascending=[False, True],
                kind="mergesort",
            )
            selected_countries = ranking.head(leftover)["country_code_3"]
            counts.loc[
                counts["country_code_3"].isin(selected_countries), "sample_quota"
            ] += 1

    if int(counts["sample_quota"].sum()) != sample_size:
        raise RuntimeError("Country quotas do not sum to the requested sample size.")
    if (counts["sample_quota"] > counts["area_count"]).any():
        raise RuntimeError("A country quota exceeds its available areas.")
    return counts[["country_code_3", "area_count", "sample_quota"]].reset_index(
        drop=True
    )


def sample_stratified_areas(
    lookup: pd.DataFrame,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Select one reproducible country-stratified area sample."""
    normalized = loco.normalize_country_lookup(lookup)
    if not 0 < sample_fraction < 1:
        raise ValueError("sample_fraction must be strictly between 0 and 1.")
    sample_size = int(round(len(normalized) * sample_fraction))
    quotas = allocate_country_area_quotas(normalized, sample_size)
    rng = np.random.default_rng(random_state)
    selected: list[pd.DataFrame] = []
    for row in quotas.itertuples(index=False):
        country_areas = (
            normalized.loc[
                normalized["country_code_3"].eq(row.country_code_3), "area_id"
            ]
            .sort_values(kind="mergesort")
            .to_numpy()
        )
        chosen = rng.choice(country_areas, size=int(row.sample_quota), replace=False)
        selected.append(
            pd.DataFrame(
                {
                    "area_id": chosen,
                    "country_code_3": row.country_code_3,
                }
            )
        )
    result = pd.concat(selected, ignore_index=True).sort_values(
        ["country_code_3", "area_id"], kind="mergesort"
    )
    result = result[["area_id", "country_code_3"]].reset_index(drop=True)
    if len(result) != sample_size or result["area_id"].nunique() != sample_size:
        raise RuntimeError("Stratified area sampling did not produce a unique exact sample.")
    return result


def area_holdout_masks(
    data: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return train/test masks that hold out every row of the selected areas."""
    loco._require_columns(data, ["area_id", "country_code_3"])
    normalized_sample = loco.normalize_country_lookup(sample)
    data_mapping = loco.normalize_country_lookup(data[["area_id", "country_code_3"]])
    checked = normalized_sample.merge(
        data_mapping,
        on="area_id",
        how="left",
        suffixes=("_sample", "_data"),
        validate="one_to_one",
        indicator=True,
    )
    if not checked["_merge"].eq("both").all():
        missing = checked.loc[checked["_merge"].ne("both"), "area_id"].tolist()
        raise ValueError(f"Sample contains area_id values absent from model data: {missing}")
    mismatched = checked["country_code_3_sample"].ne(checked["country_code_3_data"])
    if mismatched.any():
        raise ValueError("Sample country codes do not match model-data country codes.")
    test_mask = data["area_id"].isin(normalized_sample["area_id"])
    train_mask = ~test_mask
    if not bool(test_mask.any()) or not bool(train_mask.any()):
        raise ValueError("Area holdout requires non-empty train and test rows.")
    return train_mask, test_mask


def calculate_pooled_metrics(
    predictions: pd.DataFrame,
    model_name: str,
) -> dict[str, object]:
    """Calculate pooled classification and Phase 3+ R-squared metrics."""
    loco._require_columns(
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

    precision = (
        true_positive / predicted_positive_count
        if predicted_positive_count
        else np.nan
    )
    recall = true_positive / actual_positive_count if actual_positive_count else np.nan
    actual_share = predictions["phase3_test"]
    predicted_share = predictions["phase3_pred"]
    if len(predictions) < 2 or actual_share.nunique(dropna=False) < 2:
        phase3plus_r2 = np.nan
    else:
        phase3plus_r2 = float(r2_score(actual_share, predicted_share))
    return {
        "model": model_name,
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
    }


def _initialize_area_worker(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    now_train_mask: pd.Series,
    now_test_mask: pd.Series,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    split_id: str,
) -> None:
    global _AREA_WORKER_FORECASTING
    global _AREA_WORKER_NOWCASTING
    global _AREA_WORKER_TRAIN_MASK
    global _AREA_WORKER_TEST_MASK
    global _AREA_WORKER_NOW_TRAIN_MASK
    global _AREA_WORKER_NOW_TEST_MASK
    global _AREA_WORKER_GENERAL_PARAMS
    global _AREA_WORKER_PHASE3_PARAMS
    global _AREA_WORKER_SPLIT_ID
    _AREA_WORKER_FORECASTING = forecasting
    _AREA_WORKER_NOWCASTING = nowcasting
    _AREA_WORKER_TRAIN_MASK = train_mask
    _AREA_WORKER_TEST_MASK = test_mask
    _AREA_WORKER_NOW_TRAIN_MASK = now_train_mask
    _AREA_WORKER_NOW_TEST_MASK = now_test_mask
    _AREA_WORKER_GENERAL_PARAMS = general_params
    _AREA_WORKER_PHASE3_PARAMS = phase3_params
    _AREA_WORKER_SPLIT_ID = split_id


def _run_area_model_in_worker(model_name: str) -> tuple[str, pd.DataFrame]:
    required = (
        _AREA_WORKER_FORECASTING,
        _AREA_WORKER_NOWCASTING,
        _AREA_WORKER_TRAIN_MASK,
        _AREA_WORKER_TEST_MASK,
        _AREA_WORKER_NOW_TRAIN_MASK,
        _AREA_WORKER_NOW_TEST_MASK,
        _AREA_WORKER_GENERAL_PARAMS,
        _AREA_WORKER_PHASE3_PARAMS,
        _AREA_WORKER_SPLIT_ID,
    )
    if any(item is None for item in required):
        raise RuntimeError("Area-holdout worker was not initialized.")
    if model_name == "Forecasting":
        predictions = loco.fit_forecasting_split(
            _AREA_WORKER_FORECASTING,
            _AREA_WORKER_TRAIN_MASK,
            _AREA_WORKER_TEST_MASK,
            _AREA_WORKER_SPLIT_ID,
            _AREA_WORKER_GENERAL_PARAMS,
            _AREA_WORKER_PHASE3_PARAMS,
            fold_column="fold_id",
        )
    elif model_name == "Nowcasting":
        predictions = loco.fit_nowcasting_split(
            _AREA_WORKER_FORECASTING,
            _AREA_WORKER_NOWCASTING,
            _AREA_WORKER_TRAIN_MASK,
            _AREA_WORKER_TEST_MASK,
            _AREA_WORKER_NOW_TRAIN_MASK,
            _AREA_WORKER_NOW_TEST_MASK,
            _AREA_WORKER_SPLIT_ID,
            _AREA_WORKER_GENERAL_PARAMS,
            _AREA_WORKER_PHASE3_PARAMS,
            fold_column="fold_id",
        )
    else:
        raise ValueError(f"Unknown area-holdout model: {model_name}")
    return model_name, predictions


def _validate_area_prediction_coverage(
    source: pd.DataFrame,
    predictions: pd.DataFrame,
    test_mask: pd.Series,
    model_name: str,
) -> None:
    expected = source.loc[test_mask, loco.KEY_COLUMNS]
    if len(predictions) != len(expected):
        raise ValueError(
            f"{model_name} predictions have {len(predictions)} rows; expected {len(expected)}."
        )
    if predictions.duplicated(loco.KEY_COLUMNS).any():
        raise ValueError(f"{model_name} predictions contain duplicate keys.")
    observed_keys = pd.MultiIndex.from_frame(predictions[loco.KEY_COLUMNS])
    expected_keys = pd.MultiIndex.from_frame(expected)
    if set(observed_keys) != set(expected_keys):
        raise ValueError(f"{model_name} prediction keys do not match held-out area rows.")


def run_area_holdout_predictions(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    sample: pd.DataFrame,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    workers: int = DEFAULT_WORKERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    forecasting_runner: Callable[..., pd.DataFrame] = loco.fit_forecasting_split,
    nowcasting_runner: Callable[..., pd.DataFrame] = loco.fit_nowcasting_split,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit both main models once on a joint stratified area holdout."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    forecasting = loco.add_cumulative_targets(forecasting)
    train_mask, test_mask = area_holdout_masks(forecasting, sample)
    now_train_mask, now_test_mask = area_holdout_masks(nowcasting, sample)
    forecast_test_keys = pd.MultiIndex.from_frame(
        forecasting.loc[test_mask, loco.KEY_COLUMNS]
    )
    nowcast_test_keys = pd.MultiIndex.from_frame(
        nowcasting.loc[now_test_mask, loco.KEY_COLUMNS]
    )
    if set(forecast_test_keys) != set(nowcast_test_keys):
        raise ValueError("Forecasting and nowcasting held-out area keys differ.")

    general_params = dict(general_params)
    phase3_params = dict(phase3_params)
    for params in (general_params, phase3_params):
        params["random_state"] = int(random_state)
        params["n_jobs"] = 1
    split_id = SPLIT_ID_TEMPLATE.format(random_state=random_state)
    custom_runners = (
        forecasting_runner is not loco.fit_forecasting_split
        or nowcasting_runner is not loco.fit_nowcasting_split
    )
    if workers == 1 or custom_runners:
        forecast_predictions = forecasting_runner(
            forecasting,
            train_mask,
            test_mask,
            split_id,
            general_params,
            phase3_params,
            fold_column="fold_id",
        )
        nowcast_predictions = nowcasting_runner(
            forecasting,
            nowcasting,
            train_mask,
            test_mask,
            now_train_mask,
            now_test_mask,
            split_id,
            general_params,
            phase3_params,
            fold_column="fold_id",
        )
    else:
        results: dict[str, pd.DataFrame] = {}
        with ProcessPoolExecutor(
            max_workers=min(2, workers),
            initializer=_initialize_area_worker,
            initargs=(
                forecasting,
                nowcasting,
                train_mask,
                test_mask,
                now_train_mask,
                now_test_mask,
                general_params,
                phase3_params,
                split_id,
            ),
        ) as executor:
            futures = {
                executor.submit(_run_area_model_in_worker, model_name): model_name
                for model_name in ("Forecasting", "Nowcasting")
            }
            try:
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        returned_model, predictions = future.result()
                    except Exception as error:
                        for other in futures:
                            other.cancel()
                        raise RuntimeError(
                            f"Area holdout {model_name} model failed"
                        ) from error
                    if returned_model != model_name:
                        raise RuntimeError(
                            f"Area worker model mismatch: expected {model_name}, "
                            f"got {returned_model}"
                        )
                    print(f"{model_name}: complete", flush=True)
                    results[model_name] = predictions
            finally:
                for future in futures:
                    future.cancel()
        forecast_predictions = results["Forecasting"]
        nowcast_predictions = results["Nowcasting"]

    forecast_predictions = forecast_predictions.sort_values(
        ["country_code_3", "area_id", "date"], kind="mergesort"
    ).reset_index(drop=True)
    nowcast_predictions = nowcast_predictions.sort_values(
        ["country_code_3", "area_id", "date"], kind="mergesort"
    ).reset_index(drop=True)
    forecast_predictions["is_held_out_area"] = True
    nowcast_predictions["is_held_out_area"] = True
    _validate_area_prediction_coverage(
        forecasting, forecast_predictions, test_mask, "Forecasting"
    )
    _validate_area_prediction_coverage(
        nowcasting, nowcast_predictions, now_test_mask, "Nowcasting"
    )

    metrics_records = []
    n_test_areas = int(sample["area_id"].nunique())
    for model_name, predictions in (
        ("Forecasting", forecast_predictions),
        ("Nowcasting", nowcast_predictions),
    ):
        record = calculate_pooled_metrics(predictions, model_name)
        record["n_test_areas"] = n_test_areas
        record["n_train"] = int(train_mask.sum())
        record["n_train_areas"] = int(forecasting.loc[train_mask, "area_id"].nunique())
        metrics_records.append(record)
    metrics = pd.DataFrame(metrics_records)
    return forecast_predictions, nowcast_predictions, metrics


def create_metric_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Create a publication-ready 1 x 4 model-metric point figure."""
    metric_specs = [
        ("phase3plus_precision", "Phase 3+ precision"),
        ("phase3plus_recall", "Phase 3+ recall"),
        ("overall_accuracy", "Overall-phase accuracy"),
        ("phase3plus_r2", "Phase 3+ share R²"),
    ]
    loco._require_columns(
        metrics,
        ["model", "n_test", "n_test_areas", *[item[0] for item in metric_specs]],
    )
    model_order = ["Forecasting", "Nowcasting"]
    plotting = metrics.set_index("model").loc[model_order]
    loco.apply_figure_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8))
    colors = {"Forecasting": "#0072B2", "Nowcasting": "#E69F00"}
    markers = {"Forecasting": "o", "Nowcasting": "s"}
    x_positions = np.arange(len(model_order))

    for index, (axis, (column, title)) in enumerate(zip(axes, metric_specs)):
        values = plotting[column].astype(float).to_numpy()
        for x_position, model_name, value in zip(x_positions, model_order, values):
            axis.scatter(
                [x_position],
                [value],
                s=48,
                color=colors[model_name],
                marker=markers[model_name],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            if np.isfinite(value):
                horizontal_offset = 8 if x_position == x_positions[0] else -8
                horizontal_alignment = (
                    "left" if x_position == x_positions[0] else "right"
                )
                axis.annotate(
                    f"{value:.3f}",
                    (x_position, value),
                    xytext=(horizontal_offset, 8),
                    textcoords="offset points",
                    ha=horizontal_alignment,
                    va="bottom",
                    fontsize=7,
                )
        axis.set_title(title, loc="left", fontweight="normal", pad=5)
        axis.set_xticks(x_positions, ["Forecast", "Nowcast"])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.16,
            1.05,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )
        if column != "phase3plus_r2":
            finite = values[np.isfinite(values)]
            if len(finite):
                value_span = float(finite.max() - finite.min())
                padding = max(0.01, value_span * 0.75)
                lower = max(0.0, float(finite.min()) - padding)
                upper = min(1.0, float(finite.max()) + padding)
            else:
                lower, upper = 0.0, 1.0
            axis.set_ylim(lower, upper)
        else:
            finite = values[np.isfinite(values)]
            lower = min(0.0, float(finite.min())) if len(finite) else -0.05
            upper = max(0.0, float(finite.max())) if len(finite) else 0.05
            span = max(upper - lower, 0.1)
            axis.set_ylim(lower - span * 0.15, upper + span * 0.20)
            axis.axhline(0.0, color="#777777", linewidth=0.6, zorder=1)

    n_test = int(plotting["n_test"].iloc[0])
    n_areas = int(plotting["n_test_areas"].iloc[0])
    fig.suptitle(
        f"Shared 10% stratified area holdout: {n_areas} areas, {n_test:,} test rows",
        x=0.07,
        ha="left",
        fontsize=8,
        fontweight="normal",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.80, wspace=0.35)
    return fig


def _save_figure(fig: plt.Figure, output_dir: Path) -> dict[str, Path]:
    """Save the area-holdout figure in the requested reproducible formats."""
    paths = {
        "jpg": output_dir
        / "precision_recall_accuracy_p3r2_leave_area_out_10pct.jpg",
        "png": output_dir
        / "precision_recall_accuracy_p3r2_leave_area_out_10pct.png",
        "pdf": output_dir
        / "precision_recall_accuracy_p3r2_leave_area_out_10pct.pdf",
    }
    fig.savefig(paths["jpg"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _source_audit_records(
    forecasting: pd.DataFrame,
    sample: pd.DataFrame,
    forecast_predictions: pd.DataFrame,
    nowcast_predictions: pd.DataFrame,
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    sample_path: Path,
    sample_fraction: float,
    random_state: int,
    workers: int,
) -> list[dict[str, object]]:
    """Build model-level provenance and leakage-control audit records."""
    train_mask, test_mask = area_holdout_masks(forecasting, sample)
    held_areas = set(sample["area_id"].astype(int))
    train_areas = set(forecasting.loc[train_mask, "area_id"].astype(int))
    test_areas = set(forecasting.loc[test_mask, "area_id"].astype(int))
    train_countries = set(forecasting.loc[train_mask, "country_code_3"])
    test_countries = set(forecasting.loc[test_mask, "country_code_3"])
    layer1_data = loco.add_cumulative_targets(forecasting)
    split_id = SPLIT_ID_TEMPLATE.format(random_state=random_state)
    common = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "temporal_interpretation": (
            "all_dates_joint_spatial_area_holdout_not_out_of_time"
        ),
        "sampling_method": SAMPLING_METHOD,
        "sample_fraction_requested": float(sample_fraction),
        "sample_fraction_realized": len(held_areas)
        / forecasting["area_id"].nunique(),
        "random_state": int(random_state),
        "split_id": split_id,
        "n_total": int(len(forecasting)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_total_areas": int(forecasting["area_id"].nunique()),
        "n_train_areas": int(len(train_areas)),
        "n_test_areas": int(len(test_areas)),
        "total_country_count": int(forecasting["country_code_3"].nunique()),
        "train_country_count": int(len(train_countries)),
        "test_country_count": int(len(test_countries)),
        "train_excludes_held_areas": train_areas.isdisjoint(held_areas),
        "test_only_held_areas": test_areas == held_areas,
        "same_country_other_areas_allowed": True,
        "same_country_other_areas_present": bool(train_countries & test_countries),
        "layer1_feature_count": len(loco.select_layer1_features(layer1_data)),
        "layer2_feature_count": len(loco.NOWCAST_FEATURES),
        "fews_ipc_ha_in_layer1": True,
        "workers_requested": int(workers),
        "max_parallel_models": min(2, int(workers)),
        "xgboost_n_jobs": 1,
        "forecasting_input_path": str(forecasting_path.resolve()),
        "nowcasting_input_path": str(nowcasting_path.resolve()),
        "country_lookup_path": str(country_lookup_path.resolve()),
        "general_params_path": str(general_params_path.resolve()),
        "phase3_params_path": str(phase3_params_path.resolve()),
        "sample_path": str(sample_path.resolve()),
        "forecasting_input_sha256": loco._sha256_file(forecasting_path),
        "nowcasting_input_sha256": loco._sha256_file(nowcasting_path),
        "country_lookup_sha256": loco._sha256_file(country_lookup_path),
        "general_params_sha256": loco._sha256_file(general_params_path),
        "phase3_params_sha256": loco._sha256_file(phase3_params_path),
        "sample_sha256": loco._sha256_file(sample_path),
        "script_sha256": loco._sha256_file(Path(__file__)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
    }
    return [
        {
            **common,
            "model": "Forecasting",
            "prediction_row_count": int(len(forecast_predictions)),
            "nonpositive_cumulative_prediction_count": int(
                forecast_predictions["nonpositive_cumulative_prediction_sum"].sum()
            ),
        },
        {
            **common,
            "model": "Nowcasting",
            "prediction_row_count": int(len(nowcast_predictions)),
            "nonpositive_cumulative_prediction_count": int(
                nowcast_predictions["nonpositive_cumulative_prediction_sum"].sum()
            ),
        },
    ]


def run_analysis(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = DEFAULT_PHASE3_PARAMS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
    workers: int = DEFAULT_WORKERS,
    forecasting_runner: Callable[..., pd.DataFrame] = loco.fit_forecasting_split,
    nowcasting_runner: Callable[..., pd.DataFrame] = loco.fit_nowcasting_split,
) -> dict[str, Path]:
    """Run the fixed area holdout and save the complete artifact set."""
    forecasting_path = Path(forecasting_path)
    nowcasting_path = Path(nowcasting_path)
    country_lookup_path = Path(country_lookup_path)
    general_params_path = Path(general_params_path)
    phase3_params_path = Path(phase3_params_path)
    output_dir = Path(output_dir)
    lookup = loco.load_country_lookup(country_lookup_path)
    forecasting, nowcasting = loco.prepare_model_inputs(
        pd.read_csv(forecasting_path), pd.read_csv(nowcasting_path), lookup
    )
    sample = sample_stratified_areas(
        lookup,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path, phase3_params_path, random_state
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sample": output_dir / "leave_area_out_10pct_sample.csv",
        "metrics": output_dir / "leave_area_out_10pct_metrics.csv",
        "forecasting_predictions": output_dir
        / "leave_area_out_10pct_forecasting_predictions.csv",
        "nowcasting_predictions": output_dir
        / "leave_area_out_10pct_nowcasting_predictions.csv",
        "source_audit": output_dir / "leave_area_out_10pct_source_audit.csv",
    }
    sample.to_csv(paths["sample"], index=False, lineterminator="\n")

    forecast, nowcast, metrics = run_area_holdout_predictions(
        forecasting,
        nowcasting,
        sample,
        general_params,
        phase3_params,
        workers=workers,
        random_state=random_state,
        forecasting_runner=forecasting_runner,
        nowcasting_runner=nowcasting_runner,
    )
    audit = pd.DataFrame(
        _source_audit_records(
            forecasting,
            sample,
            forecast,
            nowcast,
            forecasting_path,
            nowcasting_path,
            country_lookup_path,
            general_params_path,
            phase3_params_path,
            paths["sample"],
            sample_fraction,
            random_state,
            workers,
        )
    ).sort_values("model", kind="mergesort").reset_index(drop=True)

    metrics.to_csv(paths["metrics"], index=False, float_format="%.6f")
    forecast.to_csv(
        paths["forecasting_predictions"], index=False, float_format="%.6f"
    )
    nowcast.to_csv(
        paths["nowcasting_predictions"], index=False, float_format="%.6f"
    )
    audit.to_csv(paths["source_audit"], index=False, float_format="%.6f")
    paths.update(_save_figure(create_metric_figure(metrics), output_dir))
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
    parser.add_argument(
        "--sample-fraction", type=float, default=DEFAULT_SAMPLE_FRACTION
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(
        forecasting_path=args.forecasting_input,
        nowcasting_path=args.nowcasting_input,
        country_lookup_path=args.country_lookup,
        general_params_path=args.general_params,
        phase3_params_path=args.phase3_params,
        output_dir=args.output_dir,
        sample_fraction=args.sample_fraction,
        random_state=args.random_state,
        workers=args.workers,
    )
    for artifact, path in paths.items():
        print(f"{artifact}: {path}")


if __name__ == "__main__":
    main()
