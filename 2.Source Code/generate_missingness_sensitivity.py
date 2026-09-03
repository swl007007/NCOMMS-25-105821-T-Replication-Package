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
DEFAULT_BASE_METRICS = (
    SOURCE_CODE_DIR / "produced_graph" / "simple_baseline_comparison_metrics.csv"
)
DEFAULT_GENERAL_PARAMS = SOURCE_CODE_DIR / "forecasting_hyperparameters.json"
DEFAULT_PHASE3_PARAMS = SOURCE_CODE_DIR / "forecasting_hyperparameters_p3.json"
METRICS_FILENAME = "missingness_sensitivity_metrics.csv"
FIGURE_FILENAME = "missingness_sensitivity_curves.pdf"
FIGURE_PNG_FILENAME = "missingness_sensitivity_curves.png"
THRESHOLDS = (0, 5, 10, 30, 50)
TASKS = ("Forecasting", "Nowcasting")
MODELS = ("XGBoost", "Ensemble OLS", "Ordered Probit")
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
    countries: set[str] = set()
    for data, features, train_mask in layers:
        countries.update(data["country_code_3"].astype(str))
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

    def ranking_key(iso3: str) -> tuple[object, ...]:
        if iso3 not in rates.index:
            return 1, 0.0, iso3
        return 0, -float(rates[iso3]), iso3

    return tuple(sorted(countries, key=ranking_key))


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
    for column in columns:
        result[column] = np.nan
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
    renamed = dict(zip(features, indicators, strict=True))
    indicator_frame = missing.rename(columns=renamed).astype(np.int8)
    return pd.concat([result, indicator_frame], axis=1), indicators


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
    valid_phases = np.arange(1, 6)
    if not np.isin(actual_values, valid_phases).all() or not np.isin(
        predicted_values, valid_phases
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
    xgboost_forecasting = forecasting.drop(
        columns="evaluation_phase", errors="ignore"
    )
    if task == "Forecasting":
        wide = loco.fit_forecasting_split(
            xgboost_forecasting,
            forecasting_train_mask,
            forecasting_test_mask,
            "missingness_sensitivity",
            dict(general_params),
            dict(phase3_params),
            fold_column="condition_id",
        )
    elif task == "Nowcasting":
        wide = loco.fit_nowcasting_split(
            xgboost_forecasting,
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
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    layer1_preprocessor = simple.fit_numeric_preprocessor(
        forecasting.loc[forecasting_train_mask],
        layer1_features,
        task=task,
        method="Ensemble OLS",
        layer="layer1_shared",
    )
    x1_train = layer1_preprocessor.transform(
        forecasting.loc[forecasting_train_mask]
    )
    x1_test = layer1_preprocessor.transform(forecasting.loc[forecasting_test_mask])

    layer2_preprocessor = None
    if task == "Nowcasting":
        layer2_preprocessor = simple.fit_numeric_preprocessor(
            nowcasting.loc[nowcasting_train_mask],
            layer2_features,
            task=task,
            method="Ensemble OLS",
            layer="layer2_residual",
        )

    keys = [*loco.KEY_COLUMNS, "country_code_3"]
    test_predictions = pd.DataFrame(index=forecasting.index[forecasting_test_mask])
    for phase, target_column in loco.CUMULATIVE_TARGETS.items():
        y_train = forecasting.loc[forecasting_train_mask, target_column].to_numpy(
            dtype=float
        )
        layer1_test, _ = simple.fit_ols_arrays(x1_train, y_train, x1_test)
        layer1_train, _ = simple.fit_ols_arrays(x1_train, y_train, x1_train)
        raw = layer1_test

        if task == "Nowcasting":
            residual_frame = forecasting.loc[forecasting_train_mask, keys].copy()
            residual_frame["layer1_residual"] = y_train - layer1_train
            keyed_train = nowcasting.loc[
                nowcasting_train_mask, [*keys, *layer2_features]
            ].merge(residual_frame, on=keys, how="inner", validate="one_to_one")
            expected_train = int(forecasting_train_mask.sum())
            if len(keyed_train) != expected_train:
                raise ValueError("Ensemble OLS Nowcasting lost residual training rows.")
            now_test = nowcasting.loc[
                nowcasting_test_mask, [*keys, *layer2_features]
            ]
            assert layer2_preprocessor is not None
            residual_test, _ = simple.fit_ols_arrays(
                layer2_preprocessor.transform(keyed_train),
                keyed_train["layer1_residual"].to_numpy(dtype=float),
                layer2_preprocessor.transform(now_test),
            )
            residual_predictions = now_test[keys].copy()
            residual_predictions["residual_prediction"] = residual_test
            combined = forecasting.loc[forecasting_test_mask, keys].copy()
            combined["layer1_prediction"] = layer1_test
            combined = combined.merge(
                residual_predictions, on=keys, how="inner", validate="one_to_one"
            )
            if len(combined) != int(forecasting_test_mask.sum()):
                raise ValueError("Ensemble OLS Nowcasting lost test residual rows.")
            raw = (
                combined["layer1_prediction"].to_numpy(dtype=float)
                + combined["residual_prediction"].to_numpy(dtype=float)
            )

        test_predictions[f"phase{phase}_pred_rounded"] = np.round(raw, 2)

    predicted = simple._phase_from_rounded_predictions(test_predictions)
    actual = forecasting.loc[forecasting_test_mask, "evaluation_phase"]
    return (
        prediction_frame(actual, predicted),
        int(forecasting_train_mask.sum()),
        int(forecasting_test_mask.sum()),
    )


def source_feature_contract(bundle: simple.PreparedInputs) -> dict[str, object]:
    forecast_direct = tuple(multinomial.select_feature_columns(bundle.raw_forecasting))
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
    countries = tuple(
        sorted(bundle.forecasting["country_code_3"].astype(str).unique())
    )
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


def _condition_record(
    *,
    experiment: str,
    threshold_percent: float,
    task: str,
    model: str,
    removed_feature_count: int,
    removed_country_count: int,
    removed_country_iso3: str,
    n_train: int,
    n_test: int,
    fit: Callable[[], tuple[pd.DataFrame, int, int]],
) -> dict[str, object]:
    predictions, fitted_train, fitted_test, status, reason = run_fit_safely(
        fit, n_train=n_train, n_test=n_test
    )
    return {
        "experiment": experiment,
        "threshold_percent": threshold_percent,
        "task": task,
        "model": model,
        "removed_feature_count": removed_feature_count,
        "removed_country_count": removed_country_count,
        "removed_country_iso3": removed_country_iso3,
        "n_train": fitted_train,
        "n_test": fitted_test,
        "status": status,
        "reason": reason,
        "predictions": predictions,
    }


def run_feature_removal(
    bundle: simple.PreparedInputs,
    general_params: Mapping[str, object],
    phase3_params: Mapping[str, object],
) -> list[dict[str, object]]:
    contract = source_feature_contract(bundle)
    raw_f_train, raw_f_test = simple.temporal_masks(bundle.raw_forecasting["date"])
    raw_n_train, raw_n_test = simple.temporal_masks(bundle.raw_nowcasting["date"])
    forecast_ranking = rank_features(
        bundle.raw_forecasting, contract["forecast_direct"], raw_f_train
    )
    layer2_ranking = rank_features(
        bundle.nowcasting, contract["layer2"], bundle.nowcasting_train_mask
    )
    nowcast_direct_ranking = rank_features(
        bundle.raw_nowcasting, contract["nowcast_direct"], raw_n_train
    )
    records: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        forecast_count = selection_count(threshold, len(forecast_ranking))
        layer2_count = selection_count(threshold, len(layer2_ranking))
        direct_nowcast_count = selection_count(
            threshold, len(nowcast_direct_ranking)
        )
        selected_forecast = forecast_ranking[:forecast_count]
        selected_layer1 = tuple(
            contract["forecast_source_to_layer1"][feature]
            for feature in selected_forecast
        )
        selected_layer2 = layer2_ranking[:layer2_count]
        selected_direct_nowcast = nowcast_direct_ranking[:direct_nowcast_count]

        forecasting = suppress_features(bundle.forecasting, selected_layer1)
        nowcasting = suppress_features(bundle.nowcasting, selected_layer2)
        raw_forecasting = suppress_features(
            bundle.raw_forecasting, selected_forecast
        )
        raw_nowcasting = suppress_features(
            bundle.raw_nowcasting, selected_direct_nowcast
        )

        for task in TASKS:
            for model in MODELS:
                removed_count = (
                    forecast_count
                    if task == "Forecasting"
                    else (
                        direct_nowcast_count
                        if model == "Ordered Probit"
                        else forecast_count + layer2_count
                    )
                )
                if model == "XGBoost":
                    fit = lambda task=task: fit_xgboost_task(
                        task=task,
                        forecasting=forecasting,
                        nowcasting=nowcasting,
                        forecasting_train_mask=bundle.forecasting_train_mask,
                        forecasting_test_mask=bundle.forecasting_test_mask,
                        nowcasting_train_mask=bundle.nowcasting_train_mask,
                        nowcasting_test_mask=bundle.nowcasting_test_mask,
                        general_params=general_params,
                        phase3_params=phase3_params,
                    )
                    n_train = int(bundle.forecasting_train_mask.sum())
                    n_test = int(bundle.forecasting_test_mask.sum())
                elif model == "Ensemble OLS":
                    fit = lambda task=task: fit_ensemble_ols_task(
                        task=task,
                        forecasting=forecasting,
                        nowcasting=nowcasting,
                        layer1_features=contract["layer1"],
                        layer2_features=contract["layer2"],
                        forecasting_train_mask=bundle.forecasting_train_mask,
                        forecasting_test_mask=bundle.forecasting_test_mask,
                        nowcasting_train_mask=bundle.nowcasting_train_mask,
                        nowcasting_test_mask=bundle.nowcasting_test_mask,
                    )
                    n_train = int(bundle.forecasting_train_mask.sum())
                    n_test = int(bundle.forecasting_test_mask.sum())
                else:
                    data = raw_forecasting if task == "Forecasting" else raw_nowcasting
                    features = (
                        contract["forecast_direct"]
                        if task == "Forecasting"
                        else contract["nowcast_direct"]
                    )
                    train_mask = raw_f_train if task == "Forecasting" else raw_n_train
                    test_mask = raw_f_test if task == "Forecasting" else raw_n_test
                    fit = lambda data=data, features=features, train_mask=train_mask, test_mask=test_mask, task=task: fit_ordered_probit_task(
                        data=data,
                        features=features,
                        train_mask=train_mask,
                        test_mask=test_mask,
                        task=task,
                    )
                    n_train = int(train_mask.sum())
                    n_test = int(test_mask.sum())
                records.append(
                    _condition_record(
                        experiment="feature_removal",
                        threshold_percent=threshold,
                        task=task,
                        model=model,
                        removed_feature_count=removed_count,
                        removed_country_count=0,
                        removed_country_iso3="",
                        n_train=n_train,
                        n_test=n_test,
                        fit=fit,
                    )
                )
    return records


def run_country_removal(
    bundle: simple.PreparedInputs,
    general_params: Mapping[str, object],
    phase3_params: Mapping[str, object],
) -> list[dict[str, object]]:
    contract = source_feature_contract(bundle)
    forecast_country_ranking = rank_countries(
        [(bundle.forecasting, contract["layer1"], bundle.forecasting_train_mask)]
    )
    nowcast_country_ranking = rank_countries(
        [
            (bundle.forecasting, contract["layer1"], bundle.forecasting_train_mask),
            (bundle.nowcasting, contract["layer2"], bundle.nowcasting_train_mask),
        ]
    )
    records: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        removed_count = selection_count(threshold, len(contract["countries"]))
        for task, ranking in (
            ("Forecasting", forecast_country_ranking),
            ("Nowcasting", nowcast_country_ranking),
        ):
            removed = ranking[:removed_count]
            filtered = country_filtered_bundle(bundle, removed)
            raw_data = (
                filtered.raw_forecasting
                if task == "Forecasting"
                else filtered.raw_nowcasting
            )
            raw_train, raw_test = simple.temporal_masks(raw_data["date"])
            for model in MODELS:
                if model == "XGBoost":
                    fit = lambda task=task, filtered=filtered: fit_xgboost_task(
                        task=task,
                        forecasting=filtered.forecasting,
                        nowcasting=filtered.nowcasting,
                        forecasting_train_mask=filtered.forecasting_train_mask,
                        forecasting_test_mask=filtered.forecasting_test_mask,
                        nowcasting_train_mask=filtered.nowcasting_train_mask,
                        nowcasting_test_mask=filtered.nowcasting_test_mask,
                        general_params=general_params,
                        phase3_params=phase3_params,
                    )
                    n_train = int(filtered.forecasting_train_mask.sum())
                    n_test = int(filtered.forecasting_test_mask.sum())
                elif model == "Ensemble OLS":
                    fit = lambda task=task, filtered=filtered: fit_ensemble_ols_task(
                        task=task,
                        forecasting=filtered.forecasting,
                        nowcasting=filtered.nowcasting,
                        layer1_features=contract["layer1"],
                        layer2_features=contract["layer2"],
                        forecasting_train_mask=filtered.forecasting_train_mask,
                        forecasting_test_mask=filtered.forecasting_test_mask,
                        nowcasting_train_mask=filtered.nowcasting_train_mask,
                        nowcasting_test_mask=filtered.nowcasting_test_mask,
                    )
                    n_train = int(filtered.forecasting_train_mask.sum())
                    n_test = int(filtered.forecasting_test_mask.sum())
                else:
                    features = (
                        contract["forecast_direct"]
                        if task == "Forecasting"
                        else contract["nowcast_direct"]
                    )
                    fit = lambda raw_data=raw_data, features=features, raw_train=raw_train, raw_test=raw_test, task=task: fit_ordered_probit_task(
                        data=raw_data,
                        features=features,
                        train_mask=raw_train,
                        test_mask=raw_test,
                        task=task,
                    )
                    n_train = int(raw_train.sum())
                    n_test = int(raw_test.sum())
                records.append(
                    _condition_record(
                        experiment="country_removal",
                        threshold_percent=threshold,
                        task=task,
                        model=model,
                        removed_feature_count=0,
                        removed_country_count=removed_count,
                        removed_country_iso3=";".join(removed),
                        n_train=n_train,
                        n_test=n_test,
                        fit=fit,
                    )
                )
    return records


def run_missing_indicators(
    bundle: simple.PreparedInputs,
) -> list[dict[str, object]]:
    contract = source_feature_contract(bundle)
    raw_forecasting, raw_forecast_indicators = add_missing_indicators(
        bundle.raw_forecasting, contract["forecast_direct"]
    )
    raw_nowcasting, raw_nowcast_indicators = add_missing_indicators(
        bundle.raw_nowcasting, contract["nowcast_direct"]
    )
    forecasting, layer1_indicators = add_missing_indicators(
        bundle.forecasting, contract["layer1"]
    )
    nowcasting, layer2_indicators = add_missing_indicators(
        bundle.nowcasting, contract["layer2"]
    )
    raw_f_train, raw_f_test = simple.temporal_masks(raw_forecasting["date"])
    raw_n_train, raw_n_test = simple.temporal_masks(raw_nowcasting["date"])
    layer1_features = (*contract["layer1"], *layer1_indicators)
    layer2_features = (*contract["layer2"], *layer2_indicators)
    records: list[dict[str, object]] = []

    for task in TASKS:
        for model in ("Ensemble OLS", "Ordered Probit"):
            if model == "Ensemble OLS":
                fit = lambda task=task: fit_ensemble_ols_task(
                    task=task,
                    forecasting=forecasting,
                    nowcasting=nowcasting,
                    layer1_features=layer1_features,
                    layer2_features=layer2_features,
                    forecasting_train_mask=bundle.forecasting_train_mask,
                    forecasting_test_mask=bundle.forecasting_test_mask,
                    nowcasting_train_mask=bundle.nowcasting_train_mask,
                    nowcasting_test_mask=bundle.nowcasting_test_mask,
                )
                n_train = int(bundle.forecasting_train_mask.sum())
                n_test = int(bundle.forecasting_test_mask.sum())
            else:
                data = raw_forecasting if task == "Forecasting" else raw_nowcasting
                base_features = (
                    contract["forecast_direct"]
                    if task == "Forecasting"
                    else contract["nowcast_direct"]
                )
                indicators = (
                    raw_forecast_indicators
                    if task == "Forecasting"
                    else raw_nowcast_indicators
                )
                features = (*base_features, *indicators)
                train_mask = raw_f_train if task == "Forecasting" else raw_n_train
                test_mask = raw_f_test if task == "Forecasting" else raw_n_test
                fit = lambda data=data, features=features, train_mask=train_mask, test_mask=test_mask, task=task: fit_ordered_probit_task(
                    data=data,
                    features=features,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    task=task,
                )
                n_train = int(train_mask.sum())
                n_test = int(test_mask.sum())
            records.append(
                _condition_record(
                    experiment="missing_indicators",
                    threshold_percent=np.nan,
                    task=task,
                    model=model,
                    removed_feature_count=0,
                    removed_country_count=0,
                    removed_country_iso3="",
                    n_train=n_train,
                    n_test=n_test,
                    fit=fit,
                )
            )
    return records


def record_metrics(record: Mapping[str, object]) -> dict[str, object]:
    status = str(record["status"])
    if status == "generated":
        predictions = record.get("predictions")
        if not isinstance(predictions, pd.DataFrame):
            raise ValueError("Generated conditions require a prediction frame.")
        calculated = simple.calculate_pooled_metrics(predictions)
        metrics = {name: float(calculated[name]) for name in METRIC_NAMES}
    elif status == "not_estimable":
        metrics = {name: np.nan for name in METRIC_NAMES}
    else:
        raise ValueError(f"Unknown raw condition status: {status}")
    row = {
        "experiment": record["experiment"],
        "threshold_percent": record["threshold_percent"],
        "task": record["task"],
        "model": record["model"],
        "removed_feature_count": int(record["removed_feature_count"]),
        "removed_country_count": int(record["removed_country_count"]),
        "removed_country_iso3": record["removed_country_iso3"],
        **metrics,
        "n_train": int(record["n_train"]),
        "n_test": int(record["n_test"]),
        "status": status,
        "reason": record["reason"],
    }
    for metric in METRIC_NAMES:
        row[f"delta_{metric}_vs_xgboost"] = np.nan
    return {column: row[column] for column in METRIC_COLUMNS}


def load_frozen_xgboost_references(path: Path) -> list[dict[str, object]]:
    source = pd.read_csv(path, float_precision="round_trip")
    required = {
        "task",
        "method",
        *METRIC_NAMES,
        "n_train",
        "n_test",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Frozen base metrics lack columns: {sorted(missing)}")
    selected = source.loc[source["method"].eq("Main result")]
    records: list[dict[str, object]] = []
    for task in TASKS:
        task_rows = selected.loc[selected["task"].eq(task)]
        if len(task_rows) != 1:
            raise ValueError(f"Expected one frozen XGBoost reference for {task}.")
        source_row = task_rows.iloc[0]
        row = {
            "experiment": "missing_indicators",
            "threshold_percent": np.nan,
            "task": task,
            "model": "XGBoost",
            "removed_feature_count": 0,
            "removed_country_count": 0,
            "removed_country_iso3": "",
            **{name: float(source_row[name]) for name in METRIC_NAMES},
            "n_train": int(source_row["n_train"]),
            "n_test": int(source_row["n_test"]),
            "status": "frozen_reference",
            "reason": "",
        }
        for metric in METRIC_NAMES:
            row[f"delta_{metric}_vs_xgboost"] = np.nan
        records.append({column: row[column] for column in METRIC_COLUMNS})
    if len(selected) != len(TASKS):
        unexpected = sorted(set(selected["task"].astype(str)).difference(TASKS))
        raise ValueError(f"Unexpected frozen XGBoost reference rows: {unexpected}")
    return records


def _reference_key(row: pd.Series) -> tuple[object, ...]:
    experiment = str(row["experiment"])
    if experiment == "missing_indicators":
        return experiment, str(row["task"])
    return experiment, int(row["threshold_percent"]), str(row["task"])


def add_xgboost_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    references: dict[tuple[object, ...], pd.Series] = {}
    for _, row in result.loc[result["model"].eq("XGBoost")].iterrows():
        key = _reference_key(row)
        if key in references:
            raise ValueError(f"Duplicate XGBoost reference for {key}.")
        references[key] = row
    for index, row in result.iterrows():
        key = _reference_key(row)
        if key not in references:
            raise ValueError(f"Missing XGBoost reference for {key}.")
        reference = references[key]
        for metric in METRIC_NAMES:
            value = float(row[metric])
            reference_value = float(reference[metric])
            delta_column = f"delta_{metric}_vs_xgboost"
            result.at[index, delta_column] = (
                value - reference_value
                if np.isfinite(value) and np.isfinite(reference_value)
                else np.nan
            )
    return result.loc[:, METRIC_COLUMNS]


def _blank(value: object) -> bool:
    return pd.isna(value) or str(value) == ""


def validate_metrics(metrics: pd.DataFrame) -> None:
    if tuple(metrics.columns) != METRIC_COLUMNS:
        raise ValueError("Missingness sensitivity metrics have an unexpected schema.")
    expected_counts = {
        "feature_removal": 30,
        "country_removal": 30,
        "missing_indicators": 6,
    }
    observed_counts = metrics["experiment"].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(f"Unexpected metrics row counts: {observed_counts}")

    expected_conditions = {
        (experiment, threshold, task, model)
        for experiment in ("feature_removal", "country_removal")
        for threshold in THRESHOLDS
        for task in TASKS
        for model in MODELS
    } | {
        ("missing_indicators", None, task, model)
        for task in TASKS
        for model in MODELS
    }
    observed_conditions = {
        (
            str(row.experiment),
            None
            if str(row.experiment) == "missing_indicators"
            else int(row.threshold_percent),
            str(row.task),
            str(row.model),
        )
        for row in metrics.itertuples(index=False)
    }
    if observed_conditions != expected_conditions or len(metrics) != len(
        observed_conditions
    ):
        raise ValueError("Metrics conditions are duplicated, missing, or unexpected.")
    indicator_thresholds = metrics.loc[
        metrics["experiment"].eq("missing_indicators"), "threshold_percent"
    ]
    if not indicator_thresholds.isna().all():
        raise ValueError("Missing-indicator threshold values must be missing.")

    country = metrics.loc[metrics["experiment"].eq("country_removal")]
    required_country_counts = {0: 0, 5: 2, 10: 3, 30: 9, 50: 15}
    for (task, threshold), group in country.groupby(
        ["task", "threshold_percent"], observed=True
    ):
        expected = required_country_counts[int(threshold)]
        if not group["removed_country_count"].eq(expected).all():
            raise ValueError(f"Country-removal count differs for {task} {threshold}.")
        normalized = group["removed_country_iso3"].map(
            lambda value: "" if pd.isna(value) else str(value)
        )
        if normalized.nunique() != 1:
            raise ValueError(f"Country-removal set differs for {task} {threshold}.")
        codes = [code for code in normalized.iloc[0].split(";") if code]
        if len(codes) != expected or len(set(codes)) != expected:
            raise ValueError(f"Country-removal ISO3 count differs for {task} {threshold}.")

    if (metrics[["removed_feature_count", "removed_country_count"]] < 0).any().any():
        raise ValueError("Removal counts cannot be negative.")
    valid_statuses = {"generated", "frozen_reference", "not_estimable"}
    if not set(metrics["status"]).issubset(valid_statuses):
        raise ValueError("Metrics contain an unknown status.")
    expected_frozen = metrics["experiment"].eq("missing_indicators") & metrics[
        "model"
    ].eq("XGBoost")
    if not metrics.loc[expected_frozen, "status"].eq("frozen_reference").all() or metrics.loc[
        ~expected_frozen, "status"
    ].eq("frozen_reference").any():
        raise ValueError("Frozen reference status must identify only indicator XGBoost rows.")
    for row in metrics.itertuples(index=False):
        values = np.asarray([getattr(row, name) for name in METRIC_NAMES], dtype=float)
        if np.isinf(values).any():
            raise ValueError("Metrics cannot contain infinite values.")
        if row.status == "not_estimable":
            if not np.isnan(values).all() or _blank(row.reason):
                raise ValueError("Not-estimable rows require missing metrics and a reason.")
        else:
            if not np.isfinite(values[0]) or not _blank(row.reason):
                raise ValueError("Generated/reference rows require accuracy and no reason.")
            if not 0.0 <= values[0] <= 1.0:
                raise ValueError("Overall accuracy lies outside zero and one.")
            for value in values[1:3]:
                if np.isfinite(value) and not 0.0 <= value <= 1.0:
                    raise ValueError("Precision or recall lies outside zero and one.")
            if int(row.n_train) < 1 or int(row.n_test) < 1:
                raise ValueError("Generated/reference rows require positive sample counts.")

    recalculated = add_xgboost_deltas(metrics)
    delta_columns = [f"delta_{metric}_vs_xgboost" for metric in METRIC_NAMES]
    for column in delta_columns:
        if not np.allclose(
            pd.to_numeric(metrics[column], errors="coerce"),
            pd.to_numeric(recalculated[column], errors="coerce"),
            rtol=0.0,
            atol=1e-15,
            equal_nan=True,
        ):
            raise ValueError(f"Incorrect XGBoost delta column: {column}")


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
    ("phase3above_r2", "Phase 3+ R²"),
)
MODEL_COLORS = {
    "XGBoost": "#4C78A8",
    "Ensemble OLS": "#B279A2",
    "Ordered Probit": "#F28E2B",
}


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


def create_sensitivity_figure(metrics: pd.DataFrame) -> plt.Figure:
    required = {
        "experiment",
        "threshold_percent",
        "task",
        "model",
        "removed_country_count",
        "n_test",
        *(name for name, _ in FIGURE_METRICS),
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Figure metrics lack columns: {sorted(missing)}")
    plotting = metrics.loc[
        metrics["experiment"].isin(["feature_removal", "country_removal"])
    ].copy()
    expected = {
        (experiment, task, threshold, model)
        for experiment, task in FIGURE_ROWS
        for threshold in THRESHOLDS
        for model in MODELS
    }
    observed = set(
        zip(
            plotting["experiment"],
            plotting["task"],
            plotting["threshold_percent"].astype(int),
            plotting["model"],
            strict=False,
        )
    )
    if observed != expected or len(plotting) != len(expected):
        raise ValueError("Figure requires the complete 60-row removal grid.")

    finite_r2 = plotting["phase3above_r2"].to_numpy(dtype=float)
    finite_r2 = finite_r2[np.isfinite(finite_r2)]
    if finite_r2.size:
        r2_low = min(0.0, float(finite_r2.min()))
        r2_high = max(0.0, float(finite_r2.max()))
        r2_margin = 0.05 * max(r2_high - r2_low, 0.1)
        r2_limits = (r2_low - r2_margin, r2_high + r2_margin)
    else:
        r2_limits = (-0.05, 0.05)

    apply_figure_style()
    figure, axes = plt.subplots(4, 4, figsize=(9.4, 6.25), squeeze=False)
    panel_index = 0
    for row_index, (experiment, task) in enumerate(FIGURE_ROWS):
        row_data = plotting.loc[
            plotting["experiment"].eq(experiment) & plotting["task"].eq(task)
        ]
        if experiment == "country_removal":
            tick_rows = []
            for threshold in THRESHOLDS:
                group = row_data.loc[row_data["threshold_percent"].eq(threshold)]
                if (
                    group["removed_country_count"].nunique() != 1
                    or group["n_test"].nunique() != 1
                ):
                    raise ValueError(
                        f"Country figure samples differ for {task} at {threshold}%."
                    )
                tick_rows.append(
                    (
                        int(group["removed_country_count"].iloc[0]),
                        int(group["n_test"].iloc[0]),
                    )
                )
            x_values = np.arange(len(THRESHOLDS), dtype=float)
            tick_labels = [f"{removed}\nn={n_test}" for removed, n_test in tick_rows]
            x_label = "Countries removed"
        else:
            x_values = np.arange(len(THRESHOLDS), dtype=float)
            tick_labels = [f"{threshold}%" for threshold in THRESHOLDS]
            x_label = "Most-missing features suppressed"

        for column_index, (metric, title) in enumerate(FIGURE_METRICS):
            axis = axes[row_index, column_index]
            for model in MODELS:
                series = row_data.loc[row_data["model"].eq(model)].sort_values(
                    "threshold_percent", kind="mergesort"
                )
                axis.plot(
                    x_values,
                    series[metric].to_numpy(dtype=float),
                    marker="o",
                    markersize=3.2,
                    linewidth=1.1,
                    color=MODEL_COLORS[model],
                    label=model,
                )
            if metric == "phase3above_r2":
                axis.set_ylim(*r2_limits)
                axis.axhline(
                    0.0,
                    color="#777777",
                    linewidth=0.7,
                    linestyle="--",
                    zorder=0,
                    label="_nolegend_",
                )
            else:
                axis.set_ylim(0.0, 1.0)
            axis.set_xticks(
                x_values,
                tick_labels,
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            axis.tick_params(axis="x", labelsize=6, pad=1.5)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
            axis.set_axisbelow(True)
            if row_index == 0:
                axis.set_title(title, loc="left", pad=4)
            if column_index == 0:
                removal_label = (
                    "Feature removal" if experiment == "feature_removal" else "Country removal"
                )
                axis.set_ylabel(f"{task}\n{removal_label}")
            if row_index in (1, 3):
                axis.set_xlabel(x_label)
            axis.text(
                -0.17,
                1.12,
                chr(ord("a") + panel_index),
                transform=axis.transAxes,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="top",
                clip_on=False,
            )
            panel_index += 1

    figure.legend(
        handles=axes[0, 0].lines,
        labels=("Main model", *MODELS[1:]),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
    )
    figure.suptitle(
        "Performance sensitivity to structured predictor missingness",
        x=0.08,
        y=0.995,
        ha="left",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.16,
        top=0.955,
        wspace=0.27,
        hspace=0.78,
    )
    return figure


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


def write_outputs(
    metrics: pd.DataFrame,
    figure: plt.Figure,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".missingness_sensitivity_staging_", dir=output_dir
    ) as staging_name:
        staging = Path(staging_name)
        staged_metrics = staging / METRICS_FILENAME
        staged_figure = staging / FIGURE_FILENAME
        staged_png = staging / FIGURE_PNG_FILENAME
        metrics.loc[:, METRIC_COLUMNS].to_csv(
            staged_metrics,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
            na_rep="",
        )
        reloaded = pd.read_csv(staged_metrics, float_precision="round_trip")
        validate_metrics(reloaded)
        figure.savefig(
            staged_figure,
            bbox_inches="tight",
            facecolor="white",
            metadata={"CreationDate": None, "ModDate": None},
        )
        if staged_figure.read_bytes()[:4] != b"%PDF":
            raise ValueError("Staged sensitivity figure is not a PDF.")
        figure.savefig(
            staged_png,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        if staged_png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Staged sensitivity figure is not a PNG.")
        paths = {
            "metrics_csv": output_dir / METRICS_FILENAME,
            "figure_pdf": output_dir / FIGURE_FILENAME,
            "figure_png": output_dir / FIGURE_PNG_FILENAME,
        }
        replace_with_retry(staged_metrics, paths["metrics_csv"])
        replace_with_retry(staged_figure, paths["figure_pdf"])
        replace_with_retry(staged_png, paths["figure_png"])
    return paths


def run_analysis(
    *,
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    base_metrics_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    formal_run = Path(output_dir).resolve() == DEFAULT_OUTPUT_DIR.resolve()
    if formal_run:
        frozen_main_result.assert_frozen_environment(
            ("matplotlib", "statsmodels", "patsy")
        )
        expected = selected_main.EXPECTED_FILE_SHA256
        selected_main.validate_default_inputs(
            {
                Path(forecasting_path): expected[selected_main.DEFAULT_FORECASTING_INPUT],
                Path(nowcasting_path): expected[selected_main.DEFAULT_NOWCASTING_INPUT],
                Path(country_lookup_path): expected[selected_main.DEFAULT_COUNTRY_LOOKUP],
                Path(general_params_path): expected[selected_main.DEFAULT_GENERAL_PARAMS],
                Path(phase3_params_path): expected[selected_main.DEFAULT_PHASE3_PARAMS],
                Path(base_metrics_path): expected[selected_main.DEFAULT_BASE_METRICS],
            }
        )
    bundle = simple.load_prepared_inputs(
        forecasting_path=forecasting_path,
        nowcasting_path=nowcasting_path,
        country_lookup_path=country_lookup_path,
        enforce_formal_counts=formal_run,
    )
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path,
        phase3_params_path,
        random_state=None,
        estimator_n_jobs=None,
    )
    print("Running feature-removal conditions...", flush=True)
    raw_records = run_feature_removal(bundle, general_params, phase3_params)
    print("Running country-removal conditions...", flush=True)
    raw_records.extend(run_country_removal(bundle, general_params, phase3_params))
    print("Running missing-indicator baseline conditions...", flush=True)
    raw_records.extend(run_missing_indicators(bundle))
    rows = [record_metrics(record) for record in raw_records]
    rows.extend(load_frozen_xgboost_references(base_metrics_path))
    metrics = add_xgboost_deltas(pd.DataFrame(rows, columns=METRIC_COLUMNS))
    validate_metrics(metrics)
    figure = create_sensitivity_figure(
        metrics.loc[
            metrics["experiment"].isin(["feature_removal", "country_removal"])
        ]
    )
    try:
        return write_outputs(metrics, figure, output_dir)
    finally:
        plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasting-input", type=Path, default=simple.DEFAULT_FORECASTING_INPUT
    )
    parser.add_argument(
        "--nowcasting-input", type=Path, default=simple.DEFAULT_NOWCASTING_INPUT
    )
    parser.add_argument(
        "--country-lookup", type=Path, default=simple.DEFAULT_COUNTRY_LOOKUP
    )
    parser.add_argument("--general-params", type=Path, default=DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=DEFAULT_PHASE3_PARAMS)
    parser.add_argument("--base-metrics", type=Path, default=DEFAULT_BASE_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    paths = run_analysis(
        forecasting_path=arguments.forecasting_input,
        nowcasting_path=arguments.nowcasting_input,
        country_lookup_path=arguments.country_lookup,
        general_params_path=arguments.general_params,
        phase3_params_path=arguments.phase3_params,
        base_metrics_path=arguments.base_metrics,
        output_dir=arguments.output_dir,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
