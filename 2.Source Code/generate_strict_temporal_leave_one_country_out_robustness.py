"""Generate strict temporal leave-one-country-out robustness results."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache-strict-temporal-loco"),
)

import matplotlib as mpl
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

import generate_leave_one_country_out_robustness as loco


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
CUTOFF = "2022-01-01"
EVALUATION_PROTOCOL = "fixed_hyperparameter_strict_temporal_loco"
TEMPORAL_INTERPRETATION = (
    "held_country_and_post_cutoff_labels_excluded_from_training"
)
CANONICAL_TEST_KEY_SHA256 = (
    "288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2"
)
EXPECTED_TEST_ROWS = 1170
EXPECTED_TEST_AREAS = 646
EXPECTED_NONEMPTY_COUNTRIES = 27
EXPECTED_SKIPPED_COUNTRIES = ("LSO", "ZWE")

FINAL_FILENAMES = {
    "micro_metrics": "strict_temporal_loco_micro_metrics.csv",
    "forecasting_predictions": "strict_temporal_loco_forecasting_predictions.csv",
    "nowcasting_predictions": "strict_temporal_loco_nowcasting_predictions.csv",
    "area_metrics": "strict_temporal_loco_area_metrics.csv",
    "area_macro_denominators": "strict_temporal_loco_area_macro_denominators.csv",
    "fold_audit": "strict_temporal_loco_fold_audit.csv",
    "skipped_folds": "strict_temporal_loco_skipped_folds.csv",
    "source_audit": "strict_temporal_loco_source_audit.csv",
}

_WORKER_FORECASTING: pd.DataFrame | None = None
_WORKER_NOWCASTING: pd.DataFrame | None = None
_WORKER_GENERAL_PARAMS: dict[str, object] | None = None
_WORKER_PHASE3_PARAMS: dict[str, object] | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key_sha256(data: pd.DataFrame) -> str:
    missing = sorted(set(loco.KEY_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Business-key columns are missing: {missing}")
    keys = data[loco.KEY_COLUMNS].copy()
    if keys["area_id"].isna().any() or keys["date"].isna().any():
        raise ValueError("Business keys contain missing values.")
    try:
        dates = pd.to_datetime(keys["date"], errors="raise", format="mixed")
    except (TypeError, ValueError):
        dates = pd.to_datetime(keys["date"], errors="raise")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        raise ValueError("Business-key dates must be timezone-naive.")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Business-key dates must be normalized to midnight.")
    keys["date"] = dates.dt.strftime("%Y-%m-%d")
    if keys.duplicated(loco.KEY_COLUMNS).any():
        raise ValueError("Business keys must be unique.")
    ordered = keys.sort_values(loco.KEY_COLUMNS, kind="mergesort")
    payload = ordered.to_csv(
        index=False,
        float_format="%.17g",
        na_rep="<NA>",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_temporal_masks(
    data: pd.DataFrame,
    held_out_country: str,
    cutoff: str = CUTOFF,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    required = [*loco.KEY_COLUMNS, "country_code_3"]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"Strict split columns are missing: {missing}")
    if held_out_country not in set(data["country_code_3"]):
        raise ValueError(f"Unknown held-out country: {held_out_country}")
    dates = pd.to_datetime(data["date"], errors="raise")
    boundary = pd.Timestamp(cutoff)
    held = data["country_code_3"].eq(held_out_country)
    train = (~held) & dates.lt(boundary)
    test = held & dates.ge(boundary)
    excluded = ~(train | test)
    if not train.any():
        raise ValueError(f"Strict training set is empty for {held_out_country}.")
    if not (train | test | excluded).all():
        raise ValueError(f"Strict masks do not cover all rows for {held_out_country}.")
    if (train & test).any() or (train & excluded).any() or (test & excluded).any():
        raise ValueError(f"Strict masks overlap for {held_out_country}.")
    return train, test, excluded


def subset_for_complete_split(
    data: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    active = data.loc[train_mask | test_mask].copy()
    active_train = train_mask.loc[active.index].astype(bool)
    active_test = test_mask.loc[active.index].astype(bool)
    if not (active_train | active_test).all() or (active_train & active_test).any():
        raise ValueError("Active strict split is not a complete train/test partition.")
    return active, active_train, active_test


def _partition_keys(data: pd.DataFrame, mask: pd.Series) -> set[tuple[object, object]]:
    return set(data.loc[mask, loco.KEY_COLUMNS].itertuples(index=False, name=None))


def fit_strict_country_fold(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    held_out_country: str,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_masks = strict_temporal_masks(forecasting, held_out_country)
    nowcast_masks = strict_temporal_masks(nowcasting, held_out_country)
    if not forecast_masks[1].any():
        raise ValueError(f"Strict fold has no test rows: {held_out_country}")
    for label, forecast_mask, nowcast_mask in zip(
        ("train", "test", "excluded"), forecast_masks, nowcast_masks
    ):
        if _partition_keys(forecasting, forecast_mask) != _partition_keys(
            nowcasting, nowcast_mask
        ):
            raise ValueError(
                f"Forecasting and nowcasting {label} keys differ for {held_out_country}."
            )

    forecast_active, forecast_train, forecast_test = subset_for_complete_split(
        forecasting, forecast_masks[0], forecast_masks[1]
    )
    nowcast_active, now_train, now_test = subset_for_complete_split(
        nowcasting, nowcast_masks[0], nowcast_masks[1]
    )
    forecast = loco.fit_forecasting_split(
        forecast_active,
        forecast_train,
        forecast_test,
        held_out_country,
        general_params,
        phase3_params,
        fold_column="fold_country",
    )
    nowcast = loco.fit_nowcasting_split(
        forecast_active,
        nowcast_active,
        forecast_train,
        forecast_test,
        now_train,
        now_test,
        held_out_country,
        general_params,
        phase3_params,
        fold_column="fold_country",
    )
    for predictions in (forecast, nowcast):
        predictions["cutoff"] = CUTOFF
        if not predictions["country_code_3"].eq(held_out_country).all():
            raise ValueError(f"Strict fold {held_out_country} contains another country.")
    return forecast, nowcast


def calculate_area_metrics(
    forecasting_predictions: pd.DataFrame,
    nowcasting_predictions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for model_name, predictions in (
        ("Nowcasting", nowcasting_predictions),
        ("Forecasting", forecasting_predictions),
    ):
        for area_id, group in predictions.groupby("area_id", sort=True, observed=True):
            countries = group["country_code_3"].drop_duplicates()
            if len(countries) != 1:
                raise ValueError(f"Area {area_id} maps to multiple countries.")
            metric_group = group.copy()
            metric_group["phase3_test"] = metric_group["phase3_test"].round(2)
            metric_group["phase3_pred"] = metric_group["phase3_pred"].round(2)
            metric = loco.calculate_country_metrics(
                metric_group, model_name, str(area_id)
            )
            records.append(
                {
                    "model": model_name,
                    "area_id": int(area_id),
                    "country_code_3": countries.iloc[0],
                    "n_test_dates": metric["n_test"],
                    "accuracy": metric["overall_accuracy"],
                    "precision": metric["phase3plus_precision"],
                    "recall": metric["phase3plus_recall"],
                    "R2(p3)": metric["phase3plus_r2"],
                    "actual_phase3plus_count": metric["actual_phase3plus_count"],
                    "predicted_phase3plus_count": metric["predicted_phase3plus_count"],
                    "true_positive": metric["true_positive"],
                    "false_positive": metric["false_positive"],
                    "false_negative": metric["false_negative"],
                    "true_negative": metric["true_negative"],
                    "precision_undefined_reason": metric[
                        "precision_undefined_reason"
                    ],
                    "recall_undefined_reason": metric["recall_undefined_reason"],
                    "r2_undefined_reason": metric["r2_undefined_reason"],
                    "nonpositive_cumulative_prediction_count": metric[
                        "nonpositive_cumulative_prediction_count"
                    ],
                }
            )
    return pd.DataFrame(records).sort_values(
        ["model", "area_id"], kind="mergesort"
    ).reset_index(drop=True)


def aggregate_area_metrics(
    area_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = ["accuracy", "precision", "recall", "R2(p3)"]
    order = ["Nowcasting", "Forecasting"]
    missing_models = sorted(set(order).difference(area_metrics["model"].unique()))
    if missing_models:
        raise ValueError(f"Area metrics are missing models: {missing_models}")
    macro_records = []
    denominator_records = []
    for model in order:
        group = area_metrics.loc[area_metrics["model"].eq(model)]
        macro_records.append(
            {"model": model, **{metric: group[metric].mean() for metric in metric_columns}}
        )
        denominator_records.append(
            {
                "model": model,
                "area_count_total": int(group["area_id"].nunique()),
                "accuracy_area_count_defined": int(group["accuracy"].notna().sum()),
                "precision_area_count_defined": int(group["precision"].notna().sum()),
                "recall_area_count_defined": int(group["recall"].notna().sum()),
                "r2_area_count_defined": int(group["R2(p3)"].notna().sum()),
            }
        )
    return (
        pd.DataFrame(macro_records).set_index("model")[metric_columns],
        pd.DataFrame(denominator_records).set_index("model"),
    )


def calculate_micro_metrics(
    forecasting_predictions: pd.DataFrame,
    nowcasting_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate primary metrics after pooling every test observation."""
    metric_columns = ["accuracy", "precision", "recall", "R2(p3)"]
    records: list[dict[str, object]] = []
    for model_name, predictions in (
        ("Nowcasting", nowcasting_predictions),
        ("Forecasting", forecasting_predictions),
    ):
        if predictions.empty:
            raise ValueError(f"{model_name} predictions are empty.")
        metric_predictions = predictions.copy()
        metric_predictions["phase3_test"] = metric_predictions["phase3_test"].round(2)
        metric_predictions["phase3_pred"] = metric_predictions["phase3_pred"].round(2)
        metric = loco.calculate_country_metrics(
            metric_predictions,
            model_name,
            "ALL_STRICT_TEMPORAL_TEST_ROWS",
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
    return pd.DataFrame(records).set_index("model")[metric_columns]


def _country_manifest(
    manifest_base: Mapping[str, object],
    country: str,
    train_keys: pd.DataFrame,
    test_keys: pd.DataFrame,
) -> dict[str, object]:
    return {
        **dict(manifest_base),
        "held_out_country": country,
        "train_key_sha256": canonical_key_sha256(train_keys),
        "test_key_sha256": canonical_key_sha256(test_keys),
        "n_train": int(len(train_keys)),
        "n_test": int(len(test_keys)),
    }


def _checkpoint_paths(checkpoint_dir: Path, country: str) -> dict[str, Path]:
    directory = checkpoint_dir / country
    return {
        "directory": directory,
        "forecasting": directory / "forecasting_predictions.csv",
        "nowcasting": directory / "nowcasting_predictions.csv",
        "manifest": directory / "manifest.json",
    }


def save_checkpoint(
    checkpoint_dir: Path,
    country: str,
    manifest: Mapping[str, object],
    forecast: pd.DataFrame,
    nowcast: pd.DataFrame,
) -> None:
    paths = _checkpoint_paths(checkpoint_dir, country)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    forecast_tmp = paths["forecasting"].with_suffix(".csv.tmp")
    nowcast_tmp = paths["nowcasting"].with_suffix(".csv.tmp")
    manifest_tmp = paths["manifest"].with_suffix(".json.tmp")
    forecast.to_csv(forecast_tmp, index=False, float_format="%.17g")
    nowcast.to_csv(nowcast_tmp, index=False, float_format="%.17g")
    manifest_tmp.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(forecast_tmp, paths["forecasting"])
    os.replace(nowcast_tmp, paths["nowcasting"])
    os.replace(manifest_tmp, paths["manifest"])


def load_checkpoint(
    checkpoint_dir: Path,
    country: str,
    expected_manifest: Mapping[str, object],
    expected_test_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    paths = _checkpoint_paths(checkpoint_dir, country)
    if not all(paths[name].is_file() for name in ("forecasting", "nowcasting", "manifest")):
        return None
    try:
        observed = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if observed != dict(expected_manifest):
            return None
        forecast = pd.read_csv(paths["forecasting"], float_precision="round_trip")
        nowcast = pd.read_csv(paths["nowcasting"], float_precision="round_trip")
        expected_hash = canonical_key_sha256(expected_test_keys)
        for predictions in (forecast, nowcast):
            if canonical_key_sha256(predictions) != expected_hash:
                return None
            if not predictions["country_code_3"].eq(country).all():
                return None
            if not predictions["cutoff"].eq(CUTOFF).all():
                return None
        return forecast, nowcast
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return None


def _initialize_worker(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
) -> None:
    global _WORKER_FORECASTING, _WORKER_NOWCASTING
    global _WORKER_GENERAL_PARAMS, _WORKER_PHASE3_PARAMS
    _WORKER_FORECASTING = forecasting
    _WORKER_NOWCASTING = nowcasting
    _WORKER_GENERAL_PARAMS = general_params
    _WORKER_PHASE3_PARAMS = phase3_params


def _run_country_in_worker(country: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    if any(
        value is None
        for value in (
            _WORKER_FORECASTING,
            _WORKER_NOWCASTING,
            _WORKER_GENERAL_PARAMS,
            _WORKER_PHASE3_PARAMS,
        )
    ):
        raise RuntimeError("Strict LOCO worker was not initialized.")
    forecast, nowcast = fit_strict_country_fold(
        _WORKER_FORECASTING,
        _WORKER_NOWCASTING,
        country,
        _WORKER_GENERAL_PARAMS,
        _WORKER_PHASE3_PARAMS,
    )
    return country, forecast, nowcast


def _fold_audit_records(
    forecasting: pd.DataFrame,
    country: str,
    forecast: pd.DataFrame,
    nowcast: pd.DataFrame,
    checkpoint_reused: bool,
    random_state: int,
) -> list[dict[str, object]]:
    train, test, excluded = strict_temporal_masks(forecasting, country)
    dates = pd.to_datetime(forecasting["date"])
    boundary = pd.Timestamp(CUTOFF)
    common = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "temporal_interpretation": TEMPORAL_INTERPRETATION,
        "held_out_country": country,
        "cutoff": CUTOFF,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "n_excluded": int(excluded.sum()),
        "train_area_count": int(forecasting.loc[train, "area_id"].nunique()),
        "test_area_count": int(forecasting.loc[test, "area_id"].nunique()),
        "train_country_count": int(
            forecasting.loc[train, "country_code_3"].nunique()
        ),
        "test_country_count": int(forecasting.loc[test, "country_code_3"].nunique()),
        "train_date_min": dates.loc[train].min().strftime("%Y-%m-%d"),
        "train_date_max": dates.loc[train].max().strftime("%Y-%m-%d"),
        "test_date_min": dates.loc[test].min().strftime("%Y-%m-%d"),
        "test_date_max": dates.loc[test].max().strftime("%Y-%m-%d"),
        "train_key_sha256": canonical_key_sha256(forecasting.loc[train]),
        "test_key_sha256": canonical_key_sha256(forecasting.loc[test]),
        "train_excludes_held_country": bool(
            forecasting.loc[train, "country_code_3"].ne(country).all()
        ),
        "test_only_held_country": bool(
            forecasting.loc[test, "country_code_3"].eq(country).all()
        ),
        "train_dates_before_cutoff": bool(dates.loc[train].lt(boundary).all()),
        "test_dates_on_or_after_cutoff": bool(dates.loc[test].ge(boundary).all()),
        "max_train_date_before_min_test_date": bool(
            dates.loc[train].max() < dates.loc[test].min()
        ),
        "temporal_violation_count": int(
            forecasting.loc[train, "country_code_3"].eq(country).sum()
            + dates.loc[train].ge(boundary).sum()
            + forecasting.loc[test, "country_code_3"].ne(country).sum()
            + dates.loc[test].lt(boundary).sum()
        ),
        "layer1_feature_count": len(loco.select_layer1_features(forecasting)),
        "layer2_feature_count": len(loco.NOWCAST_FEATURES),
        "fews_ipc_ha_in_layer1": True,
        "random_state": int(random_state),
        "estimator_n_jobs": 1,
        "checkpoint_reused": bool(checkpoint_reused),
    }
    if common["temporal_violation_count"]:
        raise ValueError(f"Strict isolation failed for {country}.")
    return [
        {
            **common,
            "model": "Forecasting",
            "nonpositive_cumulative_prediction_count": int(
                forecast["nonpositive_cumulative_prediction_sum"].sum()
            ),
        },
        {
            **common,
            "model": "Nowcasting",
            "nonpositive_cumulative_prediction_count": int(
                nowcast["nonpositive_cumulative_prediction_sum"].sum()
            ),
        },
    ]


def run_strict_predictions(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    countries: Sequence[str] | None,
    workers: int,
    random_state: int,
    checkpoint_dir: Path,
    resume: bool,
    manifest_base: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    forecasting = loco.add_cumulative_targets(forecasting)
    available = sorted(forecasting["country_code_3"].unique())
    requested = available if countries is None else sorted(set(countries))
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"Unknown requested countries: {unknown}")

    general_params = {**general_params, "random_state": random_state, "n_jobs": 1}
    phase3_params = {**phase3_params, "random_state": random_state, "n_jobs": 1}
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    reused: dict[str, bool] = {}
    manifests: dict[str, dict[str, object]] = {}
    skipped_records = []
    pending = []
    for country in requested:
        train, test, _ = strict_temporal_masks(forecasting, country)
        if not test.any():
            skipped_records.append(
                {
                    "held_out_country": country,
                    "n_test": 0,
                    "reason": "no_on_or_after_cutoff_observations",
                }
            )
            continue
        manifest = _country_manifest(
            manifest_base,
            country,
            forecasting.loc[train, loco.KEY_COLUMNS],
            forecasting.loc[test, loco.KEY_COLUMNS],
        )
        manifests[country] = manifest
        loaded = (
            load_checkpoint(
                checkpoint_dir,
                country,
                manifest,
                forecasting.loc[test, loco.KEY_COLUMNS],
            )
            if resume
            else None
        )
        if loaded is None:
            pending.append(country)
            reused[country] = False
        else:
            results[country] = loaded
            reused[country] = True
            print(f"[resume] {country}: checkpoint reused", flush=True)

    effective_workers = min(workers, max(1, len(pending)))
    if pending and effective_workers == 1:
        for position, country in enumerate(pending, 1):
            print(f"[{position}/{len(pending)}] {country}", flush=True)
            forecast, nowcast = fit_strict_country_fold(
                forecasting, nowcasting, country, general_params, phase3_params
            )
            results[country] = (forecast, nowcast)
            save_checkpoint(
                checkpoint_dir, country, manifests[country], forecast, nowcast
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
            for future in as_completed(futures):
                country = futures[future]
                completed += 1
                try:
                    returned_country, forecast, nowcast = future.result()
                except Exception as error:
                    raise RuntimeError(f"Strict LOCO country {country} failed") from error
                if returned_country != country:
                    raise RuntimeError(f"Worker country mismatch for {country}.")
                print(f"[{completed}/{len(pending)}] {country}: complete", flush=True)
                results[country] = (forecast, nowcast)
                save_checkpoint(
                    checkpoint_dir, country, manifests[country], forecast, nowcast
                )

    nonempty = sorted(results)
    if not nonempty:
        raise ValueError("No requested country has on-or-after-cutoff test rows.")
    forecast = pd.concat([results[c][0] for c in nonempty], ignore_index=True)
    nowcast = pd.concat([results[c][1] for c in nonempty], ignore_index=True)
    forecast = forecast.sort_values(
        ["country_code_3", *loco.KEY_COLUMNS], kind="mergesort"
    ).reset_index(drop=True)
    nowcast = nowcast.sort_values(
        ["country_code_3", *loco.KEY_COLUMNS], kind="mergesort"
    ).reset_index(drop=True)
    if canonical_key_sha256(forecast) != canonical_key_sha256(nowcast):
        raise ValueError("Forecasting and nowcasting prediction keys differ.")

    audit_records = []
    for country in nonempty:
        audit_records.extend(
            _fold_audit_records(
                forecasting,
                country,
                results[country][0],
                results[country][1],
                reused[country],
                random_state,
            )
        )
    audit = pd.DataFrame(audit_records).sort_values(
        ["model", "held_out_country"], kind="mergesort"
    ).reset_index(drop=True)
    skipped = pd.DataFrame(
        skipped_records,
        columns=["held_out_country", "n_test", "reason"],
    ).sort_values("held_out_country", kind="mergesort").reset_index(drop=True)
    return forecast, nowcast, audit, skipped


def _validate_full_prediction_population(
    forecast: pd.DataFrame,
    nowcast: pd.DataFrame,
) -> None:
    key_hashes = {}
    for model, predictions in (("Forecasting", forecast), ("Nowcasting", nowcast)):
        if len(predictions) != EXPECTED_TEST_ROWS:
            raise ValueError(f"{model} predictions have {len(predictions)} rows.")
        if predictions.duplicated(loco.KEY_COLUMNS).any():
            raise ValueError(f"{model} predictions contain duplicate keys.")
        if predictions["area_id"].nunique() != EXPECTED_TEST_AREAS:
            raise ValueError(f"{model} area coverage is incorrect.")
        if predictions["country_code_3"].nunique() != EXPECTED_NONEMPTY_COUNTRIES:
            raise ValueError(f"{model} country coverage is incorrect.")
        key_hashes[model] = canonical_key_sha256(predictions)
        if key_hashes[model] != CANONICAL_TEST_KEY_SHA256:
            raise ValueError(f"{model} prediction key hash is incorrect.")
    if key_hashes["Forecasting"] != key_hashes["Nowcasting"]:
        raise ValueError("Forecasting and nowcasting prediction keys differ.")


def _validate_primary_metrics(primary_metrics: pd.DataFrame) -> None:
    if primary_metrics.shape != (2, 4):
        raise ValueError("Primary micro table must be exactly 2 x 4.")
    if primary_metrics.index.tolist() != ["Nowcasting", "Forecasting"]:
        raise ValueError("Primary micro table model order is incorrect.")
    if primary_metrics.columns.tolist() != [
        "accuracy",
        "precision",
        "recall",
        "R2(p3)",
    ]:
        raise ValueError("Primary micro table columns are incorrect.")
    if primary_metrics.isna().any().any():
        raise ValueError("Primary micro table contains an undefined metric.")


def _validate_full_results(
    forecast: pd.DataFrame,
    nowcast: pd.DataFrame,
    area_metrics: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    denominators: pd.DataFrame,
    fold_audit: pd.DataFrame,
    skipped: pd.DataFrame,
) -> None:
    _validate_full_prediction_population(forecast, nowcast)
    if len(area_metrics) != 2 * EXPECTED_TEST_AREAS:
        raise ValueError("Area metric row count is incorrect.")
    _validate_primary_metrics(primary_metrics)
    if denominators.shape != (2, 5):
        raise ValueError("Area denominator table must be exactly 2 x 5.")
    if len(fold_audit) != 2 * EXPECTED_NONEMPTY_COUNTRIES:
        raise ValueError("Fold audit row count is incorrect.")
    if set(skipped["held_out_country"]) != set(EXPECTED_SKIPPED_COUNTRIES):
        raise ValueError("Skipped countries are incorrect.")
    if not fold_audit[
        [
            "train_excludes_held_country",
            "test_only_held_country",
            "train_dates_before_cutoff",
            "test_dates_on_or_after_cutoff",
            "max_train_date_before_min_test_date",
        ]
    ].all().all() or fold_audit["temporal_violation_count"].ne(0).any():
        raise ValueError("Fold audit contains an isolation failure.")


def _write_dataframe(name: str, data: pd.DataFrame, path: Path) -> None:
    if name == "micro_metrics":
        data.to_csv(path, index=True, index_label="model", float_format="%.6f")
    elif name == "area_macro_denominators":
        data.to_csv(path, index=True, index_label="model")
    elif name in ("forecasting_predictions", "nowcasting_predictions"):
        data.to_csv(path, index=False, float_format="%.17g")
    else:
        data.to_csv(path, index=False, float_format="%.10g")


def _replace_with_retry(source: Path, target: Path) -> None:
    """Replace a final artifact despite transient Dropbox/Windows file locks."""
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.25)


def _write_artifact_family(
    output_dir: Path,
    artifacts: Mapping[str, pd.DataFrame],
    source_context: Mapping[str, object],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {name: output_dir / filename for name, filename in FINAL_FILENAMES.items()}
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".strict-loco-stage-") as tmp:
        stage_dir = Path(tmp)
        stage_paths = {name: stage_dir / filename for name, filename in FINAL_FILENAMES.items()}
        for name, data in artifacts.items():
            _write_dataframe(name, data, stage_paths[name])
        source_row = dict(source_context)
        for name in artifacts:
            source_row[f"{name}_path"] = str(final_paths[name])
            source_row[f"{name}_sha256"] = file_sha256(stage_paths[name])
        pd.DataFrame([source_row]).to_csv(stage_paths["source_audit"], index=False)

        backup_dir = stage_dir / "backup"
        backup_dir.mkdir()
        existed = {}
        try:
            for name, final_path in final_paths.items():
                existed[name] = final_path.exists()
                if existed[name]:
                    shutil.copy2(final_path, backup_dir / final_path.name)
            for name, final_path in final_paths.items():
                _replace_with_retry(stage_paths[name], final_path)
        except Exception:
            for name, final_path in final_paths.items():
                backup = backup_dir / final_path.name
                if existed.get(name) and backup.exists():
                    _replace_with_retry(backup, final_path)
                elif not existed.get(name) and final_path.exists():
                    final_path.unlink()
            raise
    return final_paths


def aggregate_existing_strict_predictions(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Replace the primary 2 x 4 table from saved strict LOCO predictions."""
    forecasting_path = output_dir / FINAL_FILENAMES["forecasting_predictions"]
    nowcasting_path = output_dir / FINAL_FILENAMES["nowcasting_predictions"]
    source_audit_path = output_dir / FINAL_FILENAMES["source_audit"]
    for path in (forecasting_path, nowcasting_path, source_audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Strict LOCO artifact not found: {path}")

    forecast = pd.read_csv(forecasting_path, float_precision="round_trip")
    nowcast = pd.read_csv(nowcasting_path, float_precision="round_trip")
    _validate_full_prediction_population(forecast, nowcast)
    primary_metrics = calculate_micro_metrics(forecast, nowcast)
    _validate_primary_metrics(primary_metrics)

    source_audit = pd.read_csv(source_audit_path)
    if len(source_audit) != 1:
        raise ValueError("Strict LOCO source audit must contain exactly one row.")
    for artifact_name, path in (
        ("forecasting_predictions", forecasting_path),
        ("nowcasting_predictions", nowcasting_path),
    ):
        hash_column = f"{artifact_name}_sha256"
        if hash_column not in source_audit.columns:
            raise ValueError(f"Source audit is missing {hash_column}.")
        if source_audit.loc[0, hash_column] != file_sha256(path):
            raise ValueError(f"Saved {artifact_name} hash does not match source audit.")

    metrics_path = output_dir / FINAL_FILENAMES["micro_metrics"]
    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".strict-loco-micro-stage-"
    ) as tmp:
        stage_dir = Path(tmp)
        stage_metrics = stage_dir / metrics_path.name
        stage_audit = stage_dir / source_audit_path.name
        _write_dataframe("micro_metrics", primary_metrics, stage_metrics)

        source_audit.loc[0, "strict_generator_sha256"] = file_sha256(Path(__file__))
        source_audit.loc[0, "primary_metric_aggregation"] = (
            "pooled_observation_micro"
        )
        source_audit.loc[0, "primary_metric_observation_count_per_model"] = int(
            len(forecast)
        )
        source_audit.loc[0, "r2_actual_precision"] = "rounded_two_decimal"
        source_audit.loc[0, "r2_prediction_precision"] = "rounded_two_decimal"
        source_audit.loc[0, "micro_metrics_path"] = str(metrics_path.resolve())
        source_audit.loc[0, "micro_metrics_sha256"] = file_sha256(stage_metrics)
        source_audit.to_csv(stage_audit, index=False)

        backup_dir = stage_dir / "backup"
        backup_dir.mkdir()
        targets = {
            metrics_path: stage_metrics,
            source_audit_path: stage_audit,
        }
        existed: dict[Path, bool] = {}
        try:
            for target in targets:
                existed[target] = target.exists()
                if existed[target]:
                    shutil.copy2(target, backup_dir / target.name)
            for target, staged in targets.items():
                _replace_with_retry(staged, target)
        except Exception:
            for target in targets:
                backup = backup_dir / target.name
                if existed.get(target) and backup.exists():
                    _replace_with_retry(backup, target)
                elif not existed.get(target) and target.exists():
                    target.unlink()
            raise
    return metrics_path


def run_analysis(
    forecasting_path: Path = loco.DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = loco.DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = loco.DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = loco.DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = loco.DEFAULT_PHASE3_PARAMS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    countries: Sequence[str] | None = None,
    random_state: int = 0,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Path]:
    if countries is not None and output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("A partial country run requires a non-default output directory.")
    lookup = loco.load_country_lookup(country_lookup_path)
    forecasting, nowcasting = loco.prepare_model_inputs(
        pd.read_csv(forecasting_path), pd.read_csv(nowcasting_path), lookup
    )
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path,
        phase3_params_path,
        random_state=random_state,
        estimator_n_jobs=1,
    )
    manifest_base = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "temporal_interpretation": TEMPORAL_INTERPRETATION,
        "cutoff": CUTOFF,
        "forecasting_input_sha256": file_sha256(forecasting_path),
        "nowcasting_input_sha256": file_sha256(nowcasting_path),
        "country_lookup_sha256": file_sha256(country_lookup_path),
        "general_params_sha256": file_sha256(general_params_path),
        "phase3_params_sha256": file_sha256(phase3_params_path),
        "strict_generator_sha256": file_sha256(Path(__file__)),
        "reused_loco_generator_sha256": file_sha256(Path(loco.__file__)),
        "random_state": int(random_state),
        "estimator_n_jobs": 1,
    }
    checkpoint_dir = output_dir / ".strict_temporal_loco_checkpoints"
    forecast, nowcast, fold_audit, skipped = run_strict_predictions(
        forecasting,
        nowcasting,
        general_params,
        phase3_params,
        countries,
        workers,
        random_state,
        checkpoint_dir,
        resume,
        manifest_base,
    )
    primary_metrics = calculate_micro_metrics(forecast, nowcast)
    area_metrics = calculate_area_metrics(forecast, nowcast)
    _, denominators = aggregate_area_metrics(area_metrics)
    production_run = countries is None
    if production_run:
        _validate_full_results(
            forecast,
            nowcast,
            area_metrics,
            primary_metrics,
            denominators,
            fold_audit,
            skipped,
        )
    source_context = {
        **manifest_base,
        "production_run": production_run,
        "source_rows": int(len(forecasting)),
        "pre_cutoff_rows": int(
            pd.to_datetime(forecasting["date"]).lt(CUTOFF).sum()
        ),
        "test_rows": int(len(forecast)),
        "test_area_count": int(forecast["area_id"].nunique()),
        "test_country_count": int(forecast["country_code_3"].nunique()),
        "skipped_country_count": int(len(skipped)),
        "test_key_sha256": canonical_key_sha256(forecast),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "matplotlib_version": mpl.__version__,
        "workers": int(workers),
        "fewsnet_interpretation": (
            "formal_labels_held_out_but_fews_ipc_ha_available"
        ),
        "complete_country_cold_start_claim_allowed": False,
        "nowcasting_interpretation": (
            "retrospective_same_month_not_vintage_faithful"
        ),
        "nested_tuning": False,
        "primary_metric_aggregation": "pooled_observation_micro",
        "primary_metric_observation_count_per_model": int(len(forecast)),
        "r2_actual_precision": "rounded_two_decimal",
        "r2_prediction_precision": "rounded_two_decimal",
    }
    artifacts = {
        "micro_metrics": primary_metrics,
        "forecasting_predictions": forecast,
        "nowcasting_predictions": nowcast,
        "area_metrics": area_metrics,
        "area_macro_denominators": denominators,
        "fold_audit": fold_audit,
        "skipped_folds": skipped,
    }
    return _write_artifact_family(output_dir, artifacts, source_context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasting-input", type=Path, default=loco.DEFAULT_FORECASTING_INPUT
    )
    parser.add_argument(
        "--nowcasting-input", type=Path, default=loco.DEFAULT_NOWCASTING_INPUT
    )
    parser.add_argument(
        "--country-lookup", type=Path, default=loco.DEFAULT_COUNTRY_LOOKUP
    )
    parser.add_argument("--general-params", type=Path, default=loco.DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=loco.DEFAULT_PHASE3_PARAMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help=(
            "Replace only the primary pooled-micro table from saved strict LOCO "
            "predictions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate_existing:
        path = aggregate_existing_strict_predictions(args.output_dir)
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
        resume=args.resume,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
