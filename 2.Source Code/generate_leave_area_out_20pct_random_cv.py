"""Generate all-date random-area five-fold cross-validation results."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import KFold


SOURCE_CODE_DIR = Path(__file__).resolve().parent
if str(SOURCE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE_DIR))

import generate_leave_area_out_10pct_robustness as area_holdout
import generate_leave_one_country_out_robustness as loco


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECASTING_INPUT = loco.DEFAULT_FORECASTING_INPUT
DEFAULT_NOWCASTING_INPUT = loco.DEFAULT_NOWCASTING_INPUT
DEFAULT_COUNTRY_LOOKUP = loco.DEFAULT_COUNTRY_LOOKUP
DEFAULT_GENERAL_PARAMS = loco.DEFAULT_GENERAL_PARAMS
DEFAULT_PHASE3_PARAMS = loco.DEFAULT_PHASE3_PARAMS
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
FINAL_FILENAMES = {
    "folds": "leave_area_out_20pct_random_cv_area_folds.csv",
    "forecasting_predictions": "leave_area_out_20pct_random_cv_forecasting_predictions.csv",
    "nowcasting_predictions": "leave_area_out_20pct_random_cv_nowcasting_predictions.csv",
    "fold_metrics": "leave_area_out_20pct_random_cv_fold_metrics.csv",
    "metrics": "leave_area_out_20pct_random_cv_metrics.csv",
    "source_audit": "leave_area_out_20pct_random_cv_source_audit.csv",
}
SUMMARY_COLUMNS = (
    "n_test",
    "n_test_areas",
    "phase3plus_precision",
    "phase3plus_recall",
    "overall_accuracy",
    "phase3plus_r2",
)


def build_area_folds(data: pd.DataFrame) -> pd.DataFrame:
    """Assign every source-data area to one fixed random fold."""
    loco._require_columns(data, ["area_id"])
    areas = np.sort(data["area_id"].dropna().astype(int).unique())
    if len(areas) < 5:
        raise ValueError("Random five-fold CV requires at least five areas.")
    fold_ids = np.empty(len(areas), dtype=int)
    for fold_id, (_, held_index) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=0).split(areas)
    ):
        fold_ids[held_index] = fold_id
    return pd.DataFrame({"area_id": areas, "fold_id": fold_ids})


def _normalized_keys(data: pd.DataFrame, name: str) -> pd.DataFrame:
    loco._require_columns(data, loco.KEY_COLUMNS)
    keys = data[loco.KEY_COLUMNS].copy()
    if keys.isna().any().any():
        raise ValueError(f"{name} contains missing keys.")
    keys["area_id"] = keys["area_id"].astype(int)
    keys["date"] = pd.to_datetime(keys["date"], errors="raise").dt.normalize()
    if keys.duplicated(loco.KEY_COLUMNS).any():
        raise ValueError(f"{name} contains duplicate keys.")
    return keys.sort_values(loco.KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def _key_set(data: pd.DataFrame) -> set[tuple[object, object]]:
    return set(_normalized_keys(data, "Keys").itertuples(index=False, name=None))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_sha256(data: pd.DataFrame) -> str:
    keys = _normalized_keys(data, "Keys")
    keys["date"] = keys["date"].dt.strftime("%Y-%m-%d")
    return hashlib.sha256(
        keys.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _protected_10pct_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.name: _file_sha256(path)
        for path in sorted(output_dir.glob("leave_area_out_10pct_*"))
        if path.is_file()
    }


def _validate_prediction_coverage(
    predictions: pd.DataFrame, source: pd.DataFrame, model_name: str
) -> None:
    expected = _normalized_keys(source, "Source data")
    observed = _normalized_keys(predictions, f"{model_name} predictions")
    if not observed.equals(expected):
        raise ValueError(f"{model_name} prediction keys do not match the source data.")


def run_fivefold_predictions(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    forecasting_runner: Callable[..., pd.DataFrame] = loco.fit_forecasting_split,
    nowcasting_runner: Callable[..., pd.DataFrame] = loco.fit_nowcasting_split,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit both models on all-date random area folds."""
    if _key_set(forecasting) != _key_set(nowcasting):
        raise ValueError("Forecasting and nowcasting source keys differ.")
    source_keys = _normalized_keys(forecasting, "Forecasting source")
    folds = build_area_folds(forecasting)
    assignments = source_keys.merge(folds, on="area_id", how="inner", validate="many_to_one")
    if len(assignments) != len(source_keys):
        raise ValueError("Source areas are missing fold assignments.")

    forecasting = loco.add_cumulative_targets(forecasting)
    general_params = {**general_params, "random_state": 0, "n_jobs": 1}
    phase3_params = {**phase3_params, "random_state": 0, "n_jobs": 1}
    forecast_parts: list[pd.DataFrame] = []
    nowcast_parts: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []
    for fold_id in range(5):
        held_areas = set(folds.loc[folds["fold_id"].eq(fold_id), "area_id"])
        forecast_test = forecasting["area_id"].isin(held_areas)
        nowcast_test = nowcasting["area_id"].isin(held_areas)
        forecast_train = ~forecast_test
        nowcast_train = ~nowcast_test
        expected_test = assignments.loc[
            assignments["fold_id"].eq(fold_id), loco.KEY_COLUMNS
        ]
        for label, left, right in (
            ("train", forecasting.loc[forecast_train], nowcasting.loc[nowcast_train]),
            ("test", forecasting.loc[forecast_test], nowcasting.loc[nowcast_test]),
        ):
            if _key_set(left) != _key_set(right):
                raise ValueError(f"Forecasting and nowcasting {label} keys differ in fold {fold_id}.")
        if _key_set(forecasting.loc[forecast_test]) != _key_set(expected_test):
            raise ValueError(f"Forecasting test keys differ from fold {fold_id} assignments.")

        forecast_predictions = forecasting_runner(
            forecasting,
            forecast_train,
            forecast_test,
            fold_id,
            dict(general_params),
            dict(phase3_params),
            fold_column="fold_id",
        )
        nowcast_predictions = nowcasting_runner(
            forecasting,
            nowcasting,
            forecast_train,
            forecast_test,
            nowcast_train,
            nowcast_test,
            fold_id,
            dict(general_params),
            dict(phase3_params),
            fold_column="fold_id",
        )
        for model_name, predictions in (
            ("Forecasting", forecast_predictions),
            ("Nowcasting", nowcast_predictions),
        ):
            loco._require_columns(predictions, [*loco.KEY_COLUMNS, "fold_id"])
            if not predictions["fold_id"].eq(fold_id).all():
                raise ValueError(f"{model_name} predictions have an incorrect fold id.")
            if _key_set(predictions) != _key_set(expected_test):
                raise ValueError(f"{model_name} prediction keys differ in fold {fold_id}.")
            metric = area_holdout.calculate_pooled_metrics(predictions, model_name)
            metric.update(
                {
                    "fold_id": fold_id,
                    "n_train": int(forecast_train.sum()),
                    "n_train_areas": int(forecasting.loc[forecast_train, "area_id"].nunique()),
                    "n_test_areas": int(len(held_areas)),
                    "train_excludes_held_areas": not forecasting.loc[
                        forecast_train, "area_id"
                    ].isin(held_areas).any(),
                    "test_only_held_areas": forecasting.loc[
                        forecast_test, "area_id"
                    ].isin(held_areas).all(),
                }
            )
            metric_records.append(metric)
        forecast_parts.append(forecast_predictions)
        nowcast_parts.append(nowcast_predictions)

    forecast = pd.concat(forecast_parts, ignore_index=True).sort_values(
        loco.KEY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    nowcast = pd.concat(nowcast_parts, ignore_index=True).sort_values(
        loco.KEY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    _validate_prediction_coverage(forecast, forecasting, "Forecasting")
    _validate_prediction_coverage(nowcast, nowcasting, "Nowcasting")
    if _key_set(forecast) != _key_set(nowcast):
        raise ValueError("Forecasting and nowcasting prediction keys differ.")
    metrics = pd.DataFrame(metric_records).sort_values(
        ["model", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)
    return folds, forecast, nowcast, metrics


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Report fold means and sample standard deviations, never pooled metrics."""
    rows = []
    for model_name in ("Forecasting", "Nowcasting"):
        model_folds = fold_metrics.loc[fold_metrics["model"].eq(model_name)]
        if len(model_folds) != 5 or model_folds["fold_id"].nunique() != 5:
            raise ValueError(f"{model_name} must have exactly five folds.")
        row: dict[str, object] = {
            "model": model_name,
            "aggregation": "fold_mean_sample_sd",
            "n_folds": 5,
        }
        for column in SUMMARY_COLUMNS:
            row[f"fold_mean_{column}"] = model_folds[column].mean()
            row[f"fold_sd_{column}"] = model_folds[column].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_prediction_dates(predictions: pd.DataFrame) -> pd.DataFrame:
    normalized = predictions.copy()
    normalized["area_id"] = normalized["area_id"].astype(int)
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.normalize()
    return normalized


def _raise_if_frames_differ(
    observed: pd.DataFrame, expected: pd.DataFrame, message: str
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12
        )
    except AssertionError as error:
        raise ValueError(message) from error


def _validate_saved_artifacts(paths: dict[str, Path], source: pd.DataFrame) -> None:
    source_keys = _normalized_keys(source, "Source data")
    folds = pd.read_csv(paths["folds"])
    expected_folds = build_area_folds(source)
    _raise_if_frames_differ(
        folds.sort_values("area_id", kind="mergesort").reset_index(drop=True),
        expected_folds,
        "Saved fold assignments differ from the fixed KFold split.",
    )
    expected_assignments = source_keys.merge(
        expected_folds, on="area_id", how="inner", validate="many_to_one"
    ).sort_values(loco.KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    predictions_by_model = {
        "Forecasting": _normalize_prediction_dates(
            pd.read_csv(paths["forecasting_predictions"], float_precision="round_trip")
        ),
        "Nowcasting": _normalize_prediction_dates(
            pd.read_csv(paths["nowcasting_predictions"], float_precision="round_trip")
        ),
    }
    for model_name, predictions in predictions_by_model.items():
        _validate_prediction_coverage(predictions, source, model_name)
        assignment = predictions[[*loco.KEY_COLUMNS, "fold_id"]].sort_values(
            loco.KEY_COLUMNS, kind="mergesort"
        ).reset_index(drop=True)
        _raise_if_frames_differ(
            assignment,
            expected_assignments,
            f"Saved {model_name} fold assignments do not match the fold file.",
        )
    if _key_set(predictions_by_model["Forecasting"]) != _key_set(predictions_by_model["Nowcasting"]):
        raise ValueError("Saved forecasting and nowcasting prediction keys differ.")

    saved_fold_metrics = pd.read_csv(paths["fold_metrics"], float_precision="round_trip")
    required = [
        "model",
        "fold_id",
        "n_train",
        "n_train_areas",
        "n_test_areas",
        "train_excludes_held_areas",
        "test_only_held_areas",
    ]
    loco._require_columns(saved_fold_metrics, required)
    if len(saved_fold_metrics) != 10 or not saved_fold_metrics[
        ["train_excludes_held_areas", "test_only_held_areas"]
    ].all().all():
        raise ValueError("Saved fold metrics contain an area-isolation failure.")
    recomputed_rows = []
    for model_name, predictions in predictions_by_model.items():
        for fold_id, group in predictions.groupby("fold_id", sort=True):
            metric = area_holdout.calculate_pooled_metrics(group, model_name)
            metric["fold_id"] = fold_id
            recomputed_rows.append(metric)
    recomputed = pd.DataFrame(recomputed_rows).sort_values(
        ["model", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)
    metric_columns = recomputed.columns.tolist()
    _raise_if_frames_differ(
        saved_fold_metrics[metric_columns]
        .sort_values(["model", "fold_id"], kind="mergesort")
        .reset_index(drop=True),
        recomputed,
        "Saved fold metrics do not match predictions.",
    )
    expected_counts = expected_assignments.groupby("fold_id").agg(
        n_test=("area_id", "size"), n_test_areas=("area_id", "nunique")
    )
    total_areas = source_keys["area_id"].nunique()
    for row in saved_fold_metrics.itertuples(index=False):
        expected = expected_counts.loc[row.fold_id]
        if (
            row.n_test != expected.n_test
            or row.n_test_areas != expected.n_test_areas
            or row.n_train != len(source_keys) - expected.n_test
            or row.n_train_areas != total_areas - expected.n_test_areas
        ):
            raise ValueError("Saved fold row counts do not match the fold assignments.")

    saved_summary = pd.read_csv(paths["metrics"], float_precision="round_trip")
    _raise_if_frames_differ(
        saved_summary,
        summarize_metrics(saved_fold_metrics),
        "Saved summary metrics do not match fold metrics.",
    )
    audit = pd.read_csv(paths["source_audit"])
    if len(audit) != 1:
        raise ValueError("Source audit must contain exactly one row.")
    for artifact, path in paths.items():
        if artifact == "source_audit":
            continue
        column = f"{artifact}_sha256"
        if column not in audit or audit.loc[0, column] != _file_sha256(path):
            raise ValueError(f"Source audit hash mismatch for {artifact}.")


def _source_audit(
    paths: dict[str, Path],
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    source: pd.DataFrame,
    folds: pd.DataFrame,
    protected_hashes: dict[str, str],
) -> pd.DataFrame:
    assignments = _normalized_keys(source, "Source data").merge(
        folds, on="area_id", how="inner", validate="many_to_one"
    )
    return pd.DataFrame(
        [
            {
                "evaluation_protocol": "random_area_fivefold_all_dates",
                "temporal_interpretation": "all_dates_nonheld_areas_train_held_areas_test_not_out_of_time",
                "fold_assignment": "sorted_area_id_KFold_5_shuffle_true_random_state_0",
                "random_state": 0,
                "estimator_n_jobs": 1,
                "n_total": int(len(source)),
                "n_total_areas": int(source["area_id"].nunique()),
                "source_key_sha256": _key_sha256(source),
                "fold_area_counts": json.dumps(
                    folds["fold_id"].value_counts().sort_index().to_dict(), sort_keys=True
                ),
                "fold_row_counts": json.dumps(
                    assignments["fold_id"].value_counts().sort_index().to_dict(), sort_keys=True
                ),
                "fews_ipc_ha_in_layer1": "fews_ipc_ha" in source,
                "forecasting_input_path": str(forecasting_path.resolve()),
                "nowcasting_input_path": str(nowcasting_path.resolve()),
                "country_lookup_path": str(country_lookup_path.resolve()),
                "general_params_path": str(general_params_path.resolve()),
                "phase3_params_path": str(phase3_params_path.resolve()),
                "forecasting_input_sha256": _file_sha256(forecasting_path),
                "nowcasting_input_sha256": _file_sha256(nowcasting_path),
                "country_lookup_sha256": _file_sha256(country_lookup_path),
                "general_params_sha256": _file_sha256(general_params_path),
                "phase3_params_sha256": _file_sha256(phase3_params_path),
                "script_sha256": _file_sha256(Path(__file__)),
                "protected_leave_area_out_10pct_sha256": json.dumps(
                    protected_hashes, sort_keys=True
                ),
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "sklearn_version": sklearn.__version__,
                "xgboost_version": xgb.__version__,
                **{
                    f"{artifact}_path": str(path.resolve())
                    for artifact, path in paths.items()
                    if artifact != "source_audit"
                },
                **{
                    f"{artifact}_sha256": _file_sha256(path)
                    for artifact, path in paths.items()
                    if artifact != "source_audit"
                },
            }
        ]
    )


def run_analysis(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = DEFAULT_PHASE3_PARAMS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    forecasting_runner: Callable[..., pd.DataFrame] = loco.fit_forecasting_split,
    nowcasting_runner: Callable[..., pd.DataFrame] = loco.fit_nowcasting_split,
) -> dict[str, Path]:
    """Run the fixed all-date random-area CV and write six CSV artifacts."""
    forecasting_path = Path(forecasting_path)
    nowcasting_path = Path(nowcasting_path)
    country_lookup_path = Path(country_lookup_path)
    general_params_path = Path(general_params_path)
    phase3_params_path = Path(phase3_params_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protected_hashes = _protected_10pct_hashes(output_dir)
    lookup = loco.load_country_lookup(country_lookup_path)
    forecasting, nowcasting = loco.prepare_model_inputs(
        pd.read_csv(forecasting_path), pd.read_csv(nowcasting_path), lookup
    )
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path, phase3_params_path, random_state=0, estimator_n_jobs=1
    )
    folds, forecast, nowcast, fold_metrics = run_fivefold_predictions(
        forecasting,
        nowcasting,
        general_params,
        phase3_params,
        forecasting_runner=forecasting_runner,
        nowcasting_runner=nowcasting_runner,
    )
    paths = {artifact: output_dir / filename for artifact, filename in FINAL_FILENAMES.items()}
    folds.to_csv(paths["folds"], index=False)
    forecast.to_csv(paths["forecasting_predictions"], index=False, float_format="%.17g")
    nowcast.to_csv(paths["nowcasting_predictions"], index=False, float_format="%.17g")
    fold_metrics.to_csv(paths["fold_metrics"], index=False, float_format="%.17g")
    summarize_metrics(fold_metrics).to_csv(paths["metrics"], index=False, float_format="%.17g")
    _source_audit(
        paths,
        forecasting_path,
        nowcasting_path,
        country_lookup_path,
        general_params_path,
        phase3_params_path,
        forecasting,
        folds,
        protected_hashes,
    ).to_csv(paths["source_audit"], index=False)
    _validate_saved_artifacts(paths, forecasting)
    if _protected_10pct_hashes(output_dir) != protected_hashes:
        raise ValueError("A protected leave_area_out_10pct artifact changed.")
    return paths


if __name__ == "__main__":
    for artifact, path in run_analysis().items():
        print(f"{artifact}: {path}")
