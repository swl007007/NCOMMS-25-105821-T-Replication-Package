"""Generate strict-temporal random new-area five-fold robustness results."""

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


CUTOFF = pd.Timestamp("2022-01-01")
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
DEFAULT_FORECASTING_INPUT = loco.DEFAULT_FORECASTING_INPUT
DEFAULT_NOWCASTING_INPUT = loco.DEFAULT_NOWCASTING_INPUT
DEFAULT_CANONICAL_TEST = SOURCE_DATA_DIR / "All_prediction.csv"
DEFAULT_COUNTRY_LOOKUP = loco.DEFAULT_COUNTRY_LOOKUP
DEFAULT_GENERAL_PARAMS = loco.DEFAULT_GENERAL_PARAMS
DEFAULT_PHASE3_PARAMS = loco.DEFAULT_PHASE3_PARAMS
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
FINAL_FILENAMES = {
    "folds": "leave_area_out_20pct_fivefold_area_folds.csv",
    "forecasting_predictions": "leave_area_out_20pct_fivefold_forecasting_predictions.csv",
    "nowcasting_predictions": "leave_area_out_20pct_fivefold_nowcasting_predictions.csv",
    "fold_metrics": "leave_area_out_20pct_fivefold_fold_metrics.csv",
    "metrics": "leave_area_out_20pct_fivefold_metrics.csv",
    "source_audit": "leave_area_out_20pct_fivefold_source_audit.csv",
}


def build_area_folds(canonical_test: pd.DataFrame) -> pd.DataFrame:
    """Assign sorted canonical-test areas to the fixed five folds."""
    if "area_id" not in canonical_test:
        raise ValueError("Canonical test data are missing area_id.")
    areas = np.sort(canonical_test["area_id"].dropna().unique())
    fold_ids = np.empty(len(areas), dtype=int)
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    for fold_id, (_, held_index) in enumerate(splitter.split(areas)):
        fold_ids[held_index] = fold_id
    return pd.DataFrame({"area_id": areas, "fold_id": fold_ids})


def strict_temporal_area_masks(
    data: pd.DataFrame,
    held_areas: set[int],
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Keep pre-cutoff non-held rows and post-cutoff held rows only."""
    if not {"area_id", "date"}.issubset(data.columns):
        raise ValueError("Strict split data require area_id and date.")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    train_source = frame["date"].lt(cutoff) & ~frame["area_id"].isin(held_areas)
    test_source = frame["date"].ge(cutoff) & frame["area_id"].isin(held_areas)
    frame = frame.loc[train_source | test_source].reset_index(drop=True)
    return frame, frame["date"].lt(cutoff), frame["date"].ge(cutoff)


def _normalized_keys(data: pd.DataFrame, name: str) -> pd.DataFrame:
    loco._require_columns(data, loco.KEY_COLUMNS)
    keys = data[loco.KEY_COLUMNS].copy()
    if keys.isna().any().any():
        raise ValueError(f"{name} contains missing canonical keys.")
    keys["area_id"] = keys["area_id"].astype(int)
    keys["date"] = pd.to_datetime(keys["date"], errors="raise").dt.normalize()
    if keys.duplicated(loco.KEY_COLUMNS).any():
        raise ValueError(f"{name} contains duplicate canonical keys.")
    return keys.sort_values(loco.KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def _key_set(data: pd.DataFrame) -> set[tuple[object, object]]:
    return set(data[loco.KEY_COLUMNS].itertuples(index=False, name=None))


def _validate_prediction_coverage(
    predictions: pd.DataFrame,
    canonical_test: pd.DataFrame,
    model_name: str,
) -> None:
    if len(predictions) != len(canonical_test):
        raise ValueError(
            f"{model_name} has {len(predictions)} OOF predictions; expected {len(canonical_test)}."
        )
    if predictions.duplicated(loco.KEY_COLUMNS).any():
        raise ValueError(f"{model_name} OOF predictions contain duplicate keys.")
    if _key_set(predictions) != _key_set(canonical_test):
        raise ValueError(f"{model_name} OOF prediction keys do not match the canonical test set.")


def run_fivefold_predictions(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    canonical_test: pd.DataFrame,
    general_params: dict[str, object],
    phase3_params: dict[str, object],
    forecasting_runner: Callable[..., pd.DataFrame] = loco.fit_forecasting_split,
    nowcasting_runner: Callable[..., pd.DataFrame] = loco.fit_nowcasting_split,
    cutoff: pd.Timestamp = CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit each strict-temporal held-area fold and assemble OOF predictions."""
    canonical_test = _normalized_keys(canonical_test, "Canonical test data")
    if canonical_test.empty or canonical_test["date"].lt(cutoff).any():
        raise ValueError("Canonical test data must contain non-empty post-cutoff keys only.")
    forecasting = loco.add_cumulative_targets(forecasting)
    folds = build_area_folds(canonical_test)
    canonical_with_folds = canonical_test.merge(
        folds, on="area_id", how="inner", validate="many_to_one"
    )
    if len(canonical_with_folds) != len(canonical_test):
        raise ValueError("Canonical areas are missing a fold assignment.")

    general_params = {**general_params, "random_state": 0, "n_jobs": 1}
    phase3_params = {**phase3_params, "random_state": 0, "n_jobs": 1}
    forecast_parts: list[pd.DataFrame] = []
    nowcast_parts: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []
    for fold_id in range(5):
        held_areas = set(
            folds.loc[folds["fold_id"].eq(fold_id), "area_id"].astype(int)
        )
        expected_test = canonical_with_folds.loc[
            canonical_with_folds["fold_id"].eq(fold_id), loco.KEY_COLUMNS
        ]
        forecast_frame, forecast_train, forecast_test = strict_temporal_area_masks(
            forecasting, held_areas, cutoff
        )
        nowcast_frame, nowcast_train, nowcast_test = strict_temporal_area_masks(
            nowcasting, held_areas, cutoff
        )
        for label, left, right in (
            ("train", forecast_frame.loc[forecast_train], nowcast_frame.loc[nowcast_train]),
            ("test", forecast_frame.loc[forecast_test], nowcast_frame.loc[nowcast_test]),
        ):
            if _key_set(left) != _key_set(right):
                raise ValueError(f"Forecasting and nowcasting {label} keys differ in fold {fold_id}.")
        if _key_set(forecast_frame.loc[forecast_test]) != _key_set(expected_test):
            raise ValueError(f"Fold {fold_id} test keys do not match the canonical test set.")

        forecast_predictions = forecasting_runner(
            forecast_frame,
            forecast_train,
            forecast_test,
            fold_id,
            dict(general_params),
            dict(phase3_params),
            fold_column="fold_id",
        )
        nowcast_predictions = nowcasting_runner(
            forecast_frame,
            nowcast_frame,
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
                    "n_train_areas": int(
                        forecast_frame.loc[forecast_train, "area_id"].nunique()
                    ),
                    "n_test_areas": int(len(held_areas)),
                    "train_excludes_held_areas": not forecast_frame.loc[
                        forecast_train, "area_id"
                    ].isin(held_areas).any(),
                    "train_dates_before_cutoff": bool(
                        forecast_frame.loc[forecast_train, "date"].lt(cutoff).all()
                    ),
                    "test_dates_on_or_after_cutoff": bool(
                        forecast_frame.loc[forecast_test, "date"].ge(cutoff).all()
                    ),
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
    _validate_prediction_coverage(forecast, canonical_test, "Forecasting")
    _validate_prediction_coverage(nowcast, canonical_test, "Nowcasting")
    if _key_set(forecast) != _key_set(nowcast):
        raise ValueError("Forecasting and nowcasting OOF keys differ.")
    metrics = pd.DataFrame(metric_records).sort_values(
        ["model", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)
    return forecast, nowcast, metrics


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_key_sha256(data: pd.DataFrame) -> str:
    keys = _normalized_keys(data, "Keys")
    keys["date"] = keys["date"].dt.strftime("%Y-%m-%d")
    return hashlib.sha256(
        keys.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _validate_canonical_sources(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    canonical_test: pd.DataFrame,
) -> pd.DataFrame:
    canonical = _normalized_keys(canonical_test, "Canonical test data")
    if canonical.empty or canonical["date"].lt(CUTOFF).any():
        raise ValueError("Canonical test data must contain post-cutoff keys only.")
    for name, data in (("Forecasting", forecasting), ("Nowcasting", nowcasting)):
        if data[loco.OUTCOME_COLUMNS].isna().any().any():
            raise ValueError(f"{name} input has missing outcome values.")
        observed = _normalized_keys(data, f"{name} input")
        observed = observed.loc[observed["date"].ge(CUTOFF)].reset_index(drop=True)
        if not observed.equals(canonical):
            raise ValueError(f"{name} post-cutoff keys do not match All_prediction.csv.")
    return canonical


def summarize_metrics(
    fold_metrics: pd.DataFrame,
    forecast: pd.DataFrame,
    nowcast: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate pooled OOF primary metrics plus fold stability summaries."""
    rows = []
    for model_name, predictions in (("Forecasting", forecast), ("Nowcasting", nowcast)):
        model_folds = fold_metrics.loc[fold_metrics["model"].eq(model_name)]
        if len(model_folds) != 5:
            raise ValueError(f"{model_name} must have exactly five fold-metric rows.")
        row = {
            **area_holdout.calculate_pooled_metrics(predictions, model_name),
            "aggregation": "pooled_oof",
        }
        for metric in (
            "phase3plus_precision",
            "phase3plus_recall",
            "overall_accuracy",
            "phase3plus_r2",
        ):
            row[f"fold_mean_{metric}"] = model_folds[metric].mean()
            row[f"fold_sd_{metric}"] = model_folds[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _protected_10pct_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.name: _file_sha256(path)
        for path in sorted(output_dir.glob("leave_area_out_10pct_*"))
        if path.is_file()
    }


def _normalize_prediction_dates(predictions: pd.DataFrame) -> pd.DataFrame:
    normalized = predictions.copy()
    normalized["area_id"] = normalized["area_id"].astype(int)
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.normalize()
    return normalized


def _raise_if_frames_differ(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    message: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(message) from error


def _validate_saved_artifacts(
    paths: dict[str, Path],
    canonical_test: pd.DataFrame,
) -> None:
    canonical = _normalized_keys(canonical_test, "Canonical test data")
    folds = pd.read_csv(paths["folds"])
    expected_folds = build_area_folds(canonical)
    _raise_if_frames_differ(
        folds.sort_values("area_id", kind="mergesort").reset_index(drop=True),
        expected_folds,
        "Saved fold assignments differ from the fixed KFold split.",
    )
    expected_assignments = canonical.merge(
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
        _validate_prediction_coverage(predictions, canonical, model_name)
        assignment = predictions[[*loco.KEY_COLUMNS, "fold_id"]].sort_values(
            loco.KEY_COLUMNS, kind="mergesort"
        ).reset_index(drop=True)
        _raise_if_frames_differ(
            assignment,
            expected_assignments,
            f"Saved {model_name} fold assignments do not match the fold file.",
        )

    saved_fold_metrics = pd.read_csv(paths["fold_metrics"], float_precision="round_trip")
    required_fold_columns = [
        "model",
        "fold_id",
        "n_train",
        "n_train_areas",
        "n_test_areas",
        "train_excludes_held_areas",
        "train_dates_before_cutoff",
        "test_dates_on_or_after_cutoff",
    ]
    loco._require_columns(saved_fold_metrics, required_fold_columns)
    if len(saved_fold_metrics) != 10 or not saved_fold_metrics[
        [
            "train_excludes_held_areas",
            "train_dates_before_cutoff",
            "test_dates_on_or_after_cutoff",
        ]
    ].all().all():
        raise ValueError("Saved fold metrics contain an isolation failure.")
    recomputed_fold_rows = []
    for model_name, predictions in predictions_by_model.items():
        for fold_id, group in predictions.groupby("fold_id", sort=True):
            metric = area_holdout.calculate_pooled_metrics(group, model_name)
            metric["fold_id"] = fold_id
            recomputed_fold_rows.append(metric)
    recomputed_fold_metrics = pd.DataFrame(recomputed_fold_rows).sort_values(
        ["model", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)
    metric_columns = recomputed_fold_metrics.columns.tolist()
    _raise_if_frames_differ(
        saved_fold_metrics[metric_columns]
        .sort_values(["model", "fold_id"], kind="mergesort")
        .reset_index(drop=True),
        recomputed_fold_metrics,
        "Saved fold metrics do not match predictions.",
    )

    saved_summary = pd.read_csv(paths["metrics"], float_precision="round_trip")
    recomputed_summary = summarize_metrics(
        saved_fold_metrics,
        predictions_by_model["Forecasting"],
        predictions_by_model["Nowcasting"],
    )
    _raise_if_frames_differ(
        saved_summary,
        recomputed_summary,
        "Saved summary metrics do not match predictions and fold metrics.",
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
    canonical_test_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    forecasting_source: pd.DataFrame,
    canonical_test: pd.DataFrame,
    folds: pd.DataFrame,
    forecast: pd.DataFrame,
    nowcast: pd.DataFrame,
    protected_hashes: dict[str, str],
) -> pd.DataFrame:
    layer1_features = loco.select_layer1_features(
        loco.add_cumulative_targets(forecasting_source)
    )
    return pd.DataFrame(
        [
            {
                "evaluation_protocol": "strict_temporal_random_new_area_fivefold",
                "temporal_interpretation": "pre_cutoff_nonheld_areas_train_postcutoff_held_areas_test",
                "cutoff": CUTOFF.strftime("%Y-%m-%d"),
                "fold_assignment": "sorted_area_id_KFold_5_shuffle_true_random_state_0",
                "random_state": 0,
                "estimator_n_jobs": 1,
                "canonical_test_rows": int(len(canonical_test)),
                "canonical_test_areas": int(canonical_test["area_id"].nunique()),
                "canonical_test_key_sha256": _canonical_key_sha256(canonical_test),
                "fold_area_counts": json.dumps(
                    folds["fold_id"].value_counts().sort_index().to_dict(), sort_keys=True
                ),
                "fews_ipc_ha_in_layer1": "fews_ipc_ha" in layer1_features,
                "forecast_source_phase_disagreement_count": int(
                    forecast["source_overall_phase"].ne(forecast["overall_phase"]).sum()
                ),
                "nowcast_source_phase_disagreement_count": int(
                    nowcast["source_overall_phase"].ne(nowcast["overall_phase"]).sum()
                ),
                "forecasting_input_path": str(forecasting_path.resolve()),
                "nowcasting_input_path": str(nowcasting_path.resolve()),
                "canonical_test_path": str(canonical_test_path.resolve()),
                "country_lookup_path": str(country_lookup_path.resolve()),
                "general_params_path": str(general_params_path.resolve()),
                "phase3_params_path": str(phase3_params_path.resolve()),
                "forecasting_input_sha256": _file_sha256(forecasting_path),
                "nowcasting_input_sha256": _file_sha256(nowcasting_path),
                "canonical_test_sha256": _file_sha256(canonical_test_path),
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
    canonical_test_path: Path = DEFAULT_CANONICAL_TEST,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = DEFAULT_PHASE3_PARAMS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    forecasting_runner: Callable[..., pd.DataFrame] = loco.fit_forecasting_split,
    nowcasting_runner: Callable[..., pd.DataFrame] = loco.fit_nowcasting_split,
) -> dict[str, Path]:
    """Run the fixed five-fold evaluation and write its six CSV artifacts."""
    forecasting_path = Path(forecasting_path)
    nowcasting_path = Path(nowcasting_path)
    canonical_test_path = Path(canonical_test_path)
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
    canonical_test = _validate_canonical_sources(
        forecasting, nowcasting, pd.read_csv(canonical_test_path)
    )
    if canonical_test_path.resolve() == DEFAULT_CANONICAL_TEST.resolve() and (
        len(canonical_test) != 1170 or canonical_test["area_id"].nunique() != 646
    ):
        raise ValueError("The default canonical test set must contain 1,170 rows and 646 areas.")
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path, phase3_params_path, random_state=0, estimator_n_jobs=1
    )
    forecast, nowcast, fold_metrics = run_fivefold_predictions(
        forecasting,
        nowcasting,
        canonical_test,
        general_params,
        phase3_params,
        forecasting_runner=forecasting_runner,
        nowcasting_runner=nowcasting_runner,
    )
    folds = build_area_folds(canonical_test)
    summary = summarize_metrics(fold_metrics, forecast, nowcast)
    paths = {artifact: output_dir / filename for artifact, filename in FINAL_FILENAMES.items()}
    folds.to_csv(paths["folds"], index=False)
    forecast.to_csv(paths["forecasting_predictions"], index=False, float_format="%.17g")
    nowcast.to_csv(paths["nowcasting_predictions"], index=False, float_format="%.17g")
    fold_metrics.to_csv(paths["fold_metrics"], index=False, float_format="%.17g")
    summary.to_csv(paths["metrics"], index=False, float_format="%.17g")
    _source_audit(
        paths,
        forecasting_path,
        nowcasting_path,
        canonical_test_path,
        country_lookup_path,
        general_params_path,
        phase3_params_path,
        forecasting,
        canonical_test,
        folds,
        forecast,
        nowcast,
        protected_hashes,
    ).to_csv(paths["source_audit"], index=False)
    _validate_saved_artifacts(paths, canonical_test)
    if _protected_10pct_hashes(output_dir) != protected_hashes:
        raise ValueError("A protected leave_area_out_10pct artifact changed.")
    return paths


if __name__ == "__main__":
    for artifact, path in run_analysis().items():
        print(f"{artifact}: {path}")
