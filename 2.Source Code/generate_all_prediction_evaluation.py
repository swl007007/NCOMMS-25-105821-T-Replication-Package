"""Evaluate forecasting, nowcasting, and contemporaneous predictions.

Forecasting and Nowcasting use the canonical 1,170-row 2022 temporal holdout.
Contemporaneous follows ``Table1_Contemporaneous_main.ipynb``: current-month
predictors are evaluated by random five-fold row-level cross-validation. The
saved Contemporaneous sidecar contains one reproducible out-of-fold prediction
for every one of the 5,575 source rows. Metrics across these protocols are
descriptive and are not directly comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "1.Source Data" / "All_prediction.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
DEFAULT_CONTEMPORANEOUS_SOURCE_PATH = (
    REPO_ROOT / "1.Source Data" / "Nowcasting_Analysis_010825.csv"
)
DEFAULT_CONTEMPORANEOUS_PARAMS_PATH = (
    REPO_ROOT / "2.Source Code" / "contemporaneous_hyperparameters.json"
)
DEFAULT_SHEET = "All_prediction_cleaned"
EXPECTED_ROWS = 1170
EXPECTED_SOURCE_ROWS = 5575
EXPECTED_CONTEMPORANEOUS_AREAS = 1198
EXPECTED_CONTEMPORANEOUS_FOLDS = 5
EXPECTED_ROWS_PER_FOLD = 1115
RESTORED_TEST_INDICES = {3374, 3517, 3534, 3553, 3567}
DEFAULT_RANDOM_STATE = 0

TRUE_COLUMN = "overall_phase"
PREDICTION_COLUMNS = {
    "Forecasting": "overall_phase_pred",
    "Nowcasting": "nowcast_predict",
    "Contemporaneous": "contemporaneous_predict",
}
BASE_CSV_PREDICTION_COLUMNS = {
    "Forecasting": "overall_phase_pred",
    "Nowcasting": "nowcast_predict",
}
EXCEL_TRUE_COLUMN = ("IPC Offical", "Overall Phase")
EXCEL_PREDICTION_COLUMNS = {
    "Forecasting": ("12 months Forecasting", "Overall Phase"),
    "Nowcasting": ("Nowcasting", "Overall Phase"),
}
ALL_LABELS = [1, 2, 3, 4, 5]
TASK_COLORS = {
    "Forecasting": "#1F77B4",
    "Nowcasting": "#E69F00",
    "Contemporaneous": "#009E73",
}
TASK_MARKERS = {"Forecasting": "o", "Nowcasting": "s", "Contemporaneous": "D"}
CONTEMPORANEOUS_TARGETS = {
    2: "phase2_worse",
    3: "phase3_worse",
    4: "phase4_worse",
    5: "phase5_worse",
}
CONTEMPORANEOUS_KEY_COLUMNS = ["source_row_index", "area_id", "date"]
CONTEMPORANEOUS_PREDICTION_COLUMNS = [
    *CONTEMPORANEOUS_KEY_COLUMNS,
    "fold",
    "source_overall_phase",
    TRUE_COLUMN,
    *[
        column
        for phase in range(2, 6)
        for column in (f"phase{phase}_actual", f"phase{phase}_contemporaneous")
    ],
    "contemporaneous_predict",
    "evaluation_protocol",
    "evaluation_population",
    "shuffle_seed",
    "estimator_random_state",
]
CONTEMPORANEOUS_PREDICTION_STEM = (
    "all_prediction_contemporaneous_random_cv_predictions"
)
CONTEMPORANEOUS_AUDIT_STEM = (
    "all_prediction_contemporaneous_random_cv_source_audit"
)
DEFAULT_CONTEMPORANEOUS_PREDICTIONS_PATH = (
    DEFAULT_OUTPUT_DIR / f"{CONTEMPORANEOUS_PREDICTION_STEM}.csv"
)


def apply_figure_style(*, frame: str = "open") -> None:
    """Apply the repository's publication-ready figure mechanics."""
    if frame not in {"open", "boxed", "none"}:
        raise ValueError(f"Unsupported frame style: {frame}")
    boxed = frame == "boxed"
    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.linewidth": 0.6,
            "axes.spines.top": boxed,
            "axes.spines.right": boxed,
            "axes.spines.left": frame != "none",
            "axes.spines.bottom": frame != "none",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_date(values: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(values, errors="raise")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError("Prediction dates must be timezone-naive.")
    if not bool(parsed.eq(parsed.dt.normalize()).all()):
        raise ValueError("Prediction dates must contain midnight values.")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalized_contemporaneous_keys(data: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(data, CONTEMPORANEOUS_KEY_COLUMNS)
    normalized = data.copy()
    source_row_index = pd.to_numeric(
        normalized["source_row_index"], errors="coerce"
    )
    if source_row_index.isna().any() or not np.allclose(
        source_row_index, source_row_index.round()
    ):
        raise ValueError(
            f"{name} source_row_index must contain complete integer values."
        )
    normalized["source_row_index"] = source_row_index.astype(int)
    if not normalized["source_row_index"].is_unique:
        raise ValueError(f"{name} source_row_index values are not unique.")
    normalized["date"] = _normalize_date(normalized["date"])
    if normalized[["area_id", "date"]].isna().any().any():
        raise ValueError(f"{name} contains missing population keys.")
    if normalized.duplicated(["area_id", "date"]).any():
        raise ValueError(f"{name} contains duplicate area-date keys.")
    return normalized


def _population_key_sha256(data: pd.DataFrame) -> str:
    normalized = _normalized_contemporaneous_keys(data, "Population hash input")
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


def _source_row_index_sha256(data: pd.DataFrame) -> str:
    normalized = _normalized_contemporaneous_keys(data, "Source-index hash input")
    ordered = normalized[["source_row_index"]].sort_values(
        "source_row_index", kind="mergesort"
    )
    payload = ordered.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fold_assignment_sha256(data: pd.DataFrame) -> str:
    normalized = _normalized_contemporaneous_keys(data, "Fold hash input")
    _require_columns(normalized, ["fold"])
    fold = pd.to_numeric(normalized["fold"], errors="coerce")
    if fold.isna().any() or not np.allclose(fold, fold.round()):
        raise ValueError("Fold assignments must contain complete integer values.")
    ordered = normalized.assign(fold=fold.astype(int))[
        ["source_row_index", "fold"]
    ].sort_values("source_row_index", kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _add_contemporaneous_targets(data: pd.DataFrame) -> pd.DataFrame:
    required = [f"phase{phase}_percent" for phase in range(1, 6)]
    _require_columns(data, required)
    result = data.copy()
    result["phase2_worse"] = (
        result["phase2_percent"]
        + result["phase3_percent"]
        + result["phase4_percent"]
        + result["phase5_percent"]
    )
    result["phase3_worse"] = (
        result["phase3_percent"]
        + result["phase4_percent"]
        + result["phase5_percent"]
    )
    result["phase4_worse"] = result["phase4_percent"] + result["phase5_percent"]
    result["phase5_worse"] = result["phase5_percent"]
    return result


def _phase_from_cumulative(data: pd.DataFrame, suffix: str) -> np.ndarray:
    required = [f"phase{phase}_{suffix}" for phase in range(2, 6)]
    _require_columns(data, required)
    return np.select(
        [
            data[f"phase5_{suffix}"].ge(0.20),
            data[f"phase4_{suffix}"].ge(0.20),
            data[f"phase3_{suffix}"].ge(0.20),
            data[f"phase2_{suffix}"].ge(0.20),
        ],
        [5, 4, 3, 2],
        default=1,
    ).astype(int)


def generate_contemporaneous_random_cv_predictions(
    source_path: Path = DEFAULT_CONTEMPORANEOUS_SOURCE_PATH,
    params_path: Path = DEFAULT_CONTEMPORANEOUS_PARAMS_PATH,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    estimator_n_jobs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate reproducible full-OOF random five-fold Contemporaneous predictions."""
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is required to generate contemporaneous random-CV predictions."
        ) from error

    if random_state is None:
        raise ValueError(
            "A fixed random_state is required for the reproducible random-CV rerun."
        )

    source_path = Path(source_path)
    params_path = Path(params_path)
    data = pd.read_csv(source_path)
    if len(data) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROWS:,} contemporaneous source rows, "
            f"found {len(data):,}."
        )
    if "source_row_index" in data:
        raise ValueError(
            "Contemporaneous source unexpectedly contains source_row_index."
        )
    data.insert(0, "source_row_index", np.arange(len(data), dtype=int))
    _require_columns(data, ["area_id", "date", TRUE_COLUMN, "phase1_percent"])
    data = data.loc[data["phase1_percent"].notna()].copy()
    if len(data) != EXPECTED_SOURCE_ROWS:
        raise ValueError("phase1_percent filtering changed the contemporaneous population.")
    data["date"] = _normalize_date(data["date"])
    data = data.sort_values(["area_id", "date"], kind="mergesort").reset_index(
        drop=True
    )
    source_phase = pd.to_numeric(data[TRUE_COLUMN], errors="coerce")
    if source_phase.isna().any() or not np.allclose(source_phase, source_phase.round()):
        raise ValueError("Source overall_phase must contain complete integer labels.")
    data = data.rename(columns={TRUE_COLUMN: "source_overall_phase"})
    data["source_overall_phase"] = source_phase.astype(int)
    data = _add_contemporaneous_targets(data)
    data["kfolds"] = -1
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    splitter = KFold(n_splits=EXPECTED_CONTEMPORANEOUS_FOLDS, shuffle=False)
    for fold, (_, validation_index) in enumerate(splitter.split(data)):
        data.loc[validation_index, "kfolds"] = fold
    fold_counts = data["kfolds"].value_counts().sort_index()
    if fold_counts.to_dict() != {
        fold: EXPECTED_ROWS_PER_FOLD
        for fold in range(EXPECTED_CONTEMPORANEOUS_FOLDS)
    }:
        raise ValueError(f"Unexpected random-CV fold sizes: {fold_counts.to_dict()}")

    excluded = {
        "source_row_index",
        "area_id",
        "date",
        "source_overall_phase",
        *[f"phase{phase}_percent" for phase in range(1, 6)],
        *CONTEMPORANEOUS_TARGETS.values(),
    }
    feature_columns = [column for column in data.columns if column not in excluded]
    if len(feature_columns) != 174 or "kfolds" not in feature_columns:
        raise ValueError(
            "Expected 174 contemporaneous predictors including kfolds, "
            f"found {len(feature_columns)}."
        )
    features = data[feature_columns].apply(pd.to_numeric, errors="raise")
    targets = data[list(CONTEMPORANEOUS_TARGETS.values())].apply(
        pd.to_numeric, errors="raise"
    )
    if targets.isna().any().any():
        raise ValueError("Contemporaneous cumulative targets contain missing values.")

    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["random_state"] = random_state
    if estimator_n_jobs is None:
        params.pop("n_jobs", None)
    else:
        params["n_jobs"] = estimator_n_jobs

    oof_frames: list[pd.DataFrame] = []
    for fold in range(EXPECTED_CONTEMPORANEOUS_FOLDS):
        train_mask = data["kfolds"].ne(fold)
        validation_mask = data["kfolds"].eq(fold)
        fold_predictions = data.loc[
            validation_mask,
            [*CONTEMPORANEOUS_KEY_COLUMNS, "kfolds", "source_overall_phase"],
        ].rename(columns={"kfolds": "fold"})
        for phase, target_column in CONTEMPORANEOUS_TARGETS.items():
            model = xgb.XGBRegressor(**params)
            model.fit(
                features.loc[train_mask], targets.loc[train_mask, target_column]
            )
            predicted = model.predict(features.loc[validation_mask])
            if not np.isfinite(predicted).all():
                raise ValueError(
                    f"Contemporaneous Phase {phase}+ predictions are non-finite."
                )
            fold_predictions[f"phase{phase}_actual"] = targets.loc[
                validation_mask, target_column
            ].to_numpy(dtype=float)
            fold_predictions[f"phase{phase}_contemporaneous"] = np.round(
                predicted, 2
            )
        fold_predictions[TRUE_COLUMN] = _phase_from_cumulative(
            fold_predictions, "actual"
        )
        fold_predictions["contemporaneous_predict"] = _phase_from_cumulative(
            fold_predictions, "contemporaneous"
        )
        fold_predictions["evaluation_protocol"] = "random_5fold_row_cv"
        fold_predictions["evaluation_population"] = (
            "random_5fold_full_oof_5575"
        )
        fold_predictions["shuffle_seed"] = random_state
        fold_predictions["estimator_random_state"] = random_state
        oof_frames.append(fold_predictions)

    predictions = _normalized_contemporaneous_keys(
        pd.concat(oof_frames, ignore_index=True), "Contemporaneous predictions"
    ).sort_values("source_row_index", kind="mergesort")
    predictions = predictions.loc[:, CONTEMPORANEOUS_PREDICTION_COLUMNS]
    if len(predictions) != EXPECTED_SOURCE_ROWS:
        raise ValueError("Contemporaneous OOF assembly changed the source population.")
    if set(predictions["source_row_index"]) != set(range(EXPECTED_SOURCE_ROWS)):
        raise ValueError("Contemporaneous OOF predictions do not cover every source row.")
    if predictions["area_id"].nunique() != EXPECTED_CONTEMPORANEOUS_AREAS:
        raise ValueError(
            f"Expected {EXPECTED_CONTEMPORANEOUS_AREAS} contemporaneous areas, "
            f"found {predictions['area_id'].nunique()}."
        )
    actual_source_disagreements = int(
        predictions[TRUE_COLUMN].ne(predictions["source_overall_phase"]).sum()
    )

    audit = pd.DataFrame(
        [
            {
                "model": "Contemporaneous",
                "evaluation_protocol": "random_5fold_row_cv",
                "evaluation_population": "random_5fold_full_oof_5575",
                "validation_design": "random_row_cv",
                "source_rows": len(data),
                "oof_rows": len(predictions),
                "test_rows": len(predictions),
                "test_areas": predictions["area_id"].nunique(),
                "n_splits": EXPECTED_CONTEMPORANEOUS_FOLDS,
                "fold_rows": "|".join(
                    str(int(value)) for value in fold_counts.tolist()
                ),
                "feature_count": len(feature_columns),
                "kfolds_predictor_included": True,
                "shuffle_seed": random_state,
                "kfold_shuffle": False,
                "fold_assignment_sha256": _fold_assignment_sha256(predictions),
                "source_row_index_sha256": _source_row_index_sha256(predictions),
                "population_key_sha256": _population_key_sha256(predictions),
                "target_contract": "phase2plus_phase3plus_phase4plus_phase5",
                "parameter_contract": (
                    "notebook_effective_general_params_all_targets"
                ),
                "hyperparameter_selection_provenance": (
                    "legacy_notebook_files_selection_sample_not_documented"
                ),
                "historical_shuffle_seed_recorded": False,
                "rerun_interpretation": (
                    "reproducible_random_cv_rerun_not_exact_historical_last_fold"
                ),
                "actual_phase_contract": "cumulative_share_threshold_0.20",
                "source_overall_phase_disagreement_rows": actual_source_disagreements,
                "phase_round_decimals": 2,
                "phase_threshold": 0.20,
                "random_state": random_state,
                "estimator_random_state": random_state,
                "estimator_n_jobs": (
                    estimator_n_jobs if estimator_n_jobs is not None else pd.NA
                ),
                "saved_metric_aggregation": "full_oof",
                "predictions_sha256": pd.NA,
                "source_sha256": _file_sha256(source_path),
                "params_sha256": _file_sha256(params_path),
                "generator_sha256": _file_sha256(Path(__file__)),
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "xgboost_version": xgb.__version__,
                "platform": platform.platform(),
            }
        ]
    )
    return predictions.reset_index(drop=True), audit


def write_contemporaneous_random_cv_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_path: Path = DEFAULT_CONTEMPORANEOUS_SOURCE_PATH,
    params_path: Path = DEFAULT_CONTEMPORANEOUS_PARAMS_PATH,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    estimator_n_jobs: int | None = None,
) -> dict[str, Path]:
    """Generate the random-CV OOF sidecar and its source audit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, audit = generate_contemporaneous_random_cv_predictions(
        source_path,
        params_path,
        random_state=random_state,
        estimator_n_jobs=estimator_n_jobs,
    )
    paths = {
        "contemporaneous_predictions_csv": output_dir
        / f"{CONTEMPORANEOUS_PREDICTION_STEM}.csv",
        "contemporaneous_audit_csv": output_dir
        / f"{CONTEMPORANEOUS_AUDIT_STEM}.csv",
    }
    predictions.to_csv(
        paths["contemporaneous_predictions_csv"], index=False, float_format="%.10f"
    )
    audit.loc[0, "predictions_sha256"] = _file_sha256(
        paths["contemporaneous_predictions_csv"]
    )
    audit.to_csv(paths["contemporaneous_audit_csv"], index=False)
    return paths


def load_contemporaneous_random_cv_predictions(
    predictions_path: Path = DEFAULT_CONTEMPORANEOUS_PREDICTIONS_PATH,
    *,
    expected_rows: int = EXPECTED_SOURCE_ROWS,
) -> pd.DataFrame:
    """Load and validate the reproducible full-OOF Contemporaneous sidecar."""
    predictions = _normalized_contemporaneous_keys(
        pd.read_csv(predictions_path), "Contemporaneous random-CV predictions"
    )
    _require_columns(predictions, CONTEMPORANEOUS_PREDICTION_COLUMNS)
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} Contemporaneous OOF rows, "
            f"found {len(predictions):,}."
        )
    fold = pd.to_numeric(predictions["fold"], errors="coerce")
    if fold.isna().any() or not np.allclose(fold, fold.round()):
        raise ValueError("Contemporaneous fold values must be complete integers.")
    predictions["fold"] = fold.astype(int)
    expected_folds = set(range(EXPECTED_CONTEMPORANEOUS_FOLDS))
    if set(predictions["fold"]) != expected_folds:
        raise ValueError("Contemporaneous OOF sidecar does not contain all five folds.")
    fold_counts = predictions["fold"].value_counts()
    expected_fold_size = expected_rows // EXPECTED_CONTEMPORANEOUS_FOLDS
    if expected_rows % EXPECTED_CONTEMPORANEOUS_FOLDS != 0 or not fold_counts.eq(
        expected_fold_size
    ).all():
        raise ValueError(
            "Contemporaneous OOF folds do not have equal expected row counts."
        )
    for column in (TRUE_COLUMN, "source_overall_phase", "contemporaneous_predict"):
        predictions[column] = _validated_labels(predictions, column)
    numeric_columns = [
        column
        for phase in range(2, 6)
        for column in (f"phase{phase}_actual", f"phase{phase}_contemporaneous")
    ]
    numeric = predictions[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Contemporaneous cumulative values must be finite.")
    predictions[numeric_columns] = numeric
    if set(predictions["evaluation_protocol"].astype(str)) != {
        "random_5fold_row_cv"
    }:
        raise ValueError("Contemporaneous evaluation protocol is not random five-fold CV.")
    if set(predictions["evaluation_population"].astype(str)) != {
        "random_5fold_full_oof_5575"
    }:
        raise ValueError("Contemporaneous evaluation population is not full OOF.")
    if set(pd.to_numeric(predictions["shuffle_seed"], errors="raise")) != {
        DEFAULT_RANDOM_STATE
    }:
        raise ValueError("Contemporaneous shuffle seed differs from the formal rerun.")
    return predictions.sort_values("source_row_index", kind="mergesort").reset_index(
        drop=True
    )


def _require_columns(data: pd.DataFrame, required: list[object]) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")


def _validated_labels(data: pd.DataFrame, column: object) -> pd.Series:
    values = pd.to_numeric(data[column], errors="raise")
    if values.isna().any():
        raise ValueError(f"Column {column} contains missing labels.")
    if not np.allclose(values.to_numpy(), values.round().to_numpy()):
        raise ValueError(f"Column {column} contains non-integer labels.")
    labels = values.astype(int)
    unexpected = sorted(set(labels).difference(ALL_LABELS))
    if unexpected:
        raise ValueError(f"Column {column} contains labels outside 1--5: {unexpected}")
    return labels


def load_predictions(
    input_path: Path = DEFAULT_INPUT_PATH,
    sheet_name: str = DEFAULT_SHEET,
    contemporaneous_predictions_path: Path | None = None,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Load the shared temporal-holdout truth and Forecasting/Nowcasting labels."""
    del contemporaneous_predictions_path
    input_path = Path(input_path)
    if input_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        data = pd.read_excel(input_path, sheet_name=sheet_name, header=[0, 1])
        _require_columns(data, [EXCEL_TRUE_COLUMN, *EXCEL_PREDICTION_COLUMNS.values()])
        if len(data) == 0:
            raise ValueError("Workbook contains no prediction rows.")
        uid_column = next(
            (column for column in data.columns if column[0] == "UID"), None
        )
        if uid_column is not None and (
            data[uid_column].isna().any() or not data[uid_column].is_unique
        ):
            raise ValueError("UID must be complete and unique for row-level alignment.")
        true_column = EXCEL_TRUE_COLUMN
        prediction_columns = EXCEL_PREDICTION_COLUMNS
    else:
        data = pd.read_csv(input_path)
        required = [
            "test_index",
            "area_id",
            "date",
            TRUE_COLUMN,
            *BASE_CSV_PREDICTION_COLUMNS.values(),
        ]
        _require_columns(data, required)
        if len(data) != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} prediction rows, found {len(data):,}."
            )
        if data["test_index"].isna().any() or not data["test_index"].is_unique:
            raise ValueError("test_index must be complete and unique.")
        if data[["area_id", "date"]].isna().any().any() or data.duplicated(
            ["area_id", "date"]
        ).any():
            raise ValueError("area_id and date must form complete unique keys.")
        observed_indices = set(
            pd.to_numeric(data["test_index"], errors="raise").astype(int)
        )
        missing_restored = sorted(RESTORED_TEST_INDICES.difference(observed_indices))
        if missing_restored:
            raise ValueError(
                f"Canonical predictions are missing restored indices: {missing_restored}"
            )
        true_column = TRUE_COLUMN
        prediction_columns = BASE_CSV_PREDICTION_COLUMNS

    y_true = _validated_labels(data, true_column)
    predictions = {
        task: _validated_labels(data, column)
        for task, column in prediction_columns.items()
    }
    return y_true, predictions


def _validated_label_series(values: object, name: str) -> pd.Series:
    series = pd.Series(values).reset_index(drop=True)
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not np.allclose(numeric, numeric.round()):
        raise ValueError(f"{name} must contain complete integer labels.")
    labels = numeric.astype(int)
    unexpected = sorted(set(labels).difference(ALL_LABELS))
    if unexpected:
        raise ValueError(f"{name} contains labels outside 1--5: {unexpected}")
    return labels


def calculate_task_metrics(
    evaluations: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Calculate metrics while retaining each task's own truth and population."""
    metric_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}

    for task, evaluation in evaluations.items():
        required = {
            "y_true",
            "y_pred",
            "evaluation_protocol",
            "evaluation_population",
        }
        missing = sorted(required.difference(evaluation))
        if missing:
            raise ValueError(f"{task} evaluation bundle is missing: {missing}")
        y_true = _validated_label_series(evaluation["y_true"], f"{task} truth")
        y_pred = _validated_label_series(evaluation["y_pred"], f"{task} predictions")
        if len(y_true) != len(y_pred) or len(y_true) == 0:
            raise ValueError(f"{task} truth and predictions must have equal non-zero length.")
        evaluation_protocol = str(evaluation["evaluation_protocol"])
        evaluation_population = str(evaluation["evaluation_population"])
        display_label = str(evaluation.get("display_label", task))
        support_counts = y_true.value_counts().reindex(ALL_LABELS, fill_value=0)
        macro_labels = support_counts.loc[support_counts > 0].index.tolist()
        if macro_labels == list(range(macro_labels[0], macro_labels[-1] + 1)):
            macro_label_text = f"IPC Phases {macro_labels[0]}–{macro_labels[-1]}"
        else:
            macro_label_text = f"IPC Phases {', '.join(map(str, macro_labels))}"
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=ALL_LABELS,
            average=None,
            zero_division=0,
        )
        class_metrics = pd.DataFrame(
            {
                "phase": ALL_LABELS,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support.astype(int),
            }
        ).set_index("phase")

        metric_rows.append(
            {
                "task": task,
                "display_label": display_label,
                "evaluation_protocol": evaluation_protocol,
                "evaluation_population": evaluation_population,
                "n_observations": len(y_true),
                "macro_label_set": macro_label_text,
                "macro_precision": class_metrics.loc[macro_labels, "precision"].mean(),
                "macro_recall": class_metrics.loc[macro_labels, "recall"].mean(),
                "macro_f1": class_metrics.loc[macro_labels, "f1"].mean(),
                "zero_division": 0,
                **{
                    f"phase_{phase}_support": int(support_counts.loc[phase])
                    for phase in ALL_LABELS
                },
            }
        )
        for phase, row in class_metrics.iterrows():
            per_class_rows.append(
                {
                    "task": task,
                    "display_label": display_label,
                    "evaluation_protocol": evaluation_protocol,
                    "evaluation_population": evaluation_population,
                    "n_observations": len(y_true),
                    "phase": phase,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "support": int(row["support"]),
                }
            )

        matrix = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
        matrices[task] = matrix
        row_support = matrix.sum(axis=1)
        for actual_index, actual_phase in enumerate(ALL_LABELS):
            for predicted_index, predicted_phase in enumerate(ALL_LABELS):
                count = int(matrix[actual_index, predicted_index])
                row_percentage = (
                    count / row_support[actual_index]
                    if row_support[actual_index] > 0
                    else np.nan
                )
                confusion_rows.append(
                    {
                        "task": task,
                        "display_label": display_label,
                        "evaluation_protocol": evaluation_protocol,
                        "evaluation_population": evaluation_population,
                        "n_observations": len(y_true),
                        "actual_phase": actual_phase,
                        "predicted_phase": predicted_phase,
                        "count": count,
                        "actual_phase_support": int(row_support[actual_index]),
                        "row_percentage": row_percentage,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(per_class_rows)
    confusion_long = pd.DataFrame(confusion_rows)
    return metrics, per_class, confusion_long, matrices


def calculate_metrics(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Compatibility wrapper for tasks sharing one evaluation population."""
    evaluations = {
        task: {
            "y_true": y_true,
            "y_pred": y_pred,
            "evaluation_protocol": "shared_evaluation_protocol",
            "evaluation_population": "shared_evaluation_population",
            "display_label": task,
        }
        for task, y_pred in predictions.items()
    }
    return calculate_task_metrics(evaluations)


def _ordered_tasks(task_names: object) -> list[str]:
    observed = set(task_names)
    return [task for task in PREDICTION_COLUMNS if task in observed]


def _macro_figure_title(metrics: pd.DataFrame) -> str:
    if (
        metrics["evaluation_protocol"].nunique() > 1
        or metrics["evaluation_population"].nunique() > 1
    ):
        return "Descriptive macro metrics under task-specific validation protocols"
    indexed = metrics.set_index("task")
    leaders = {
        "precision": indexed["macro_precision"].idxmax(),
        "recall": indexed["macro_recall"].idxmax(),
        "F1": indexed["macro_f1"].idxmax(),
    }
    if len(set(leaders.values())) == 1:
        return f"{leaders['precision']} is highest across all three macro metrics"
    if leaders["recall"] == leaders["F1"]:
        return (
            f"{leaders['precision']} leads precision; "
            f"{leaders['recall']} leads recall and F1"
        )
    return "The three models show distinct macro-metric trade-offs"


def _evaluation_protocol_note(metrics: pd.DataFrame) -> str:
    indexed = metrics.set_index("task")
    if {"Forecasting", "Nowcasting", "Contemporaneous"}.issubset(indexed.index):
        return (
            "Forecasting and Nowcasting: fixed 2022 temporal holdout "
            f"(n = {int(indexed.loc['Forecasting', 'n_observations']):,} each).\n"
            "Contemporaneous: random five-fold full-OOF row CV "
            f"(n = {int(indexed.loc['Contemporaneous', 'n_observations']):,}). "
            "Protocols and populations differ; metrics are not directly comparable."
        )
    parts = [
        f"{row.display_label}: {row.evaluation_protocol}, n = {int(row.n_observations):,}"
        for row in metrics.itertuples(index=False)
    ]
    return "; ".join(parts)


def create_metrics_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Create a compact comparison of macro precision, recall, and F1."""
    apply_figure_style(frame="open")
    metric_specs = [
        ("Precision", "macro_precision"),
        ("Recall", "macro_recall"),
        ("F1", "macro_f1"),
    ]
    x = np.arange(len(metric_specs), dtype=float)
    task_order = _ordered_tasks(metrics["task"])
    offset_values = (
        np.linspace(-0.18, 0.18, len(task_order))
        if len(task_order) > 1
        else np.array([0.0])
    )
    offsets = dict(zip(task_order, offset_values))

    fig, ax = plt.subplots(figsize=(7.2, 3.75), constrained_layout=True)
    indexed = metrics.set_index("task")
    all_values: list[float] = []
    for task_index, task in enumerate(task_order):
        values = np.array([indexed.loc[task, column] for _, column in metric_specs])
        all_values.extend(values.tolist())
        positions = x + offsets[task]
        ax.vlines(
            positions,
            0,
            values,
            color=TASK_COLORS[task],
            linewidth=1.1,
            alpha=0.55,
            zorder=1,
        )
        ax.scatter(
            positions,
            values,
            s=46,
            marker=TASK_MARKERS[task],
            color=TASK_COLORS[task],
            edgecolor="white",
            linewidth=0.6,
            label=str(indexed.loc[task, "display_label"]),
            zorder=3,
        )
        value_label_offset = 0.013 + (0.015 if task_index % 2 else 0.0)
        for position, value in zip(positions, values):
            ax.text(
                position,
                value + value_label_offset,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#262626",
            )

    ax.set_title(_macro_figure_title(metrics), loc="left", pad=34)
    ax.text(
        0,
        1.015,
        _evaluation_protocol_note(metrics),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        linespacing=1.25,
        color="#4D4D4D",
    )
    ax.set_xticks(x, [label for label, _ in metric_specs])
    ax.set_ylabel("Macro score")
    ax.set_ylim(0, max(0.3, max(all_values) * 1.13))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#D0D0D0", alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.text(
        0.99,
        0.95,
        "higher is better  ↑",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    ax.margins(x=0.16)
    return fig


def _row_proportions(matrix: np.ndarray) -> np.ndarray:
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals > 0,
    )


def create_confusion_figure(
    matrices: dict[str, np.ndarray], metrics: pd.DataFrame | None = None
) -> plt.Figure:
    """Create five-class confusion matrices with counts and row shares."""
    apply_figure_style(frame="boxed")
    task_order = _ordered_tasks(matrices)
    if not task_order:
        raise ValueError("At least one confusion matrix is required.")
    fig = plt.figure(
        figsize=(3.15 * len(task_order) + 0.6, 3.9), constrained_layout=True
    )
    grid = fig.add_gridspec(
        2,
        len(task_order) + 1,
        height_ratios=[1, 0.06],
        width_ratios=[1] * len(task_order) + [0.045],
    )
    axes = [fig.add_subplot(grid[0, 0])]
    for panel_index in range(1, len(task_order)):
        axes.append(
            fig.add_subplot(
                grid[0, panel_index], sharex=axes[0], sharey=axes[0]
            )
        )
    colorbar_ax = fig.add_subplot(grid[0, len(task_order)])
    caption_ax = fig.add_subplot(grid[1, :])
    caption_ax.axis("off")
    images = []
    phase_labels = [f"Phase {phase}" for phase in ALL_LABELS]
    metric_lookup = metrics.set_index("task") if metrics is not None else None

    for panel_index, (ax, task) in enumerate(zip(axes, task_order)):
        matrix = matrices[task]
        proportions = _row_proportions(matrix)
        image = ax.imshow(proportions, cmap="cividis", vmin=0, vmax=1, aspect="equal")
        images.append(image)

        for row in range(len(ALL_LABELS)):
            support = matrix[row].sum()
            for column in range(len(ALL_LABELS)):
                share = proportions[row, column]
                percentage = f"{share:.1%}" if support > 0 else "—"
                color = "#FFFFFF" if share < 0.48 else "#111111"
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n{percentage}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

        if metric_lookup is None:
            panel_title = task
        else:
            panel_title = (
                f"{metric_lookup.loc[task, 'display_label']}\n"
                f"n = {int(metric_lookup.loc[task, 'n_observations']):,}"
            )
        ax.set_title(panel_title, loc="center", pad=7)
        ax.set_xticks(range(len(ALL_LABELS)), phase_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(ALL_LABELS)), phase_labels)
        ax.set_xlabel("Predicted IPC phase")
        ax.tick_params(length=0)
        ax.text(
            -0.16,
            1.02,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    axes[0].set_ylabel("Actual IPC phase")
    concentrated_in_two_three = all(
        int(matrix[:, [0, 3, 4]].sum()) == 0 for matrix in matrices.values()
    )
    mixed_protocols = bool(
        metrics is not None
        and (
            metrics["evaluation_protocol"].nunique() > 1
            or metrics["evaluation_population"].nunique() > 1
        )
    )
    if mixed_protocols:
        title = "Five-class confusion matrices under task-specific validation protocols"
    else:
        title = (
            "All models concentrate predictions in IPC Phases 2 and 3"
            if concentrated_in_two_three
            else "Five-class confusion matrices"
        )
    fig.suptitle(
        title,
        x=0.01,
        ha="left",
        fontsize=9,
        fontweight="normal",
    )
    colorbar = fig.colorbar(images[0], cax=colorbar_ax)
    colorbar.set_label("Share of actual class")
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    caption = "Cells show count and row percentage."
    zero_support_notes = []
    for task in task_order:
        zero_support_phases = [
            phase
            for phase, support in zip(ALL_LABELS, matrices[task].sum(axis=1))
            if support == 0
        ]
        if zero_support_phases:
            zero_support_notes.append(
                f"{task}: Phase {', '.join(map(str, zero_support_phases))} has no actual observations"
            )
    if zero_support_notes:
        caption += " " + "; ".join(zero_support_notes) + " (—)."
    if mixed_protocols:
        caption += (
            " Forecasting/Nowcasting use the 2022 temporal holdout; "
            "Contemporaneous uses random five-fold full-OOF row CV. "
            "The panels are descriptive and not directly comparable."
        )
    caption_ax.text(
        0,
        0.5,
        caption,
        ha="left",
        va="center",
        fontsize=7,
        color="#4D4D4D",
    )
    return fig


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, Path]:
    paths = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("jpg", "png", "pdf")}
    fig.savefig(paths["jpg"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def run_analysis(
    input_path: Path = DEFAULT_INPUT_PATH,
    sheet_name: str = DEFAULT_SHEET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    contemporaneous_source_path: Path = DEFAULT_CONTEMPORANEOUS_SOURCE_PATH,
    contemporaneous_params_path: Path = DEFAULT_CONTEMPORANEOUS_PARAMS_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
    estimator_n_jobs: int | None = None,
) -> dict[str, Path]:
    """Calculate metrics and save tabular and figure artifacts."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = write_contemporaneous_random_cv_artifacts(
        output_dir,
        contemporaneous_source_path,
        contemporaneous_params_path,
        random_state=random_state,
        estimator_n_jobs=estimator_n_jobs,
    )
    y_true, predictions = load_predictions(input_path, sheet_name)
    contemporaneous = load_contemporaneous_random_cv_predictions(
        paths["contemporaneous_predictions_csv"]
    )
    temporal_protocol = (
        "fixed_2022_temporal_holdout"
        if input_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}
        else "legacy_workbook_evaluation"
    )
    temporal_population = (
        "canonical_1170_temporal_test"
        if input_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}
        else "legacy_workbook_rows"
    )
    evaluations = {
        task: {
            "y_true": y_true,
            "y_pred": y_pred,
            "evaluation_protocol": temporal_protocol,
            "evaluation_population": temporal_population,
            "display_label": task,
        }
        for task, y_pred in predictions.items()
    }
    evaluations["Contemporaneous"] = {
        "y_true": contemporaneous[TRUE_COLUMN],
        "y_pred": contemporaneous["contemporaneous_predict"],
        "evaluation_protocol": "random_5fold_row_cv",
        "evaluation_population": "random_5fold_full_oof_5575",
        "display_label": "Contemporaneous (random CV)",
    }
    metrics, per_class, confusion_long, matrices = calculate_task_metrics(evaluations)

    paths.update(
        {
            "metrics_csv": output_dir / "all_prediction_macro_metrics.csv",
            "per_class_csv": output_dir / "all_prediction_per_class_metrics.csv",
            "confusion_csv": output_dir
            / "all_prediction_confusion_matrix_long.csv",
        }
    )
    metrics.to_csv(paths["metrics_csv"], index=False, float_format="%.6f")
    per_class.to_csv(paths["per_class_csv"], index=False, float_format="%.6f")
    confusion_long.to_csv(paths["confusion_csv"], index=False, float_format="%.6f")

    metric_paths = _save_figure(
        create_metrics_figure(metrics),
        output_dir / "all_prediction_macro_metrics",
    )
    confusion_paths = _save_figure(
        create_confusion_figure(matrices, metrics),
        output_dir / "all_prediction_five_class_confusion_matrices",
    )
    paths.update({f"metrics_{key}": value for key, value in metric_paths.items()})
    paths.update({f"confusion_{key}": value for key, value in confusion_paths.items()})
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--contemporaneous-source",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_SOURCE_PATH,
    )
    parser.add_argument(
        "--contemporaneous-params",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_PARAMS_PATH,
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--estimator-n-jobs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(
        args.input,
        args.sheet,
        args.output_dir,
        args.contemporaneous_source,
        args.contemporaneous_params,
        args.random_state,
        args.estimator_n_jobs,
    )
    for artifact, path in paths.items():
        print(f"{artifact}: {path}")


if __name__ == "__main__":
    main()
