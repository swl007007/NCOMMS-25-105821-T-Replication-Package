"""Fit multinomial baselines and compare them with the frozen Main result.

The baseline follows the replication notebooks' outcome and evaluation logic:
phase labels are reconstructed from cumulative population shares using a 0.20
cutoff, training observations precede 2022-01-01, and the temporal holdout begins
on that date. Precision and recall refer to IPC Phase 3 or above.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import main_result_figure1_v1 as frozen_main_result


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
DEFAULT_CUTOFF_DATE = "2022-01-01"
PHASE_SHARE_CUTOFF = 0.20

INPUT_PATHS = {
    "Forecasting": SOURCE_DATA_DIR / "Forecasting_Analysis_010825.csv",
    "Nowcasting": SOURCE_DATA_DIR / "Nowcasting_Analysis_010825.csv",
}

NON_FEATURE_COLUMNS = {
    "date",
    "area_id",
    "overall_phase",
    "phase1_percent",
    "phase2_percent",
    "phase3_percent",
    "phase4_percent",
    "phase5_percent",
}

MAIN_RESULT_REFERENCES = frozen_main_result.classification_references()


def apply_figure_style() -> None:
    """Apply publication-ready sizing, typography, ticks, and export settings."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
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


def _require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"Input data are missing required columns: {sorted(missing)}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_phase_labels(
    data: pd.DataFrame,
    cutoff: float = PHASE_SHARE_CUTOFF,
) -> pd.Series:
    """Reconstruct the notebook's overall IPC phase from cumulative shares."""
    phase_columns = [f"phase{i}_percent" for i in range(1, 6)]
    _require_columns(data, phase_columns)

    phase2_worse = data[[f"phase{i}_percent" for i in range(2, 6)]].sum(axis=1)
    phase3_worse = data[[f"phase{i}_percent" for i in range(3, 6)]].sum(axis=1)
    phase4_worse = data[["phase4_percent", "phase5_percent"]].sum(axis=1)
    phase5_worse = data["phase5_percent"]

    labels = np.select(
        [
            phase5_worse.ge(cutoff),
            phase4_worse.ge(cutoff),
            phase3_worse.ge(cutoff),
            phase2_worse.ge(cutoff),
        ],
        [5, 4, 3, 2],
        default=1,
    )
    return pd.Series(labels, index=data.index, name="overall_phase_evaluation").astype(int)


def calculate_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate five-class accuracy and Phase 3+ binary metrics."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    true_phase3plus = y_true_array >= 3
    pred_phase3plus = y_pred_array >= 3
    return {
        "overall_accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "phase3plus_accuracy": float(
            accuracy_score(true_phase3plus, pred_phase3plus)
        ),
        "phase3plus_recall": float(
            recall_score(true_phase3plus, pred_phase3plus, zero_division=0)
        ),
        "phase3plus_precision": float(
            precision_score(true_phase3plus, pred_phase3plus, zero_division=0)
        ),
    }


def select_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return the original workflow's predictors, excluding keys and outcomes."""
    _require_columns(data, NON_FEATURE_COLUMNS)
    return [column for column in data.columns if column not in NON_FEATURE_COLUMNS]


def temporal_masks(
    dates: pd.Series,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
) -> tuple[pd.Series, pd.Series]:
    """Return train/test masks with the cutoff date assigned to the holdout."""
    parsed_dates = pd.to_datetime(dates, errors="raise")
    boundary = pd.Timestamp(cutoff_date)
    train_mask = parsed_dates < boundary
    test_mask = parsed_dates >= boundary
    return train_mask, test_mask


def _reconstruct_original_phase3plus_accuracy(
    y_true: pd.Series,
    recall: float,
    precision: float,
) -> float:
    """Recover binary accuracy from exact stored rates and the holdout labels."""
    true_positive_count = int((y_true >= 3).sum())
    true_negative_count = int((y_true < 3).sum())
    tp = int(round(recall * true_positive_count))
    predicted_positive_count = int(round(tp / precision))
    fp = predicted_positive_count - tp
    tn = true_negative_count - fp
    if min(tp, fp, tn) < 0:
        raise ValueError("Stored precision/recall are inconsistent with holdout labels.")
    return (tp + tn) / len(y_true)


def fit_multinomial_baseline(
    data: pd.DataFrame,
    task: str,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit one leakage-controlled temporal multinomial logistic baseline."""
    if data.empty:
        raise ValueError("Input data contain no observations.")
    _require_columns(data, NON_FEATURE_COLUMNS)

    target = derive_phase_labels(data)
    train_mask, test_mask = temporal_masks(data["date"], cutoff_date)
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Temporal split must contain both training and test observations.")

    feature_columns = select_feature_columns(data)
    numeric_data = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    usable_features = [
        column for column in feature_columns if numeric_data.loc[train_mask, column].notna().any()
    ]
    if not usable_features:
        raise ValueError("No usable predictor columns remain after training-data checks.")

    non_numeric = [
        column
        for column in usable_features
        if not pd.api.types.is_numeric_dtype(numeric_data[column])
    ]
    if non_numeric:
        raise ValueError(f"Baseline predictors must be numeric: {non_numeric}")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    l1_ratio=0.0,
                    C=1.0,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(numeric_data.loc[train_mask, usable_features], target.loc[train_mask])
    predictions = pipeline.predict(numeric_data.loc[test_mask, usable_features])
    y_test = target.loc[test_mask]
    metrics = calculate_classification_metrics(y_test, predictions)

    record: dict[str, object] = {
        "task": task,
        "model": "Multinomial logistic baseline",
        "model_type": "Baseline",
        **metrics,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "feature_count": len(usable_features),
        "cutoff_date": cutoff_date,
        "phase_share_cutoff": PHASE_SHARE_CUTOFF,
        "target_definition": (
            "highest cumulative IPC phase with population share >= 0.20"
        ),
        "metric_source": "Generated from source CSV by this script",
        "main_result_reference_id": frozen_main_result.FREEZE_ID,
        "environment_relation": "same_frozen_python_core_stack_and_test_population",
    }

    prediction_frame = pd.DataFrame(
        {
            "task": task,
            "source_index": data.index[test_mask],
            "area_id": data.loc[test_mask, "area_id"].to_numpy(),
            "date": pd.to_datetime(data.loc[test_mask, "date"]).dt.strftime("%Y-%m-%d"),
            "overall_phase_evaluation": y_test.to_numpy(dtype=int),
            "overall_phase_pred": predictions.astype(int),
            "phase3plus_evaluation": (y_test.to_numpy() >= 3).astype(int),
            "phase3plus_pred": (predictions >= 3).astype(int),
        }
    )
    return record, prediction_frame


def build_comparison_metrics(
    baseline_records: list[dict[str, object]],
    holdout_labels: dict[str, pd.Series],
) -> pd.DataFrame:
    """Combine generated baselines with the frozen Figure 1 main result."""
    original_records: list[dict[str, object]] = []
    for task, stored in MAIN_RESULT_REFERENCES.items():
        labels = holdout_labels[task]
        phase3plus_accuracy = _reconstruct_original_phase3plus_accuracy(
            labels,
            recall=stored["phase3plus_recall"],
            precision=stored["phase3plus_precision"],
        )
        original_records.append(
            {
                "task": task,
                "model": "Main result",
                "model_type": "Main result",
                "overall_accuracy": stored["overall_accuracy"],
                "phase3plus_accuracy": phase3plus_accuracy,
                "phase3plus_recall": stored["phase3plus_recall"],
                "phase3plus_precision": stored["phase3plus_precision"],
                "n_train": 4405,
                "n_test": len(labels),
                "feature_count": np.nan,
                "cutoff_date": DEFAULT_CUTOFF_DATE,
                "phase_share_cutoff": PHASE_SHARE_CUTOFF,
                "target_definition": (
                    "highest cumulative IPC phase with population share >= 0.20"
                ),
                "metric_source": (
                    f"{stored['metric_source']}; Phase 3+ accuracy reconstructed "
                    "from exact stored recall/precision and holdout class counts"
                ),
                "main_result_reference_id": frozen_main_result.FREEZE_ID,
                "environment_relation": "frozen_main_result_reference",
            }
        )

    combined = pd.DataFrame(original_records + baseline_records)
    task_order = pd.Categorical(
        combined["task"], categories=["Forecasting", "Nowcasting"], ordered=True
    )
    model_order = pd.Categorical(
        combined["model_type"], categories=["Main result", "Baseline"], ordered=True
    )
    return (
        combined.assign(_task_order=task_order, _model_order=model_order)
        .sort_values(["_task_order", "_model_order"])
        .drop(columns=["_task_order", "_model_order"])
        .reset_index(drop=True)
    )


def create_comparison_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Create a precision-recall scatter styled after the original figure."""
    apply_figure_style()
    colors = {"Forecasting": "#1F77B4", "Nowcasting": "#E69F00"}
    markers = {"Forecasting": "o", "Nowcasting": "s"}

    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    for task in ["Forecasting", "Nowcasting"]:
        task_rows = metrics.loc[metrics["task"] == task].set_index("model_type")
        ax.plot(
            task_rows["phase3plus_recall"],
            task_rows["phase3plus_precision"],
            color=colors[task],
            linewidth=1.2,
            alpha=0.55,
            zorder=1,
        )
        for model_type in ["Main result", "Baseline"]:
            row = task_rows.loc[model_type]
            is_main_result = model_type == "Main result"
            ax.scatter(
                row["phase3plus_recall"],
                row["phase3plus_precision"],
                marker=markers[task],
                s=78,
                facecolor=colors[task] if is_main_result else "white",
                edgecolor=colors[task],
                linewidth=1.35,
                zorder=3,
            )

    label_specs = {
        ("Forecasting", "Main result"): (-12, -29, "right"),
        ("Nowcasting", "Main result"): (-92, 17, "right"),
        ("Forecasting", "Baseline"): (11, -26, "left"),
        ("Nowcasting", "Baseline"): (11, 15, "left"),
    }
    for row in metrics.itertuples(index=False):
        dx, dy, alignment = label_specs[(row.task, row.model_type)]
        model_label = (
            "main result" if row.model_type == "Main result" else "multinomial"
        )
        label = f"{row.task} {model_label}\n3+ accuracy = {row.phase3plus_accuracy:.2f}"
        ax.annotate(
            label,
            xy=(row.phase3plus_recall, row.phase3plus_precision),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=7,
            color="#262626",
            arrowprops={
                "arrowstyle": "-",
                "color": colors[row.task],
                "linewidth": 0.6,
                "shrinkA": 2,
                "shrinkB": 5,
            },
        )

    ax.set_title(
        "Multinomial baselines trade substantially lower recall for slightly higher precision",
        loc="left",
        pad=17,
    )
    ax.text(
        0,
        1.02,
        "Temporal holdout: train before 2022; test Jan–Nov 2022 (n = 1,170)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#4D4D4D",
    )
    ax.set_xlabel("Sensitivity (recall), IPC Phase 3+")
    ax.set_ylabel("Precision, IPC Phase 3+")
    ax.set_xlim(0.55, 0.975)
    ax.set_ylim(0.765, 0.807)
    ax.grid(True, linestyle="--", linewidth=0.55, color="#D0D0D0", alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.985,
        0.96,
        "higher is better  ↗",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    ax.margins(0.04)
    return fig


def run_analysis(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
) -> dict[str, Path]:
    """Fit both baselines and save metrics, predictions, and comparison figures."""
    frozen_main_result.assert_frozen_environment(("matplotlib",))
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_records: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    holdout_labels: dict[str, pd.Series] = {}

    for task, input_path in INPUT_PATHS.items():
        data = pd.read_csv(input_path)
        record, predictions = fit_multinomial_baseline(data, task, cutoff_date)
        baseline_records.append(record)
        prediction_frames[task] = predictions
        _, test_mask = temporal_masks(data["date"], cutoff_date)
        holdout_labels[task] = derive_phase_labels(data).loc[test_mask]

    metrics = build_comparison_metrics(baseline_records, holdout_labels)
    fig = create_comparison_figure(metrics)

    source_audit = pd.DataFrame(
        [
            {
                "task": task,
                "main_result_reference_id": frozen_main_result.FREEZE_ID,
                "main_result_environment_id": frozen_main_result.ENVIRONMENT[
                    "environment_id"
                ],
                "main_result_overall_accuracy": MAIN_RESULT_REFERENCES[task][
                    "overall_accuracy"
                ],
                "main_result_phase3plus_precision": MAIN_RESULT_REFERENCES[task][
                    "phase3plus_precision"
                ],
                "main_result_phase3plus_recall": MAIN_RESULT_REFERENCES[task][
                    "phase3plus_recall"
                ],
                "main_result_phase3plus_r2": MAIN_RESULT_REFERENCES[task][
                    "phase3plus_r2"
                ],
                "comparison_scope": "same_1170_row_temporal_test_population",
                "baseline_model": "multinomial_logistic_regression",
                "baseline_random_state": 42,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "sklearn_version": sklearn.__version__,
                "matplotlib_version": mpl.__version__,
                "input_path": str(INPUT_PATHS[task]),
                "input_sha256": file_sha256(INPUT_PATHS[task]),
                "generator_sha256": file_sha256(Path(__file__)),
                "freeze_source_path": str(Path(frozen_main_result.__file__)),
                "freeze_source_sha256": file_sha256(
                    Path(frozen_main_result.__file__)
                ),
                "n_train": int(metrics.loc[metrics["task"].eq(task), "n_train"].max()),
                "n_test": int(metrics.loc[metrics["task"].eq(task), "n_test"].max()),
            }
            for task in INPUT_PATHS
        ]
    )

    paths = {
        "metrics": output_dir / "multinomial_baseline_metrics.csv",
        "source_audit": output_dir / "multinomial_baseline_source_audit.csv",
        "forecasting_predictions": output_dir
        / "multinomial_baseline_forecasting_predictions.csv",
        "nowcasting_predictions": output_dir
        / "multinomial_baseline_nowcasting_predictions.csv",
        "jpg": output_dir / "precision_recall_scatter_with_multinomial_baselines.jpg",
        "png": output_dir / "precision_recall_scatter_with_multinomial_baselines.png",
        "pdf": output_dir / "precision_recall_scatter_with_multinomial_baselines.pdf",
    }
    metrics.to_csv(paths["metrics"], index=False)
    source_audit.to_csv(paths["source_audit"], index=False)
    prediction_frames["Forecasting"].to_csv(paths["forecasting_predictions"], index=False)
    prediction_frames["Nowcasting"].to_csv(paths["nowcasting_predictions"], index=False)
    fig.savefig(paths["jpg"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff-date", default=DEFAULT_CUTOFF_DATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(args.output_dir, args.cutoff_date)
    for artifact, path in paths.items():
        print(f"{artifact}: {path}")


if __name__ == "__main__":
    main()
