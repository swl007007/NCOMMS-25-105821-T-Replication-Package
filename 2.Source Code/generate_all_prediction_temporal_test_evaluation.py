"""Evaluate two temporal-holdout models and random-CV Contemporaneous results.

The historical ``all_prediction_temporal_test_*`` prefix is retained for artifact
continuity. Forecasting and Nowcasting use the canonical 1,170-row 2022 temporal
holdout; Contemporaneous uses a reproducible 5,575-row random five-fold full-OOF
rerun. Figures and tables explicitly mark the protocols as not directly comparable.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache-all-prediction")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate_all_prediction_temporal_test as temporal_contract
from generate_all_prediction_evaluation import (
    ALL_LABELS,
    CONTEMPORANEOUS_PREDICTION_STEM,
    DEFAULT_CONTEMPORANEOUS_PARAMS_PATH,
    DEFAULT_CONTEMPORANEOUS_PREDICTIONS_PATH,
    DEFAULT_CONTEMPORANEOUS_SOURCE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    PREDICTION_COLUMNS as EVALUATION_PREDICTION_COLUMNS,
    TASK_COLORS,
    TASK_MARKERS,
    _evaluation_protocol_note,
    _save_figure,
    _validated_labels,
    apply_figure_style,
    calculate_task_metrics,
    create_confusion_figure as create_base_confusion_figure,
    create_metrics_figure,
    load_contemporaneous_random_cv_predictions,
    write_contemporaneous_random_cv_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "1.Source Data" / "All_prediction.csv"
EXPECTED_ROWS = 1170
OUTPUT_PREFIX = "all_prediction_temporal_test"
ID_COLUMN = "test_index"
KEY_COLUMNS = ["area_id", "date"]
TRUE_COLUMN = "overall_phase"
PREDICTION_COLUMNS = dict(EVALUATION_PREDICTION_COLUMNS)


def create_confusion_figure(
    matrices: dict[str, np.ndarray], metrics: pd.DataFrame
) -> plt.Figure:
    """Keep the mixed-protocol confusion export compact and fully vector."""
    fig = create_base_confusion_figure(matrices, metrics)
    for axis in fig.axes:
        for image in list(axis.images):
            values = np.asarray(image.get_array())
            x_limits = axis.get_xlim()
            y_limits = axis.get_ylim()
            axis.pcolormesh(
                np.arange(values.shape[1] + 1) - 0.5,
                np.arange(values.shape[0] + 1) - 0.5,
                values,
                cmap=image.get_cmap(),
                norm=image.norm,
                shading="flat",
                edgecolors="none",
                rasterized=False,
            )
            image.remove()
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
        for collection in axis.collections:
            collection.set_rasterized(False)
    caption_axes = [axis for axis in fig.axes if not axis.axison]
    if len(caption_axes) != 1 or len(caption_axes[0].texts) != 1:
        raise ValueError("Expected one confusion-matrix caption axis and text block.")
    caption_text = caption_axes[0].texts[0]
    caption_text.set_text(textwrap.fill(caption_text.get_text(), width=105))
    caption_text.set_linespacing(1.25)
    return fig


def load_predictions(
    input_path: Path = DEFAULT_INPUT_PATH,
    contemporaneous_predictions_path: Path | None = (
        DEFAULT_CONTEMPORANEOUS_PREDICTIONS_PATH
    ),
    expected_contemporaneous_rows: int = 5575,
) -> dict[str, dict[str, object]]:
    """Load task-specific evaluation bundles without forcing shared populations."""
    data = pd.read_csv(input_path)
    required = [
        *temporal_contract.CANONICAL_OUTPUT_COLUMNS,
        ID_COLUMN,
        TRUE_COLUMN,
        *KEY_COLUMNS,
    ]
    missing = list(dict.fromkeys(column for column in required if column not in data))
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")
    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} prediction rows, found {len(data):,}."
        )
    if data[ID_COLUMN].isna().any():
        raise ValueError("test_index must be complete and unique.")
    if not data[ID_COLUMN].is_unique:
        raise ValueError("test_index must be complete and unique.")

    validated = temporal_contract.validate_canonical_prediction_artifact(
        data, expected_rows=EXPECTED_ROWS
    )
    if contemporaneous_predictions_path is None:
        raise ValueError("Contemporaneous random-CV predictions are required.")
    contemporaneous = load_contemporaneous_random_cv_predictions(
        contemporaneous_predictions_path,
        expected_rows=expected_contemporaneous_rows,
    )
    temporal_truth = validated[TRUE_COLUMN].astype(int)
    evaluations = {
        task: {
            "y_true": temporal_truth,
            "y_pred": _validated_labels(validated, column),
            "evaluation_protocol": "fixed_2022_temporal_holdout",
            "evaluation_population": "canonical_1170_temporal_test",
            "display_label": task,
        }
        for task, column in PREDICTION_COLUMNS.items()
        if task in {"Forecasting", "Nowcasting"}
    }
    evaluations["Contemporaneous"] = {
        "y_true": contemporaneous[TRUE_COLUMN],
        "y_pred": contemporaneous[PREDICTION_COLUMNS["Contemporaneous"]],
        "evaluation_protocol": "random_5fold_row_cv",
        "evaluation_population": "random_5fold_full_oof_5575",
        "display_label": "Contemporaneous (random CV)",
    }
    return evaluations


def build_class_specific_precision_recall(
    per_class: pd.DataFrame,
    evaluations: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Preserve undefined class precision separately from macro zero filling."""
    rows: list[dict[str, object]] = []
    indexed = per_class.set_index(["task", "phase"])
    for task, evaluation in evaluations.items():
        y_pred = pd.Series(evaluation["y_pred"])
        predicted_support = y_pred.value_counts().reindex(ALL_LABELS, fill_value=0)
        for phase in ALL_LABELS:
            source = indexed.loc[(task, phase)]
            is_defined = int(predicted_support.loc[phase]) > 0
            rows.append(
                {
                    "task": task,
                    "display_label": source["display_label"],
                    "evaluation_protocol": source["evaluation_protocol"],
                    "evaluation_population": source["evaluation_population"],
                    "n_observations": int(source["n_observations"]),
                    "phase": phase,
                    "actual_support": int(source["support"]),
                    "predicted_support": int(predicted_support.loc[phase]),
                    "precision_defined": is_defined,
                    "precision_class_specific": (
                        float(source["precision"]) if is_defined else np.nan
                    ),
                    "precision_used_for_macro": float(source["precision"]),
                    "recall": float(source["recall"]),
                }
            )
    return pd.DataFrame(rows)


def create_class_specific_precision_recall_figure(
    class_metrics: pd.DataFrame,
) -> plt.Figure:
    """Plot class-specific precision and recall for all prediction tasks."""
    apply_figure_style(frame="open")
    fig = plt.figure(figsize=(8.6, 4.45), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.17])
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[0, 1], sharey=axes[0]))
    caption_ax = fig.add_subplot(grid[1, :])
    caption_ax.axis("off")

    x = np.arange(1, 6, dtype=float)
    task_order = [
        task for task in PREDICTION_COLUMNS if task in set(class_metrics["task"])
    ]
    offset_values = (
        np.linspace(-0.17, 0.17, len(task_order))
        if len(task_order) > 1
        else np.array([0.0])
    )
    offsets = dict(zip(task_order, offset_values))
    panel_specs = [
        ("Class-specific precision", "precision_class_specific"),
        ("Class-specific recall", "recall"),
    ]

    for panel_index, (ax, (title, metric_column)) in enumerate(
        zip(axes, panel_specs)
    ):
        for task_index, task in enumerate(task_order):
            task_rows = class_metrics.loc[class_metrics["task"].eq(task)].set_index(
                "phase"
            )
            positions = x + offsets[task]
            values = task_rows.loc[ALL_LABELS, metric_column].to_numpy(dtype=float)
            defined = np.isfinite(values)
            positive = defined & (values > 0)
            ax.vlines(
                positions[positive],
                0,
                values[positive],
                color=TASK_COLORS[task],
                linewidth=1.0,
                alpha=0.5,
                zorder=1,
            )
            ax.scatter(
                positions[defined],
                values[defined],
                s=42,
                marker=TASK_MARKERS[task],
                color=TASK_COLORS[task],
                edgecolor="white",
                linewidth=0.6,
                label=(
                    str(task_rows["display_label"].iloc[0])
                    if panel_index == 0
                    else None
                ),
                zorder=3,
            )
            value_label_offset = 0.025 + 0.055 * task_index
            for position, value in zip(positions[positive], values[positive]):
                ax.text(
                    position,
                    value + value_label_offset,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="#262626",
                )
            for position in positions[~defined]:
                ax.text(
                    position,
                    0.015 + 0.032 * task_index,
                    "n.d.",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=TASK_COLORS[task],
                )

        ax.set_title(title, loc="left", pad=8)
        ax.set_xticks(x, [f"Phase {phase}" for phase in ALL_LABELS])
        ax.set_xlim(0.55, 5.45)
        ax.set_xlabel("Actual IPC phase")
        ax.set_ylim(-0.055, 1.18)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.55,
            color="#D0D0D0",
            alpha=0.75,
        )
        ax.set_axisbelow(True)
        ax.margins(x=0.08)
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

    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper left", ncol=min(3, len(task_order)))
    axes[1].text(
        0.99,
        0.96,
        "higher is better  ↑",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    fig.suptitle(
        "Class-specific precision and recall under task-specific validation protocols",
        x=0.01,
        ha="left",
        fontsize=9,
        fontweight="normal",
    )
    protocol_metadata = class_metrics.drop_duplicates("task")
    caption_ax.text(
        0,
        0.5,
        _evaluation_protocol_note(protocol_metadata)
        + "\nn.d. = precision undefined because predicted support is zero; "
        "macro calculations use zero_division = 0.",
        ha="left",
        va="center",
        fontsize=6.7,
        linespacing=1.25,
        color="#4D4D4D",
    )
    return fig


def run_analysis(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    contemporaneous_source_path: Path = DEFAULT_CONTEMPORANEOUS_SOURCE_PATH,
    contemporaneous_params_path: Path = DEFAULT_CONTEMPORANEOUS_PARAMS_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
    estimator_n_jobs: int | None = None,
) -> dict[str, Path]:
    """Save the mixed-protocol evaluation family and random-CV OOF sidecar."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contemporaneous_paths = write_contemporaneous_random_cv_artifacts(
        output_dir,
        contemporaneous_source_path,
        contemporaneous_params_path,
        random_state=random_state,
        estimator_n_jobs=estimator_n_jobs,
    )
    contemporaneous_predictions_path = contemporaneous_paths[
        "contemporaneous_predictions_csv"
    ]
    evaluations = load_predictions(input_path, contemporaneous_predictions_path)
    metrics, per_class, confusion_long, matrices = calculate_task_metrics(
        evaluations
    )
    class_precision_recall = build_class_specific_precision_recall(
        per_class, evaluations
    )
    for task, evaluation in evaluations.items():
        expected_rows = len(pd.Series(evaluation["y_true"]))
        task_confusion_total = int(
            confusion_long.loc[confusion_long["task"].eq(task), "count"].sum()
        )
        if task_confusion_total != expected_rows:
            raise ValueError(
                f"Confusion totals do not equal {expected_rows} for {task}."
            )
        task_classes = class_precision_recall.loc[
            class_precision_recall["task"].eq(task)
        ]
        if int(task_classes["actual_support"].sum()) != expected_rows:
            raise ValueError(f"Actual supports do not equal {expected_rows} for {task}.")
        if int(task_classes["predicted_support"].sum()) != expected_rows:
            raise ValueError(
                f"Predicted supports do not equal {expected_rows} for {task}."
            )

    paths = {
        **contemporaneous_paths,
        "metrics_csv": output_dir / f"{OUTPUT_PREFIX}_macro_metrics.csv",
        "per_class_csv": output_dir / f"{OUTPUT_PREFIX}_per_class_metrics.csv",
        "confusion_csv": output_dir / f"{OUTPUT_PREFIX}_confusion_matrix_long.csv",
        "class_precision_recall_csv": output_dir
        / f"{OUTPUT_PREFIX}_class_specific_precision_recall.csv",
    }
    metrics.to_csv(paths["metrics_csv"], index=False, float_format="%.6f")
    per_class.to_csv(paths["per_class_csv"], index=False, float_format="%.6f")
    confusion_long.to_csv(paths["confusion_csv"], index=False, float_format="%.6f")
    class_precision_recall.to_csv(
        paths["class_precision_recall_csv"], index=False, float_format="%.6f"
    )

    metric_paths = _save_figure(
        create_metrics_figure(metrics), output_dir / f"{OUTPUT_PREFIX}_macro_metrics"
    )
    confusion_paths = _save_figure(
        create_confusion_figure(matrices, metrics),
        output_dir / f"{OUTPUT_PREFIX}_five_class_confusion_matrices",
    )
    class_precision_recall_paths = _save_figure(
        create_class_specific_precision_recall_figure(class_precision_recall),
        output_dir / f"{OUTPUT_PREFIX}_class_specific_precision_recall",
    )
    paths.update({f"metrics_{key}": value for key, value in metric_paths.items()})
    paths.update({f"confusion_{key}": value for key, value in confusion_paths.items()})
    paths.update(
        {
            f"class_precision_recall_{key}": value
            for key, value in class_precision_recall_paths.items()
        }
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = run_analysis(
        args.input,
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
