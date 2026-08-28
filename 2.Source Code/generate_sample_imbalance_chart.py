"""Generate the area-level sample imbalance chart for the replication package."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "1.Source Data" / "Forecasting_Analysis_010825.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
OUTPUT_STEM = "sample_imbalance_area_frequency"


def apply_figure_style() -> None:
    """Apply a compact, publication-ready Matplotlib style."""
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


def _validate_input(data: pd.DataFrame) -> None:
    required = {"area_id", "overall_phase"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input data are missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Input data contain no observations.")
    if data["area_id"].isna().any():
        raise ValueError("area_id contains missing values; area frequencies are undefined.")


def build_area_frequency_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Count areas by observation frequency and observed phase diversity.

    Each ``area_id`` contributes exactly once. Its observation frequency is the
    number of rows in the full dataset, while phase diversity is the number of
    distinct non-missing ``overall_phase`` values recorded for that area.
    """
    _validate_input(data)

    area_summary = (
        data.groupby("area_id", sort=False, observed=True)
        .agg(
            observation_frequency=("area_id", "size"),
            distinct_phase_count=("overall_phase", "nunique"),
        )
        .reset_index(drop=True)
    )

    distribution = (
        area_summary.groupby(
            ["observation_frequency", "distinct_phase_count"],
            observed=True,
        )
        .size()
        .rename("area_count")
        .reset_index()
        .sort_values(["observation_frequency", "distinct_phase_count"])
        .reset_index(drop=True)
    )
    return distribution


def _build_area_summary(data: pd.DataFrame) -> pd.DataFrame:
    _validate_input(data)
    return (
        data.groupby("area_id", sort=False, observed=True)
        .agg(
            observation_frequency=("area_id", "size"),
            distinct_phase_count=("overall_phase", "nunique"),
        )
        .reset_index()
    )


def create_sample_imbalance_figure(
    data: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Create the solid discrete-frequency bar chart and return its audit data."""
    apply_figure_style()
    distribution = build_area_frequency_distribution(data)
    area_summary = _build_area_summary(data)

    min_frequency = int(area_summary["observation_frequency"].min())
    max_frequency = int(area_summary["observation_frequency"].max())
    frequencies = np.arange(min_frequency, max_frequency + 1)
    heights = (
        distribution.groupby("observation_frequency", observed=True)["area_count"]
        .sum()
        .reindex(frequencies, fill_value=0)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.bar(
        frequencies,
        heights,
        width=0.82,
        color="#4C78A8",
        edgecolor="none",
    )

    total_areas = len(area_summary)
    total_observations = len(data)
    low_frequency_count = int((area_summary["observation_frequency"] <= 2).sum())
    low_frequency_share = 100 * low_frequency_count / total_areas
    median_frequency = float(area_summary["observation_frequency"].median())

    ax.set_title(
        "Observation frequency is highly uneven across analysis areas",
        loc="left",
        pad=17,
    )
    ax.text(
        0,
        1.02,
        f"{total_observations:,} observations across {total_areas:,} unique area_id values",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#4D4D4D",
    )
    ax.set_xlabel("Observations per analysis area (area_id)")
    ax.set_ylabel("Number of analysis areas")
    ax.set_xticks(frequencies)
    ax.set_xlim(min_frequency - 0.65, max_frequency + 0.65)
    ax.set_ylim(0, max(heights) * 1.08)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    annotation = (
        f"{low_frequency_count:,} areas ({low_frequency_share:.1f}%) appear at most twice\n"
        f"Median = {median_frequency:g}; maximum = {max_frequency} observations"
    )
    ax.text(
        0.985,
        0.58,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BDBDBD"},
    )
    ax.margins(y=0.04)
    return fig, distribution


def save_sample_imbalance_chart(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Read source data and save PNG, PDF, and audit CSV outputs."""
    data = pd.read_csv(input_path, usecols=["area_id", "overall_phase"])
    fig, distribution = create_sample_imbalance_figure(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_dir / f"{OUTPUT_STEM}.png",
        "pdf": output_dir / f"{OUTPUT_STEM}.pdf",
        "csv": output_dir / f"{OUTPUT_STEM}.csv",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    distribution.to_csv(paths["csv"], index=False)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = save_sample_imbalance_chart(args.input, args.output_dir)
    for artifact_type, path in paths.items():
        print(f"{artifact_type}: {path}")


if __name__ == "__main__":
    main()
