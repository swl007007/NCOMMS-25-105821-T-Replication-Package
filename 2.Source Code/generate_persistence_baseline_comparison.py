"""Generate horizon-aligned IPC persistence baselines and comparison figures."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

import main_result_figure1_v1 as frozen_main_result

from generate_multinomial_baseline_comparison import (
    DEFAULT_CUTOFF_DATE,
    DEFAULT_OUTPUT_DIR,
    INPUT_PATHS,
    _reconstruct_original_phase3plus_accuracy,
    apply_figure_style,
    calculate_classification_metrics,
    derive_phase_labels,
    file_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CODESPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_COUNTRY_LOOKUP = (
    CODESPACE_ROOT / "0.Archived" / "new_merge_0108_with_country_code.csv"
)
EARTH_RADIUS_KM = 6371.0088
MAIN_RESULT_REFERENCES = frozen_main_result.classification_references()


def load_country_lookup(path: Path = DEFAULT_COUNTRY_LOOKUP) -> pd.DataFrame:
    """Load and validate the stable area-to-country mapping."""
    lookup = pd.read_csv(path, usecols=["area_id", "country_code_3"])
    if lookup[["area_id", "country_code_3"]].isna().any().any():
        raise ValueError("Country lookup contains missing area_id or country_code_3 values.")
    lookup["area_id"] = lookup["area_id"].astype(int)
    country_counts = lookup.groupby("area_id")["country_code_3"].nunique()
    ambiguous = country_counts[country_counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            f"Some area_id values map to multiple countries: {ambiguous.index.tolist()}"
        )
    return lookup.drop_duplicates("area_id").sort_values("area_id").reset_index(drop=True)


def haversine_distance_km(
    target_lat: float,
    target_lon: float,
    candidate_lat: pd.Series | np.ndarray,
    candidate_lon: pd.Series | np.ndarray,
) -> np.ndarray:
    """Calculate great-circle distances from one target to many candidates."""
    lat1 = np.radians(float(target_lat))
    lon1 = np.radians(float(target_lon))
    lat2 = np.radians(np.asarray(candidate_lat, dtype=float))
    lon2 = np.radians(np.asarray(candidate_lon, dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        delta_lon / 2
    ) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _select_nearest_candidate(
    target: pd.Series,
    candidates: pd.DataFrame,
) -> tuple[pd.Series, float]:
    if candidates.empty:
        raise ValueError("Nearest-neighbor selection requires at least one candidate.")
    distances = haversine_distance_km(
        target["lat"], target["lon"], candidates["lat"], candidates["lon"]
    )
    minimum = float(distances.min())
    tied = candidates.loc[np.isclose(distances, minimum, rtol=0, atol=1e-9)].copy()
    selected_index = tied.sort_values("area_id").index[0]
    return candidates.loc[selected_index], minimum


def _month_gap(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def _prepare_inputs(
    data: pd.DataFrame,
    country_lookup: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "date",
        "area_id",
        "lat",
        "lon",
        "phase1_percent",
        "phase2_percent",
        "phase3_percent",
        "phase4_percent",
        "phase5_percent",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input data are missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Input data contain no observations.")

    prepared = data.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    prepared["area_id"] = prepared["area_id"].astype(int)
    if prepared.duplicated(["area_id", "date"]).any():
        raise ValueError("Input data contain duplicate (area_id, date) rows.")
    if prepared[["lat", "lon"]].isna().any().any():
        raise ValueError("Input data contain missing coordinates.")
    prepared["evaluation_phase"] = derive_phase_labels(prepared)

    countries = country_lookup[["area_id", "country_code_3"]].copy()
    countries["area_id"] = countries["area_id"].astype(int)
    if countries.groupby("area_id")["country_code_3"].nunique().gt(1).any():
        raise ValueError("Country lookup maps at least one area_id to multiple countries.")
    countries = countries.drop_duplicates("area_id")
    prepared = prepared.merge(countries, on="area_id", how="left", validate="many_to_one")
    if prepared["country_code_3"].isna().any():
        missing_areas = sorted(prepared.loc[prepared["country_code_3"].isna(), "area_id"].unique())
        raise ValueError(f"Country lookup is missing area_id values: {missing_areas}")
    return prepared.sort_values(["date", "area_id"]).reset_index(names="source_index")


def build_persistence_predictions(
    data: pd.DataFrame,
    country_lookup: pd.DataFrame,
    horizon: str,
    target_start: str = DEFAULT_CUTOFF_DATE,
) -> pd.DataFrame:
    """Predict each holdout phase from the latest allowable own or neighbor phase.

    ``nowcasting`` uses observations strictly before the target date. ``forecasting``
    uses observations on or before the issue date, defined as target date minus
    twelve calendar months. If no own history is available, the method searches
    backward to the latest date with a same-country candidate; only areas with no
    historical same-country candidate use a global neighbor on the latest
    available date. Equal-distance ties select the smallest ``area_id``.
    """
    if horizon not in {"forecasting", "nowcasting"}:
        raise ValueError("horizon must be either 'forecasting' or 'nowcasting'.")
    prepared = _prepare_inputs(data, country_lookup)
    target_boundary = pd.Timestamp(target_start)
    targets = prepared.loc[prepared["date"] >= target_boundary].copy()
    if targets.empty:
        raise ValueError("No target observations occur on or after target_start.")

    records: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        target_date = target["date"]
        if horizon == "forecasting":
            issue_date = target_date - pd.DateOffset(months=12)
            available = prepared.loc[prepared["date"] <= issue_date]
        else:
            issue_date = target_date
            available = prepared.loc[prepared["date"] < target_date]
        if available.empty:
            raise ValueError(f"No historical observations are available for {target_date.date()}.")

        own_history = available.loc[available["area_id"] == target["area_id"]]
        distance_km = 0.0
        if not own_history.empty:
            source_date = own_history["date"].max()
            candidates = own_history.loc[own_history["date"] == source_date]
            source = candidates.sort_values("area_id").iloc[0]
            source_method = "own_history"
        else:
            same_country = available.loc[
                available["country_code_3"] == target["country_code_3"]
            ]
            if not same_country.empty:
                source_date = same_country["date"].max()
                candidates = same_country.loc[same_country["date"] == source_date]
                source, distance_km = _select_nearest_candidate(target, candidates)
                source_method = "same_country_neighbor"
            else:
                source_date = available["date"].max()
                candidates = available.loc[available["date"] == source_date]
                source, distance_km = _select_nearest_candidate(target, candidates)
                source_method = "global_neighbor"

        records.append(
            {
                "task": "Forecasting" if horizon == "forecasting" else "Nowcasting",
                "target_index": int(target["source_index"]),
                "area_id": int(target["area_id"]),
                "target_country_code_3": target["country_code_3"],
                "target_date": target_date.strftime("%Y-%m-%d"),
                "issue_date": issue_date.strftime("%Y-%m-%d"),
                "actual_phase": int(target["evaluation_phase"]),
                "predicted_phase": int(source["evaluation_phase"]),
                "source_method": source_method,
                "source_area_id": int(source["area_id"]),
                "source_country_code_3": source["country_code_3"],
                "source_date": source["date"].strftime("%Y-%m-%d"),
                "distance_km": float(distance_km),
                "target_gap_months": _month_gap(target_date, source["date"]),
                "issue_gap_months": _month_gap(issue_date, source["date"]),
            }
        )
    return pd.DataFrame(records)


def _baseline_record(predictions: pd.DataFrame) -> dict[str, object]:
    metrics = calculate_classification_metrics(
        predictions["actual_phase"], predictions["predicted_phase"]
    )
    counts = predictions["source_method"].value_counts()
    return {
        "task": predictions["task"].iloc[0],
        "model": "Persistence baseline",
        "model_type": "Persistence",
        **metrics,
        "n_test": len(predictions),
        "own_history_count": int(counts.get("own_history", 0)),
        "same_country_neighbor_count": int(counts.get("same_country_neighbor", 0)),
        "global_neighbor_count": int(counts.get("global_neighbor", 0)),
        "target_definition": "highest cumulative IPC phase with share >= 0.20",
        "metric_source": "Generated from historical source phases by this script",
        "main_result_reference_id": frozen_main_result.FREEZE_ID,
        "environment_relation": "same_frozen_python_core_stack_and_test_population",
    }


def build_comparison_metrics(
    forecasting_predictions: pd.DataFrame,
    nowcasting_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Combine both persistence baselines with the frozen Main-result points."""
    baseline_records = [
        _baseline_record(forecasting_predictions),
        _baseline_record(nowcasting_predictions),
    ]
    main_result_records: list[dict[str, object]] = []
    for predictions in [forecasting_predictions, nowcasting_predictions]:
        task = predictions["task"].iloc[0]
        stored = MAIN_RESULT_REFERENCES[task]
        main_result_records.append(
            {
                "task": task,
                "model": "Main result",
                "model_type": "Main result",
                "overall_accuracy": stored["overall_accuracy"],
                "phase3plus_accuracy": _reconstruct_original_phase3plus_accuracy(
                    predictions["actual_phase"],
                    stored["phase3plus_recall"],
                    stored["phase3plus_precision"],
                ),
                "phase3plus_recall": stored["phase3plus_recall"],
                "phase3plus_precision": stored["phase3plus_precision"],
                "n_test": len(predictions),
                "own_history_count": np.nan,
                "same_country_neighbor_count": np.nan,
                "global_neighbor_count": np.nan,
                "target_definition": "highest cumulative IPC phase with share >= 0.20",
                "metric_source": (
                    f"{stored['metric_source']}; Phase 3+ accuracy reconstructed "
                    "from frozen recall/precision and holdout class counts"
                ),
                "main_result_reference_id": frozen_main_result.FREEZE_ID,
                "environment_relation": "frozen_main_result_reference",
            }
        )
    combined = pd.DataFrame(main_result_records + baseline_records)
    task_order = pd.Categorical(
        combined["task"], categories=["Forecasting", "Nowcasting"], ordered=True
    )
    model_order = pd.Categorical(
        combined["model_type"], categories=["Main result", "Persistence"], ordered=True
    )
    return (
        combined.assign(_task=task_order, _model=model_order)
        .sort_values(["_task", "_model"])
        .drop(columns=["_task", "_model"])
        .reset_index(drop=True)
    )


def build_source_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage, distance, and temporal gap by fallback source."""
    audit = (
        predictions.groupby(["task", "source_method"], observed=True)
        .agg(
            observation_count=("area_id", "size"),
            distance_km_median=("distance_km", "median"),
            distance_km_max=("distance_km", "max"),
            target_gap_months_median=("target_gap_months", "median"),
            target_gap_months_max=("target_gap_months", "max"),
            issue_gap_months_median=("issue_gap_months", "median"),
            issue_gap_months_max=("issue_gap_months", "max"),
        )
        .reset_index()
    )
    totals = audit.groupby("task")["observation_count"].transform("sum")
    audit["observation_share"] = audit["observation_count"] / totals
    return audit


def create_comparison_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Create the dedicated four-point precision-recall comparison."""
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
        for model_type in ["Main result", "Persistence"]:
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
        ("Forecasting", "Main result"): (-12, -27, "right"),
        ("Nowcasting", "Main result"): (-94, 17, "right"),
        ("Forecasting", "Persistence"): (11, 15, "left"),
        ("Nowcasting", "Persistence"): (11, -22, "left"),
    }
    for row in metrics.itertuples(index=False):
        dx, dy, alignment = label_specs[(row.task, row.model_type)]
        method = "main result" if row.model_type == "Main result" else "persistence"
        ax.annotate(
            f"{row.task} {method}\n3+ accuracy = {row.phase3plus_accuracy:.2f}",
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
        "Persistence baselines improve precision but miss more Phase 3+ crises",
        loc="left",
        pad=17,
    )
    ax.text(
        0,
        1.02,
        "Horizon-aligned 2022 holdout: 12-month forecasting and current-date nowcasting",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#4D4D4D",
    )
    ax.set_xlabel("Sensitivity (recall), IPC Phase 3+")
    ax.set_ylabel("Precision, IPC Phase 3+")
    ax.set_xlim(0.58, 0.975)
    ax.set_ylim(0.76, 0.87)
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
    return fig


def run_analysis(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
) -> dict[str, Path]:
    """Generate predictions, audits, metrics, and the dedicated comparison figure."""
    frozen_main_result.assert_frozen_environment(("matplotlib",))
    data = pd.read_csv(INPUT_PATHS["Forecasting"])
    countries = load_country_lookup(country_lookup_path)
    forecasting = build_persistence_predictions(data, countries, "forecasting")
    nowcasting = build_persistence_predictions(data, countries, "nowcasting")
    metrics = build_comparison_metrics(forecasting, nowcasting)
    audit = build_source_audit(pd.concat([forecasting, nowcasting], ignore_index=True))
    audit["main_result_reference_id"] = frozen_main_result.FREEZE_ID
    audit["main_result_environment_id"] = frozen_main_result.ENVIRONMENT[
        "environment_id"
    ]
    audit["main_result_overall_accuracy"] = audit["task"].map(
        {
            task: reference["overall_accuracy"]
            for task, reference in MAIN_RESULT_REFERENCES.items()
        }
    )
    audit["main_result_phase3plus_precision"] = audit["task"].map(
        {
            task: reference["phase3plus_precision"]
            for task, reference in MAIN_RESULT_REFERENCES.items()
        }
    )
    audit["main_result_phase3plus_recall"] = audit["task"].map(
        {
            task: reference["phase3plus_recall"]
            for task, reference in MAIN_RESULT_REFERENCES.items()
        }
    )
    audit["main_result_phase3plus_r2"] = audit["task"].map(
        {
            task: reference["phase3plus_r2"]
            for task, reference in MAIN_RESULT_REFERENCES.items()
        }
    )
    audit["python_version"] = platform.python_version()
    audit["numpy_version"] = np.__version__
    audit["pandas_version"] = pd.__version__
    audit["sklearn_version"] = sklearn.__version__
    audit["matplotlib_version"] = mpl.__version__
    audit["input_path"] = str(INPUT_PATHS["Forecasting"])
    audit["input_sha256"] = file_sha256(INPUT_PATHS["Forecasting"])
    audit["country_lookup_path"] = str(country_lookup_path)
    audit["country_lookup_sha256"] = file_sha256(country_lookup_path)
    audit["generator_sha256"] = file_sha256(Path(__file__))
    audit["freeze_source_path"] = str(Path(frozen_main_result.__file__))
    audit["freeze_source_sha256"] = file_sha256(Path(frozen_main_result.__file__))
    fig = create_comparison_figure(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "persistence_baseline_metrics.csv",
        "audit": output_dir / "persistence_baseline_source_audit.csv",
        "forecasting_predictions": output_dir
        / "persistence_baseline_forecasting_predictions.csv",
        "nowcasting_predictions": output_dir
        / "persistence_baseline_nowcasting_predictions.csv",
        "jpg": output_dir / "precision_recall_scatter_with_persistence_baselines.jpg",
        "png": output_dir / "precision_recall_scatter_with_persistence_baselines.png",
        "pdf": output_dir / "precision_recall_scatter_with_persistence_baselines.pdf",
    }
    metrics.to_csv(paths["metrics"], index=False)
    audit.to_csv(paths["audit"], index=False)
    forecasting.to_csv(paths["forecasting_predictions"], index=False)
    nowcasting.to_csv(paths["nowcasting_predictions"], index=False)
    fig.savefig(paths["jpg"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--country-lookup", type=Path, default=DEFAULT_COUNTRY_LOOKUP
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(args.output_dir, args.country_lookup)
    for artifact, path in paths.items():
        print(f"{artifact}: {path}")


if __name__ == "__main__":
    main()
