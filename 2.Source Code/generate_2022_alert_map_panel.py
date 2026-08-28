"""Generate a replicated 2x3 panel of 2022 food-crisis alert maps.

The aggregation intentionally preserves the original supplementary notebook:
one record per area is selected by maximum ``phase3_pred``. The Phase 3+
forecasting, nowcasting, and actual alerts all come from that selected record.
Actual severity is separately merged from each area's maximum observed share of
the population in Phase 3 or above. An optional comparison mode independently
selects each area's annual minimum forecasting, nowcasting, and actual severity.
Top-30% alerts use the 70th percentile and ``>=`` exactly as in the original code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import generate_all_prediction_temporal_test as temporal_contract


YEAR = 2022
TOP30_QUANTILE = 0.70
FIXED_SEVERITY_CUTOFF = 0.20


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_prediction_input(
    path: Path, require_canonical: bool = True
) -> pd.DataFrame:
    """Load predictions and enforce the current 1,170-row contract by default."""
    data = pd.read_csv(path)
    if not require_canonical:
        return data
    return temporal_contract.validate_canonical_prediction_artifact(
        data, expected_rows=temporal_contract.EXPECTED_TEST_ROWS
    )


def prepare_area_records(
    predictions: pd.DataFrame, actual: pd.DataFrame, year: int = YEAR
) -> pd.DataFrame:
    """Reproduce the original per-area record selection and phase alerts."""
    prediction_columns = {
        "area_id",
        "date",
        "phase3_pred",
        "phase3_nowcast",
        "overall_phase",
        "overall_phase_pred",
        "nowcast_predict",
        "lat",
        "lon",
    }
    actual_columns = {
        "area_id",
        "date",
        "phase3_percent",
        "phase4_percent",
        "phase5_percent",
    }
    _require_columns(predictions, prediction_columns, "predictions")
    _require_columns(actual, actual_columns, "actual data")

    selected = predictions.copy()
    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected[selected["date"].dt.year == year].copy()
    selected["area_id"] = pd.to_numeric(selected["area_id"], errors="raise").astype(int)
    selected = selected.sort_values("phase3_pred", ascending=False).drop_duplicates(
        "area_id"
    )

    actual_severity = actual[list(actual_columns)].copy()
    actual_severity["date"] = pd.to_datetime(actual_severity["date"])
    actual_severity = actual_severity[actual_severity["date"].dt.year == year].copy()
    actual_severity["area_id"] = pd.to_numeric(
        actual_severity["area_id"], errors="raise"
    ).astype(int)
    actual_severity["phase3_actual"] = actual_severity[
        ["phase3_percent", "phase4_percent", "phase5_percent"]
    ].sum(axis=1)
    actual_severity = (
        actual_severity.sort_values("phase3_actual", ascending=False)
        .drop_duplicates("area_id")[["area_id", "phase3_actual"]]
    )

    selected = selected.merge(
        actual_severity, on="area_id", how="left", validate="one_to_one"
    )
    if selected["phase3_actual"].isna().any():
        missing = selected.loc[selected["phase3_actual"].isna(), "area_id"].tolist()
        raise ValueError(f"Missing actual severity for area_id values: {missing[:10]}")

    selected["crisis_forecast"] = (selected["overall_phase_pred"] >= 3).astype(int)
    selected["crisis_nowcast"] = (selected["nowcast_predict"] >= 3).astype(int)
    selected["crisis_actual"] = (selected["overall_phase"] >= 3).astype(int)
    return selected.sort_values("area_id").reset_index(drop=True)


def prepare_minimum_severity_area_records(
    predictions: pd.DataFrame, actual: pd.DataFrame, year: int = YEAR
) -> pd.DataFrame:
    """Select each area's annual minimum severity independently by modality."""
    prediction_columns = {"area_id", "date", "phase3_pred", "phase3_nowcast"}
    actual_columns = {
        "area_id",
        "date",
        "phase3_percent",
        "phase4_percent",
        "phase5_percent",
    }
    _require_columns(predictions, prediction_columns, "predictions")
    _require_columns(actual, actual_columns, "actual data")

    prediction_severity = predictions[list(prediction_columns)].copy()
    prediction_severity["date"] = pd.to_datetime(prediction_severity["date"])
    prediction_severity = prediction_severity[
        prediction_severity["date"].dt.year == year
    ].copy()
    prediction_severity["area_id"] = pd.to_numeric(
        prediction_severity["area_id"], errors="raise"
    ).astype(int)
    prediction_severity = (
        prediction_severity.groupby("area_id", as_index=False)[
            ["phase3_pred", "phase3_nowcast"]
        ].min()
    )

    actual_severity = actual[list(actual_columns)].copy()
    actual_severity["date"] = pd.to_datetime(actual_severity["date"])
    actual_severity = actual_severity[actual_severity["date"].dt.year == year].copy()
    actual_severity["area_id"] = pd.to_numeric(
        actual_severity["area_id"], errors="raise"
    ).astype(int)
    actual_severity["phase3_actual"] = actual_severity[
        ["phase3_percent", "phase4_percent", "phase5_percent"]
    ].sum(axis=1)
    actual_severity = actual_severity.groupby("area_id", as_index=False)[
        "phase3_actual"
    ].min()

    selected = prediction_severity.merge(
        actual_severity, on="area_id", how="left", validate="one_to_one"
    )
    severity_fields = ["phase3_pred", "phase3_nowcast", "phase3_actual"]
    if selected[severity_fields].isna().any(axis=None):
        missing = selected.loc[
            selected[severity_fields].isna().any(axis=1), "area_id"
        ].tolist()
        raise ValueError(f"Missing minimum severity for area_id values: {missing[:10]}")
    return selected.sort_values("area_id").reset_index(drop=True)


def add_top30_flags(
    records: pd.DataFrame, quantile: float = TOP30_QUANTILE
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add original-style top-30% flags for the three severity proxies."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    severity_fields = {
        "forecasting": ("phase3_pred", "top30_forecast"),
        "nowcasting": ("phase3_nowcast", "top30_nowcast"),
        "actual": ("phase3_actual", "top30_actual"),
    }
    _require_columns(
        records,
        {field for field, _ in severity_fields.values()},
        "area records",
    )

    flagged = records.copy()
    thresholds: dict[str, float] = {}
    for modality, (severity_field, flag_field) in severity_fields.items():
        threshold = float(flagged[severity_field].quantile(quantile))
        thresholds[modality] = threshold
        flagged[flag_field] = (flagged[severity_field] >= threshold).astype(int)
    return flagged, thresholds


def add_fixed_severity_alert_flags(
    records: pd.DataFrame, cutoff: float = FIXED_SEVERITY_CUTOFF
) -> pd.DataFrame:
    """Threshold the same Phase 3+ severity fields used by the top-30% row."""
    if not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between 0 and 1")

    severity_fields = {
        "phase3_pred": "severity_alert_forecast",
        "phase3_nowcast": "severity_alert_nowcast",
        "phase3_actual": "severity_alert_actual",
    }
    _require_columns(records, set(severity_fields), "area records")

    flagged = records.copy()
    for severity_field, flag_field in severity_fields.items():
        flagged[flag_field] = (flagged[severity_field] >= cutoff).astype(int)
    return flagged


def attach_geometry(records: pd.DataFrame, shapefile: Path):
    """Attach the archived EPSG:4326 polygons through the verified area_id key."""
    import geopandas as gpd

    boundaries = gpd.read_file(shapefile)
    _require_columns(boundaries, {"area_id", "geometry"}, "boundary shapefile")
    boundaries["area_id"] = pd.to_numeric(
        boundaries["area_id"], errors="raise"
    ).astype(int)
    boundary_columns = ["area_id", "geometry"]
    if {"lat", "lon"}.issubset(boundaries.columns):
        boundary_columns.extend(["lat", "lon"])
    boundaries = boundaries[boundary_columns].drop_duplicates("area_id")

    joined = boundaries.merge(
        records,
        on="area_id",
        how="inner",
        suffixes=("_boundary", ""),
        validate="one_to_one",
    )
    if len(joined) != len(records):
        missing = sorted(set(records["area_id"]) - set(joined["area_id"]))
        raise ValueError(f"Boundary coverage is incomplete; missing area_id: {missing[:10]}")

    gdf = gpd.GeoDataFrame(joined, geometry="geometry", crs=boundaries.crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(epsg=3857)


def plot_panel(
    gdf,
    output_path: Path,
    add_basemap: bool = True,
    severity_aligned_first_row: bool = False,
    annual_minimum_severity: bool = False,
) -> None:
    """Render the requested forecasting/nowcasting/actual 2x3 panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if annual_minimum_severity:
        panels = [
            ("severity_alert_forecast", "Forecasting: annual min Phase 3+ proxy >=20%", "fixed"),
            ("severity_alert_nowcast", "Nowcasting: annual min Phase 3+ proxy >=20%", "fixed"),
            ("severity_alert_actual", "Actual: annual min Phase 3+ share >=20%", "fixed"),
            ("top30_forecast", "Forecasting: annual min Phase 3+ proxy >=Q70", "top30"),
            ("top30_nowcast", "Nowcasting: annual min Phase 3+ proxy >=Q70", "top30"),
            ("top30_actual", "Actual: annual min Phase 3+ share >=Q70", "top30"),
        ]
    elif severity_aligned_first_row:
        panels = [
            ("severity_alert_forecast", "Forecasting: Phase 3+ proxy >=20% (2022)", "fixed"),
            ("severity_alert_nowcast", "Nowcasting: Phase 3+ proxy >=20% (2022)", "fixed"),
            ("severity_alert_actual", "Actual: max Phase 3+ share >=20% (2022)", "fixed"),
            ("top30_forecast", "Forecasting: Phase 3+ proxy >=Q70 (2022)", "top30"),
            ("top30_nowcast", "Nowcasting: Phase 3+ proxy >=Q70 (2022)", "top30"),
            ("top30_actual", "Actual: max Phase 3+ share >=Q70 (2022)", "top30"),
        ]
    else:
        panels = [
            ("crisis_forecast", "Forecasting Alert in 2022", "alert"),
            ("crisis_nowcast", "Nowcasting Alert in 2022", "alert"),
            ("crisis_actual", "Actual Alert in 2022", "alert"),
            ("top30_forecast", "Forecasting Alert Top 30% in 2022", "top30"),
            ("top30_nowcast", "Nowcasting Alert Top 30% in 2022", "top30"),
            ("top30_actual", "Actual Alert Top 30% in 2022", "top30"),
        ]
    legends = {
        "alert": (
            "Has Alert (Overall Phase >=3)",
            [Patch(color="green", label="No Crisis"), Patch(color="red", label="Crisis")],
        ),
        "top30": (
            (
                "Phase 3+ severity rank"
                if severity_aligned_first_row or annual_minimum_severity
                else "Crisis Top 30%"
            ),
            [
                Patch(
                    color="green",
                    label=(
                        "Below Q70"
                        if severity_aligned_first_row or annual_minimum_severity
                        else "Not in Top 30%"
                    ),
                ),
                Patch(
                    color="red",
                    label=(
                        "At/above Q70"
                        if severity_aligned_first_row or annual_minimum_severity
                        else "Top 30%"
                    ),
                ),
            ],
        ),
        "fixed": (
            "Phase 3+ severity cutoff",
            [Patch(color="green", label="<20%"), Patch(color="red", label=">=20%")],
        ),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 7.2))
    minx, miny, maxx, maxy = gdf.total_bounds
    basemap_warning = None

    for ax, (field, title, legend_kind) in zip(axes.flat, panels):
        gdf.plot(ax=ax, color="white", alpha=0.5, edgecolor="black", linewidth=0.1)
        for value, color in ((0, "green"), (1, "red")):
            subset = gdf[gdf[field] == value]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.1)

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        if add_basemap:
            try:
                import contextily as ctx

                ctx.add_basemap(
                    ax,
                    crs=gdf.crs.to_string(),
                    source=ctx.providers.CartoDB.Positron,
                    attribution=False,
                )
            except Exception as exc:  # Basemap is cosmetic; retain data map offline.
                basemap_warning = str(exc)
        ax.set_axis_off()
        legend_title, handles = legends[legend_kind]
        ax.legend(handles=handles, title=legend_title, loc="upper left", fontsize=8)
        ax.set_title(title, fontsize=12)

    if annual_minimum_severity:
        figure_title = "2022 Annual-Minimum Phase 3+ Severity: Fixed 20% vs Q70 Cutoffs"
    elif severity_aligned_first_row:
        figure_title = "2022 Phase 3+ Severity Alerts: Fixed 20% vs Q70 Cutoffs"
    else:
        figure_title = "2022 Food-Crisis Alert Maps: Forecasting, Nowcasting, and Actual"
    fig.suptitle(figure_title, fontsize=16)
    fig.text(
        0.005,
        0.005,
        "(C) OpenStreetMap contributors (C) CARTO" if add_basemap else "",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.94), pad=0.5, w_pad=0.35, h_pad=0.7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if basemap_warning:
        print(f"Warning: basemap unavailable; plotted data layers only: {basemap_warning}")


def build_diagnostics(
    gdf,
    thresholds: dict[str, float],
    severity_aligned_first_row: bool = False,
    annual_minimum_severity: bool = False,
) -> pd.DataFrame:
    """Create a compact audit table for thresholds and selected-area counts."""
    if severity_aligned_first_row or annual_minimum_severity:
        specifications = [
            ("forecasting", "phase3_pred", "severity_alert_forecast", "top30_forecast"),
            ("nowcasting", "phase3_nowcast", "severity_alert_nowcast", "top30_nowcast"),
            ("actual", "phase3_actual", "severity_alert_actual", "top30_actual"),
        ]
    else:
        specifications = [
            ("forecasting", "phase3_pred", "crisis_forecast", "top30_forecast"),
            ("nowcasting", "phase3_nowcast", "crisis_nowcast", "top30_nowcast"),
            ("actual", "phase3_actual", "crisis_actual", "top30_actual"),
        ]
    rows = []
    for modality, severity_field, alert_field, top30_field in specifications:
        rows.append(
            {
                "modality": modality,
                "areas": int(len(gdf)),
                "severity_field": severity_field,
                "top30_cutoff_q70": thresholds[modality],
                "phase3plus_alert_areas": int(gdf[alert_field].sum()),
                "top30_alert_areas": int(gdf[top30_field].sum()),
                "first_row_cutoff": (
                    (
                        FIXED_SEVERITY_CUTOFF
                        if severity_aligned_first_row or annual_minimum_severity
                        else "overall_phase>=3"
                    )
                ),
                "selection_record": (
                    "independent annual minimum severity per modality"
                    if annual_minimum_severity
                    else "area row with maximum phase3_pred"
                ),
            }
        )
    return pd.DataFrame(rows)


def default_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    codespace_root = repo_root.parents[1]
    return {
        "predictions": repo_root / "1.Source Data" / "All_prediction.csv",
        "actual": repo_root / "1.Source Data" / "Forecasting_Analysis_010825.csv",
        "shapefile": codespace_root
        / "0.Archived"
        / "New_analysis_dataset_for_vis"
        / "world_analysis.shp",
        "output": repo_root / "1.Source Data" / "2022_alert_map_panel_2x3.png",
        "aligned_output": repo_root
        / "1.Source Data"
        / "2022_alert_map_panel_2x3_severity_aligned.png",
        "diagnostics": repo_root
        / "1.Source Data"
        / "2022_alert_map_panel_2x3_diagnostics.csv",
        "aligned_diagnostics": repo_root
        / "1.Source Data"
        / "2022_alert_map_panel_2x3_severity_aligned_diagnostics.csv",
        "minimum_output": repo_root
        / "1.Source Data"
        / "2022_alert_map_panel_2x3_minimum_severity.png",
        "minimum_diagnostics": repo_root
        / "1.Source Data"
        / "2022_alert_map_panel_2x3_minimum_severity_diagnostics.csv",
    }


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=paths["predictions"])
    parser.add_argument("--actual", type=Path, default=paths["actual"])
    parser.add_argument("--shapefile", type=Path, default=paths["shapefile"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument(
        "--severity-aligned-first-row",
        action="store_true",
        help="Use the three Phase 3+ severity fields with a fixed 0.20 cutoff in row 1.",
    )
    parser.add_argument(
        "--minimum-severity-per-area",
        action="store_true",
        help=(
            "Independently select each area's annual minimum forecasting, "
            "nowcasting, and actual Phase 3+ severity for both rows."
        ),
    )
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Skip the CARTO basemap when running without network access.",
    )
    args = parser.parse_args()
    severity_aligned_first_row = (
        args.severity_aligned_first_row or args.minimum_severity_per_area
    )
    if args.output is None:
        if args.minimum_severity_per_area:
            args.output = paths["minimum_output"]
        elif args.severity_aligned_first_row:
            args.output = paths["aligned_output"]
        else:
            args.output = paths["output"]
    if args.diagnostics is None:
        if args.minimum_severity_per_area:
            args.diagnostics = paths["minimum_diagnostics"]
        elif args.severity_aligned_first_row:
            args.diagnostics = paths["aligned_diagnostics"]
        else:
            args.diagnostics = paths["diagnostics"]

    predictions = load_prediction_input(args.predictions)
    actual = pd.read_csv(args.actual)
    if args.minimum_severity_per_area:
        records = prepare_minimum_severity_area_records(predictions, actual, year=YEAR)
    else:
        records = prepare_area_records(predictions, actual, year=YEAR)
    records = add_fixed_severity_alert_flags(records)
    records, thresholds = add_top30_flags(records)
    gdf = attach_geometry(records, args.shapefile)
    plot_panel(
        gdf,
        args.output,
        add_basemap=not args.no_basemap,
        severity_aligned_first_row=severity_aligned_first_row,
        annual_minimum_severity=args.minimum_severity_per_area,
    )

    diagnostics = build_diagnostics(
        gdf,
        thresholds,
        severity_aligned_first_row=severity_aligned_first_row,
        annual_minimum_severity=args.minimum_severity_per_area,
    )
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.diagnostics, index=False)
    print(diagnostics.to_string(index=False))
    print(f"Map: {args.output}")
    print(f"Diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
