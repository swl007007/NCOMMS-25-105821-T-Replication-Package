"""Aggregate actual and modeled area-level Phase 3+ population to country-date.

Repeated model geometries that carry the same IPC population profile are averaged
and counted once before country aggregation. The output covers only the analyzed
IPC population present in ``All_prediction.csv``; it must not be interpreted as
the full national population or summed across dates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import generate_all_prediction_temporal_test as temporal_contract


PREDICTION_COLUMNS = {"area_id", "date", "phase3_pred", "phase3_nowcast"}
PROFILE_COLUMNS = [
    "estimated_population",
    "overall_phase",
    "phase1_population",
    "phase2_population",
    "phase3_population",
    "phase4_population",
    "phase5_population",
]
POPULATION_COLUMNS = {
    "area_id",
    "date",
    "country_code_3",
    "country_en",
    *PROFILE_COLUMNS,
}
KEY_COLUMNS = ["area_id", "date"]


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["area_id"] = pd.to_numeric(
        normalized["area_id"], errors="raise"
    ).astype(int)
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise")
    return normalized


def _reject_duplicate_keys(frame: pd.DataFrame, label: str) -> None:
    duplicates = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"{label} has duplicate area_id/date keys: {examples}")


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


def aggregate_country_phase3plus_population(
    predictions: pd.DataFrame, population_lookup: pd.DataFrame
) -> pd.DataFrame:
    """Sum predicted Phase 3+ people across covered areas by country and date."""
    _require_columns(predictions, PREDICTION_COLUMNS, "predictions")
    _require_columns(population_lookup, POPULATION_COLUMNS, "population lookup")

    prediction_data = _normalize_keys(predictions[list(PREDICTION_COLUMNS)])
    population_data = _normalize_keys(population_lookup[list(POPULATION_COLUMNS)])
    _reject_duplicate_keys(prediction_data, "predictions")
    _reject_duplicate_keys(population_data, "population lookup")

    for field in ("phase3_pred", "phase3_nowcast"):
        prediction_data[field] = pd.to_numeric(
            prediction_data[field], errors="raise"
        )
        if prediction_data[field].isna().any() or not prediction_data[field].between(
            0, 1
        ).all():
            raise ValueError(f"{field} must contain non-missing proportions in [0, 1]")

    for field in PROFILE_COLUMNS:
        population_data[field] = pd.to_numeric(population_data[field], errors="raise")

    merged = prediction_data.merge(
        population_data,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        examples = merged.loc[unmatched, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"Missing population lookup for area_id/date keys: {examples}")
    merged = merged.drop(columns="_merge")

    if merged[PROFILE_COLUMNS].isna().any(axis=None):
        raise ValueError("IPC population profile fields must be non-missing")
    if not (merged["estimated_population"] > 0).all():
        raise ValueError("estimated_population must contain positive, non-missing counts")
    if merged[["country_code_3", "country_en"]].isna().any(axis=None):
        raise ValueError("country_code_3 and country_en must be non-missing")
    names_per_code = merged.groupby("country_code_3")["country_en"].nunique()
    if (names_per_code > 1).any():
        invalid_codes = names_per_code[names_per_code > 1].index.tolist()
        raise ValueError(f"Multiple country_en values for ISO3 codes: {invalid_codes}")

    unit_keys = ["country_code_3", "country_name", "date", *PROFILE_COLUMNS]
    population_units = (
        merged.rename(columns={"country_en": "country_name"})
        .groupby(unit_keys, as_index=False)
        .agg(
            source_geometry_count=("area_id", "nunique"),
            phase3_pred=("phase3_pred", "mean"),
            phase3_nowcast=("phase3_nowcast", "mean"),
        )
    )
    population_units["forecast_phase3plus_population"] = (
        population_units["phase3_pred"]
        * population_units["estimated_population"]
    )
    population_units["nowcast_phase3plus_population"] = (
        population_units["phase3_nowcast"]
        * population_units["estimated_population"]
    )
    population_units["actual_phase3plus_population"] = population_units[
        ["phase3_population", "phase4_population", "phase5_population"]
    ].sum(axis=1)

    aggregated = population_units.groupby(
        ["country_code_3", "country_name", "date"], as_index=False
    ).agg(
        source_geometry_count=("source_geometry_count", "sum"),
        population_profile_count=("estimated_population", "size"),
        analyzed_population=("estimated_population", "sum"),
        actual_phase3plus_population=("actual_phase3plus_population", "sum"),
        forecast_phase3plus_population=("forecast_phase3plus_population", "sum"),
        nowcast_phase3plus_population=("nowcast_phase3plus_population", "sum"),
    )
    aggregated["actual_phase3plus_share"] = (
        aggregated["actual_phase3plus_population"]
        / aggregated["analyzed_population"]
    )
    aggregated["forecast_phase3plus_share"] = (
        aggregated["forecast_phase3plus_population"]
        / aggregated["analyzed_population"]
    )
    aggregated["nowcast_phase3plus_share"] = (
        aggregated["nowcast_phase3plus_population"]
        / aggregated["analyzed_population"]
    )
    column_order = [
        "country_code_3",
        "country_name",
        "date",
        "source_geometry_count",
        "population_profile_count",
        "analyzed_population",
        "actual_phase3plus_population",
        "actual_phase3plus_share",
        "forecast_phase3plus_population",
        "forecast_phase3plus_share",
        "nowcast_phase3plus_population",
        "nowcast_phase3plus_share",
    ]
    return aggregated[column_order].sort_values(
        ["date", "country_code_3"]
    ).reset_index(drop=True)


def default_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    codespace_root = repo_root.parents[1]
    return {
        "predictions": repo_root / "1.Source Data" / "All_prediction.csv",
        "population": codespace_root / "0.Archived" / "new_merge_0107.csv",
        "output": repo_root
        / "1.Source Data"
        / "All_prediction_country_phase3plus_population.csv",
    }


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=paths["predictions"])
    parser.add_argument("--population-data", type=Path, default=paths["population"])
    parser.add_argument("--output", type=Path, default=paths["output"])
    args = parser.parse_args()

    predictions = load_prediction_input(args.predictions)
    population_lookup = pd.read_csv(
        args.population_data,
        usecols=sorted(POPULATION_COLUMNS),
        low_memory=False,
    )
    result = aggregate_country_phase3plus_population(
        predictions, population_lookup
    )
    output = result.copy()
    population_fields = [
        "analyzed_population",
        "actual_phase3plus_population",
        "forecast_phase3plus_population",
        "nowcast_phase3plus_population",
    ]
    share_fields = [
        "actual_phase3plus_share",
        "forecast_phase3plus_share",
        "nowcast_phase3plus_share",
    ]
    output[population_fields] = output[population_fields].round(2)
    output[share_fields] = output[share_fields].round(6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(
        f"Wrote {len(output)} country-date rows for "
        f"{output['country_code_3'].nunique()} countries: {args.output}"
    )


if __name__ == "__main__":
    main()
