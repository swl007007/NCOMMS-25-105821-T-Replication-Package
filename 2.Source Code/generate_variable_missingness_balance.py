"""Report predictor-level missingness by country, year, and month."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
OUTPUT_PATH = REPO_ROOT / "2.Source Code" / "produced_graph" / "variable_missingness_balance.csv"
MODEL_INPUTS = {
    "Forecasting": SOURCE_DATA_DIR / "Forecasting_Analysis_010825.csv",
    "Nowcasting": SOURCE_DATA_DIR / "Nowcasting_Analysis_010825.csv",
}
COUNTRY_LOOKUP_PATH = SOURCE_DATA_DIR / "area_country_lookup.csv"
KEY_COLUMNS = ["area_id", "date"]
EXCLUDED_COLUMNS = {
    *KEY_COLUMNS,
    "country_code_3",
    "overall_phase",
    *(f"phase{i}_percent" for i in range(1, 6)),
    *(f"phase{i}_worse" for i in range(2, 6)),
}
GROUP_COLUMNS = {"country": "country_code_3", "year": "year", "month": "month"}
COUNTRY_MIN_N = 20
EXPECTED_ROWS = 5_575
EXPECTED_PREDICTOR_COUNTS = {"Forecasting": 106, "Nowcasting": 173}
EFFECT_SIZE_THRESHOLD = 0.10
RATE_DIFFERENCE_THRESHOLD = 0.10


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return BH-adjusted p-values without changing missing values."""
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return adjusted
    scaled = valid.to_numpy(dtype=float) * len(valid) / np.arange(1, len(valid) + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1].clip(0.0, 1.0)
    adjusted.loc[valid.index] = scaled
    return adjusted


def summarize_variable_dimension(
    data: pd.DataFrame, *, model: str, variable: str, dimension: str
) -> pd.DataFrame:
    """Return group-level missingness and one shared balance result."""
    group_column = GROUP_COLUMNS[dimension]
    missing = data[variable].isna()
    summary = (
        pd.DataFrame({group_column: data[group_column], "missing": missing})
        .groupby(group_column, as_index=False, observed=True)
        .agg(n=("missing", "size"), missing_n=("missing", "sum"))
    )
    summary["nonmissing_n"] = summary["n"] - summary["missing_n"]
    summary["missing_rate"] = summary["missing_n"] / summary["n"]
    summary["eligible_for_balance"] = (
        summary["n"] >= COUNTRY_MIN_N if dimension == "country" else True
    )
    analysis = summary.loc[summary["eligible_for_balance"]].copy()
    analysis_n = int(analysis["n"].sum())
    test_group_count = len(analysis)
    overall_missing_rate = (
        float(analysis["missing_n"].sum() / analysis_n) if analysis_n else math.nan
    )
    max_rate_difference = (
        float((analysis["missing_rate"] - overall_missing_rate).abs().max())
        if analysis_n
        else math.nan
    )
    chi2 = dof = p_value = cramers_v = math.nan
    test_status = "tested"
    material_imbalance = False

    if test_group_count < 2:
        test_status = "untestable_fewer_than_two_eligible_groups"
    elif analysis["missing_n"].sum() in (0, analysis_n):
        test_status = "untestable_constant_missingness"
    else:
        contingency = analysis[["missing_n", "nonmissing_n"]].to_numpy()
        chi2, p_value, dof, expected = chi2_contingency(contingency, correction=False)
        cramers_v = math.sqrt(chi2 / (analysis_n * min(contingency.shape[0] - 1, 1)))
        material_imbalance = bool(
            cramers_v >= EFFECT_SIZE_THRESHOLD
            and max_rate_difference >= RATE_DIFFERENCE_THRESHOLD
        )
        if (expected < 5).any():
            p_value = math.nan
            test_status = "tested_sparse_expected_counts"

    summary = summary.rename(columns={group_column: "group"})
    summary["group"] = summary["group"].astype(str)
    summary.insert(0, "dimension", dimension)
    summary.insert(0, "variable", variable)
    summary.insert(0, "model", model)
    summary["test_population_n"] = analysis_n
    summary["test_group_count"] = test_group_count
    summary["overall_missing_rate"] = overall_missing_rate
    summary["chi2"] = chi2
    summary["degrees_of_freedom"] = dof
    summary["p_value"] = p_value
    summary["cramers_v"] = cramers_v
    summary["max_abs_missing_rate_diff"] = max_rate_difference
    summary["material_imbalance"] = material_imbalance
    summary["test_status"] = test_status
    return summary


def _validate_keys(data: pd.DataFrame, model: str) -> None:
    if set(KEY_COLUMNS).difference(data.columns):
        raise ValueError(f"{model} is missing required keys: {KEY_COLUMNS}")
    if data[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{model} contains missing area_id or date values.")
    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{model} contains duplicate (area_id, date) keys.")


def _load_model_data(
    model: str, path: Path, country_lookup: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(path)
    _validate_keys(data, model)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"{model} has {len(data)} rows, expected {EXPECTED_ROWS}.")
    features = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    if len(features) != EXPECTED_PREDICTOR_COUNTS[model]:
        raise ValueError(
            f"{model} has {len(features)} predictors, expected "
            f"{EXPECTED_PREDICTOR_COUNTS[model]}."
        )
    data = data.merge(country_lookup, on="area_id", how="left", validate="many_to_one")
    if data["country_code_3"].isna().any():
        raise ValueError(f"{model} has area_id values without country mappings.")
    date = pd.to_datetime(data.pop("date"), format="%Y-%m-%d", errors="raise")
    data = pd.concat(
        [
            data,
            date.rename("date"),
            date.dt.year.rename("year"),
            date.dt.month.rename("month"),
        ],
        axis=1,
    )
    return data, features


def build_missingness_report() -> pd.DataFrame:
    """Build the complete long-form balance table from the package inputs."""
    country_lookup = pd.read_csv(COUNTRY_LOOKUP_PATH)
    if country_lookup.columns.tolist() != ["area_id", "country_code_3"]:
        raise ValueError("Country lookup must contain only area_id and country_code_3.")
    if country_lookup.isna().any().any() or country_lookup.duplicated("area_id").any():
        raise ValueError("Country lookup must be complete and one country per area_id.")

    prepared = {
        model: _load_model_data(model, path, country_lookup)
        for model, path in MODEL_INPUTS.items()
    }
    first, second = (prepared[model][0] for model in MODEL_INPUTS)
    if not first.set_index(KEY_COLUMNS).index.equals(second.set_index(KEY_COLUMNS).index):
        raise ValueError("Forecasting and Nowcasting inputs do not have the same keys.")

    rows = []
    for model, (data, features) in prepared.items():
        for variable in features:
            for dimension in GROUP_COLUMNS:
                rows.append(
                    summarize_variable_dimension(
                        data, model=model, variable=variable, dimension=dimension
                    )
                )
    report = pd.concat(rows, ignore_index=True)
    tests = report[["model", "variable", "dimension", "p_value"]].drop_duplicates()
    tests["p_value_bh"] = tests.groupby(
        ["model", "dimension"], group_keys=False
    )["p_value"].apply(benjamini_hochberg)
    report = report.merge(tests, on=["model", "variable", "dimension", "p_value"], how="left")
    return report.sort_values(
        ["model", "variable", "dimension", "group"], kind="mergesort"
    ).reset_index(drop=True)


def validate_report(report: pd.DataFrame) -> None:
    """Reject a report that no longer accounts for every source observation."""
    expected = sum(
        EXPECTED_PREDICTOR_COUNTS[model]
        * sum(report.loc[report["model"] == model, "dimension"].eq(dimension).sum() // EXPECTED_PREDICTOR_COUNTS[model] for dimension in GROUP_COLUMNS)
        for model in MODEL_INPUTS
    )
    if len(report) != expected:
        raise ValueError("Report row count does not match its model-variable-group coverage.")
    grouped_n = report.groupby(["model", "variable", "dimension"], observed=True)["n"].sum()
    if not grouped_n.eq(EXPECTED_ROWS).all():
        raise ValueError("At least one variable-dimension result lost source observations.")
    if not report["missing_n"].between(0, report["n"]).all():
        raise ValueError("Missing counts fall outside their group sizes.")


def main() -> None:
    report = build_missingness_report()
    validate_report(report)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False, float_format="%.12g")
    print(f"Wrote {len(report)} rows to {OUTPUT_PATH}")
    print(
        report.groupby(["model", "dimension"], observed=True)["material_imbalance"]
        .any()
        .to_string()
    )


if __name__ == "__main__":
    main()
