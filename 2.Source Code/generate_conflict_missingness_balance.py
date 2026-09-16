"""Compare observation-level predictor missingness by conflict severity."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import generate_variable_missingness_balance as missingness


OUTPUT_PATH = (
    missingness.REPO_ROOT
    / "2.Source Code"
    / "produced_graph"
    / "missingness_sensitivity"
    / "conflict_missingness_balance_test.csv"
)
CONFLICT_COLUMNS = (
    "fatalities_battles",
    "fatalities_explosions",
    "fatalities_violence",
)
SEVERE_THRESHOLD = 10.0
EXPECTED_SEVERE_N = 256
EXPECTED_NONSEVERE_N = 5_319


def summarize_group(values: pd.Series, area_id: pd.Series) -> dict[str, float | int]:
    return {
        "n_observations": int(len(values)),
        "n_areas": int(area_id.nunique()),
        "missing_mean": float(values.mean()),
        "missing_sd": float(values.std(ddof=1)),
        "missing_median": float(values.median()),
        "missing_q1": float(values.quantile(0.25)),
        "missing_q3": float(values.quantile(0.75)),
    }


def balance_classification(smd: float) -> str:
    magnitude = abs(smd)
    if magnitude < 0.10:
        return "balanced"
    if magnitude <= 0.20:
        return "slight_imbalance"
    return "clear_imbalance"


def analyze_task(
    task: str,
    data: pd.DataFrame,
    features: list[str],
    conflict_profile: pd.DataFrame,
) -> dict[str, object]:
    analysis = data.merge(
        conflict_profile,
        on=missingness.KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if analysis["severe_conflict"].isna().any():
        raise ValueError(f"{task} observations are missing conflict profiles.")
    analysis["missing_feature_count"] = (
        analysis[features]
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        .sum(axis=1)
        .astype(int)
    )

    severe = analysis["severe_conflict"].astype(bool)
    severe_stats = summarize_group(
        analysis.loc[severe, "missing_feature_count"],
        analysis.loc[severe, "area_id"],
    )
    nonsevere_stats = summarize_group(
        analysis.loc[~severe, "missing_feature_count"],
        analysis.loc[~severe, "area_id"],
    )
    difference = severe_stats["missing_mean"] - nonsevere_stats["missing_mean"]
    pooled_sd = np.sqrt(
        (
            (severe_stats["n_observations"] - 1) * severe_stats["missing_sd"] ** 2
            + (nonsevere_stats["n_observations"] - 1)
            * nonsevere_stats["missing_sd"] ** 2
        )
        / (len(analysis) - 2)
    )
    smd = float(difference / pooled_sd)

    design = pd.DataFrame(
        {
            "const": 1.0,
            "severe_conflict": severe.astype(float),
        }
    )
    fitted = sm.OLS(analysis["missing_feature_count"].astype(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis["area_id"], "use_correction": True},
        use_t=True,
    )
    interval = fitted.conf_int().loc["severe_conflict"]

    row: dict[str, object] = {
        "task": task,
        "feature_count": len(features),
        "n_observations": len(analysis),
        "n_areas": int(analysis["area_id"].nunique()),
        "severe_conflict_definition": (
            "fatalities_battles + fatalities_explosions + fatalities_violence >= 10"
        ),
    }
    for prefix, values in (("severe", severe_stats), ("nonsevere", nonsevere_stats)):
        row.update({f"{prefix}_{name}": value for name, value in values.items()})
    row.update(
        {
            "mean_difference_severe_minus_nonsevere": float(difference),
            "cluster_robust_se": float(fitted.bse["severe_conflict"]),
            "ci95_low": float(interval.iloc[0]),
            "ci95_high": float(interval.iloc[1]),
            "p_value": float(fitted.pvalues["severe_conflict"]),
            "smd": smd,
            "balance_classification": balance_classification(smd),
            "cluster_count": int(analysis["area_id"].nunique()),
            "cluster_inference_df": int(analysis["area_id"].nunique() - 1),
        }
    )
    return row


def build_results() -> pd.DataFrame:
    country_lookup = pd.read_csv(missingness.COUNTRY_LOOKUP_PATH)
    prepared = {
        task: missingness._load_model_data(task, path, country_lookup)
        for task, path in missingness.MODEL_INPUTS.items()
    }
    nowcasting = prepared["Nowcasting"][0]
    if nowcasting[list(CONFLICT_COLUMNS)].isna().any().any():
        raise ValueError("Conflict-profile variables contain missing values.")
    conflict_profile = nowcasting[missingness.KEY_COLUMNS].copy()
    conflict_profile["severe_conflict"] = (
        nowcasting[list(CONFLICT_COLUMNS)].sum(axis=1).ge(SEVERE_THRESHOLD).astype(int)
    )
    counts = conflict_profile["severe_conflict"].value_counts().to_dict()
    if counts != {0: EXPECTED_NONSEVERE_N, 1: EXPECTED_SEVERE_N}:
        raise ValueError(f"Conflict-profile group counts changed: {counts}")

    result = pd.DataFrame(
        analyze_task(task, data, features, conflict_profile)
        for task, (data, features) in prepared.items()
    )
    result["p_value_holm"] = multipletests(result["p_value"], method="holm")[1]
    return result


def validate_results(result: pd.DataFrame) -> None:
    if result["task"].tolist() != ["Forecasting", "Nowcasting"]:
        raise ValueError("Expected one ordered result row per task.")
    if result["feature_count"].tolist() != [106, 173]:
        raise ValueError("Predictor counts changed.")
    if not result["n_observations"].eq(missingness.EXPECTED_ROWS).all():
        raise ValueError("A task lost source observations.")
    if not (
        result["severe_n_observations"].eq(EXPECTED_SEVERE_N).all()
        and result["nonsevere_n_observations"].eq(EXPECTED_NONSEVERE_N).all()
    ):
        raise ValueError("Conflict groups do not cover the approved population.")
    numeric = result.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Balance results contain non-finite values.")
    if not result["p_value"].between(0.0, 1.0).all() or not result[
        "p_value_holm"
    ].between(0.0, 1.0).all():
        raise ValueError("P-values lie outside zero and one.")


def main() -> None:
    result = build_results()
    validate_results(result)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".conflict_balance_", dir=OUTPUT_PATH.parent) as name:
        staged = Path(name) / OUTPUT_PATH.name
        result.to_csv(staged, index=False, float_format="%.17g", lineterminator="\n")
        validate_results(pd.read_csv(staged, float_precision="round_trip"))
        os.replace(staged, OUTPUT_PATH)
    print(result.to_string(index=False))
    print(f"Wrote {len(result)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
