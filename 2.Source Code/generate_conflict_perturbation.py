"""Exploratory +10% conflict-input sensitivity, paired within one frozen fit.

Uses the pinned Figure 1 Windows environment. Fits four forecasting regressors
once, requires recovery of the saved baseline, then predicts both input states.
No main-result artifact or manuscript is changed.
"""

from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import xgboost as xgb

import generate_filtered_main_result_metrics as selected
import generate_leave_one_country_out_robustness as loco


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "2.Source Code/produced_graph/conflict_perturbation_10pct"
REFERENCE = ROOT / "2.Source Code/produced_graph/direct_phase3_vs_phase45_rescue/direct_phase3_vs_phase45_rescue_benchmark_predictions.csv"
REFERENCE_SHA256 = "b608e162672558b971e92dad906ce863d38814c0e532d5faf20bf88b1b7aad41"
FACTOR = 1.1


def summarize_transitions(before, after):
    before, after = np.asarray(before), np.asarray(after)
    assert before.shape == after.shape and len(before) > 0
    assert set(before) <= {1, 2, 3, 4, 5} and set(after) <= {1, 2, 3, 4, 5}
    n = len(before)
    summary = {"n_test": n}
    conditions = {
        "promotion": after > before,
        "demotion": after < before,
        "unchanged": after == before,
        "crisis_promotion": (before < 3) & (after >= 3),
        "crisis_demotion": (before >= 3) & (after < 3),
    }
    for name, mask in conditions.items():
        summary[name + "_count"] = int(mask.sum())
        summary[name + "_fraction_all"] = float(mask.mean())
    rows = []
    for phase in range(1, 6):
        denominator = int((before == phase).sum())
        for target in range(1, 6):
            count = int(((before == phase) & (after == target)).sum())
            rows.append({
                "baseline_predicted_phase": phase, "perturbed_predicted_phase": target,
                "count": count, "baseline_phase_denominator": denominator,
                "fraction_of_baseline_phase": count / denominator if denominator else np.nan,
                "fraction_of_all_test": count / n,
            })
    assert sum(summary[k + "_count"] for k in ("promotion", "demotion", "unchanged")) == n
    return summary, pd.DataFrame(rows)


def main():
    # Small runnable check: denominators are baseline predictions, never truth.
    check, cells = summarize_transitions([2, 2, 3, 4], [3, 2, 2, 4])
    assert check["promotion_count"] == check["demotion_count"] == 1
    assert cells.loc[(cells.baseline_predicted_phase == 2) & (cells.perturbed_predicted_phase == 3), "fraction_of_baseline_phase"].item() == 0.5
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing exploratory run: {OUTPUT}")
    selected.validate_environment()
    inputs = {p: selected.EXPECTED_FILE_SHA256[p] for p in (
        selected.DEFAULT_FORECASTING_INPUT, selected.DEFAULT_NOWCASTING_INPUT,
        selected.DEFAULT_COUNTRY_LOOKUP, selected.DEFAULT_GENERAL_PARAMS,
        selected.DEFAULT_PHASE3_PARAMS,
    )}
    inputs[REFERENCE] = REFERENCE_SHA256
    selected.validate_default_inputs(inputs)
    protected = [ROOT / "1.Source Data/All_prediction.csv", *ROOT.glob("writing/*.tex"),
                 *ROOT.glob("writing/*.docx"), *ROOT.glob("writing/*.pdf")]
    protected_hashes = {p: selected.file_sha256(p) for p in protected}
    forecasting, nowcasting = selected.load_model_inputs(
        selected.DEFAULT_FORECASTING_INPUT, selected.DEFAULT_NOWCASTING_INPUT,
        repeated_area_filter=False,
    )
    forecasting, _ = loco.prepare_model_inputs(
        forecasting, nowcasting, loco.load_country_lookup(selected.DEFAULT_COUNTRY_LOOKUP),
    )
    forecasting = loco.add_cumulative_targets(forecasting)
    train = pd.to_datetime(forecasting.date).lt("2022-01-01")
    test = ~train
    assert train.sum() == 4405 and test.sum() == 1170
    features = loco.select_layer1_features(forecasting)
    conflict = [c for c in features if c.startswith(("fatalities_", "event_count_"))]
    expected_conflict = {
        f"{measure}_{event}{suffix}"
        for measure in ("fatalities", "event_count")
        for event in ("battles", "explosions", "violence")
        for suffix in ("_l12", "_w5_l12", "_w10_l12", "_s12_l12", "_w5_s12_l12", "_w10_s12_l12")
    }
    assert len(features) == 106 and set(conflict) == expected_conflict
    original_x = forecasting.loc[test, features].reset_index(drop=True)
    perturbed_x = original_x.copy()
    perturbed_x[conflict] = perturbed_x[conflict] * FACTOR
    pd.testing.assert_frame_equal(original_x.drop(columns=conflict), perturbed_x.drop(columns=conflict))
    np.testing.assert_array_equal(original_x.isna(), perturbed_x.isna())
    np.testing.assert_array_equal(original_x[conflict].eq(0), perturbed_x[conflict].eq(0))
    general, phase3 = loco.load_hyperparameters(
        selected.DEFAULT_GENERAL_PARAMS, selected.DEFAULT_PHASE3_PARAMS,
        random_state=None, estimator_n_jobs=None,
    )
    models = []

    def capture_model(**params):
        print(f"Fitting phase {len(models) + 2}+ once...", flush=True)
        model = xgb.XGBRegressor(**params)
        models.append(model)
        return model

    baseline = loco.fit_forecasting_split(
        forecasting, train, test, "temporal_2022", general, phase3,
        estimator_factory=capture_model,
    )
    assert len(models) == 4 and len(baseline) == 1170
    assert selected.canonical_key_sha256(baseline) == selected.EXPECTED_FULL_COUNTS["test_key_sha256"]
    reference = pd.read_csv(REFERENCE, float_precision="round_trip")
    reference = reference.loc[reference.task.eq("Forecasting") & reference.method.eq("frozen_base")]
    for frame in (baseline, reference):
        frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    matched = baseline.merge(reference, on=["area_id", "date"], validate="one_to_one", suffixes=("", "_reference"))
    assert len(matched) == 1170
    np.testing.assert_array_equal(matched.overall_phase, matched.reconstructed_overall_phase)
    np.testing.assert_array_equal(matched.overall_phase_pred, matched.base_overall_phase_pred)
    maximum_share_difference = {}
    for phase in range(2, 6):
        pred, saved = matched[f"phase{phase}_pred"], matched[f"phase{phase}_pred_rounded"]
        np.testing.assert_allclose(pred, saved, rtol=0, atol=1e-7)
        np.testing.assert_allclose(matched[f"phase{phase}_test"], matched[f"phase{phase}_test_reference"], rtol=0, atol=1e-12)
        maximum_share_difference[str(phase)] = float(np.max(np.abs(pred - saved)))
    print("Baseline recovered: all 1,170 labels and four cumulative-share predictions match frozen references.", flush=True)
    # The exact same fitted estimators predict both states. No fit below here.
    perturbed = baseline.copy()
    paired = baseline[["area_id", "date", "country_code_3", "overall_phase", "overall_phase_pred"]].rename(
        columns={"overall_phase": "y_true", "overall_phase_pred": "baseline_predicted_phase"})
    for phase, model in zip(range(2, 6), models):
        original_raw = model.predict(original_x)
        modified_raw = model.predict(perturbed_x)
        np.testing.assert_array_equal(original_raw.round(2), baseline[f"phase{phase}_pred"])
        perturbed[f"phase{phase}_pred"] = modified_raw
        paired[f"phase{phase}_actual_share"] = baseline[f"phase{phase}_test"]
        paired[f"phase{phase}_baseline_raw"] = original_raw
        paired[f"phase{phase}_perturbed_raw"] = modified_raw
        paired[f"phase{phase}_baseline_rounded"] = baseline[f"phase{phase}_pred"]
        paired[f"phase{phase}_perturbed_rounded"] = modified_raw.round(2)
    perturbed = loco.wide_predictions_to_phases(perturbed)
    paired["perturbed_predicted_phase"] = perturbed.overall_phase_pred
    paired["phase_change"] = paired.perturbed_predicted_phase - paired.baseline_predicted_phase
    paired["direction"] = np.select([paired.phase_change.gt(0), paired.phase_change.lt(0)], ["promotion", "demotion"], default="unchanged")
    summary, transitions = summarize_transitions(paired.baseline_predicted_phase, paired.perturbed_predicted_phase)
    summary.update({"model": "Forecasting", "factor": FACTOR, "n_train": 4405,
                    "n_conflict_features": len(conflict), "n_model_fits": len(models)})
    summary["n_test_rows_with_changed_conflict_input"] = int((original_x[conflict].ne(perturbed_x[conflict]) & original_x[conflict].notna()).any(axis=1).sum())
    for name, source in (("crisis_promotion", paired.baseline_predicted_phase.lt(3)),
                         ("crisis_demotion", paired.baseline_predicted_phase.ge(3))):
        summary[name + "_baseline_denominator"] = int(source.sum())
        summary[name + "_fraction_baseline_group"] = summary[name + "_count"] / int(source.sum()) if source.any() else np.nan
    for phase in range(1, 6):
        summary[f"baseline_phase{phase}_n"] = int(paired.baseline_predicted_phase.eq(phase).sum())
        summary[f"perturbed_phase{phase}_n"] = int(paired.perturbed_predicted_phase.eq(phase).sum())
    for label, col in (("raw", "raw"), ("rounded", "rounded")):
        delta = paired[f"phase3_perturbed_{col}"] - paired[f"phase3_baseline_{col}"]
        summary[f"mean_phase3_share_delta_{label}"] = float(delta.mean())
        summary[f"median_phase3_share_delta_{label}"] = float(delta.median())
    selected.validate_default_inputs({**inputs, **protected_hashes})
    OUTPUT.mkdir(parents=True)
    paired.to_csv(OUTPUT / "paired_predictions.csv", index=False, float_format="%.17g")
    transitions.to_csv(OUTPUT / "phase_transitions.csv", index=False, float_format="%.17g")
    pd.DataFrame([summary]).to_csv(OUTPUT / "summary_metrics.csv", index=False, float_format="%.17g")
    for phase, model in zip(range(2, 6), models):
        model.save_model(OUTPUT / f"forecasting_phase{phase}_model.json")
    audit = {
        "status": "exploratory_completed", "frozen_baseline_recovered": True,
        "calibration_label_mismatches": 0, "calibration_max_share_difference": maximum_share_difference,
        "environment": selected.EXPECTED_ENVIRONMENT, "platform": platform.platform(),
        "general_params": general, "phase3_params": phase3, "features_in_order": features,
        "conflict_features_in_order": conflict, "input_multiplier": FACTOR,
        "feature_intervention": "multiply stored conflict inputs including spatial and s12-derived columns; zero and NaN unchanged",
        "comparison": "paired perturbed prediction versus baseline prediction, never versus y_true",
        "phase_rule": "same frozen rule: round predicted cumulative shares to 2 decimals, choose highest phase with share >=0.20; no clipping or monotonic repair",
        "causal_interpretation": False, "manuscript_modified": False,
        "input_hashes": {str(p.relative_to(ROOT)): h for p, h in inputs.items()},
        "protected_hashes": {str(p.relative_to(ROOT)): h for p, h in protected_hashes.items()},
        "artifact_hashes": {p.name: selected.file_sha256(p) for p in OUTPUT.iterdir()},
    }
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved exploratory run to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
