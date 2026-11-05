"""Package saved data by accepted figure/table number; no model fitting.

Run from any directory: python "2.Source Code/build_source_data.py"
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
PG = "2.Source Code/produced_graph/"
DATA = "1.Source Data/"
TEMPORAL = PG + "direct_phase3_vs_phase45_rescue/direct_phase3_vs_phase45_rescue_"
CONTEMP = PG + "direct_phase3_vs_phase45_rescue_contemporaneous/direct_phase3_vs_phase45_rescue_contemporaneous_"
SENSITIVITY = PG + "missingness_sensitivity/"
MEMBERS = {}
SOURCES = {}
BLOCKS = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record_source(path):
    SOURCES[path] = digest((ROOT / path).read_bytes())


def load(path):
    record_source(path)
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def block(path, kind, select=lambda row: True, **labels):
    return [dict(_record_type=kind, _source_file=path, _source_row=str(i), **labels, **r)
            for i, r in enumerate(load(path), 2) if select(r)]


def save(name, rows):
    assert rows, name
    columns = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    MEMBERS[name] = stream.getvalue().encode("utf-8-sig")
    counts = {}
    for row in rows:
        key = (row.get("_record_type", ""), row.get("_source_file", ""))
        counts[key] = counts.get(key, 0) + 1
    BLOCKS.extend(dict(file=name, record_type=k[0], source_file=k[1], rows=n)
                  for k, n in counts.items())


def metric_check(rows, actual, predicted, expected):
    y = [float(r[actual]) for r in rows]
    pred = [float(r[predicted]) for r in rows]
    tp = sum(a >= 3 and b >= 3 for a, b in zip(y, pred))
    values = (sum(a == b for a, b in zip(y, pred)) / len(y),
              tp / sum(a >= 3 for a in y), tp / sum(b >= 3 for b in pred))
    assert all(abs(a - b) < 0.00000051 for a, b in zip(values, expected)), values


def build():
    MEMBERS.clear()
    SOURCES.clear()
    BLOCKS.clear()
    main = block(PG + "main_result_figure1_v1.csv", "summary_metric")
    for task in ("Forecasting", "Nowcasting"):
        predictions = block(TEMPORAL + "benchmark_predictions.csv", "classification_prediction",
                            lambda r: r["task"] == task and r["method"] == "frozen_base")
        assert len(predictions) == 1170
        metric = next(r for r in main if r.get("model") == task)
        metric_check(predictions, "reconstructed_overall_phase", "final_overall_phase_pred",
                     tuple(float(metric[k]) for k in ("overall_accuracy", "phase3plus_recall", "phase3plus_precision")))
        main += predictions
        main += block(DATA + f"r2_frame_{task.lower()}.csv", "scatter_point", _reported_model=task)
    contemporary = block(PG + "all_prediction_contemporaneous_random_cv_predictions.csv",
                         "oof_prediction", _reported_model="Contemporaneous")
    assert len(contemporary) == 5575
    metric_check(contemporary, "overall_phase", "contemporaneous_predict",
                 (0.6934529147982063, 0.9043707832208859, 0.7972588569950866))
    main += contemporary
    main += block(PG + "spatial_feature_comparison_metrics.csv", "summary_metric",
                  lambda r: r["condition"] == "baseline_with_lat_lon" and r["model"] == "Contemporaneous")
    save("Figure 1.csv", main)

    inputs = []
    for task in ("Forecasting", "Nowcasting"):
        rows = block(DATA + task + "_Analysis_010825.csv", "model_input", _reported_model=task)
        assert len(rows) == 5575
        for row in rows:
            row["_split"] = "training_2017_2021" if row["date"][:10] < "2022-01-01" else "test_2022"
        assert sum(r["_split"] == "training_2017_2021" for r in rows) == 4405
        inputs += rows
    for task in ("forecasting", "nowcasting"):
        for suffix, kind in (("signed_sums", "feature_signed_shap_sum"),
                             ("top10", "plotted_top10_shap"), ("groups", "normalized_group_shap")):
            inputs += block(f"writing/final_artwork/data/shap_{task}_{suffix}.csv", kind,
                            _reported_model=task.capitalize())
    record_source("writing/final_artwork/data/shap_provenance.json")
    save("Figure 2.csv", inputs)
    MEMBERS["Figure 3.txt"] = (
        "Figure 3: Ensemble Model Procedure Overview\n"
        "Conceptual diagram; no empirical observations or statistical measurements.\n"
        "Four regressors predict cumulative population shares for IPC 2+, 3+, 4+, and 5.\n"
        "The highest phase with a rounded cumulative share >= 0.20 is assigned; otherwise phase 1.\n"
        "Implementation: 2.Source Code/food_crisis_functions.py; artwork: Figure3_Flowchart.ipynb.\n"
    ).encode()
    raw = load(DATA + "Nowcasting_Analysis_010825.csv")
    save("Figure 4.csv", [dict(_record_type="observation", _source_file=DATA + "Nowcasting_Analysis_010825.csv",
                              _source_row=i, area_id=r["area_id"], date=r["date"],
                              year=r["date"][:4], overall_phase=r["overall_phase"])
                          for i, r in enumerate(raw, 2)])
    # Historical changed-phase predictions were not saved; retain reported values as such.
    changed = [dict(_record_type="reported_changed_phase_metric", _source_file="accepted main manuscript, Table 1",
                    model=task, overall_accuracy=a, phase3plus_recall=r, phase3plus_precision=p)
               for task, a, r, p in [("Forecasting", ".593", ".990", ".702"), ("Nowcasting", ".571", ".964", ".733")]]
    save("Table 1.csv", main + changed)

    obs = block(DATA + "Forecasting_Analysis_010825.csv", "observation")
    slim = [{k: r[k] for k in ("_record_type", "_source_file", "_source_row", "area_id", "date", "overall_phase")}
            for r in obs]
    save("Supplementary Figure 1.csv", slim + block(PG + "sample_imbalance_area_frequency.csv", "histogram_bin"))

    # Match prepare_area_records/add_top30_flags without importing their model-training dependencies.
    pred = pd.DataFrame(load(DATA + "All_prediction.csv"))
    pred["phase3_pred"] = pd.to_numeric(pred["phase3_pred"])
    pred = pred.loc[pred.date.str.startswith("2022")].sort_values("phase3_pred", ascending=False).drop_duplicates("area_id")
    actual = pd.DataFrame(load(DATA + "Forecasting_Analysis_010825.csv"))
    actual = actual.loc[actual.date.str.startswith("2022")].copy()
    actual["phase3_actual"] = actual[[f"phase{i}_percent" for i in (3, 4, 5)]].apply(pd.to_numeric).sum(axis=1)
    actual = actual.sort_values("phase3_actual", ascending=False).drop_duplicates("area_id")
    mapped = pred.merge(actual[["area_id", "phase3_actual"]], on="area_id", validate="one_to_one")
    assert len(mapped) == 646 and mapped.phase3_actual.notna().all()
    for field in ("phase3_pred", "phase3_actual"):
        threshold = mapped[field].quantile(.7)
        mapped[field + "_q70"] = threshold
        mapped[field + "_top30"] = (mapped[field] >= threshold).astype(int)
    assert mapped.phase3_pred_top30.sum() == 207 and mapped.phase3_actual_top30.sum() == 195
    mapped = mapped[["area_id", "date", "lat", "lon", "phase3_pred", "phase3_actual",
                     "phase3_pred_q70", "phase3_actual_q70", "phase3_pred_top30", "phase3_actual_top30"]]
    save("Supplementary Figure 2.csv", [dict(_record_type="mapped_area", **r) for r in mapped.to_dict("records")])

    temporal_rescue = block(TEMPORAL + "benchmark_predictions.csv", "prediction",
                           lambda r: r["method"] in ("frozen_base", "direct_phase45_sqrt_balance"))
    cont_rescue = block(CONTEMP + "oof_predictions.csv", "prediction",
                       lambda r: r["method"] == "direct_phase45_full_balance")
    assert len(temporal_rescue) == 4680 and len(cont_rescue) == 5575
    save("Supplementary Figure 3.csv", temporal_rescue + cont_rescue)
    scatter = block(PG + "phase_cumulative_three_model_scatter_predictions.csv", "scatter_point")
    assert len(scatter) == 23745
    save("Supplementary Figure 4.csv", scatter + block(PG + "phase_cumulative_three_model_scatter_metrics.csv", "panel_metric"))
    binary = block(TEMPORAL + "binary_confusion_matrices.csv", "confusion_cell",
                   lambda r: r["population_scope"] == "benchmark" and r["method"].startswith("direct_phase45_"))
    binary += block(CONTEMP + "binary_confusion_matrices.csv", "confusion_cell",
                    lambda r: r["population_scope"] == "full_oof")
    assert len(binary) == 36
    binary += block(TEMPORAL + "benchmark_predictions.csv", "prediction", lambda r: r["method"].startswith("direct_phase45_"))
    binary += block(CONTEMP + "oof_predictions.csv", "prediction")
    save("Supplementary Figure 5.csv", binary)
    save("Supplementary Figure 6.csv", [dict(_record_type="design_fold", fold=i, training_start_year=2017,
                                           training_end_year=year-1, validation_year=year, role="tuning")
                                      for i, year in enumerate(range(2018, 2022), 1)] +
         [dict(_record_type="design_fold", fold="final", training_start_year=2017,
               training_end_year=2021, validation_year=2022, role="final_holdout")])

    sensitivity = SENSITIVITY + "missingness_sensitivity_metrics.csv"
    stable1 = block(PG + "persistence_baseline_metrics.csv", "summary_metric",
                    lambda r: r["task"] == "Forecasting" and r["model"] == "Persistence baseline", _panel="A")
    for method, panel in [("Ordered Probit", "B"), ("Ensemble OLS", "C"), ("Main model", "E")]:
        stable1 += block(sensitivity, "summary_metric", lambda r: r["experiment"] == "feature_removal" and
                         r["threshold_percent"] == "0" and r["model"] == method, _panel=panel)
    stable1 += block(PG + "selected_figure1_repeated_area_refit_metrics.csv", "summary_metric",
                     lambda r: r["method"] == "Main result (repeated-area refit)", _panel="D")
    stable1 += block(PG + "persistence_baseline_forecasting_predictions.csv", "prediction", _panel="A")
    save("Supplementary Table 1.csv", stable1)
    spatial = lambda r: r["condition"] != "baseline_with_lat_lon"
    save("Supplementary Table 2.csv", block(PG + "spatial_feature_comparison_metrics.csv", "summary_metric", spatial) +
         block(PG + "spatial_feature_comparison_predictions.csv", "prediction", spatial))

    spatial_validation = []
    for panel, prefix, task, label in [
        ("A", "strict_temporal_loco", "Forecasting", "Forecasting"),
        ("A", "strict_temporal_loco", "Nowcasting", "Nowcasting"),
        ("A", "leave_one_country_out", "Nowcasting", "Contemporaneous"),
        ("B", "leave_area_out_20pct_fivefold", "Forecasting", "Forecasting"),
        ("B", "leave_area_out_20pct_fivefold", "Nowcasting", "Nowcasting"),
        ("B", "leave_area_out_20pct_random_cv", "Nowcasting", "Contemporaneous")]:
        suffix = "micro_metrics" if panel == "A" else "metrics"
        spatial_validation += block(PG + prefix + "_" + suffix + ".csv", "summary_metric",
                                    lambda r: r["model"] == task, _panel=panel, _reported_model=label)
        predictions = block(PG + prefix + "_" + task.lower() + "_predictions.csv", "prediction",
                            _panel=panel, _reported_model=label)
        assert len(predictions) == (5575 if label == "Contemporaneous" else 1170)
        spatial_validation += predictions
    spatial_validation += block(PG + "leave_area_out_20pct_random_cv_fold_metrics.csv", "fold_metric",
                                lambda r: r["model"] == "Nowcasting", _panel="B", _reported_model="Contemporaneous")
    save("Supplementary Table 3.csv", spatial_validation)

    dictionary = DATA + "Dataset_Preparation.xlsx"
    record_source(dictionary)
    workbook = load_workbook(ROOT / dictionary, read_only=True, data_only=True)
    rows = list(workbook["Sheet1"].values)
    headers = rows[0]
    save("Supplementary Table 4.csv", [dict(_record_type="variable_dictionary", _source_file=dictionary,
                                          _source_row=i, **{str(k): v for k, v in zip(headers, r) if k is not None})
                                      for i, r in enumerate(rows[1:], 2) if any(v is not None for v in r)])
    workbook.close()
    grid = dict(max_depth=[3, 7, 9, 11], n_estimators=[200, 500, 800], learning_rate=[.01, .05, .08, .10],
                subsample=[.5, .7, 1.], colsample_bytree=[.5, .7, 1.], gamma=[0, .1, .2],
                min_child_weight=[0, 1, 2], scale_pos_weight=[1, 2, 3])
    assert math.prod(len(v) for v in grid.values()) == 11664
    hyper = [dict(_record_type="search_grid", parameter=k, value=v) for k, values in grid.items() for v in values]
    for name in ("forecasting_hyperparameters", "forecasting_hyperparameters_p3", "contemporaneous_hyperparameters"):
        path = "2.Source Code/" + name + ".json"
        record_source(path)
        hyper += [dict(_record_type="effective_parameter", _source_file=path, parameter=k, value=v)
                  for k, v in json.loads((ROOT / path).read_text()).items()]
    save("Supplementary Table 5.csv", hyper)
    stable6 = []
    for method, panel in [("Ordered Probit", "A"), ("Ensemble OLS", "B")]:
        stable6 += block(SENSITIVITY + "missing_indicator_baseline_comparison_metrics.csv", "summary_metric",
                         lambda r: r["model"] == method, _panel=panel)
    for experiment, method, panel in [("feature_removal", "Ordered Probit", "C"),
                                       ("feature_removal", "Ensemble OLS", "D"),
                                       ("country_removal", "Ordered Probit", "E"),
                                       ("country_removal", "Ensemble OLS", "F"),
                                       ("country_removal", "Main model", "H")]:
        stable6 += block(sensitivity, "summary_metric", lambda r: r["experiment"] == experiment and
                         r["threshold_percent"] == "10" and r["model"] == method, _panel=panel)
    stable6 += block(sensitivity, "summary_metric", lambda r: r["experiment"] == "feature_removal" and
                     r["threshold_percent"] == "0" and r["model"] == "Main model", _panel="G")
    assert len(stable6) == 24
    save("Supplementary Table 6.csv", stable6)

    assert len(MEMBERS) == 17
    MEMBERS["README.txt"] = (ROOT / "SOURCE_DATA.md").read_bytes()
    record_source("2.Source Code/build_source_data.py")
    record_source("SOURCE_DATA.md")
    manifest = dict(title="Source Data", article="Predicting Acute Food Insecurity with Machine Learning",
                    input_sha256=SOURCES, output_sha256={k: digest(v) for k, v in MEMBERS.items()}, blocks=BLOCKS)
    MEMBERS["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = ROOT / "Source Data.zip"
    staged = output.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in MEMBERS.items():
                entry = zipfile.ZipInfo("Source Data/" + name, date_time=(2026, 11, 4, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, payload)
        with zipfile.ZipFile(staged) as archive:
            assert archive.testzip() is None
            for name, expected in manifest["output_sha256"].items():
                assert digest(archive.read("Source Data/" + name)) == expected
        for path, expected in SOURCES.items():
            assert digest((ROOT / path).read_bytes()) == expected, path
        staged.replace(output)
    finally:
        if staged.exists():
            staged.unlink()
    print(f"Verified {len(MEMBERS)-2} figure/table files; {output.stat().st_size:,} bytes; SHA256 {digest(output.read_bytes())}")


if __name__ == "__main__":
    build()
