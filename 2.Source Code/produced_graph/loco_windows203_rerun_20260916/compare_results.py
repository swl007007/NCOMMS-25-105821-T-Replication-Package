"""Compare fresh Windows-2.0.3 LOCO reruns with preserved historical artifacts."""
from pathlib import Path
from decimal import Decimal
import csv
import hashlib
import json
import math
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
ARCHIVE = ROOT / "0.Archived/2026-09-17_pre_windows203_loco"
TOL = Decimal("0.001")
METRICS = ["accuracy", "precision", "recall", "R2(p3)"]


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def keyed(path, columns):
    rows = read(path)
    result = {tuple(row[c] for c in columns): row for row in rows}
    assert len(result) == len(rows), f"Duplicate keys: {path}"
    return result


def numeric(value):
    return None if value.strip().lower() in {"", "nan", "none"} else Decimal(value)


def within(left, right):
    if left is None or right is None:
        return left is right
    return abs(left - right) <= TOL


def main():
    assert within(Decimal("0"), Decimal("0.001"))
    assert not within(Decimal("0"), Decimal("0.001001"))
    assert within(None, None) and not within(None, Decimal("0"))
    context = json.loads((BASE / "run_context.json").read_text())
    status = json.loads((BASE / "run_status.json").read_text())
    assert status["ordinary"] == status["strict_temporal"] == 0, status
    for path, digest in context["protected_sha256"].items():
        assert hashlib.sha256(((ARCHIVE / path) if (ARCHIVE / path).is_file() else (ROOT / path)).read_bytes()).hexdigest() == digest, path
    configurations = [
        ("strict_temporal", ARCHIVE / "2.Source Code/produced_graph", "strict_temporal_loco",
         "area_metrics", ["model", "area_id"], METRICS, 1170, 646, 27),
        ("ordinary", ROOT / "0.Archived/2026-09-08_obsolete_presentation/2.Source Code/produced_graph",
         "leave_one_country_out", "country_metrics", ["model", "country_code_3"],
         ["overall_accuracy", "phase3plus_precision", "phase3plus_recall", "phase3plus_r2"], 5575, 1198, 29),
    ]
    comparisons, populations, checks = [], [], []
    # NumPy is used only to reproduce the documented two-decimal rounding rule.
    import numpy as np
    for family, old_dir, prefix, detail_name, detail_keys, detail_metrics, n, areas, countries in configurations:
        new_dir = BASE / family
        for level, suffix, keys, columns in [
            ("primary", "micro_metrics", ["model"], METRICS),
            ("detail", detail_name, detail_keys, detail_metrics),
        ]:
            old = keyed(old_dir / f"{prefix}_{suffix}.csv", keys)
            new = keyed(new_dir / f"{prefix}_{suffix}.csv", keys)
            assert old.keys() == new.keys(), f"Metric key coverage: {family}/{level}"
            for key in sorted(old):
                for column in columns:
                    left, right = numeric(old[key][column]), numeric(new[key][column])
                    passed = within(left, right)
                    difference = None if left is None or right is None else abs(right - left)
                    comparisons.append(dict(
                        family=family, level=level, key="|".join(key), metric=column,
                        old=old[key][column], new=new[key][column],
                        absolute_difference="" if difference is None else str(difference),
                        old_3dp="" if left is None else f"{left:.3f}",
                        new_3dp="" if right is None else f"{right:.3f}",
                        both_undefined=left is None and right is None,
                        definition_changed=(left is None) != (right is None),
                        passed=passed))
        new_metrics = keyed(new_dir / f"{prefix}_micro_metrics.csv", ["model"])
        for model in ["forecasting", "nowcasting"]:
            filename = f"{prefix}_{model}_predictions.csv"
            old = keyed(old_dir / filename, ["area_id", "date"])
            new = keyed(new_dir / filename, ["area_id", "date"])
            assert len(old) == len(new) == n and old.keys() == new.keys(), filename
            assert len({row["area_id"] for row in new.values()}) == areas
            assert len({row["country_code_3"] for row in new.values()}) == countries
            for key in old:
                for column in ["country_code_3", "fold_country", "source_row_index",
                               "overall_phase", "source_overall_phase"] + [f"phase{i}_test" for i in range(2, 6)]:
                    left, right = old[key][column], new[key][column]
                    if column in {"country_code_3", "fold_country"}:
                        assert left == right, (filename, key, column)
                    else:
                        assert numeric(left) == numeric(right), (filename, key, column)
                if family == "strict_temporal":
                    assert new[key]["date"][:10] >= "2022-01-01"
            ordered = list(new.values())
            y = np.array([float(r["overall_phase"]) for r in ordered])
            pred = np.array([float(r["overall_phase_pred"]) for r in ordered])
            truth_share = np.round([float(r["phase3_test"]) for r in ordered], 2)
            pred_share = np.round([float(r["phase3_pred"]) for r in ordered], 2)
            tp = int(np.sum((y >= 3) & (pred >= 3)))
            values = [float(np.mean(y == pred)), tp / int(np.sum(pred >= 3)),
                      tp / int(np.sum(y >= 3)),
                      float(1 - np.sum((truth_share - pred_share) ** 2) /
                            np.sum((truth_share - np.mean(truth_share)) ** 2))]
            for metric, value in zip(METRICS, values):
                saved = float(new_metrics[(model.capitalize(),)][metric])
                assert math.isclose(saved, value, rel_tol=0, abs_tol=0.00000051), (family, model, metric, saved, value)
                checks.append(dict(family=family, model=model, metric=metric, saved=saved, recomputed=value))
            populations.append(dict(family=family, model=model, rows=n, areas=areas, countries=countries,
                                    full_keys_and_truth_unchanged=True))
        audit = read(new_dir / f"{prefix}_source_audit.csv")
        assert all(r["python_version"] == "3.11.3" and r["xgboost_version"] == "2.0.3" for r in audit)
        for row in audit:
            if "checkpoint_reused" in row:
                assert row["checkpoint_reused"].lower() in {"false", "0"}
    with (BASE / "metric_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {}
    for family, *_ in configurations:
        summary[family] = {}
        for level in ["primary", "detail"]:
            rows = [r for r in comparisons if r["family"] == family and r["level"] == level]
            finite = [Decimal(r["absolute_difference"]) for r in rows if r["absolute_difference"]]
            summary[family][level] = dict(
                cells=len(rows), failures=sum(not r["passed"] for r in rows),
                both_undefined=sum(r["both_undefined"] for r in rows),
                definition_changes=sum(r["definition_changed"] for r in rows),
                max_absolute_difference=str(max(finite, default=Decimal(0))),
                passed=all(r["passed"] for r in rows))
    result = dict(tolerance=str(TOL), decision_rule="unrounded absolute difference <= 0.001; display 3 decimals",
                  summary=summary, populations=populations, independently_recomputed_primary=checks,
                  protected_hashes_unchanged=True, limitation=context["limitation"])
    (BASE / "comparison_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# LOCO Windows XGBoost 2.0.3 rerun comparison", "",
             "Tolerance: absolute unrounded difference <= 0.001. Display: three decimals.",
             "Both undefined cells match; a one-sided undefined value fails. No rows were dropped.",
             "Historical sources remain unchanged in archives. Both rerun families were promoted to produced_graph on 2026-09-17.", "",
             "## Primary pooled metrics", "",
             "| Family | Model | Metric | Historical | Rerun | Absolute difference | Pass |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for r in comparisons:
        if r["level"] == "primary":
            lines.append(f"| {r['family']} | {r['key']} | {r['metric']} | {r['old_3dp']} | {r['new_3dp']} | {r['absolute_difference']} | {'PASS' if r['passed'] else 'FAIL'} |")
    lines += ["", "## Separate primary and detailed verdicts", ""]
    for family, levels in summary.items():
        for level, stats in levels.items():
            lines.append(f"- {family} / {level}: {'PASS' if stats['passed'] else 'FAIL'}; {stats['failures']}/{stats['cells']} failures; max delta {stats['max_absolute_difference']}; both undefined {stats['both_undefined']}; definition changes {stats['definition_changes']}.")
    lines += ["", "Full precision comparisons: metric_comparison.csv. Machine-readable checks: comparison_summary.json.",
              "Full population and truth keys verified; primary metrics independently recomputed from saved predictions.",
              "", "## Provenance limitation", "", context["limitation"],
              "", "Training environment and exact commands: run_context.json. Logs: strict_temporal.log and ordinary.log.",
              "Recheck with: python3 -B compare_results.py (requires NumPy; does not retrain).", ""]
    (BASE / "README.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
