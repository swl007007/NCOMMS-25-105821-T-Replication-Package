"""Export seven compact tables from the existing Source Data selections."""

import csv
import io
import json
import math
from pathlib import Path

import build_source_data as source

OUT = source.ROOT / "tables"


def metrics(rows, table):
    result = []
    for row in rows:
        if row["_record_type"] not in ("summary_metric", "reported_changed_phase_metric"):
            continue
        panel = row.get("_panel", "")
        task = row.get("_reported_model") or row.get("task") or row["model"]
        fields = ("overall_accuracy", "phase3plus_recall", "phase3plus_precision", "phase3plus_r2")
        aggregation = "pooled"
        if table == 3 and panel == "A":
            fields = ("accuracy", "recall", "precision", "R2(p3)")
        elif table == 3 and task == "Contemporaneous":
            fields = tuple("fold_mean_" + field for field in fields)
            aggregation = "mean of five folds"
        elif table == 1 and panel == "D":
            fields = (*fields[:3], "phase3plus_continuous_r2")
        elif table in (1, 6) and row.get("phase3plus_population_share_r2"):
            fields = (*fields[:3], "phase3plus_population_share_r2")
        changed = row["_record_type"] == "reported_changed_phase_metric"
        method = row.get("method") or (row.get("model") if row.get("task") else "Main model")
        record = dict(panel=panel, model=task, method=method,
                      population="changed-phase observations" if changed else "all evaluation observations",
                      aggregation="reported historical summary" if changed else aggregation)
        if table == 1 and panel == "A":
            record["model"] = "Forecasting / Nowcasting"
        if table == 1 and panel == "D":
            record["population"] = "repeated-area temporal test (n=1,103)"
        if table == 2:
            record["panel"] = {"no_lat_lon": "A", "knn5_spatial_means": "B", "d200_spatial_means": "C"}[row["condition"]]
            record["condition"] = row["condition_label"]
        if table == 6:
            record["condition"] = row.get("experiment") or "missing_indicators"
            record["threshold_percent"] = row.get("threshold_percent", "")
        for label, field in zip(("accuracy", "sensitivity", "precision", "r_squared_phase3plus"), fields):
            value = row.get(field, "")
            assert value or label == "r_squared_phase3plus", (table, panel, field)
            if value:
                assert math.isfinite(float(value)), (label, value)
                if label != "r_squared_phase3plus":
                    assert 0 <= float(value) <= 1, (label, value)
            record[label] = value
        record.update(source_file=row["_source_file"], source_row=row.get("_source_row", ""))
        result.append(record)
    return sorted(result, key=lambda row: row["panel"])


def main():
    source.build()
    OUT.mkdir(exist_ok=True)
    outputs = {}
    expected = {0: 5, 1: 12, 2: 9, 3: 6, 4: 76, 5: 8, 6: 24}
    for table in range(7):
        name = "Table 1.csv" if table == 0 else f"Supplementary Table {table}.csv"
        payload = source.MEMBERS[name]
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        if table == 4:
            rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
        elif table == 5:
            grid = {}
            for row in rows:
                if row["_record_type"] == "search_grid":
                    grid.setdefault(row["parameter"], []).append(row["value"])
            assert math.prod(map(len, grid.values())) == 11664
            rows = [dict(parameter=key, candidate_values=", ".join(values)) for key, values in grid.items()]
        else:
            rows = metrics(rows, table)
        assert len(rows) == expected[table], (name, len(rows))
        if table == 0:
            contemporary = next(row for row in rows if row["model"] == "Contemporaneous")
            assert [f"{float(contemporary[k]):.3f}" for k in
                    ("accuracy", "sensitivity", "precision", "r_squared_phase3plus")] == ["0.693", "0.904", "0.797", "0.642"]
        path = OUT / name.replace(" ", "_")
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        outputs[path.name] = dict(rows=len(rows), sha256=source.digest(path.read_bytes()),
                                  source_member=name, source_sha256=source.digest(payload))
    audit = dict(source_archive_sha256=source.digest((source.ROOT / "Source Data.zip").read_bytes()),
                 generator_sha256=source.digest(Path(__file__).read_bytes()), tables=outputs)
    (OUT / "table_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Verified seven table exports; sensitivity=recall; original numeric precision retained.")


if __name__ == "__main__":
    main()
