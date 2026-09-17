"""Verify the isolated fresh run against Figure 1's preserved predictions."""
from pathlib import Path
import csv
import hashlib
import json

base = Path(__file__).resolve().parent
root = base.parents[2]
graph = base.parent
filename = "all_prediction_contemporaneous_random_cv_predictions.csv"
assert (base / filename).read_bytes() == (graph / filename).read_bytes()
with (base / filename).open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len({(r["area_id"], r["date"]) for r in rows}) == 5575
assert all(r["shuffle_seed"] == r["estimator_random_state"] == "0" for r in rows)
with (base / "contemporaneous_main_metrics.csv").open(newline="") as stream:
    metrics = next(csv.DictReader(stream))
assert [f"{float(metrics[k]):.3f}" for k in
        ["accuracy", "recall", "precision", "phase3plus_r2"]] == ["0.693", "0.904", "0.797", "0.642"]
archive = root / "0.Archived/2026-09-16_legacy_contemporaneous_notebook"
for path, digest in json.loads((archive / "protected_before.json").read_text()).items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
with (archive / "inventory.csv").open(newline="") as stream:
    record = next(csv.DictReader(stream))
assert hashlib.sha256((root / record["archive_path"]).read_bytes()).hexdigest() == record["sha256"]
assert not (root / record["original_path"]).exists()
print("PASS: fresh 5,575-row predictions exactly match Figure 1; original artifacts preserved.")
