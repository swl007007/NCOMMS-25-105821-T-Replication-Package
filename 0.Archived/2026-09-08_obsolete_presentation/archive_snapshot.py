"""One-off archive manifest and integrity check; never runs scientific models.

Run from any directory: python archive_snapshot.py [--apply | --verify].
Default previews counts. --apply refuses to overwrite an existing inventory.
"""
import argparse
import csv
import hashlib
import os
import re
from collections import Counter
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[1]
INVENTORY = ARCHIVE / "inventory.csv"
FIGURES = {
    "accuracy.jpg": "Fig. 1a", "precision_sensitivity.jpg": "Fig. 1b",
    "R2.jpg": "Fig. 1c", "Picture6.jpg": "Fig. 2a",
    "Picture8.jpg": "Fig. 2b", "Picture7.jpg": "Fig. 2c",
    "Picture9.jpg": "Fig. 2d", "Picture3.jpg": "Fig. 3",
    "Picture1.jpg": "Fig. 4a", "Picture2.jpg": "Fig. 4b",
    "sample_imbalance_area_frequency.png": "Supplementary Fig. 1",
    "2022_alert_map_panel_2x3.png": "Supplementary Fig. 2",
    "Picture4.jpg": "Supplementary Fig. 3",
    "direct_phase3_vs_phase45_rescue_five_class_confusion_atlas.png": "Supplementary Fig. 4",
    "phase_cumulative_actual_vs_predicted_forecasting_nowcasting_contemporaneous.png": "Supplementary Fig. 5",
    "direct_phase3_vs_phase45_rescue_binary_confusion_atlas.png": "Supplementary Fig. 6",
}
AMBIGUOUS = {
    "shap_values_forecasting_phase3_phase4_colored.jpg",
    "shap_values_grouped_horizontal_bars.jpg",
    "shap_values_nowcasting_phase3_phase4_colored.jpg",
    "shap_values_grouped_horizontal_bars_nowcasting.jpg",
    "shap_values_nowcasting_phase3_phase4_L1.jpg",
    "shap_values_nowcasting_bar_phase3_phase4_L1.jpg",
    "ipc_flowchart_final_nature.jpg", "ipc_flowchart_rules_transparent_cluster.jpg",
}
OLD_IMAGES = {
    "Picture5.jpg", "Picture10.jpg", "Picture_flow.jpg", "Picture_flow_edited.jpg",
    "Actual_alert_map.png", "Forward-looking finetuning.jpg",
    "ipc_flowchart_with_rules_cluster.jpg", "sensitivity_precision_threshold.jpg",
    "shap_values_forecasting_phase3_phase4.jpg",
    "shap_values_forecasting_bar_phase3_phase4.jpg",
    "shap_values_nowcasting_phase3_phase4.jpg",
    "shap_values_nowcasting_phase3_phase4_L2.jpg",
    "shap_values_nowcasting_bar_phase3_phase4.jpg",
    "shap_values_nowcasting_bar_phase3_phase4_L2.jpg",
}
ARTIFACT_EXT = {".csv", ".gz", ".xlsx", ".jpg", ".jpeg", ".png", ".pdf",
                ".eps", ".svg", ".tif", ".tiff", ".drawio", ".json"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def classify(rel):
    name = rel.name
    if any(part.lower() in {"archive", "0.archived", "old_graph"} for part in rel.parts):
        return "existing_archive", "Already separated from active results"
    if rel.as_posix() == "2.Source Code/Conflit_Simulation.ipynb":
        return "archive", "Withdrawn legacy conflict simulation; not a paired prediction transition"
    if rel.parts[0] == "writing":
        if name in {"Picture5.jpg", "Picture5B.jpg", "fig.eps", "empty.eps"}:
            return "archive", "No reference in current manuscript or Response"
        return "retain_writing", FIGURES.get(name, "Current writing or template/support resource; sn-article.pdf is audit-bound")
    if "conflict_perturbation_10pct" in rel.parts or name == "generate_conflict_perturbation.py":
        return "retain_user_exception", "Corrected exploratory perturbation explicitly retained by author"
    if rel.parts[0] == "1.Source Data":
        if name.startswith(("2022_alert_map_panel_2x3_severity_aligned", "2022_alert_map_panel_2x3_minimum_severity")):
            return "archive", "Unselected map alternatives and their diagnostics; current SI Fig. 2 uses unsuffixed map"
        return "retain_source", "Source input or upstream data dependency; current map and diagnostics retained"
    if "flowchart_SI" in rel.parts:
        return "retain_uncertain", "Editable figure sources; lineage to current artwork not fully resolved"
    if "produced_graph" in rel.parts:
        if name.startswith(("leave_one_country_out_", "leave_area_out_10pct_", "leave_area_out_20pct_random_cv_")):
            return "archive", "Superseded spatial protocol; current SI Table 3 uses strict temporal LOCO/fivefold"
        if any(part in {"nowcasting_calibration", "nowcasting_calibration_full2021"} for part in rel.parts):
            return "archive", "Uncited calibration family; restore jointly with archived temporal evaluation for saved-audit checks"
        if name.startswith("all_prediction_") and not name.startswith("all_prediction_contemporaneous_random_cv_"):
            return "archive", "Alternative F/N evaluation, not current manuscript/R1; underlying All_prediction.csv retained"
        if name in OLD_IMAGES or name in {"feature_importance_leave_one_out_022025.csv", "forecasting_region_sub.csv"}:
            return "archive", "Uncited legacy output; no current computational reader found"
        if name.startswith(("precision_recall_scatter_leave_one_country_out", "precision_recall_accuracy_p3r2_leave_area_out_10pct",
                            "precision_recall_scatter_with_persistence_baselines", "precision_recall_scatter_with_multinomial_baselines",
                            "phase2plus_actual_vs_predicted_", "phase4plus_actual_vs_predicted_", "phase5_actual_vs_predicted_")):
            return "archive", "Unselected presentation variant; supporting current tables/three-model grid retained"
        if name in AMBIGUOUS:
            return "retain_uncertain", "Current Figure 2/3 notebook output; exact lineage to renamed writing artwork unresolved"
        return "retain_lineage", "Current figure/table source, companion export, or complete audit/hash-bound dependency family"
    if rel.parts[0] == "2.Source Code" and name == "shap_values_forecasting_phase3_colored.jpg":
        return "archive", "Unreferenced single-phase SHAP image; no current reader or writer found"
    if name in AMBIGUOUS:
        return "retain_uncertain", "SHAP source version differs from writing; retain pending figure lineage review"
    return "retain_code_support", "Code, configuration, documentation, tests or other non-presentation support"


def writing_check():
    tex = (ROOT / "writing/Predict_food_crisis_NC_RandR.tex").read_text()
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    assert used == set(FIGURES), (used - set(FIGURES), set(FIGURES) - used)
    assert all((ROOT / "writing" / name).is_file() for name in used)


def verify():
    with INVENTORY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({r["original_path"] for r in rows}) == len(rows)
    for row in rows:
        source = ROOT / row["original_path"]
        target = ROOT / row["archive_path"] if row["archive_path"] else source
        assert target.is_file(), target
        if row["sha256"]:
            assert digest(target) == row["sha256"], target
        if row["archive_path"]:
            assert not source.exists(), source
    writing_check()
    print("Verified all manifest paths and recorded hashes; all 16 manuscript image references exist.")
    print(dict(Counter(r["status"] for r in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    assert not INVENTORY.exists(), "Snapshot already exists; use --verify"
    writing_check()
    rows = []
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {"__pycache__", "node_modules", ".venv", "venv", "0.Archived"})
        for name in sorted(files):
            path = Path(directory) / name
            if name.startswith(".") or path.is_symlink():
                continue
            rel = path.relative_to(ROOT)
            status, reason = classify(rel)
            destination = ARCHIVE / rel if status == "archive" else None
            if destination:
                assert not destination.exists(), destination
            rows.append(dict(original_path=rel.as_posix(), status=status, reason=reason,
                             bytes=path.stat().st_size,
                             sha256=digest(path) if status == "archive" or path.suffix.lower() in ARTIFACT_EXT or rel.parts[0] == "writing" else "",
                             archive_path=destination.relative_to(ROOT).as_posix() if destination else ""))
    print(dict(Counter(r["status"] for r in rows)))
    for row in rows:
        if row["status"] == "archive":
            print(row["original_path"])
    if args.apply:
        # Save recovery instructions before the first move; never overwrite files.
        with INVENTORY.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            if row["archive_path"]:
                source, destination = ROOT / row["original_path"], ROOT / row["archive_path"]
                assert digest(source) == row["sha256"] and not destination.exists()
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
        verify()


if __name__ == "__main__":
    main()
