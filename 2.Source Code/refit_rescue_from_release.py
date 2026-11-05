"""Run temporal rescue with the later, expanded spatial-results bundle.

The published spatial CSVs added Contemporaneous rows after rescue was frozen.
Prove the temporal prediction bytes are unchanged before rebinding those three
whole-file hashes. All original source, environment and predecessor gates remain.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import generate_direct_phase3_vs_phase45_rescue as rescue

SPATIAL_HASHES = {
    "spatial_feature_comparison_predictions.csv": "6bc28fe3f2bd438c3a95a07ea5ac73d053c050f35daa3902f6fe411cfcd9fbc3",
    "spatial_feature_comparison_metrics.csv": "8d307c4e1bf479b4e7fec193ada011366362221a13a9b7f0ab4cc950b8b167d6",
    "spatial_feature_comparison_source_audit.csv": "926d10294ffb0c6781b3cd293fbe52b0e9d72b44bde547353389da76ab1354cf",
}
TEMPORAL_PREDICTIONS_SHA256 = "bdef15d0b1facf892dcbfdb019e119463cb93ee79bd384915ff6c45fa9f1a7a7"


def check_release_inputs():
    predecessor = rescue.predecessor
    directory = predecessor.SPATIAL_PREDICTIONS.parent
    for name, expected in SPATIAL_HASHES.items():
        if rescue.file_sha256(directory / name) != expected:
            raise ValueError(f"Published spatial input hash mismatch: {name}")
    lines = predecessor.SPATIAL_PREDICTIONS.read_bytes().splitlines(keepends=True)
    records = list(csv.reader(line.decode("utf-8") for line in lines))
    assert len(lines) == len(records) == 31661
    model_index = records[0].index("model")
    temporal = [line for line, row in zip(lines[1:], records[1:])
                if row[model_index] in ("Forecasting", "Nowcasting")]
    if len(temporal) != 9360 or hashlib.sha256(lines[0] + b"".join(temporal)).hexdigest() != TEMPORAL_PREDICTIONS_SHA256:
        raise ValueError("Temporal predictions differ from the original frozen rescue input.")
    predecessor.EXPECTED_SOURCE_HASHES.update(
        {directory / name: expected for name, expected in SPATIAL_HASHES.items()})
    rescue.verify_frozen_sources()
    rescue.load_contemporaneous_figure_evidence()
    rescue.assert_formal_environment()
    predecessor.load_and_validate_frozen_base_predictions()
    print("PASS: expanded spatial inputs verified; original temporal prediction bytes unchanged; frozen gates passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=rescue.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    check_release_inputs()
    if not args.check_only:
        rescue.run_generation(args.output_dir)


if __name__ == "__main__":
    main()
