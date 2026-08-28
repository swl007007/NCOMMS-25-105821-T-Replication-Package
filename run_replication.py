#!/usr/bin/env python3
"""Run the dependency-ordered core replication workflow.

The default action executes the four controlled generators used for the shared
temporal-test, random-CV Contemporaneous, and cumulative-scatter artifacts.
Use ``--check-only`` for a fast, non-model readiness and publication-hygiene
check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_CODE = REPO_ROOT / "2.Source Code"
PRODUCED_GRAPH = SOURCE_CODE / "produced_graph"
RELEASE_ASSET = PRODUCED_GRAPH / "spatial_feature_interpolation_audit.csv.gz"
RELEASE_ASSET_SHA256 = (
    "c6769b453538c19be0745e66772ad7c794960eb2c77e96e914d35a422f8e39b4"
)

REQUIRED_PATHS = (
    REPO_ROOT / "1.Source Data" / "Forecasting_Analysis_010825.csv",
    REPO_ROOT / "1.Source Data" / "Nowcasting_Analysis_010825.csv",
    REPO_ROOT / "1.Source Data" / "area_country_lookup.csv",
    REPO_ROOT / "1.Source Data" / "r2_frame_forecasting.csv",
    REPO_ROOT / "1.Source Data" / "r2_frame_nowcasting.csv",
    SOURCE_CODE / "forecasting_hyperparameters.json",
    SOURCE_CODE / "forecasting_hyperparameters_p3.json",
    SOURCE_CODE / "contemporaneous_hyperparameters.json",
    SOURCE_CODE / "contemporaneous_hyperparameters_p3.json",
)

WORKFLOW = (
    (
        "canonical temporal-test predictions",
        SOURCE_CODE / "generate_all_prediction_temporal_test.py",
        ("--workers", "1"),
    ),
    (
        "shared three-model evaluation and Contemporaneous OOF sidecar",
        SOURCE_CODE / "generate_all_prediction_evaluation.py",
        (),
    ),
    (
        "temporal-test-prefixed evaluation artifacts",
        SOURCE_CODE / "generate_all_prediction_temporal_test_evaluation.py",
        (),
    ),
    (
        "cumulative-phase scatter artifacts",
        SOURCE_CODE / "generate_phase_cumulative_scatter_comparison.py",
        ("--workers", "1"),
    ),
)

ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/mnt/[a-z]/Users/|/home/[^/\s]+/)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _iter_code_sources(notebook: dict) -> Iterable[tuple[int, str]]:
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            yield index, "".join(cell.get("source", []))


def check_readiness() -> list[str]:
    """Return blocking readiness errors; warnings are printed immediately."""
    errors: list[str] = []

    for path in (*REQUIRED_PATHS, *(step[1] for step in WORKFLOW)):
        if not path.is_file():
            errors.append(f"missing required file: {_display(path)}")

    notebooks = sorted(SOURCE_CODE.glob("*.ipynb"))
    if len(notebooks) != 12:
        errors.append(f"expected 12 analysis notebooks, found {len(notebooks)}")

    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid notebook JSON: {_display(path)} ({exc})")
            continue

        for index, source in _iter_code_sources(notebook):
            if ABSOLUTE_PATH_PATTERN.search(source):
                errors.append(
                    f"absolute executable path in {_display(path)}, code cell {index}"
                )

        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None:
                errors.append(
                    f"saved execution count in {_display(path)}, code cell {index}"
                )
            if cell.get("outputs"):
                errors.append(f"saved output in {_display(path)}, code cell {index}")

    if RELEASE_ASSET.is_file():
        observed = _sha256(RELEASE_ASSET)
        if observed != RELEASE_ASSET_SHA256:
            errors.append(
                "release asset checksum mismatch: "
                f"expected {RELEASE_ASSET_SHA256}, found {observed}"
            )
        else:
            print(
                "Release asset present and checksum verified: "
                f"{_display(RELEASE_ASSET)}"
            )
    else:
        print(
            "Warning: release-only lineage asset is not restored. Download it "
            "from GitHub release v1.2.0 before auditing spatial interpolation."
        )

    if errors:
        return errors

    print(
        f"Readiness checks passed: {len(notebooks)} notebooks and "
        f"{len(WORKFLOW)} workflow steps are structurally ready."
    )
    return []


def run_workflow() -> None:
    for number, (label, script, arguments) in enumerate(WORKFLOW, start=1):
        command = [sys.executable, str(script), *arguments]
        print(f"[{number}/{len(WORKFLOW)}] {label}", flush=True)
        print("  " + " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate files and notebook hygiene without fitting models",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    errors = check_readiness()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if arguments.check_only:
        return 0

    run_workflow()
    print("Core replication workflow completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
