"""Frozen Figure 1 main-result contract shared by generated paper artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
from typing import Mapping, Sequence


FREEZE_ID = "main-result-figure1-v1"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"
ENVIRONMENT_LOCK_PATH = (
    REPO_ROOT / "2.Source Code" / "main_result_figure1_v1_environment_lock.txt"
)

EVALUATION: Mapping[str, object] = {
    "cutoff_date": "2022-01-01",
    "source_rows": 5575,
    "train_rows": 4405,
    "test_rows": 1170,
    "test_areas": 646,
    "actual_phase_counts": {"1": 22, "2": 269, "3": 764, "4": 113, "5": 2},
    "target_definition": (
        "highest cumulative IPC phase with population share >= 0.20"
    ),
    "overall_accuracy_definition": "five_class_exact_match_pooled_over_1170_rows",
    "phase3plus_precision_definition": "pooled_binary_tp_over_tp_plus_fp",
    "phase3plus_recall_definition": "pooled_binary_tp_over_tp_plus_fn",
    "phase3plus_r2_definition": (
        "sklearn r2_score on two-decimal Phase 3+ predictions and unrounded actual shares"
    ),
}

ENVIRONMENT: Mapping[str, object] = {
    "environment_id": "windows_py3113_xgb203_defaultthreads_no_explicit_seed",
    "platform_family": "Windows",
    "python_version": "3.11.3",
    "numpy_version": "1.26.4",
    "pandas_version": "2.2.3",
    "scipy_version": "1.17.1",
    "scikit_learn_version": "1.5.2",
    "xgboost_version": "2.0.3",
    "xgboost_n_jobs_override": None,
    "xgboost_random_state_override": None,
    "outer_model_workers": 1,
    "python_installer_sha256": (
        "63649443c026d4c88c63fd9ea9931b9ecde3e671092439b6f55b3be3dcca6da3"
    ),
    "xgboost_dll_sha256": (
        "ca6f7a13af14d3ed08da0a12164ca042302151999191bb81062768cffbc95ce1"
    ),
}

GENERATOR_ENVIRONMENT_EXTENSIONS: Mapping[str, str] = {
    "matplotlib": "3.10.1",
    "statsmodels": "0.14.6",
    "patsy": "1.0.1",
}

_CORE_PACKAGE_VERSIONS: Mapping[str, str] = {
    "numpy": str(ENVIRONMENT["numpy_version"]),
    "pandas": str(ENVIRONMENT["pandas_version"]),
    "scipy": str(ENVIRONMENT["scipy_version"]),
    "scikit-learn": str(ENVIRONMENT["scikit_learn_version"]),
    "xgboost": str(ENVIRONMENT["xgboost_version"]),
}

RESULTS: Mapping[str, Mapping[str, object]] = {
    "Forecasting": {
        "overall_accuracy": 0.6495726495726496,
        "phase3plus_precision": 0.7750702905342081,
        "phase3plus_recall": 0.9408418657565415,
        "phase3plus_r2": 0.2489985704986828,
        "serialized_anchor_recomputed_r2": 0.24899856935748543,
        "correct_rows": 760,
        "true_positive": 827,
        "false_positive": 240,
        "false_negative": 52,
        "true_negative": 51,
        "aggregate_notebook_path": "2.Source Code/Table1_Forecasting_main.ipynb",
        "aggregate_notebook_cell_index": 1,
        "row_anchor_generator_path": (
            "2.Source Code/Figure2_Feature_Importance_Forecasting.ipynb"
        ),
        "row_anchor_path": "1.Source Data/r2_frame_forecasting.csv",
        "row_anchor_sha256": (
            "1f165b8755b729751d177599837d021148766f472d81bb97e9f0af18eb2013fb"
        ),
        "phase_long_sha256": (
            "d0c6cefd6bc6962b710cf21fe27246924bd8d8ca9ebfbb3afa8f2e52fb4eb2b1"
        ),
        "converted_predictions_sha256": (
            "65233dfbd6e164c956d5342dd45f546b8cef25bed881c0942e08a890fc4eaaa8"
        ),
        "metric_source": (
            "main-result-figure1-v1; Table1_Forecasting_main.ipynb stored output; "
            "Figure 1 anchor"
        ),
    },
    "Nowcasting": {
        "overall_accuracy": 0.6538461538461539,
        "phase3plus_precision": 0.7774647887323943,
        "phase3plus_recall": 0.9419795221843004,
        "phase3plus_r2": 0.252784701177865,
        "serialized_anchor_recomputed_r2": 0.25278470025242084,
        "correct_rows": 765,
        "true_positive": 828,
        "false_positive": 237,
        "false_negative": 51,
        "true_negative": 54,
        "aggregate_notebook_path": (
            "2.Source Code/Figure2_Nowcasting_two_layer_feature_importance.ipynb"
        ),
        "aggregate_notebook_cell_index": 1,
        "row_anchor_generator_path": (
            "2.Source Code/Figure2_Nowcasting_two_layer_feature_importance.ipynb"
        ),
        "row_anchor_path": "1.Source Data/r2_frame_nowcasting.csv",
        "row_anchor_sha256": (
            "981deb2afb05c1864655e8ef23e8ca2337af21a01025bf7a9a3bd12d18633dd8"
        ),
        "phase_long_sha256": (
            "042220e9573a0317db67160fc362087aa90306b5b6da2e3423338081095f3cab"
        ),
        "converted_predictions_sha256": (
            "cc0b45cd48d0748504428abe43955fb99346e4ddab674b523c26d1a629c22532"
        ),
        "metric_source": (
            "main-result-figure1-v1; "
            "Figure2_Nowcasting_two_layer_feature_importance.ipynb stored output; "
            "Figure 1 anchor"
        ),
    },
}

TABLE1_NOWCASTING_ALTERNATIVE: Mapping[str, object] = {
    "overall_accuracy": 0.6666666666666666,
    "phase3plus_precision": 0.8035892323030908,
    "phase3plus_recall": 0.9169510807736063,
    "phase3plus_r2": 0.27554513043222684,
    "source": "2.Source Code/Table1_Nowcasting_two_layer.ipynb stored output",
    "selection_status": "nonselected_alternative_lineage",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_references() -> dict[str, dict[str, object]]:
    """Return independent classification-reference dictionaries for consumers."""
    return {
        model: {
            "reference_id": FREEZE_ID,
            "environment_id": ENVIRONMENT["environment_id"],
            "overall_accuracy": result["overall_accuracy"],
            "phase3plus_precision": result["phase3plus_precision"],
            "phase3plus_recall": result["phase3plus_recall"],
            "phase3plus_r2": result["phase3plus_r2"],
            "metric_source": result["metric_source"],
        }
        for model, result in RESULTS.items()
    }


def current_environment_record(
    required_extensions: Sequence[str] = (),
) -> dict[str, object]:
    """Return the live environment fields needed by frozen-result generators."""
    unknown = sorted(set(required_extensions).difference(GENERATOR_ENVIRONMENT_EXTENSIONS))
    if unknown:
        raise ValueError(f"Unknown generator environment extensions: {unknown}")
    package_names = [*_CORE_PACKAGE_VERSIONS, *required_extensions]
    package_versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = None
    return {
        "platform_family": platform.system(),
        "python_version": platform.python_version(),
        "package_versions": package_versions,
    }


def assert_frozen_environment(
    required_extensions: Sequence[str] = (),
) -> dict[str, object]:
    """Reject formal artifact generation outside the selected frozen environment."""
    record = current_environment_record(required_extensions)
    expected_packages = {
        **_CORE_PACKAGE_VERSIONS,
        **{
            package_name: GENERATOR_ENVIRONMENT_EXTENSIONS[package_name]
            for package_name in required_extensions
        },
    }
    mismatches: list[str] = []
    if record["platform_family"] != ENVIRONMENT["platform_family"]:
        mismatches.append(
            f"platform={record['platform_family']} expected={ENVIRONMENT['platform_family']}"
        )
    if record["python_version"] != ENVIRONMENT["python_version"]:
        mismatches.append(
            f"python={record['python_version']} expected={ENVIRONMENT['python_version']}"
        )
    actual_packages = record["package_versions"]
    assert isinstance(actual_packages, dict)
    for package_name, expected_version in expected_packages.items():
        actual_version = actual_packages.get(package_name)
        if actual_version != expected_version:
            mismatches.append(
                f"{package_name}={actual_version} expected={expected_version}"
            )
    if mismatches:
        raise RuntimeError(
            "Formal Main-result artifacts require the frozen generator environment: "
            + "; ".join(mismatches)
        )
    return record


def spatial_references() -> dict[str, dict[str, object]]:
    """Return the four frozen metrics in the spatial-generator compatibility schema."""
    return {
        model: {
            "reference_id": FREEZE_ID,
            "environment_id": ENVIRONMENT["environment_id"],
            "notebook_path": result["aggregate_notebook_path"],
            "cell_index": result["aggregate_notebook_cell_index"],
            "phase3plus_precision": result["phase3plus_precision"],
            "phase3plus_recall": result["phase3plus_recall"],
            "overall_accuracy": result["overall_accuracy"],
            "phase3plus_r2": result["phase3plus_r2"],
        }
        for model, result in RESULTS.items()
    }


def manifest_payload() -> dict[str, object]:
    return {
        "freeze_id": FREEZE_ID,
        "schema_version": SCHEMA_VERSION,
        "evaluation": dict(EVALUATION),
        "environment": dict(ENVIRONMENT),
        "generator_environment_extensions": dict(GENERATOR_ENVIRONMENT_EXTENSIONS),
        "environment_lock_path": str(ENVIRONMENT_LOCK_PATH.relative_to(REPO_ROOT)),
        "environment_lock_sha256": file_sha256(ENVIRONMENT_LOCK_PATH),
        "results": {model: dict(result) for model, result in RESULTS.items()},
        "nonselected_table1_nowcasting_alternative": dict(
            TABLE1_NOWCASTING_ALTERNATIVE
        ),
        "freeze_source_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "freeze_source_sha256": file_sha256(Path(__file__)),
    }


def write_manifest_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "main_result_figure1_v1.json"
    csv_path = output_dir / "main_result_figure1_v1.csv"
    json_path.write_text(
        json.dumps(manifest_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "freeze_id",
        "model",
        "overall_accuracy",
        "phase3plus_precision",
        "phase3plus_recall",
        "phase3plus_r2",
        "serialized_anchor_recomputed_r2",
        "correct_rows",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "environment_id",
        "metric_source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for model, result in RESULTS.items():
            writer.writerow(
                {
                    "freeze_id": FREEZE_ID,
                    "model": model,
                    **{field: result[field] for field in fields if field in result},
                    "environment_id": ENVIRONMENT["environment_id"],
                }
            )
    return {"json": json_path, "csv": csv_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    for name, path in write_manifest_artifacts(arguments.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
