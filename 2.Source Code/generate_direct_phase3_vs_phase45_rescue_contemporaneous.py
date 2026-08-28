"""Generate Contemporaneous random-five-fold Phase-4/5 rescue evidence.

The frozen Contemporaneous base predictions use seed-0 row-level random
five-fold full-OOF cross-validation.  This generator fits the same three
Phase-4/5 rescue weighting candidates on those folds, selects one pooled-OOF
threshold per candidate, and preserves the base action gate: only a base
Phase-3 prediction may be promoted, and it may only be promoted to Phase 4.

Because threshold selection and descriptive evaluation use the same pooled
OOF population, these results are sensitivity/tuning evidence rather than an
independent benchmark and are not directly comparable with the fixed 2022
temporal holdout used by Forecasting and Nowcasting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb

import generate_all_prediction_evaluation as evaluation
import generate_direct_phase3_vs_phase45_rescue as rescue


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE_DIR = REPO_ROOT / "2.Source Code"
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
PRODUCED_GRAPH_DIR = SOURCE_CODE_DIR / "produced_graph"
DEFAULT_OUTPUT_DIR = (
    PRODUCED_GRAPH_DIR / "direct_phase3_vs_phase45_rescue_contemporaneous"
)
SOURCE_PATH = evaluation.DEFAULT_CONTEMPORANEOUS_SOURCE_PATH
BASE_PREDICTIONS_PATH = evaluation.DEFAULT_CONTEMPORANEOUS_PREDICTIONS_PATH
BASE_AUDIT_PATH = (
    PRODUCED_GRAPH_DIR
    / f"{evaluation.CONTEMPORANEOUS_AUDIT_STEM}.csv"
)
PARAMS_PATH = SOURCE_CODE_DIR / "contemporaneous_hyperparameters_p3.json"

EXPERIMENT_ID = "direct_phase3_vs_phase45_rescue_contemporaneous_random5fold"
REFERENCE_ENVIRONMENT_ID = rescue.REFERENCE_ENVIRONMENT_ID
TASK = "Contemporaneous"
PREFIX = "direct_phase3_vs_phase45_rescue_contemporaneous_"
EVALUATION_PROTOCOL = "random_5fold_row_cv"
EVALUATION_POPULATION = "random_5fold_full_oof_5575"
THRESHOLD_SELECTION_POPULATION = (
    "pooled_random_5fold_full_oof_base_phase3_gate"
)
EVALUATION_INTERPRETATION = (
    "apparent_oof_after_pooled_threshold_selection_not_nested_or_independent"
)
ACTION_CONTRACT = "base_phase3_only_to_phase4"
BASE_PREDICTION_CONTRACT = "frozen_contemporaneous_random_5fold_full_oof"

EXPECTED_ROWS = evaluation.EXPECTED_SOURCE_ROWS
EXPECTED_AREAS = evaluation.EXPECTED_CONTEMPORANEOUS_AREAS
EXPECTED_FOLDS = evaluation.EXPECTED_CONTEMPORANEOUS_FOLDS
EXPECTED_ROWS_PER_FOLD = evaluation.EXPECTED_ROWS_PER_FOLD
EXPECTED_FEATURES = 174
EXPECTED_OOF_ROWS = EXPECTED_ROWS * len(rescue.CANDIDATE_ORDER)
EXPECTED_BINARY_CONFUSION_ROWS = len(rescue.CANDIDATE_ORDER) * 2 * 4
EXPECTED_FIVE_CLASS_CONFUSION_ROWS = len(rescue.CANDIDATE_ORDER) * 25
EXPECTED_MODEL_AUDIT_ROWS = len(rescue.CANDIDATE_ORDER) * EXPECTED_FOLDS

OOF_COLUMNS = (
    "task",
    "candidate_id",
    "method",
    "source_row_index",
    "area_id",
    "date",
    "fold",
    "source_overall_phase",
    "reconstructed_overall_phase",
    "severe_rescue_target",
    "base_overall_phase_pred",
    "in_base_phase3_gate",
    "direct_phase45_score",
    "threshold",
    "threshold_source",
    "triggered",
    "final_overall_phase_pred",
    "fold_training_rows",
    "fold_training_negative_count",
    "fold_training_positive_count",
    "fold_class_ratio",
    "positive_row_weight",
    "selection_status",
    "primary_selected",
    "evaluation_protocol",
    "evaluation_population",
    "threshold_selection_population",
    "evaluation_interpretation",
    "action_contract",
    "base_prediction_contract",
)
FEATURE_MANIFEST_COLUMNS = (
    "task",
    "model_component",
    "feature_order",
    "feature_name",
    "source_dtype",
    "native_missingness_preserved",
    "feature_order_sha256",
)
MODEL_AUDIT_COLUMNS = (
    "task",
    "candidate_id",
    "fold",
    "model_component",
    "estimator_class",
    "objective",
    "run_status",
    "training_rows",
    "training_negative_count",
    "training_positive_count",
    "class_ratio",
    "negative_row_weight",
    "positive_row_weight",
    "scale_pos_weight",
    "sample_weight_sha256",
    "feature_count",
    "feature_order_sha256",
    "parameter_sha256",
    "random_state",
    "n_jobs",
    "training_source_row_index_sha256",
    "target_sha256",
    "validation_rows",
    "validation_source_row_index_sha256",
    "validation_score_sha256",
)
SOURCE_AUDIT_COLUMNS = (
    "run_status",
    "experiment_id",
    "reference_environment_id",
    "evaluation_protocol",
    "evaluation_population",
    "threshold_selection_population",
    "evaluation_interpretation",
    "action_contract",
    "base_prediction_contract",
    "source_rows",
    "oof_rows",
    "areas",
    "n_splits",
    "fold_rows",
    "feature_count",
    "feature_order_sha256",
    "fold_assignment_sha256",
    "source_row_index_sha256",
    "population_key_sha256",
    "source_overall_phase_disagreement_rows",
    "source_path",
    "source_sha256",
    "base_predictions_path",
    "base_predictions_sha256",
    "base_audit_path",
    "base_audit_sha256",
    "params_path",
    "params_sha256",
    "evaluation_generator_path",
    "evaluation_generator_sha256",
    "rescue_generator_path",
    "rescue_generator_sha256",
    "generator_path",
    "generator_sha256",
    "platform_family",
    "python_version",
    "numpy_version",
    "pandas_version",
    "scipy_version",
    "sklearn_version",
    "xgboost_version",
    "matplotlib_version",
    "xgboost_dll_sha256",
    "random_state",
    "n_jobs",
    "protected_manifest_sha256_before",
    "protected_manifest_sha256_after",
    "protected_manifest_match",
    "artifact_manifest_json",
    "artifact_manifest_sha256",
)

CSV_SCHEMAS = {
    f"{PREFIX}oof_predictions.csv": OOF_COLUMNS,
    f"{PREFIX}threshold_frontier.csv": rescue.THRESHOLD_COLUMNS,
    f"{PREFIX}selected_policies.csv": rescue.SELECTED_POLICY_COLUMNS,
    f"{PREFIX}gate_pooled_metrics.csv": rescue.GATE_METRIC_COLUMNS,
    f"{PREFIX}binary_confusion_matrices.csv": rescue.BINARY_CONFUSION_COLUMNS,
    f"{PREFIX}five_class_confusion_matrices.csv": rescue.FIVE_CLASS_CONFUSION_COLUMNS,
    f"{PREFIX}feature_manifest.csv": FEATURE_MANIFEST_COLUMNS,
    f"{PREFIX}model_audit.csv": MODEL_AUDIT_COLUMNS,
    f"{PREFIX}source_audit.csv": SOURCE_AUDIT_COLUMNS,
}
DATA_BASENAMES = tuple(name for name in CSV_SCHEMAS if not name.endswith("source_audit.csv"))
CONFIGURATION_BASENAME = f"{PREFIX}configuration.json"
SOURCE_AUDIT_BASENAME = f"{PREFIX}source_audit.csv"
EXPECTED_ARTIFACTS = (*DATA_BASENAMES, CONFIGURATION_BASENAME, SOURCE_AUDIT_BASENAME)


def _series_sha256(values: Sequence[object], name: str) -> str:
    frame = pd.DataFrame({name: pd.Series(values).reset_index(drop=True)})
    payload = frame.to_csv(
        index=False,
        float_format="%.17g",
        na_rep="<NA>",
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_base_predictions_and_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not BASE_PREDICTIONS_PATH.is_file() or not BASE_AUDIT_PATH.is_file():
        raise FileNotFoundError(
            "Frozen Contemporaneous random-CV predictions and audit are required."
        )
    base = evaluation.load_contemporaneous_random_cv_predictions(
        BASE_PREDICTIONS_PATH
    )
    audit = pd.read_csv(BASE_AUDIT_PATH)
    if len(audit) != 1:
        raise ValueError("Contemporaneous base audit must contain one row.")
    expected_hash = str(audit.loc[0, "predictions_sha256"])
    if rescue.file_sha256(BASE_PREDICTIONS_PATH) != expected_hash:
        raise ValueError("Contemporaneous base predictions do not match their audit.")
    if (
        str(audit.loc[0, "evaluation_protocol"]) != EVALUATION_PROTOCOL
        or str(audit.loc[0, "evaluation_population"]) != EVALUATION_POPULATION
        or int(audit.loc[0, "shuffle_seed"]) != 0
    ):
        raise ValueError("Contemporaneous base protocol drifted.")
    return base, audit


def prepare_model_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    base, base_audit = _load_base_predictions_and_audit()
    data = pd.read_csv(SOURCE_PATH)
    if len(data) != EXPECTED_ROWS or "source_row_index" in data:
        raise ValueError("Contemporaneous source row contract failed.")
    data.insert(0, "source_row_index", np.arange(len(data), dtype=int))
    evaluation._require_columns(
        data,
        ["area_id", "date", evaluation.TRUE_COLUMN, "phase1_percent"],
    )
    data = data.loc[data["phase1_percent"].notna()].copy()
    if len(data) != EXPECTED_ROWS:
        raise ValueError("phase1_percent filtering changed the source population.")
    data["date"] = evaluation._normalize_date(data["date"])
    data = data.sort_values(["area_id", "date"], kind="mergesort").reset_index(
        drop=True
    )
    source_phase = pd.to_numeric(data[evaluation.TRUE_COLUMN], errors="coerce")
    if source_phase.isna().any() or not np.allclose(source_phase, source_phase.round()):
        raise ValueError("Source overall_phase must contain complete integer labels.")
    data = data.rename(columns={evaluation.TRUE_COLUMN: "source_overall_phase"})
    data["source_overall_phase"] = source_phase.astype(int)
    data = evaluation._add_contemporaneous_targets(data)
    data["kfolds"] = -1
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    splitter = KFold(n_splits=EXPECTED_FOLDS, shuffle=False)
    for fold, (_, validation_index) in enumerate(splitter.split(data)):
        data.loc[validation_index, "kfolds"] = fold

    fold_counts = data["kfolds"].value_counts().sort_index().to_dict()
    if fold_counts != {fold: EXPECTED_ROWS_PER_FOLD for fold in range(EXPECTED_FOLDS)}:
        raise ValueError(f"Unexpected random-CV fold sizes: {fold_counts}")

    excluded = {
        "source_row_index",
        "area_id",
        "date",
        "source_overall_phase",
        *[f"phase{phase}_percent" for phase in range(1, 6)],
        *evaluation.CONTEMPORANEOUS_TARGETS.values(),
    }
    feature_columns = [column for column in data.columns if column not in excluded]
    if len(feature_columns) != EXPECTED_FEATURES or "kfolds" not in feature_columns:
        raise ValueError(
            "Expected 174 contemporaneous predictors including kfolds, "
            f"found {len(feature_columns)}."
        )
    features = data[feature_columns].apply(pd.to_numeric, errors="raise")
    targets = data[list(evaluation.CONTEMPORANEOUS_TARGETS.values())].apply(
        pd.to_numeric, errors="raise"
    )
    if targets.isna().any().any():
        raise ValueError("Contemporaneous cumulative targets contain missing values.")
    actual_frame = pd.DataFrame(
        {
            f"phase{phase}_actual": targets[target_name]
            for phase, target_name in evaluation.CONTEMPORANEOUS_TARGETS.items()
        }
    )
    data["reconstructed_overall_phase"] = evaluation._phase_from_cumulative(
        actual_frame, "actual"
    )

    aligned_base = data[["source_row_index"]].merge(
        base[
            [
                "source_row_index",
                "area_id",
                "date",
                "fold",
                "source_overall_phase",
                evaluation.TRUE_COLUMN,
                "contemporaneous_predict",
            ]
        ],
        on="source_row_index",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(aligned_base) != len(data) or aligned_base.isna().any().any():
        raise ValueError("Frozen base predictions failed complete source-row alignment.")
    if not np.array_equal(
        aligned_base["source_row_index"].to_numpy(dtype=int),
        data["source_row_index"].to_numpy(dtype=int),
    ):
        raise ValueError("Frozen base alignment changed source-row order.")
    comparisons = {
        "area_id": pd.to_numeric(data["area_id"], errors="raise"),
        "date": data["date"].astype(str),
        "fold": data["kfolds"].astype(int),
        "source_overall_phase": data["source_overall_phase"].astype(int),
        evaluation.TRUE_COLUMN: data["reconstructed_overall_phase"].astype(int),
    }
    for column, expected in comparisons.items():
        observed = aligned_base[column]
        if column == "area_id":
            equal = np.isclose(
                pd.to_numeric(observed, errors="raise").to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
            if not bool(equal.all()):
                raise ValueError(f"Frozen base alignment drifted for {column}.")
            continue
        if column in {"fold", "source_overall_phase", evaluation.TRUE_COLUMN}:
            equal = observed.astype(int).eq(expected)
        else:
            equal = observed.astype(str).eq(expected)
        if not bool(equal.all()):
            raise ValueError(f"Frozen base alignment drifted for {column}.")

    data["fold"] = aligned_base["fold"].astype(int)
    data["base_overall_phase_pred"] = aligned_base[
        "contemporaneous_predict"
    ].astype(int)
    data["in_base_phase3_gate"] = data["base_overall_phase_pred"].eq(3)
    data["severe_rescue_target"] = rescue.build_severe_rescue_target(
        data["reconstructed_overall_phase"]
    )
    return data, features, feature_columns, base_audit


def classifier_parameters() -> dict[str, object]:
    parameters = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    parameters.update(
        {
            "objective": "binary:logistic",
            "random_state": 0,
            "n_jobs": 1,
            "scale_pos_weight": 1,
        }
    )
    return parameters


def build_feature_manifest(
    features: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    order_hash = rescue.json_sha256(list(feature_columns))
    rows = [
        {
            "task": TASK,
            "model_component": "direct_contemporaneous_phase45_classifier",
            "feature_order": order,
            "feature_name": feature,
            "source_dtype": str(features[feature].dtype),
            "native_missingness_preserved": True,
            "feature_order_sha256": order_hash,
        }
        for order, feature in enumerate(feature_columns, start=1)
    ]
    result = pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS)
    if len(result) != EXPECTED_FEATURES or result.duplicated("feature_order").any():
        raise ValueError("Contemporaneous feature-manifest contract failed.")
    return result


def generate_oof_scores(
    data: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameters = classifier_parameters()
    feature_hash = rescue.json_sha256(list(feature_columns))
    parameter_hash = rescue.json_sha256(parameters)
    rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for fold in range(EXPECTED_FOLDS):
        validation_mask = data["fold"].eq(fold)
        training_mask = data["fold"].ne(fold) & data[
            "reconstructed_overall_phase"
        ].isin([3, 4, 5])
        target = data.loc[training_mask, "severe_rescue_target"].reset_index(
            drop=True
        )
        training_matrix = features.loc[training_mask].reset_index(drop=True)
        validation_matrix = features.loc[validation_mask].reset_index(drop=True)
        if len(validation_matrix) != EXPECTED_ROWS_PER_FOLD:
            raise ValueError(f"Fold {fold} validation support drifted.")
        for candidate in rescue.CANDIDATE_ORDER:
            sample_weight, weight_audit = rescue.build_candidate_sample_weight(
                target, candidate
            )
            model = xgb.XGBClassifier(**parameters)
            model.fit(training_matrix, target, sample_weight=sample_weight)
            score = np.asarray(model.predict_proba(validation_matrix)[:, 1], dtype=float)
            if not np.isfinite(score).all():
                raise ValueError(f"{candidate}/fold-{fold} produced non-finite scores.")
            output = data.loc[
                validation_mask,
                [
                    "source_row_index",
                    "area_id",
                    "date",
                    "fold",
                    "source_overall_phase",
                    "reconstructed_overall_phase",
                    "severe_rescue_target",
                    "base_overall_phase_pred",
                    "in_base_phase3_gate",
                ],
            ].copy()
            output.insert(0, "method", rescue.CANDIDATE_METHODS[candidate])
            output.insert(0, "candidate_id", candidate)
            output.insert(0, "task", TASK)
            output["direct_phase45_score"] = score
            output["fold_training_rows"] = len(training_matrix)
            output["fold_training_negative_count"] = weight_audit["negative_count"]
            output["fold_training_positive_count"] = weight_audit["positive_count"]
            output["fold_class_ratio"] = weight_audit["class_ratio"]
            output["positive_row_weight"] = weight_audit["positive_row_weight"]
            rows.append(output)
            audit_rows.append(
                {
                    "task": TASK,
                    "candidate_id": candidate,
                    "fold": fold,
                    "model_component": "direct_contemporaneous_phase45_classifier",
                    "estimator_class": "XGBClassifier",
                    "objective": "binary:logistic",
                    "run_status": "fitted_oof_not_published",
                    "training_rows": len(training_matrix),
                    "training_negative_count": weight_audit["negative_count"],
                    "training_positive_count": weight_audit["positive_count"],
                    "class_ratio": weight_audit["class_ratio"],
                    "negative_row_weight": 1.0,
                    "positive_row_weight": weight_audit["positive_row_weight"],
                    "scale_pos_weight": 1.0,
                    "sample_weight_sha256": _series_sha256(
                        sample_weight, "sample_weight"
                    ),
                    "feature_count": len(feature_columns),
                    "feature_order_sha256": feature_hash,
                    "parameter_sha256": parameter_hash,
                    "random_state": 0,
                    "n_jobs": 1,
                    "training_source_row_index_sha256": _series_sha256(
                        data.loc[training_mask, "source_row_index"],
                        "source_row_index",
                    ),
                    "target_sha256": _series_sha256(target, "severe_rescue_target"),
                    "validation_rows": len(validation_matrix),
                    "validation_source_row_index_sha256": _series_sha256(
                        data.loc[validation_mask, "source_row_index"],
                        "source_row_index",
                    ),
                    "validation_score_sha256": _series_sha256(
                        score, "direct_phase45_score"
                    ),
                }
            )
    oof = pd.concat(rows, ignore_index=True).sort_values(
        ["candidate_id", "source_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    if len(oof) != EXPECTED_OOF_ROWS or oof.duplicated(
        ["candidate_id", "source_row_index"]
    ).any():
        raise ValueError("Contemporaneous rescue OOF score contract failed.")
    for candidate, group in oof.groupby("candidate_id", sort=True, observed=True):
        if len(group) != EXPECTED_ROWS or set(group["fold"]) != set(range(EXPECTED_FOLDS)):
            raise ValueError(f"{candidate} does not cover the full five-fold population.")
    audit = pd.DataFrame(audit_rows, columns=MODEL_AUDIT_COLUMNS).sort_values(
        ["candidate_id", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    if len(audit) != EXPECTED_MODEL_AUDIT_ROWS or audit.duplicated(
        ["candidate_id", "fold"]
    ).any():
        raise ValueError("Contemporaneous model-audit contract failed.")
    return oof, audit


def select_policies_and_apply(
    oof_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_frontiers = []
    for candidate in rescue.CANDIDATE_ORDER:
        frontier = rescue.build_threshold_frontier(
            oof_scores.loc[oof_scores["candidate_id"].eq(candidate)]
        )
        frontier["threshold_source"] = THRESHOLD_SELECTION_POPULATION
        candidate_frontiers.append(frontier)
    frontier, policies = rescue.select_task_policies(
        pd.concat(candidate_frontiers, ignore_index=True)
    )
    frontier = frontier.sort_values(
        ["candidate_id", "threshold_rank"], kind="mergesort"
    ).reset_index(drop=True)
    policies = policies.sort_values("candidate_id", kind="mergesort").reset_index(
        drop=True
    )
    policy_index = policies.set_index("candidate_id")
    completed: list[pd.DataFrame] = []
    for candidate in rescue.CANDIDATE_ORDER:
        frame = oof_scores.loc[oof_scores["candidate_id"].eq(candidate)].copy()
        threshold = float(policy_index.loc[candidate, "threshold"])
        final_phase = rescue.apply_phase45_rescue(
            frame["base_overall_phase_pred"],
            frame["direct_phase45_score"],
            threshold,
        )
        frame["threshold"] = threshold
        frame["threshold_source"] = THRESHOLD_SELECTION_POPULATION
        frame["triggered"] = (
            frame["base_overall_phase_pred"].to_numpy(dtype=int) != final_phase
        )
        frame["final_overall_phase_pred"] = final_phase
        frame["selection_status"] = str(
            policy_index.loc[candidate, "selection_status"]
        )
        frame["primary_selected"] = bool(
            policy_index.loc[candidate, "primary_selected"]
        )
        frame["evaluation_protocol"] = EVALUATION_PROTOCOL
        frame["evaluation_population"] = EVALUATION_POPULATION
        frame["threshold_selection_population"] = THRESHOLD_SELECTION_POPULATION
        frame["evaluation_interpretation"] = EVALUATION_INTERPRETATION
        frame["action_contract"] = ACTION_CONTRACT
        frame["base_prediction_contract"] = BASE_PREDICTION_CONTRACT
        completed.append(frame.loc[:, list(OOF_COLUMNS)])
    predictions = pd.concat(completed, ignore_index=True).sort_values(
        ["candidate_id", "source_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    if len(predictions) != EXPECTED_OOF_ROWS:
        raise ValueError("Completed Contemporaneous OOF predictions changed support.")
    return predictions, frontier.loc[:, list(rescue.THRESHOLD_COLUMNS)], policies


def build_metric_and_confusion_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate_rows: list[dict[str, object]] = []
    binary_rows: list[pd.DataFrame] = []
    five_rows: list[pd.DataFrame] = []
    for candidate in rescue.CANDIDATE_ORDER:
        method = rescue.CANDIDATE_METHODS[candidate]
        frame = predictions.loc[predictions["candidate_id"].eq(candidate)].copy()
        frame["method_role"] = "candidate_sensitivity"
        gate_rows.append(rescue._gate_metric_record(frame))
        for scope, scoped in (
            ("gate", frame.loc[frame["in_base_phase3_gate"].astype(bool)]),
            ("full_oof", frame),
        ):
            actual = scoped["reconstructed_overall_phase"].to_numpy(dtype=int)
            predicted = scoped["final_overall_phase_pred"].to_numpy(dtype=int)
            cells = rescue.build_binary_confusion_cells(
                np.isin(actual, [4, 5]).astype(int),
                np.isin(predicted, [4, 5]).astype(int),
            )
            cells.insert(0, "method", method)
            cells.insert(0, "task", TASK)
            cells.insert(0, "population_scope", scope)
            binary_rows.append(cells.loc[:, list(rescue.BINARY_CONFUSION_COLUMNS)])
        five = rescue.build_five_class_confusion_cells(
            frame["reconstructed_overall_phase"],
            frame["final_overall_phase_pred"],
        )
        five.insert(0, "method", method)
        five.insert(0, "task", TASK)
        five_rows.append(five.loc[:, list(rescue.FIVE_CLASS_CONFUSION_COLUMNS)])
    gate_metrics = pd.DataFrame(gate_rows, columns=rescue.GATE_METRIC_COLUMNS)
    binary = pd.concat(binary_rows, ignore_index=True)
    five = pd.concat(five_rows, ignore_index=True)
    if len(gate_metrics) != 3 or gate_metrics.duplicated(["task", "method"]).any():
        raise ValueError("Contemporaneous gate-metric contract failed.")
    if len(binary) != EXPECTED_BINARY_CONFUSION_ROWS or binary.duplicated(
        ["population_scope", "task", "method", "actual_binary", "predicted_binary"]
    ).any():
        raise ValueError("Contemporaneous binary-confusion contract failed.")
    if len(five) != EXPECTED_FIVE_CLASS_CONFUSION_ROWS or five.duplicated(
        ["task", "method", "actual_phase", "predicted_phase"]
    ).any():
        raise ValueError("Contemporaneous five-class-confusion contract failed.")
    return gate_metrics, binary, five


def _configuration_payload(
    environment: Mapping[str, object],
    feature_columns: Sequence[str],
    policies: pd.DataFrame,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    package_versions = environment["package_versions"]
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "environment": {
            "environment_id": REFERENCE_ENVIRONMENT_ID,
            "platform_family": environment["platform_family"],
            "python_version": environment["python_version"],
            "numpy_version": package_versions["numpy"],
            "pandas_version": package_versions["pandas"],
            "scipy_version": package_versions["scipy"],
            "scikit_learn_version": package_versions["scikit-learn"],
            "xgboost_version": package_versions["xgboost"],
            "matplotlib_version": package_versions["matplotlib"],
            "xgboost_dll_sha256": environment["xgboost_dll_sha256"],
        },
        "decisions": {
            "truth": "reconstructed_fixed_0.20_phase",
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "evaluation_population": EVALUATION_POPULATION,
            "training_population": "outer_fold_training_rows_with_actual_phase_3_4_5",
            "positive_class": "reconstructed_phase_4_or_5",
            "negative_training_class": "reconstructed_phase_3",
            "threshold_selection_population": THRESHOLD_SELECTION_POPULATION,
            "selection_objective": "pooled_oof_phase45_f2_first",
            "evaluation_interpretation": EVALUATION_INTERPRETATION,
            "allowed_action": ACTION_CONTRACT,
            "base_prediction_contract": BASE_PREDICTION_CONTRACT,
            "final_refit_included": False,
            "manuscript_adoption_authorized": False,
            "direct_comparison_with_temporal_holdout_authorized": False,
        },
        "folds": {
            "n_splits": EXPECTED_FOLDS,
            "rows_per_fold": EXPECTED_ROWS_PER_FOLD,
            "shuffle_seed": 0,
            "splitter_shuffle": False,
            "row_level": True,
        },
        "candidates": {
            "order": list(rescue.CANDIDATE_ORDER),
            "methods": [
                rescue.CANDIDATE_METHODS[candidate]
                for candidate in rescue.CANDIDATE_ORDER
            ],
            "weight_formulas": {
                "unweighted": "1",
                "sqrt_balance": "sqrt(n_phase3/n_phase45)",
                "full_balance": "n_phase3/n_phase45",
            },
            "negative_row_weight": 1.0,
            "scale_pos_weight": 1.0,
            "selected_policies": rescue.json_safe(policies.to_dict("records")),
        },
        "features": {
            "count": len(feature_columns),
            "includes_kfolds": "kfolds" in feature_columns,
            "order_sha256": rescue.json_sha256(list(feature_columns)),
        },
        "parameters": classifier_parameters(),
        "sources": {
            "source_path": rescue.relative_path(SOURCE_PATH),
            "source_sha256": rescue.file_sha256(SOURCE_PATH),
            "base_predictions_path": rescue.relative_path(BASE_PREDICTIONS_PATH),
            "base_predictions_sha256": rescue.file_sha256(BASE_PREDICTIONS_PATH),
            "base_audit_path": rescue.relative_path(BASE_AUDIT_PATH),
            "base_audit_sha256": rescue.file_sha256(BASE_AUDIT_PATH),
            "params_path": rescue.relative_path(PARAMS_PATH),
            "params_sha256": rescue.file_sha256(PARAMS_PATH),
            "evaluation_generator_path": rescue.relative_path(
                Path(evaluation.__file__)
            ),
            "evaluation_generator_sha256": rescue.file_sha256(
                Path(evaluation.__file__)
            ),
            "rescue_generator_path": rescue.relative_path(Path(rescue.__file__)),
            "rescue_generator_sha256": rescue.file_sha256(Path(rescue.__file__)),
            "generator_path": rescue.relative_path(Path(__file__)),
            "generator_sha256": rescue.file_sha256(Path(__file__)),
        },
        "row_contracts": {
            "oof_predictions": EXPECTED_OOF_ROWS,
            "threshold_frontier": "sum(distinct_finite_pooled_gate_scores + 1)",
            "selected_policies": len(rescue.CANDIDATE_ORDER),
            "gate_pooled_metrics": len(rescue.CANDIDATE_ORDER),
            "binary_confusion_matrices": EXPECTED_BINARY_CONFUSION_ROWS,
            "five_class_confusion_matrices": EXPECTED_FIVE_CLASS_CONFUSION_ROWS,
            "feature_manifest": EXPECTED_FEATURES,
            "model_audit": EXPECTED_MODEL_AUDIT_ROWS,
        },
        "csv_schemas": {
            name: list(columns) for name, columns in sorted(CSV_SCHEMAS.items())
        },
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
    }


def _source_audit_row(
    environment: Mapping[str, object],
    base_audit: pd.DataFrame,
    feature_columns: Sequence[str],
    predictions: pd.DataFrame,
    protected_manifest: str,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    package_versions = environment["package_versions"]
    reference = predictions.loc[predictions["candidate_id"].eq("unweighted")]
    source_disagreements = int(
        reference["source_overall_phase"].ne(
            reference["reconstructed_overall_phase"]
        ).sum()
    )
    manifest_json = json.dumps(
        dict(sorted(artifact_hashes.items())),
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "run_status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "reference_environment_id": REFERENCE_ENVIRONMENT_ID,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "evaluation_population": EVALUATION_POPULATION,
        "threshold_selection_population": THRESHOLD_SELECTION_POPULATION,
        "evaluation_interpretation": EVALUATION_INTERPRETATION,
        "action_contract": ACTION_CONTRACT,
        "base_prediction_contract": BASE_PREDICTION_CONTRACT,
        "source_rows": EXPECTED_ROWS,
        "oof_rows": len(predictions),
        "areas": int(reference["area_id"].nunique()),
        "n_splits": EXPECTED_FOLDS,
        "fold_rows": "|".join([str(EXPECTED_ROWS_PER_FOLD)] * EXPECTED_FOLDS),
        "feature_count": len(feature_columns),
        "feature_order_sha256": rescue.json_sha256(list(feature_columns)),
        "fold_assignment_sha256": str(base_audit.loc[0, "fold_assignment_sha256"]),
        "source_row_index_sha256": str(
            base_audit.loc[0, "source_row_index_sha256"]
        ),
        "population_key_sha256": str(base_audit.loc[0, "population_key_sha256"]),
        "source_overall_phase_disagreement_rows": source_disagreements,
        "source_path": rescue.relative_path(SOURCE_PATH),
        "source_sha256": rescue.file_sha256(SOURCE_PATH),
        "base_predictions_path": rescue.relative_path(BASE_PREDICTIONS_PATH),
        "base_predictions_sha256": rescue.file_sha256(BASE_PREDICTIONS_PATH),
        "base_audit_path": rescue.relative_path(BASE_AUDIT_PATH),
        "base_audit_sha256": rescue.file_sha256(BASE_AUDIT_PATH),
        "params_path": rescue.relative_path(PARAMS_PATH),
        "params_sha256": rescue.file_sha256(PARAMS_PATH),
        "evaluation_generator_path": rescue.relative_path(Path(evaluation.__file__)),
        "evaluation_generator_sha256": rescue.file_sha256(
            Path(evaluation.__file__)
        ),
        "rescue_generator_path": rescue.relative_path(Path(rescue.__file__)),
        "rescue_generator_sha256": rescue.file_sha256(Path(rescue.__file__)),
        "generator_path": rescue.relative_path(Path(__file__)),
        "generator_sha256": rescue.file_sha256(Path(__file__)),
        "platform_family": environment["platform_family"],
        "python_version": environment["python_version"],
        "numpy_version": package_versions["numpy"],
        "pandas_version": package_versions["pandas"],
        "scipy_version": package_versions["scipy"],
        "sklearn_version": package_versions["scikit-learn"],
        "xgboost_version": package_versions["xgboost"],
        "matplotlib_version": package_versions["matplotlib"],
        "xgboost_dll_sha256": environment["xgboost_dll_sha256"],
        "random_state": 0,
        "n_jobs": 1,
        "protected_manifest_sha256_before": protected_manifest,
        "protected_manifest_sha256_after": protected_manifest,
        "protected_manifest_match": True,
        "artifact_manifest_json": manifest_json,
        "artifact_manifest_sha256": rescue.manifest_sha256(artifact_hashes),
    }


def write_artifacts(
    staging_dir: Path,
    *,
    environment: Mapping[str, object],
    protected_manifest: str,
    base_audit: pd.DataFrame,
    feature_columns: Sequence[str],
    predictions: pd.DataFrame,
    frontier: pd.DataFrame,
    policies: pd.DataFrame,
    gate_metrics: pd.DataFrame,
    binary_confusion: pd.DataFrame,
    five_confusion: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    model_audit: pd.DataFrame,
) -> None:
    payloads = {
        f"{PREFIX}oof_predictions.csv": predictions,
        f"{PREFIX}threshold_frontier.csv": frontier,
        f"{PREFIX}selected_policies.csv": policies,
        f"{PREFIX}gate_pooled_metrics.csv": gate_metrics,
        f"{PREFIX}binary_confusion_matrices.csv": binary_confusion,
        f"{PREFIX}five_class_confusion_matrices.csv": five_confusion,
        f"{PREFIX}feature_manifest.csv": feature_manifest,
        f"{PREFIX}model_audit.csv": model_audit,
    }
    for name, frame in payloads.items():
        rescue._write_csv(frame, staging_dir / name, CSV_SCHEMAS[name])
    data_hashes = {
        name: rescue.file_sha256(staging_dir / name) for name in DATA_BASENAMES
    }
    configuration = _configuration_payload(
        environment, feature_columns, policies, data_hashes
    )
    rescue._write_json(configuration, staging_dir / CONFIGURATION_BASENAME)
    manifest = {
        **data_hashes,
        CONFIGURATION_BASENAME: rescue.file_sha256(
            staging_dir / CONFIGURATION_BASENAME
        ),
    }
    audit = pd.DataFrame(
        [
            _source_audit_row(
                environment,
                base_audit,
                feature_columns,
                predictions,
                protected_manifest,
                manifest,
            )
        ],
        columns=SOURCE_AUDIT_COLUMNS,
    )
    rescue._write_csv(
        audit, staging_dir / SOURCE_AUDIT_BASENAME, SOURCE_AUDIT_COLUMNS
    )
    validate_artifact_contract(staging_dir)


def validate_artifact_contract(directory: Path) -> None:
    directory = Path(directory)
    files = sorted(path.name for path in directory.iterdir() if path.is_file())
    directories = [path.name for path in directory.iterdir() if path.is_dir()]
    if files != sorted(EXPECTED_ARTIFACTS) or directories:
        raise ValueError(f"Artifact basename contract failed: {files}, {directories}")
    expected_rows = {
        f"{PREFIX}oof_predictions.csv": EXPECTED_OOF_ROWS,
        f"{PREFIX}selected_policies.csv": len(rescue.CANDIDATE_ORDER),
        f"{PREFIX}gate_pooled_metrics.csv": len(rescue.CANDIDATE_ORDER),
        f"{PREFIX}binary_confusion_matrices.csv": EXPECTED_BINARY_CONFUSION_ROWS,
        f"{PREFIX}five_class_confusion_matrices.csv": EXPECTED_FIVE_CLASS_CONFUSION_ROWS,
        f"{PREFIX}feature_manifest.csv": EXPECTED_FEATURES,
        f"{PREFIX}model_audit.csv": EXPECTED_MODEL_AUDIT_ROWS,
        SOURCE_AUDIT_BASENAME: 1,
    }
    loaded: dict[str, pd.DataFrame] = {}
    for name, columns in CSV_SCHEMAS.items():
        frame = pd.read_csv(
            directory / name,
            na_values=["<NA>"],
            keep_default_na=True,
            float_precision="round_trip",
        )
        if tuple(frame.columns) != tuple(columns):
            raise ValueError(f"{name} schema drifted.")
        loaded[name] = frame
        if name in expected_rows and len(frame) != expected_rows[name]:
            raise ValueError(f"{name} row count drifted: {len(frame)}")
    predictions = loaded[f"{PREFIX}oof_predictions.csv"]
    if predictions.duplicated(["candidate_id", "source_row_index"]).any():
        raise ValueError("OOF prediction keys are duplicated.")
    if set(predictions["task"]) != {TASK} or set(predictions["method"]) != set(
        rescue.CANDIDATE_METHODS.values()
    ):
        raise ValueError("OOF task or method set drifted.")
    if not predictions["evaluation_protocol"].eq(EVALUATION_PROTOCOL).all():
        raise ValueError("OOF evaluation protocol drifted.")
    for (_, _), group in predictions.groupby(
        ["candidate_id", "fold"], sort=True, observed=True
    ):
        if len(group) != EXPECTED_ROWS_PER_FOLD:
            raise ValueError("A candidate/fold OOF block changed support.")
        rescue.predecessor.assert_postclassification_invariants(
            group["base_overall_phase_pred"], group["final_overall_phase_pred"]
        )
    frontier = loaded[f"{PREFIX}threshold_frontier.csv"]
    for candidate in rescue.CANDIDATE_ORDER:
        scores = predictions.loc[
            predictions["candidate_id"].eq(candidate)
            & predictions["in_base_phase3_gate"].astype(bool),
            "direct_phase45_score",
        ].to_numpy(dtype=float)
        group = frontier.loc[frontier["candidate_id"].eq(candidate)]
        if len(group) != len(np.unique(scores)) + 1:
            raise ValueError(f"{candidate} threshold-frontier formula failed.")
        if int(group["within_candidate_selected"].astype(bool).sum()) != 1:
            raise ValueError(f"{candidate} requires one selected threshold.")
    configuration = json.loads(
        (directory / CONFIGURATION_BASENAME).read_text(encoding="utf-8")
    )
    if set(configuration["artifact_hashes"]) != set(DATA_BASENAMES):
        raise ValueError("Configuration artifact manifest drifted.")
    for name, expected_hash in configuration["artifact_hashes"].items():
        if rescue.file_sha256(directory / name) != expected_hash:
            raise ValueError(f"Configuration hash mismatch for {name}.")
    audit = loaded[SOURCE_AUDIT_BASENAME].iloc[0]
    manifest = json.loads(str(audit["artifact_manifest_json"]))
    if set(manifest) != set((*DATA_BASENAMES, CONFIGURATION_BASENAME)):
        raise ValueError("Source-audit artifact manifest drifted.")
    if rescue.manifest_sha256(manifest) != str(audit["artifact_manifest_sha256"]):
        raise ValueError("Source-audit manifest digest drifted.")
    for name, expected_hash in manifest.items():
        if rescue.file_sha256(directory / name) != expected_hash:
            raise ValueError(f"Source-audit hash mismatch for {name}.")


def _generate_staging_bundle(
    staging_dir: Path,
    *,
    environment: Mapping[str, object],
    protected_manifest: str,
) -> dict[str, object]:
    data, features, feature_columns, base_audit = prepare_model_inputs()
    feature_manifest = build_feature_manifest(features, feature_columns)
    oof_scores, model_audit = generate_oof_scores(
        data, features, feature_columns
    )
    predictions, frontier, policies = select_policies_and_apply(oof_scores)
    gate_metrics, binary_confusion, five_confusion = (
        build_metric_and_confusion_tables(predictions)
    )
    write_artifacts(
        staging_dir,
        environment=environment,
        protected_manifest=protected_manifest,
        base_audit=base_audit,
        feature_columns=feature_columns,
        predictions=predictions,
        frontier=frontier,
        policies=policies,
        gate_metrics=gate_metrics,
        binary_confusion=binary_confusion,
        five_confusion=five_confusion,
        feature_manifest=feature_manifest,
        model_audit=model_audit,
    )
    primary = policies.loc[policies["primary_selected"].astype(bool)]
    return {
        "policies": policies,
        "primary_selected": (
            str(primary["candidate_id"].iloc[0]) if not primary.empty else None
        ),
    }


def run_generation(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_dir = Path(output_dir)
    rescue.validate_generation_target(output_dir)
    environment = rescue.assert_formal_environment()
    protected_before = rescue.protected_artifact_manifest_sha256(
        excluded_paths=(output_dir,)
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    first_staging = Path(
        tempfile.mkdtemp(prefix=".contemporaneous-phase45-a-", dir=output_dir.parent)
    )
    second_staging = Path(
        tempfile.mkdtemp(prefix=".contemporaneous-phase45-b-", dir=output_dir.parent)
    )
    published = False
    try:
        result = _generate_staging_bundle(
            first_staging,
            environment=environment,
            protected_manifest=protected_before,
        )
        _generate_staging_bundle(
            second_staging,
            environment=environment,
            protected_manifest=protected_before,
        )
        validate_artifact_contract(first_staging)
        validate_artifact_contract(second_staging)
        rescue.assert_byte_identical_artifacts(
            first_staging, second_staging, EXPECTED_ARTIFACTS
        )
        rescue._rmtree_with_retry(second_staging)
        if output_dir.exists():
            output_dir.rmdir()
        rescue._replace_with_retry(first_staging, output_dir)
        published = True
        protected_after = rescue.protected_artifact_manifest_sha256(
            excluded_paths=(output_dir,)
        )
        if protected_after != protected_before:
            rescue._rmtree_with_retry(output_dir)
            raise RuntimeError(
                "Protected produced_graph artifacts changed during generation."
            )
        validate_artifact_contract(output_dir)
        policies = result["policies"]
        return {
            "output_dir": output_dir,
            "primary_selected": result["primary_selected"],
            "selected_thresholds": {
                str(row.candidate_id): float(row.threshold)
                for row in policies.itertuples(index=False)
            },
        }
    except BaseException:
        for staging in (first_staging, second_staging):
            if staging.exists():
                try:
                    rescue._rmtree_with_retry(staging)
                except PermissionError:
                    pass
        if published and output_dir.exists():
            try:
                rescue._rmtree_with_retry(output_dir)
            except PermissionError:
                pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate seed-0 random-five-fold Contemporaneous Phase-4/5 "
            "rescue sensitivity artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Absent or empty write-once target directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    result = run_generation(arguments.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(result["output_dir"]),
                "primary_selected": result["primary_selected"],
                "selected_thresholds": result["selected_thresholds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
