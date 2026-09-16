"""Recompute Table R1 from frozen A1/SI Fig. 4 predictions; never fit models."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from generate_all_prediction_evaluation import calculate_task_metrics
from main_result_figure1_v1 import RESULTS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "2.Source Code/produced_graph/response_table_r1_macro_metrics.csv"
TEMPORAL = "2.Source Code/produced_graph/direct_phase3_vs_phase45_rescue/direct_phase3_vs_phase45_rescue_benchmark_predictions.csv"
CONTEMPORANEOUS = "2.Source Code/produced_graph/all_prediction_contemporaneous_random_cv_predictions.csv"
HASHES = {
    TEMPORAL: "b608e162672558b971e92dad906ce863d38814c0e532d5faf20bf88b1b7aad41",
    CONTEMPORANEOUS: "4eb561baedaf22a51d534177b768afb82865f44f7b781cc334a481d40759cde3",
}


def main():
    for relative, expected in HASHES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    temporal = pd.read_csv(ROOT / TEMPORAL, float_precision="round_trip")
    contemporary = pd.read_csv(ROOT / CONTEMPORANEOUS, float_precision="round_trip")
    evaluations = {}
    provenance = []
    for task in ("Forecasting", "Nowcasting", "Contemporaneous"):
        is_temporal = task != "Contemporaneous"
        if is_temporal:
            frame = temporal.loc[temporal.task.eq(task) & temporal.method.eq("frozen_base")]
            source, truth, prediction = TEMPORAL, "reconstructed_overall_phase", "final_overall_phase_pred"
            expected_n, expected_accuracy = 1170, RESULTS[task]["overall_accuracy"]
            assert frame[prediction].eq(frame.base_overall_phase_pred).all()
        else:
            frame = contemporary
            source, truth, prediction = CONTEMPORANEOUS, "overall_phase", "contemporaneous_predict"
            expected_n, expected_accuracy = 5575, 0.6934529147982063
            assert frame.fold.value_counts().sort_index().tolist() == [1115] * 5
        assert len(frame) == expected_n and not frame.duplicated(["area_id", "date"]).any()
        assert set(frame[truth]) == {1, 2, 3, 4, 5}
        assert accuracy_score(frame[truth], frame[prediction]) == expected_accuracy
        evaluations[task] = {
            "y_true": frame[truth], "y_pred": frame[prediction],
            "evaluation_protocol": "fixed_2022_temporal_holdout" if is_temporal else "random_5fold_row_cv",
            "evaluation_population": "full_1170_temporal_test" if is_temporal else "random_5fold_full_oof_5575",
        }
        provenance.append({
            "source_predictions": source, "source_sha256": HASHES[source],
            "source_filter": f"task={task};method=frozen_base" if is_temporal else "all_rows",
            "y_true_column": truth, "y_pred_column": prediction,
            "overall_accuracy": expected_accuracy,
        })
    metrics, _, _, _ = calculate_task_metrics(evaluations)
    # Independent explicit macro call checks the shared utility's aggregation.
    for i, evaluation in enumerate(evaluations.values()):
        explicit = precision_recall_fscore_support(
            evaluation["y_true"], evaluation["y_pred"],
            labels=[1, 2, 3, 4, 5], average="macro", zero_division=0,
        )[:3]
        np.testing.assert_allclose(metrics.loc[i, ["macro_precision", "macro_recall", "macro_f1"]].astype(float), explicit)
    metrics["labels"] = "1,2,3,4,5"
    metrics["average"] = "macro"
    metrics["pooled"] = True
    metrics["prediction_stage"] = "frozen_base_before_rescue"
    metrics = pd.concat([metrics, pd.DataFrame(provenance)], axis=1)
    metrics.to_csv(OUTPUT, index=False)
    print(metrics[["task", "n_observations", "macro_precision", "macro_recall", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
