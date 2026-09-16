"""Display-only regression check; uses saved counts and never fits models."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

path = Path(__file__).resolve().parents[1] / "2.Source Code/generate_direct_phase3_vs_phase45_rescue_five_class_confusion.py"
spec = importlib.util.spec_from_file_location("rescue_confusion_display", path)
atlas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas)


class RescueDisplayTests(unittest.TestCase):
    def test_combined_prediction_column_preserves_all_actual_rows(self):
        source, methods = atlas.load_and_validate_source()
        original = source.copy(deep=True)
        figure = atlas.render_figure(source, methods)
        try:
            for index, axis in enumerate(figure.axes[1:]):
                task = atlas.TASK_ORDER[index // 2]
                rescued = index % 2 == 1
                method = methods[task] if rescued else atlas.BASE_METHOD
                groups = [(1,), (2,), (3,), (4, 5)] if rescued else [(p,) for p in atlas.PHASES]
                labels = [t.get_text() for t in axis.get_xticklabels()]
                self.assertEqual(labels, ["P1", "P2", "P3", "Combined\nP4/5"] if rescued else [f"P{p}" for p in atlas.PHASES])
                self.assertEqual(len(axis.patches), 5 * len(groups))
                matrix = source.loc[source.task.eq(task) & source.method.eq(method)]
                plotted = {text.get_position(): text.get_text() for text in axis.texts[:-1]}
                for actual in atlas.PHASES:
                    for col, group in enumerate(groups):
                        cells = matrix.loc[matrix.actual_phase.eq(actual) & matrix.predicted_phase.isin(group)]
                        self.assertEqual(plotted[(col + 0.5, actual - 0.5)],
                                         f"{int(cells['count'].sum())}\n{float(cells.actual_row_share.sum()):.1%}")
            pd.testing.assert_frame_equal(source, original)
        finally:
            atlas.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
