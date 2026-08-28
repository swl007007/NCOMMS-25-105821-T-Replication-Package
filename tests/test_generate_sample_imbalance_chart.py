import sys
import unittest
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_sample_imbalance_chart as imbalance_chart


class AreaFrequencyDistributionTests(unittest.TestCase):
    def test_counts_each_area_once_by_total_frequency_and_phase_diversity(self):
        data = pd.DataFrame(
            {
                "area_id": [1, 1, 2, 2, 2, 3, 4, 4, 4],
                "overall_phase": [2, 2, 2, 3, 3, 4, 1, 2, 3],
            }
        )

        result = imbalance_chart.build_area_frequency_distribution(data)

        observed = {
            (row.observation_frequency, row.distinct_phase_count): row.area_count
            for row in result.itertuples(index=False)
        }
        self.assertEqual(
            observed,
            {
                (1, 1): 1,
                (2, 1): 1,
                (3, 2): 1,
                (3, 3): 1,
            },
        )
        self.assertEqual(result["area_count"].sum(), 4)


if __name__ == "__main__":
    unittest.main()
