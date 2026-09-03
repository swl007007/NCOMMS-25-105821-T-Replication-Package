import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_variable_missingness_balance as balance


class VariableMissingnessBalanceTests(unittest.TestCase):
    def test_country_keeps_small_group_but_excludes_it_from_balance_decision(self):
        data = pd.DataFrame(
            {
                "country_code_3": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 2,
                "feature": [np.nan] * 2 + [1.0] * 18 + [np.nan] * 12 + [1.0] * 8 + [np.nan] * 2,
            }
        )

        result = balance.summarize_variable_dimension(
            data, model="Forecasting", variable="feature", dimension="country"
        )

        observed = result.set_index("group")
        self.assertEqual(observed.loc["CCC", "n"], 2)
        self.assertEqual(observed.loc["CCC", "missing_rate"], 1.0)
        self.assertFalse(observed.loc["CCC", "eligible_for_balance"])
        self.assertTrue(observed.loc["AAA", "eligible_for_balance"])
        self.assertTrue(observed.loc["BBB", "eligible_for_balance"])
        self.assertTrue((result["test_population_n"] == 40).all())
        self.assertTrue(result["material_imbalance"].all())

    def test_all_missing_variable_is_reported_but_not_tested(self):
        data = pd.DataFrame(
            {
                "year": [2021, 2021, 2022, 2022],
                "feature": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        result = balance.summarize_variable_dimension(
            data, model="Nowcasting", variable="feature", dimension="year"
        )

        self.assertTrue((result["missing_rate"] == 1.0).all())
        self.assertTrue((result["test_status"] == "untestable_constant_missingness").all())
        self.assertTrue(result["cramers_v"].isna().all())
        self.assertFalse(result["material_imbalance"].any())

    def test_sparse_expected_counts_keep_effect_size_but_suppress_p_value(self):
        data = pd.DataFrame(
            {
                "year": [2021] * 20 + [2022] * 20,
                "feature": [np.nan] + [1.0] * 39,
            }
        )

        result = balance.summarize_variable_dimension(
            data, model="Forecasting", variable="feature", dimension="year"
        )

        self.assertTrue((result["test_status"] == "tested_sparse_expected_counts").all())
        self.assertTrue(result["p_value"].isna().all())
        self.assertTrue(result["cramers_v"].notna().all())

    def test_benjamini_hochberg_preserves_missing_p_values(self):
        adjusted = balance.benjamini_hochberg(pd.Series([0.01, np.nan, 0.04, 0.03]))

        self.assertAlmostEqual(adjusted.iloc[0], 0.03)
        self.assertTrue(math.isnan(adjusted.iloc[1]))
        self.assertAlmostEqual(adjusted.iloc[2], 0.04)
        self.assertAlmostEqual(adjusted.iloc[3], 0.04)


if __name__ == "__main__":
    unittest.main()
