"""Compatibility entry point for the current 1,170-row prediction evaluator.

The canonical evaluator is ``generate_all_prediction_temporal_test_evaluation.py``.
This legacy filename is retained only so existing commands use the same 1,170-row
input contract and output names instead of failing on the former 1,165-row check.
"""

from generate_all_prediction_temporal_test_evaluation import *  # noqa: F401,F403
from generate_all_prediction_temporal_test_evaluation import main


if __name__ == "__main__":
    main()
