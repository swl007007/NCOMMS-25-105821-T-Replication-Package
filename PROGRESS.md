# Progress

Plan: `docs/2026-09-01-strict-temporal-new-area-fivefold-implementation-plan.md`

- [x] Preflight: isolated worktree confirmed; existing 10% test passes in Windows Python 3.11.3 / XGBoost 2.0.3.
- [x] Task 1: strict-temporal folds and OOF core (five contract tests).
- [x] Task 2: summary, artifacts, and audit (post-write tamper check included).
- [x] Task 3: controlled run and independent verification (1,170 OOF rows/model; protected 10% hashes unchanged).

Ruling: Use the dependency-pinned Windows interpreter for tests because WSL Python lacks `xgboost`; cost if wrong: platform-specific behavior would be missed.

Ruling: Run the five folds sequentially and omit a worker-control surface; the required 60 fits remain deterministic and no performance requirement justifies process-pool complexity.

Ruling: Bootstrap the script directory in `sys.path` for direct Windows execution from WSL; cost if wrong: the entrypoint cannot import its sibling package modules.

Evidence: real run exited 0 in Python 3.11.3 / XGBoost 2.0.3; an independent CSV-only verifier recomputed phases, fold metrics, pooled metrics, artifact hashes, and protected hashes.

Known unrelated platform result: two `test_generate_leave_one_country_out_robustness.py` dtype-only assertions fail in Windows because `numpy.dtype(int)` is `int32` while their fixtures assert `int64`; no shared helper was changed.
