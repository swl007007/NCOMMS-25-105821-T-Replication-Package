# Strict-Temporal New-Area Five-Fold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible strict-temporal, area-disjoint five-fold OOF robustness generator for Forecasting and cascading two-layer Nowcasting.

**Architecture:** Add one sibling generator. It builds the 646-area map from canonical 2022 keys, creates one complete train/test frame per held fold, and calls the existing split helpers. It writes only the six approved CSV files and independently reloads them for validation.

**Tech Stack:** Python, pandas, NumPy, scikit-learn `KFold`, XGBoost, unittest.

**Spec:** `docs/2026-09-01-strict-temporal-new-area-fivefold-design.md`

## Global Constraints

- Keep `leave_area_out_10pct_*` artifacts byte-identical.
- Fold sorted canonical 646 test areas with `KFold(n_splits=5, shuffle=True, random_state=0)`.
- For fold `k`, train only pre-2022 rows from non-held areas and test only 2022 rows from held areas.
- Reuse the existing Forecasting and two-layer Nowcasting split helpers; retain `fews_ipc_ha`, in-sample Layer-1 residuals, and the existing cumulative phase conversion.
- Use XGBoost `random_state=0` and `n_jobs=1`; add no dependency, figure, checkpoint/resume path, or notebook edit.
- Main metrics are pooled 1,170-row OOF metrics; fold mean and sample SD are supplemental.
- Do not commit or push unless the user asks.

---

### Task 1: Five-fold strict-temporal split and OOF core

**Files:**
- Create: `2.Source Code/generate_leave_area_out_20pct_fivefold_robustness.py`
- Create: `tests/test_generate_leave_area_out_20pct_fivefold_robustness.py`

**Interfaces:**
- Consumes: canonical keys from `All_prediction.csv`; prepared source tables via `loco.prepare_model_inputs`; `loco.fit_forecasting_split`; `loco.fit_nowcasting_split`.
- Produces: `build_area_folds(canonical_test: pd.DataFrame) -> pd.DataFrame`, `strict_temporal_area_masks(data: pd.DataFrame, held_areas: set[int], cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, pd.Series]`, and `run_fivefold_predictions(forecasting: pd.DataFrame, nowcasting: pd.DataFrame, canonical_test: pd.DataFrame, general_params: dict[str, object], phase3_params: dict[str, object], forecasting_runner: Callable[..., pd.DataFrame] = loco.fit_forecasting_split, nowcasting_runner: Callable[..., pd.DataFrame] = loco.fit_nowcasting_split, cutoff: pd.Timestamp = CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`.

- [x] **Step 1: Write failing split tests**

The test fixture has three areas, a 2021 history row and a 2022 row per area. The canonical test table contains only the three 2022 keys. Assert these concrete behaviors:

```python
def test_build_area_folds_assigns_each_area_to_one_of_five_folds():
    canonical = pd.DataFrame({"area_id": range(1, 647), "date": ["2022-01-01"] * 646})
    folds = fivefold.build_area_folds(canonical)
    self.assertEqual(len(folds), 646)
    self.assertEqual(folds["area_id"].nunique(), 646)
    self.assertEqual(set(folds["fold_id"]), set(range(5)))
    self.assertTrue(folds.groupby("area_id")["fold_id"].nunique().eq(1).all())

def test_strict_temporal_area_masks_remove_held_history_and_nonheld_2022_rows():
    frame, train, test = fivefold.strict_temporal_area_masks(data, {1}, pd.Timestamp("2022-01-01"))
    self.assertEqual(set(frame.loc[train, "area_id"]), {2, 3})
    self.assertEqual(set(frame.loc[test, "area_id"]), {1})
    self.assertTrue(frame.loc[train, "date"].lt("2022-01-01").all())
    self.assertTrue(frame.loc[test, "date"].ge("2022-01-01").all())
```

- [x] **Step 2: Run the new test module and verify RED**

Run: `python3 -m unittest discover -s tests -p 'test_generate_leave_area_out_20pct_fivefold_robustness.py' -v`

Expected: import failure because the new generator module does not exist.

- [x] **Step 3: Implement only the split helpers needed by the tests**

```python
def build_area_folds(canonical_test: pd.DataFrame) -> pd.DataFrame:
    areas = np.sort(canonical_test["area_id"].unique())
    fold_ids = np.empty(len(areas), dtype=int)
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    for fold_id, (_, held_index) in enumerate(splitter.split(areas)):
        fold_ids[held_index] = fold_id
    return pd.DataFrame({"area_id": areas, "fold_id": fold_ids})

def strict_temporal_area_masks(data: pd.DataFrame, held_areas: set[int], cutoff: pd.Timestamp):
    train_source = data["date"].lt(cutoff) & ~data["area_id"].isin(held_areas)
    test_source = data["date"].ge(cutoff) & data["area_id"].isin(held_areas)
    frame = data.loc[train_source | test_source].copy().reset_index(drop=True)
    return frame, frame["date"].lt(cutoff), frame["date"].ge(cutoff)
```

- [x] **Step 4: Run split tests and verify GREEN**

Run: `python3 -m unittest discover -s tests -p 'test_generate_leave_area_out_20pct_fivefold_robustness.py' -v`

Expected: both tests pass.

- [x] **Step 5: Write a failing OOF/no-leak test using existing perfect runners**

Use the existing 10% test module's `perfect_forecasting_runner` and `perfect_nowcasting_runner` pattern. For a ten-area fixture with one 2021 and one 2022 row per area, assert:

```python
forecast, nowcast, fold_metrics = fivefold.run_fivefold_predictions(
    forecasting, nowcasting, canonical_test, general_params, phase3_params,
    forecasting_runner=perfect_forecasting_runner,
    nowcasting_runner=perfect_nowcasting_runner,
)
self.assertEqual(len(forecast), 10)
self.assertFalse(forecast.duplicated(["area_id", "date"]).any())
self.assertEqual(set(forecast["fold_id"]), set(range(5)))
self.assertEqual(set(map(tuple, forecast[["area_id", "date"]].to_numpy())), canonical_keys)
self.assertEqual(len(fold_metrics), 10)
self.assertTrue(fold_metrics["train_excludes_held_areas"].all())
```

- [x] **Step 6: Run the OOF test and verify RED**

Run: `python3 -m unittest discover -s tests -p 'test_generate_leave_area_out_20pct_fivefold_robustness.py' -v`

Expected: attribute failure because `run_fivefold_predictions` does not exist.

- [x] **Step 7: Implement sequential fold fitting with existing helpers**

For each fold, build complete local Forecasting and Nowcasting frames with `strict_temporal_area_masks`; assert their train and test `(area_id, date)` sets are equal. Invoke the existing helpers with copied parameter dictionaries containing `random_state=0`, `n_jobs=1`, and `fold_column="fold_id"`. Concatenate predictions and reject duplicate or missing canonical keys. For each model/fold, calculate `area_holdout.calculate_pooled_metrics(...)` and append `fold_id`, `n_train`, `n_train_areas`, `n_test_areas`, and `train_excludes_held_areas`.

- [x] **Step 8: Run core and adjacent 10% tests**

Run: `python3 -m unittest discover -s tests -p 'test_generate_leave_area_out_20pct_fivefold_robustness.py' -v`

Run: `python3 -m unittest discover -s tests -p 'test_generate_leave_area_out_10pct_robustness.py' -v`

Expected: both modules pass.

### Task 2: Metrics summary, approved artifacts, and source audit

**Files:**
- Modify: `2.Source Code/generate_leave_area_out_20pct_fivefold_robustness.py`
- Modify: `tests/test_generate_leave_area_out_20pct_fivefold_robustness.py`

**Interfaces:**
- Consumes: five-fold OOF predictions and per-fold metric rows from Task 1.
- Produces: `summarize_metrics(fold_metrics: pd.DataFrame, forecast: pd.DataFrame, nowcast: pd.DataFrame) -> pd.DataFrame`, `run_analysis(forecasting_path: Path = DEFAULT_FORECASTING_INPUT, nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT, canonical_test_path: Path = DEFAULT_CANONICAL_TEST, country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP, general_params_path: Path = DEFAULT_GENERAL_PARAMS, phase3_params_path: Path = DEFAULT_PHASE3_PARAMS, output_dir: Path = DEFAULT_OUTPUT_DIR, forecasting_runner: Callable[..., pd.DataFrame] = loco.fit_forecasting_split, nowcasting_runner: Callable[..., pd.DataFrame] = loco.fit_nowcasting_split) -> dict[str, Path]`, and the six approved CSVs.

- [x] **Step 1: Write failing summary and artifact tests**

The fixture writes small Forecasting, Nowcasting, lookup, parameter JSON, and canonical-test CSVs to `TemporaryDirectory`. It calls `run_analysis` with perfect runners and asserts:

```python
artifacts = fivefold.run_analysis(
    forecasting_path=forecasting_path,
    nowcasting_path=nowcasting_path,
    canonical_test_path=canonical_path,
    country_lookup_path=lookup_path,
    general_params_path=general_path,
    phase3_params_path=phase3_path,
    output_dir=output_dir,
    forecasting_runner=perfect_forecasting_runner,
    nowcasting_runner=perfect_nowcasting_runner,
)
self.assertEqual(set(artifacts), {"folds", "forecasting_predictions", "nowcasting_predictions", "fold_metrics", "metrics", "source_audit"})
self.assertTrue(all(path.is_file() for path in artifacts.values()))
summary = pd.read_csv(artifacts["metrics"])
self.assertTrue(summary["aggregation"].eq("pooled_oof").all())
self.assertEqual(summary["n_test"].tolist(), [10, 10])
self.assertIn("fold_mean_overall_accuracy", summary.columns)
self.assertIn("fold_sd_overall_accuracy", summary.columns)
```

- [x] **Step 2: Run summary/artifact tests and verify RED**

Run: `python3 -m unittest discover -s tests -p 'test_generate_leave_area_out_20pct_fivefold_robustness.py' -v`

Expected: attribute failure because `run_analysis` and `summarize_metrics` do not exist.

- [x] **Step 3: Implement the minimum summary and artifact family**

```python
def summarize_metrics(fold_metrics, forecast, nowcast):
    rows = []
    for model, predictions in (("Forecasting", forecast), ("Nowcasting", nowcast)):
        pooled = area_holdout.calculate_pooled_metrics(predictions, model)
        model_folds = fold_metrics.loc[fold_metrics["model"].eq(model)]
        row = {**pooled, "aggregation": "pooled_oof"}
        for metric in ("phase3plus_precision", "phase3plus_recall", "overall_accuracy", "phase3plus_r2"):
            row[f"fold_mean_{metric}"] = model_folds[metric].mean()
            row[f"fold_sd_{metric}"] = model_folds[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)
```

`run_analysis` loads the three source CSVs, verifies canonical key equality, prepares inputs, calls the Task 1 runner, writes exactly the six named CSVs, and writes the source audit last. The audit hashes the fold map, two prediction files, fold metrics, and summary metrics; it need not hash itself.

- [x] **Step 4: Add the smallest post-write validation**

Implement `_validate_saved_artifacts(paths, canonical_test)` that reloads the two prediction files and both metric files, recomputes `summarize_metrics`, verifies their canonical keys/fold IDs, and raises on a changed value. Before calling `run_analysis`, record SHA-256 values of existing `leave_area_out_10pct_*` files; compare after validation and raise if any value changed.

- [x] **Step 5: Run new and adjacent test modules and verify GREEN**

Run: `python3 -m unittest discover -s tests -p 'test_generate_leave_area_out_20pct_fivefold_robustness.py' -v`

Run: `python3 -m unittest discover -s tests -p 'test_generate_leave_area_out_10pct_robustness.py' -v`

Expected: both modules pass.

### Task 3: Controlled run and evidence handoff

**Files:**
- Create: the six approved `2.Source Code/produced_graph/leave_area_out_20pct_fivefold_*` CSVs
- Modify: `docs/2026-09-01-strict-temporal-new-area-fivefold-implementation-plan.md` (check completed steps)

**Interfaces:**
- Consumes: passing generator/tests and the frozen Windows-compatible environment.
- Produces: completed reproducibility artifacts and execution evidence.

- [x] **Step 1: Run structural validation in the selected Windows/XGBoost environment**

Run: `C:\\Users\\swl00\\AppData\\Local\\NCOMMSFigure1\\venvs\\xgb203-generators\\Scripts\\python.exe -m py_compile "2.Source Code\\generate_leave_area_out_20pct_fivefold_robustness.py" "tests\\test_generate_leave_area_out_20pct_fivefold_robustness.py"`

Run: `C:\\Users\\swl00\\AppData\\Local\\NCOMMSFigure1\\venvs\\xgb203-generators\\Scripts\\python.exe -m unittest discover -s tests -p test_generate_leave_area_out_20pct_fivefold_robustness.py -v`

Expected: syntax and synthetic-contract tests pass.

- [x] **Step 2: Run the 60-fit generator only in the selected Windows/XGBoost 2.0.3 environment**

Run: `C:\\Users\\swl00\\AppData\\Local\\NCOMMSFigure1\\venvs\\xgb203-generators\\Scripts\\python.exe "2.Source Code\\generate_leave_area_out_20pct_fivefold_robustness.py"`

Expected: six new CSVs, exactly 1,170 OOF predictions per model, and no overwrite of 10% artifacts.

- [x] **Step 3: Independently verify generated CSVs and protected hashes**

Run a read-only verifier that reloads all six artifacts, checks the spec acceptance rules, and compares pre/post hashes for existing `leave_area_out_10pct_*` artifacts.

- [x] **Step 4: Mark completed checkboxes and report evidence**

Record commands, row/key counts, metrics, source-audit hashes, environment, and any platform limitation. Do not commit or push unless the user asks.

## Execution evidence

- Ran on Windows Python 3.11.3, pandas 2.2.3, scikit-learn 1.5.2, and XGBoost
  2.0.3. The generator exited 0 after its 20 Forecasting and 40 Nowcasting fits.
- Canonical population: 1,170 unique keys, 646 areas, key SHA-256
  `288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2`;
  the fold counts are 130, 129, 129, 129, and 129.
- Pooled OOF metrics: Forecasting precision/recall/accuracy/R2(p3) =
  0.7980/0.9078/0.6487/0.1646; Nowcasting =
  0.8022/0.9135/0.6556/0.1696. The source audit records every input,
  parameter, fold, script, and output hash.
- An independent CSV-only verifier recomputed OOF keys, phase reconstruction,
  every fold metric, pooled metrics, and all audit hashes. The five protected
  `leave_area_out_10pct_*` SHA-256 values were unchanged.
- The new and adjacent 10% tests pass. The strict-temporal LOCO tests pass.
  Two unrelated LOCO tests remain Windows-only dtype assertion failures:
  `numpy.dtype(int)` is `int32` there, while their fixtures require `int64`.
  No shared helper was changed. No commit or push was made.
