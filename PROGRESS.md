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

## 2026-11-04 — Final artwork for the accepted article

Scope authorized by the authors: regenerate all main/SI artwork, define
shadings/symbols/colours/abbreviations, collect replacement files in one folder,
and push the changes. Latest editable manuscript source remains with the authors.

- [x] Export four main and six supplementary figures to `writing/final_artwork/`
  as complete PDF/SVG figures with 300-dpi PNG previews, 180-mm width and 5–7-pt
  sans-serif text. All chart, map and flowchart elements are vector objects.
- [x] Restore SHAP numeric exports using the original notebook computations in
  Windows Python 3.11.3 / XGBoost 2.0.3, adding SHAP 0.46.0 and compatible
  dependencies without changing the frozen model stack. Preserve original
  feature ordering, hyperparameter carryover, signed-sum aggregation and groups.
  All four top-ten rankings agree with accepted artwork; old numeric SHAP arrays
  are unavailable for byte-for-byte historical comparison. A second fresh export
  reproduced all six CSVs and the provenance JSON byte-for-byte.
- [x] Preserve the separate Figure 1 and supplementary-map prediction lineages,
  all evaluation populations and saved numerical results. Bundle the map's 646
  full-resolution analysis polygons and a public-domain Natural Earth basemap.
- [x] Redraw both flowcharts, including the correct 2022 final holdout. Add
  internal colour/symbol keys, confusion-matrix shading scales and black-frame
  definitions. Supply ten English captions in Markdown and LaTeX; the authors
  will apply these to the Overleaf source.
- [x] Add Figure 2 numerical contributions to `Source Data.zip` and rebuild its
  17 figure/table files with verified source/output hashes.
- [x] Run numerical population/metric checks and export checks for page size,
  font size, embedded fonts, editable SVG text, absent raster objects, text
  within the page and the complete caption inventory. Evidence is recorded in
  `artwork_audit.json` and `file_verification.json` in the artwork folder.
- [x] Independently inspect all ten previews/captions and recompute Figure 1,
  map, SHAP aggregation and supplementary matrix/scatter checks. No numerical
  or population error found; add the requested map boundary/water key.
- [ ] Commit and verify remote push.

Reproduction: `export_final_artwork_shap.py` runs only in the frozen model
environment; `export_final_artwork.py` renders the saved numbers in the recorded
plotting environment; `verify_final_artwork.py` validates the delivered files.
Unrelated pre-existing CSV/log changes are excluded from this delivery.
