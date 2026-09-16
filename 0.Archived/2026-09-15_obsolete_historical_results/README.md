# Obsolete historical results — 15 September 2026

The author reviewed the historical files identified by `RESULT_SCRIPT_AUDIT.md`
and confirmed that they are obsolete. Archive them unless active scripts
explicitly consume them.

Thirteen files were moved here, preserving their package-relative paths:
the twelve previously unresolved derived files plus `1.Source Data/r2_frame_cv.csv`.
Their bytes were not changed. No scientific models were rerun.

Two files remain at their original locations because active consumers require them:

- `1.Source Data/r2_frame_forecasting.csv`
- `1.Source Data/r2_frame_nowcasting.csv`

Consumers include `Figure1_multiple_figures.ipynb` (zero-based cell 1,
source lines 15–16), `generate_phase_cumulative_scatter_comparison.py`
(default anchors at lines 43–44, read at line 1347), `run_replication.py`
(required paths at lines 34–35), and calibration integrity checks.
The notebooks' same-name exports under `produced_graph/` do not replace these inputs.

`inventory.csv` records all 15 decisions, original/current paths and SHA-256
hashes. `consumer_check.json` preserves the static search scope and reader
evidence. Documentation references, output-only writes, generic CLI file
arguments and an Excel sheet name were not counted as active input dependencies.
Older missingness artifacts already in their historical archive were left there.

To restore an archived file, verify its SHA-256 and copy `current_path` back
to `original_path` from the package root. Resolve any existing destination
before copying; keep this archive and manifest as the historical record.
