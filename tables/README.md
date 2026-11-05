# Tables

Rebuild all seven CSV files with `python "2.Source Code/export_final_tables.py"`.
This also rebuilds `Source Data.zip` using the existing figure/table selections.
CSV values retain source precision; round to three decimals for display.
Sensitivity means recall. Blank values were not reported, not zero.

| File | Content |
| --- | --- |
| `Table_1.csv` | Main model metrics and reported changed-phase summaries |
| `Supplementary_Table_1.csv` | Persistence, Ordered Probit, Ensemble OLS, repeated-area refit, main model |
| `Supplementary_Table_2.csv` | Coordinate removal and spatial feature variants |
| `Supplementary_Table_3.csv` | Country and 20% area holdouts |
| `Supplementary_Table_4.csv` | Full 76-row variable dictionary underlying the manuscript's grouped catalogue |
| `Supplementary_Table_5.csv` | Eight hyperparameter grids (11,664 combinations) |
| `Supplementary_Table_6.csv` | Missing indicators and feature/country removal |

Table 1's changed-phase rows are historical reported summaries: original keyed
predictions were not retained. SI Table 1 panel D uses continuous Phase 3+ share
R-squared. SI Table 3 panel B uses pooled temporal metrics for Forecasting and
Nowcasting and fold means for Contemporaneous (random area CV using the Nowcasting
dataset). SI Table 6 panel H uses country removal at the 10% threshold, following
the existing Source Data selection. SI Table 4 is the complete data dictionary,
not a typeset copy of the grouped manuscript table.

Metric rows identify their original file and CSV line. `table_audit.json` binds
the exports to their Source Data members and generator.
