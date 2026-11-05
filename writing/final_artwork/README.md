# Final artwork

Replace the figures in Overleaf with the **10 PDF files in this folder**. Each
PDF is a complete figure, including its panels and internal keys. The matching
SVG contains editable text; PNG is a 300-dpi preview. All figures are 180 mm
wide, 88–185 mm high, with embedded sans-serif fonts at 5–7 pt. Insert at the
intended two-column width; reducing the width also reduces the text size.

| Accepted figure | Replacement PDF | Previous artwork |
| --- | --- | --- |
| Figure 1 | `Figure_1.pdf` | `accuracy.jpg`, `precision_sensitivity.jpg`, `R2.jpg` |
| Figure 2 | `Figure_2.pdf` | `Picture6.jpg`, `Picture8.jpg`, `Picture7.jpg`, `Picture9.jpg` |
| Figure 3 | `Figure_3.pdf` | `Picture3.jpg` |
| Figure 4 | `Figure_4.pdf` | `Picture1.jpg`, `Picture2.jpg` |
| Supplementary Figures 1–6 | `Supplementary_Figure_1.pdf` … `Supplementary_Figure_6.pdf` | Corresponding figures in the final SI PDF |

Use the replacement captions in **`figure_legends.tex`** (or the readable
`figure_legends.md`). They define the visible colours, shading, symbols and
abbreviations in words and contain the Source Data statement. No coloured
glyphs are needed in captions. These are caption snippets, not edits to the
manuscript source. Existing figure labels and cross-references can be retained.

## Rebuild

From the repository root:

```bash
python "2.Source Code/export_final_artwork.py"
```

The renderer uses saved results and bundled geometry; it does not fit models.
The checked rendering environment is Python 3.12.3, NumPy 2.4.4, pandas 3.0.3,
Matplotlib 3.10.9 and GeoPandas 1.1.3. Exact versions, input/output hashes,
physical dimensions, text bounds and numerical checks are in
`artwork_audit.json`. Run `python "2.Source Code/verify_final_artwork.py"` to
check the saved PDFs and SVGs using Poppler's `pdfinfo`, `pdffonts` and
`pdfimages` tools.

Figure 2's numerical export is a separate step:

```bash
python "2.Source Code/export_final_artwork_shap.py"
```

Run that step only in the frozen Windows Python 3.11.3 / XGBoost 2.0.3 model
environment, with SHAP 0.46.0, numba 0.60.0, llvmlite 0.43.0, slicer 0.0.8,
cloudpickle 3.0.0 and tqdm 4.66.5. It executes the original notebooks' training
and grouping statements without their historical file-writing cells. The
four top-ten rankings agree with the accepted artwork; recovered bar magnitudes
and group shares agree at the resolution visible in those images. No historical
numeric SHAP arrays exist for a byte-for-byte comparison. The recovered
signed sums, top-ten values and group shares are in `data/` and Figure 2 of
`Source Data.zip`; `data/shap_provenance.json` records their sources/environment.
A second run reproduced all six SHAP CSVs and the provenance JSON byte-for-byte.

The map preserves all 646 analysis polygons at their original resolution,
and the original selection rules and alert counts. Its former raster basemap
is replaced with public-domain Natural Earth country geometry. Geometry and
provenance are bundled in `data/`; no online tile service is required. Both
flowcharts are drawn by the renderer, including the final 2022 holdout.

The source/result files used for the other figures are unchanged. In particular,
Figure 1 uses the frozen main-result lineage; the supplementary map retains its
separate `All_prediction.csv` lineage. The package does not rerun LOCO or the
other model experiments.
