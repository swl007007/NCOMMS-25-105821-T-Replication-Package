# Predicting Acute Food Insecurity with Machine Learning

Standalone replication package for the accepted Nature Communications article.
Includes four main and six supplementary figures, seven tables, their scripts,
model-ready inputs and saved predictions. Historical archives and development
files are excluded.

## Reproduce figures and tables

Use Python 3.12 in a fresh environment. From this folder:

```bash
python -m venv .venv
# Activate .venv (Windows: .venv\Scripts\activate; Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements-render.txt
python "2.Source Code/export_final_artwork.py"
python "2.Source Code/export_final_tables.py"
```

- `writing/final_artwork/`: ten complete figures in PDF/SVG, 300-dpi PNG previews,
  captions and figure-specific provenance. Use the PDFs for publication.
- `tables/`: seven CSV tables; see its README for column definitions.
- `Source Data.zip`: one data file for each of the 17 main/SI figures and tables.
- `1.Source Data/`: required analysis inputs.
- `2.Source Code/produced_graph/`: saved numeric results and dependencies.

These commands render saved results and rebuild Source Data without model fitting.
The map geometry is bundled; no tile service or data download is needed.
Install Poppler (`pdfinfo`, `pdffonts`, `pdfimages`) to additionally run
`python "2.Source Code/verify_final_artwork.py"`.

See [REPRODUCE.md](REPRODUCE.md) for the output-to-script map and optional model
refitting in the frozen Windows environment. `release_manifest.json` records
every packaged file's SHA-256. Preserve file bytes and relative paths.
