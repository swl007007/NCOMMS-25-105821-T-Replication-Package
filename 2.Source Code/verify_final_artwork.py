"""Check the delivered artwork, manifest and legend inventory without refitting."""

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "writing" / "final_artwork"


def output(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main():
    audit = json.loads((OUT / "artwork_audit.json").read_text())
    assert len(audit["figures"]) == 10 and len(audit["outputs"]) == 30
    for name, expected in audit["outputs"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in audit["sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    reports = {}
    for stem, checks in audit["figures"].items():
        pdf = OUT / (stem + ".pdf")
        info = output(["pdfinfo", str(pdf)])
        assert re.search(r"Pages:\s+1\b", info), stem
        match = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
        size_mm = [float(value) * 25.4 / 72 for value in match.groups()]
        assert abs(size_mm[0] - 180) < .01 and size_mm[1] <= 185.01, (stem, size_mm)
        assert all(abs(a-b) < .01 for a,b in zip(size_mm,checks["size_mm"]))
        fonts = output(["pdffonts", str(pdf)])
        font_rows = fonts.splitlines()[2:]
        assert font_rows and "Type 3" not in fonts, stem
        assert all(re.search(r"\byes\s+yes\s+yes\b", line) for line in font_rows), fonts
        images = output(["pdfimages", "-list", str(pdf)]).splitlines()[2:]
        assert not images, (stem, "embedded raster images", images)
        tree = ET.parse(OUT / (stem + ".svg"))
        assert not list(tree.iter("{http://www.w3.org/2000/svg}image")), stem
        texts = list(tree.iter("{http://www.w3.org/2000/svg}text"))
        assert len(texts) >= (5 if stem == "Supplementary_Figure_2" else 10), (stem, "missing editable SVG text")
        assert checks["font_size_pt"][0] >= 5 and checks["font_size_pt"][1] <= 7
        assert not checks["out_of_page_text"] and pdf.stat().st_size < 50_000_000
        with Image.open(OUT / (stem + ".png")) as preview:
            assert preview.mode in ("RGB", "RGBA")
            assert all(abs(dpi - 300) < .1 for dpi in preview.info["dpi"])
            assert all(abs(pixels - mm / 25.4 * 300) <= 1 for pixels,mm in zip(preview.size,size_mm))
        reports[stem] = {"size_mm": size_mm, "embedded_fonts": len(font_rows),
                         "raster_images": 0, "editable_svg_text_elements": len(texts),
                         "pdf_bytes": pdf.stat().st_size}
    md = (OUT / "figure_legends.md").read_text()
    tex = (OUT / "figure_legends.tex").read_text()
    assert md.count("\n## ") == tex.count("\\caption{") == 10
    assert md.count("Source data are provided as a Source Data file.") == 10
    assert tex.count("Source data are provided as a Source Data file.") == 10
    assert "\\textcolor" not in tex and "??" not in tex
    (OUT / "file_verification.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("PASS: 10 single-page vector PDFs; 180 mm wide; 5–7 pt fonts; embedded TrueType fonts;")
    print("no raster images; 10 SVGs with editable text; all input/output hashes; 10 complete captions.")
    for stem, report in reports.items():
        print(f"{stem}: {report['size_mm'][0]:.0f} × {report['size_mm'][1]:.0f} mm, {report['pdf_bytes']:,} bytes")


if __name__ == "__main__":
    main()
