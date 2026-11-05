"""Build the curated v1.4.0 tree and ZIP from the reviewed explicit file list."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = "NCOMMS-replication-v1.4.0"
OUT = ROOT / "release_build" / NAME


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Refuse to overwrite a previous release or a user's regeneration outputs.
    OUT.mkdir(parents=True, exist_ok=False)
    files = {}
    for line in (ROOT / "release" / "files.tsv").read_text().splitlines():
        source, target = line.split("\t")
        assert not Path(target).is_absolute() and ".." not in Path(target).parts
        assert target not in files, target
        assert not (ROOT / source).is_symlink(), source
        path = OUT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, path)
        assert sha256(path) == sha256(ROOT / source), source
        files[target] = dict(sha256=sha256(path), bytes=path.stat().st_size, source=source)
    assert len(list((OUT / "writing/final_artwork").glob("*.pdf"))) == 10
    assert len(list((OUT / "tables").glob("*.csv"))) == 7
    assert not any("0.Archived" in p or ".ipynb_checkpoints" in p or p.endswith(".log") for p in files)
    manifest = dict(version="v1.4.0", source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), files=files)
    (OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive_path = OUT.parent / (NAME + ".zip")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                entry = zipfile.ZipInfo(NAME + "/" + path.relative_to(OUT).as_posix(), (2026, 11, 5, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, path.read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name, metadata in files.items():
            assert hashlib.sha256(archive.read(NAME + "/" + name)).hexdigest() == metadata["sha256"]
    checksum = f"{sha256(archive_path)}  {archive_path.name}\n"
    (OUT.parent / "SHA256SUMS.txt").write_text(checksum, encoding="utf-8")
    print(f"Verified {len(files)} files; ZIP {archive_path.stat().st_size:,} bytes\n{checksum}")


if __name__ == "__main__":
    main()
