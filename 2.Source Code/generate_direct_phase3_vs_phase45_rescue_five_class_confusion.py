from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence


os.environ.setdefault(
    "MPLCONFIGDIR", tempfile.mkdtemp(prefix="ncomms-five-class-confusion-mpl-")
)

import matplotlib


matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib import colors as mpl_colors
from matplotlib import patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCED_GRAPH_DIR = Path(__file__).resolve().parent / "produced_graph"
FORMAL_RESCUE_DIR = PRODUCED_GRAPH_DIR / "direct_phase3_vs_phase45_rescue"
SOURCE_BASENAME = "direct_phase3_vs_phase45_rescue_five_class_confusion_matrices.csv"
CONFIGURATION_BASENAME = "direct_phase3_vs_phase45_rescue_configuration.json"
DEFAULT_INPUT_CSV = FORMAL_RESCUE_DIR / SOURCE_BASENAME
DEFAULT_CONFIGURATION = FORMAL_RESCUE_DIR / CONFIGURATION_BASENAME
CONTEMPORANEOUS_RESCUE_DIR = (
    PRODUCED_GRAPH_DIR / "direct_phase3_vs_phase45_rescue_contemporaneous"
)
CONTEMPORANEOUS_PREFIX = "direct_phase3_vs_phase45_rescue_contemporaneous_"
DEFAULT_CONTEMPORANEOUS_INPUT_CSV = (
    CONTEMPORANEOUS_RESCUE_DIR
    / f"{CONTEMPORANEOUS_PREFIX}five_class_confusion_matrices.csv"
)
DEFAULT_CONTEMPORANEOUS_CONFIGURATION = (
    CONTEMPORANEOUS_RESCUE_DIR / f"{CONTEMPORANEOUS_PREFIX}configuration.json"
)
DEFAULT_OUTPUT_DIR = (
    PRODUCED_GRAPH_DIR / "direct_phase3_vs_phase45_rescue_five_class_confusion"
)

OUTPUT_PREFIX = "direct_phase3_vs_phase45_rescue_five_class_confusion_atlas"
OUTPUT_PDF = f"{OUTPUT_PREFIX}.pdf"
OUTPUT_PNG = f"{OUTPUT_PREFIX}.png"
EXPECTED_OUTPUTS = (OUTPUT_PDF, OUTPUT_PNG)

TEMPORAL_TASK_ORDER = ("Forecasting", "Nowcasting")
TASK_ORDER = (*TEMPORAL_TASK_ORDER, "Contemporaneous")
TEMPORAL_SOURCE_METHOD_ORDER = (
    "frozen_base",
    "legacy_direct_exact_phase4_050",
    "direct_phase45_unweighted",
    "direct_phase45_sqrt_balance",
    "direct_phase45_full_balance",
)
METHOD_ORDER = (
    "direct_phase45_unweighted",
    "direct_phase45_sqrt_balance",
    "direct_phase45_full_balance",
)
METHOD_DISPLAY_NAMES = {
    "direct_phase45_unweighted": "P4/5 unweighted",
    "direct_phase45_sqrt_balance": "P4/5 sqrt-balance",
    "direct_phase45_full_balance": "P4/5 full-balance",
}
METHOD_COLORS = {
    "direct_phase45_unweighted": "#61DDAA",
    "direct_phase45_sqrt_balance": "#D89C00",
    "direct_phase45_full_balance": "#E8684A",
}
PHASES = (1, 2, 3, 4, 5)
SOURCE_COLUMNS = (
    "task",
    "method",
    "actual_phase",
    "predicted_phase",
    "count",
    "actual_row_total",
    "actual_row_share",
)
TASK_POPULATIONS = {
    "Forecasting": (1170, "fixed 2022 temporal holdout"),
    "Nowcasting": (1170, "fixed 2022 temporal holdout"),
    "Contemporaneous": (5575, "seed-0 random five-fold row-CV full OOF"),
}
FIGURE_SIZE = (6.5, 10.5)
PNG_DPI = 600
PNG_SIZE = (3900, 6300)

FIGURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.7,
    "savefig.bbox": None,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generation_target(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Five-class confusion output must be absent or empty: {output_dir}"
        )


def load_and_validate_source(
    input_csv: Path = DEFAULT_INPUT_CSV,
    configuration_path: Path = DEFAULT_CONFIGURATION,
    contemporaneous_input_csv: Path = DEFAULT_CONTEMPORANEOUS_INPUT_CSV,
    contemporaneous_configuration_path: Path = (
        DEFAULT_CONTEMPORANEOUS_CONFIGURATION
    ),
) -> tuple[pd.DataFrame, dict[str, str]]:
    input_csv = Path(input_csv)
    configuration_path = Path(configuration_path)
    contemporaneous_input_csv = Path(contemporaneous_input_csv)
    contemporaneous_configuration_path = Path(
        contemporaneous_configuration_path
    )
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    expected_hash = configuration.get("artifact_hashes", {}).get(input_csv.name)
    if expected_hash is None or file_sha256(input_csv) != expected_hash:
        raise ValueError("Five-class confusion source is not bound to the formal manifest.")

    temporal = pd.read_csv(
        input_csv,
        na_values=["<NA>"],
        keep_default_na=True,
        float_precision="round_trip",
    )
    if tuple(temporal.columns) != SOURCE_COLUMNS:
        raise ValueError(f"Unexpected temporal source schema: {tuple(temporal.columns)}")
    if len(temporal) != 250 or temporal.duplicated(
        ["task", "method", "actual_phase", "predicted_phase"]
    ).any():
        raise ValueError("Temporal source violates the 250-cell formal contract.")
    if set(temporal["task"]) != set(TEMPORAL_TASK_ORDER) or set(
        temporal["method"]
    ) != set(
        TEMPORAL_SOURCE_METHOD_ORDER
    ):
        raise ValueError("Temporal task or method set drifted.")

    contemporaneous_configuration = json.loads(
        contemporaneous_configuration_path.read_text(encoding="utf-8")
    )
    contemporaneous_expected_hash = contemporaneous_configuration.get(
        "artifact_hashes", {}
    ).get(contemporaneous_input_csv.name)
    if (
        contemporaneous_expected_hash is None
        or file_sha256(contemporaneous_input_csv)
        != contemporaneous_expected_hash
    ):
        raise ValueError(
            "Contemporaneous five-class confusion source is not bound to its manifest."
        )
    decisions = contemporaneous_configuration.get("decisions", {})
    if (
        decisions.get("evaluation_protocol") != "random_5fold_row_cv"
        or decisions.get("evaluation_population")
        != "random_5fold_full_oof_5575"
        or decisions.get("direct_comparison_with_temporal_holdout_authorized")
        is not False
    ):
        raise ValueError("Contemporaneous evaluation-protocol contract drifted.")
    contemporaneous = pd.read_csv(
        contemporaneous_input_csv,
        na_values=["<NA>"],
        keep_default_na=True,
        float_precision="round_trip",
    )
    if tuple(contemporaneous.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"Unexpected Contemporaneous source schema: {tuple(contemporaneous.columns)}"
        )
    if (
        len(contemporaneous) != 75
        or set(contemporaneous["task"]) != {"Contemporaneous"}
        or set(contemporaneous["method"]) != set(METHOD_ORDER)
        or contemporaneous.duplicated(
            ["task", "method", "actual_phase", "predicted_phase"]
        ).any()
    ):
        raise ValueError("Contemporaneous source violates the 75-cell contract.")

    frame = pd.concat(
        [
            temporal.loc[
                temporal["task"].isin(TEMPORAL_TASK_ORDER)
                & temporal["method"].isin(METHOD_ORDER)
            ],
            contemporaneous,
        ],
        ignore_index=True,
    )
    if len(frame) != 225 or frame.duplicated(
        ["task", "method", "actual_phase", "predicted_phase"]
    ).any():
        raise ValueError("Combined source violates the 225-cell display contract.")
    if set(frame["task"]) != set(TASK_ORDER) or set(frame["method"]) != set(
        METHOD_ORDER
    ):
        raise ValueError("Combined task or method set drifted.")
    if set(frame["actual_phase"]) != set(PHASES) or set(
        frame["predicted_phase"]
    ) != set(PHASES):
        raise ValueError("Actual or predicted phase support is incomplete.")

    for (task, method), group in frame.groupby(
        ["task", "method"], sort=True, observed=True
    ):
        expected_rows = TASK_POPULATIONS[str(task)][0]
        if len(group) != 25 or int(group["count"].sum()) != expected_rows:
            raise ValueError(
                f"{task}/{method} does not contain a complete {expected_rows:,}-row matrix."
            )
        for actual_phase, row in group.groupby("actual_phase", sort=True):
            totals = set(row["actual_row_total"].astype(int))
            if len(totals) != 1:
                raise ValueError(
                    f"{task}/{method}/P{actual_phase} row-total metadata drifted."
                )
            row_total = totals.pop()
            if int(row["count"].sum()) != row_total:
                raise ValueError(
                    f"{task}/{method}/P{actual_phase} row counts do not sum."
                )
            expected_share = row["count"].to_numpy(dtype=float) / row_total
            if not np.allclose(
                row["actual_row_share"].to_numpy(dtype=float),
                expected_share,
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError(
                    f"{task}/{method}/P{actual_phase} row shares do not reconstruct."
                )

    primary_methods: dict[str, str] = {}
    for task, source_configuration in (
        ("Forecasting", configuration),
        ("Nowcasting", configuration),
        ("Contemporaneous", contemporaneous_configuration),
    ):
        records = source_configuration.get("candidates", {}).get(
            "selected_policies", []
        )
        selected = [
            str(record["method"])
            for record in records
            if str(record.get("task")) == task
            and bool(record.get("primary_selected"))
            and str(record.get("method")) in METHOD_ORDER
        ]
        if len(selected) != 1:
            raise ValueError(f"{task} requires exactly one primary rescue policy.")
        primary_methods[task] = selected[0]
    return frame, primary_methods


def _text_color(facecolor: tuple[float, float, float, float]) -> str:
    red, green, blue = facecolor[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111111" if luminance > 0.56 else "#FFFFFF"


def render_figure(
    source: pd.DataFrame, primary_methods: Mapping[str, str]
) -> mpl.figure.Figure:
    with mpl.rc_context(FIGURE_RC):
        figure = plt.figure(figsize=FIGURE_SIZE)
        grid = figure.add_gridspec(
            3,
            2,
            left=0.20,
            right=0.88,
            bottom=0.18,
            top=0.89,
            wspace=0.12,
            hspace=0.38,
            width_ratios=(1.0, 0.045),
        )
        colorbar_axis = figure.add_subplot(grid[:, 1])

        normalization = mpl_colors.Normalize(vmin=0.0, vmax=1.0)
        colormap = mpl.colormaps["cividis"]
        row_protocol_labels = {
            "Forecasting": "fixed 2022 temporal holdout",
            "Nowcasting": "fixed 2022 temporal holdout",
            "Contemporaneous": "seed-0 random five-fold full OOF",
        }
        for row_index, task in enumerate(TASK_ORDER):
            method = primary_methods[task]
            axis = figure.add_subplot(grid[row_index, 0])
            matrix = source.loc[
                source["task"].eq(task) & source["method"].eq(method)
            ].set_index(["actual_phase", "predicted_phase"])
            if len(matrix) != 25:
                raise ValueError(f"{task}/{method} matrix is incomplete.")

            for actual_phase in PHASES:
                for predicted_phase in PHASES:
                    cell = matrix.loc[(actual_phase, predicted_phase)]
                    share = float(cell["actual_row_share"])
                    facecolor = colormap(normalization(share))
                    axis.add_patch(
                        mpl_patches.Rectangle(
                            (predicted_phase - 1, actual_phase - 1),
                            1,
                            1,
                            facecolor=facecolor,
                            edgecolor="white",
                            linewidth=0.75,
                        )
                    )
                    axis.text(
                        predicted_phase - 0.5,
                        actual_phase - 0.5,
                        f"{int(cell['count'])}\n{share:.1%}",
                        ha="center",
                        va="center",
                        fontsize=5.2,
                        color=_text_color(facecolor),
                    )

            axis.set_xlim(0, 5)
            axis.set_ylim(5, 0)
            axis.set_aspect("equal")
            axis.set_xticks(
                np.arange(0.5, 5.0, 1.0), [f"P{phase}" for phase in PHASES]
            )
            axis.set_yticks(
                np.arange(0.5, 5.0, 1.0),
                [f"P{phase}" for phase in PHASES],
            )
            axis.tick_params(length=0, pad=1.8)

            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(METHOD_COLORS[method])
                spine.set_linewidth(1.35)

            support, _ = TASK_POPULATIONS[task]
            axis.text(
                -0.14,
                1.14,
                chr(ord("a") + row_index),
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
            axis.text(
                0.5,
                1.14,
                task,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )
            axis.text(
                0.5,
                1.085,
                f"{METHOD_DISPLAY_NAMES[method]} (primary)",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=METHOD_COLORS[method],
            )
            axis.text(
                0.5,
                1.035,
                f"{row_protocol_labels[task]}; n = {support:,}",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.4,
                color="#444444",
            )

        figure.supxlabel("Predicted IPC phase", x=0.505, y=0.125, fontsize=8)
        figure.supylabel("Actual IPC phase", x=0.10, y=0.535, fontsize=8)

        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(norm=normalization, cmap=colormap),
            cax=colorbar_axis,
        )
        colorbar.set_label("Share within actual class", fontsize=7)
        colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        colorbar.ax.tick_params(labelsize=6, length=2)

        figure.text(
            0.12,
            0.045,
            (
                "Each panel shows the task-specific OOF-selected primary rescue policy.\n"
                "Cells show count and percentage within each actual phase; rows are "
                "actual phases and columns are predicted phases.\n"
                "Forecasting and Nowcasting: fixed 2022 temporal holdout "
                "(n = 1,170 each).\n"
                "Contemporaneous: seed-0 random row-level five-fold full OOF "
                "(n = 5,575).\n"
                "Its threshold uses pooled OOF selection (not nested/independent).\n"
                "Protocols and populations differ; results are not directly comparable."
            ),
            ha="left",
            va="center",
            fontsize=6.2,
            color="#444444",
        )
        figure.suptitle(
            "Five-class errors for the selected primary Phase-4/5 rescue policies",
            fontsize=11,
            fontweight="bold",
            y=0.975,
        )
        return figure


def save_figure(figure: mpl.figure.Figure, output_dir: Path) -> None:
    pdf_metadata = {
        "Title": "Direct Phase-3 versus Phase-4/5 five-class confusion atlas",
        "Author": "",
        "Subject": (
            "Five-class confusion matrices for the task-specific primary rescue "
            "policies across three validation protocols"
        ),
        "Keywords": "",
        "Creator": "NCOMMS five-class confusion generator",
        "CreationDate": None,
        "ModDate": None,
    }
    png_metadata = {"Software": "NCOMMS five-class confusion generator"}
    with mpl.rc_context(FIGURE_RC):
        figure.savefig(
            output_dir / OUTPUT_PDF,
            format="pdf",
            metadata=pdf_metadata,
            facecolor="white",
        )
        figure.savefig(
            output_dir / OUTPUT_PNG,
            format="png",
            dpi=PNG_DPI,
            metadata=png_metadata,
            facecolor="white",
        )
    plt.close(figure)


def _read_png_contract(path: Path) -> tuple[int, int, float | None]:
    payload = Path(path).read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature.")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    offset = 8
    dpi = None
    while offset + 12 <= len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_data = payload[offset + 8 : offset + 8 + length]
        if chunk_type == b"pHYs" and len(chunk_data) == 9 and chunk_data[8] == 1:
            dpi = int.from_bytes(chunk_data[:4], "big") * 0.0254
            break
        offset += 12 + length
    return width, height, dpi


def validate_outputs(output_dir: Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    directories = [path.name for path in output_dir.iterdir() if path.is_dir()]
    if files != sorted(EXPECTED_OUTPUTS) or directories:
        raise ValueError(f"Output contract failed: files={files}, dirs={directories}")
    pdf_payload = (output_dir / OUTPUT_PDF).read_bytes()
    if (
        not pdf_payload.startswith(b"%PDF")
        or b"CreationDate" in pdf_payload
        or b"ModDate" in pdf_payload
    ):
        raise ValueError("PDF metadata contract failed.")
    width, height, dpi = _read_png_contract(output_dir / OUTPUT_PNG)
    if (
        (width, height) != PNG_SIZE
        or dpi is None
        or abs(dpi - PNG_DPI) > 1.0
    ):
        raise ValueError(
            f"PNG contract failed: width={width}, height={height}, dpi={dpi}"
        )
    return {name: file_sha256(output_dir / name) for name in EXPECTED_OUTPUTS}


def _replace_with_retry(
    source: Path, destination: Path, attempts: int = 20
) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            gc.collect()
            time.sleep(0.25)


def _rmtree_with_retry(path: Path, attempts: int = 40) -> None:
    path = Path(path)
    if not path.exists():
        return
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            gc.collect()
            time.sleep(0.25)


def run_generation(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    input_csv: Path = DEFAULT_INPUT_CSV,
    configuration_path: Path = DEFAULT_CONFIGURATION,
    contemporaneous_input_csv: Path = DEFAULT_CONTEMPORANEOUS_INPUT_CSV,
    contemporaneous_configuration_path: Path = (
        DEFAULT_CONTEMPORANEOUS_CONFIGURATION
    ),
) -> dict[str, object]:
    output_dir = Path(output_dir)
    validate_generation_target(output_dir)
    source, primary_methods = load_and_validate_source(
        input_csv,
        configuration_path,
        contemporaneous_input_csv,
        contemporaneous_configuration_path,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".five-class-confusion-staging-", dir=output_dir.parent)
    )
    published = False
    try:
        save_figure(render_figure(source, primary_methods), staging_dir)
        hashes = validate_outputs(staging_dir)
        if output_dir.exists():
            _rmtree_with_retry(output_dir)
        _replace_with_retry(staging_dir, output_dir)
        published = True
        validate_outputs(output_dir)
        return {
            "output_dir": output_dir,
            "source_csv": Path(input_csv),
            "source_sha256": file_sha256(Path(input_csv)),
            "contemporaneous_source_csv": Path(contemporaneous_input_csv),
            "contemporaneous_source_sha256": file_sha256(
                Path(contemporaneous_input_csv)
            ),
            "primary_methods": primary_methods,
            "output_sha256": hashes,
        }
    except BaseException:
        if staging_dir.exists():
            _rmtree_with_retry(staging_dir)
        if published and output_dir.exists():
            _rmtree_with_retry(output_dir)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the five-class confusion atlas for the isolated Direct "
            "Phase-3 versus Phase-4/5 rescue experiment."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--configuration", type=Path, default=DEFAULT_CONFIGURATION
    )
    parser.add_argument(
        "--contemporaneous-input-csv",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_INPUT_CSV,
    )
    parser.add_argument(
        "--contemporaneous-configuration",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_CONFIGURATION,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    result = run_generation(
        output_dir=arguments.output_dir,
        input_csv=arguments.input_csv,
        configuration_path=arguments.configuration,
        contemporaneous_input_csv=arguments.contemporaneous_input_csv,
        contemporaneous_configuration_path=(
            arguments.contemporaneous_configuration
        ),
    )
    print(
        json.dumps(
            {
                "output_dir": str(result["output_dir"]),
                "source_csv": str(result["source_csv"]),
                "source_sha256": result["source_sha256"],
                "contemporaneous_source_csv": str(
                    result["contemporaneous_source_csv"]
                ),
                "contemporaneous_source_sha256": result[
                    "contemporaneous_source_sha256"
                ],
                "primary_methods": result["primary_methods"],
                "output_sha256": result["output_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
