"""Command Line Interface for Cardiac Fat Quantification and Federated Learning.

Provides end-to-end processing, anatomy segmentation, EAT/PVAT mask computation,
CSV export, and federated training routines matching the MGH CardiacFatQuant specification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from cardiac_fat_fl.anatomy_segmenter import UnifiedAnatomySegmenter
from cardiac_fat_fl.fl_client_algo import CardiacFatClientAlgo
from cardiac_fat_fl.fl_stats_executor import CardiacFatStatsExecutor
from cardiac_fat_fl.io_nrrd import (
    discover_participants,
    read_nrrd_as_metatensor,
    write_eat_pvat_seg_nrrd,
)
from cardiac_fat_fl.measurement_schema import (
    DEFAULT_PVAT_RADIAL_DISTANCE_MM,
    MEASUREMENT_CSV_COLUMNS,
    validate_measurement_dataframe,
)
from cardiac_fat_fl.transforms import ComputeEatPvatMasksd, QuantifyAdiposeMetricsd

app = typer.Typer(
    name="cardiac-fat-fl",
    help="Modernized MONAI & NVFLARE pipeline for EAT/PVAT quantification from Cardiac CT.",
    add_completion=False,
)
console = Console()


@app.command("quantify")
def quantify(
    source_dir: Path = typer.Argument(
        ...,
        help="Root directory containing participant subfolders (source_dir/<participant_id>/scan.nrrd).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the aggregate CSV. Defaults to source_dir.",
    ),
    save_seg: bool = typer.Option(
        True,
        "--save-seg/--no-save-seg",
        help="Whether to write `eat_pvat.seg.nrrd` inside each participant directory.",
    ),
    csv_filename: str = typer.Option(
        "cardiac_fat_measurements.csv",
        "--csv-filename",
        "-c",
        help="Filename for the exported measurements CSV.",
    ),
    pvat_radius: float = typer.Option(
        DEFAULT_PVAT_RADIAL_DISTANCE_MM,
        "--pvat-radius",
        "-r",
        help="Radial dilation distance in mm for pericoronary adipose tissue (PVAT).",
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Optional path to PyTorch or ONNX model weights for anatomy segmentation.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Optional maximum number of participants to process (useful for large cohorts).",
    ),
) -> None:
    """Execute end-to-end quantification across participants in source_dir."""
    console.print(f"[bold cyan]Scanning directory:[/bold cyan] {source_dir}")

    participants = discover_participants(source_dir)
    if not participants:
        console.print("[yellow]No participant scans found. Ensure layout is source_dir/<id>/scan.nrrd or contains DICOM series[/yellow]")
        raise typer.Exit(code=1)

    if limit is not None and limit > 0:
        participants = participants[:limit]
        console.print(f"[green]Processing limited subset of {len(participants)} participant(s).[/green]")
    else:
        console.print(f"[green]Discovered {len(participants)} participant(s) to process.[/green]")

    out_root = output_dir if output_dir is not None else source_dir
    out_root.mkdir(parents=True, exist_ok=True)

    mask_transform = ComputeEatPvatMasksd(
        keys=["image"],
        pvat_radial_distance_mm=pvat_radius,
    )
    quant_transform = QuantifyAdiposeMetricsd(keys=["image"])

    rows = []
    with console.status("[bold green]Processing cardiac scans..."):
        for idx, (pid, scan_path) in enumerate(participants, start=1):
            console.print(f"[{idx}/{len(participants)}] Processing: [bold]{pid}[/bold] ({scan_path.name})")

            meta_ct = read_nrrd_as_metatensor(scan_path)
            data_dict = {
                "image": meta_ct,
                "participant_id": pid,
            }

            # Step 1: Compute 8-compartment EAT/PVAT masks
            data_dict = mask_transform(data_dict)
            seg_mask_8ch = data_dict["pred"]

            # Save eat_pvat.seg.nrrd in participant directory if requested
            if save_seg:
                if scan_path.is_dir():
                    participant_dir = scan_path if scan_path.parent == source_dir else scan_path.parent
                else:
                    participant_dir = scan_path.parent

                seg_path = participant_dir / "eat_pvat.seg.nrrd"
                ref_header = meta_ct.meta.get("nrrd_header")
                spacing = tuple(meta_ct.meta.get("spacing", (1.0, 1.0, 1.0)))
                origin = tuple(meta_ct.meta.get("origin", (0.0, 0.0, 0.0))) if "origin" in meta_ct.meta else None

                write_eat_pvat_seg_nrrd(
                    seg_data=seg_mask_8ch,
                    output_path=seg_path,
                    reference_header=ref_header,
                    spacing=spacing,
                    origin=origin,
                )
                console.print(f"  -> Saved segmentation: [dim]{seg_path}[/dim]")

            # Step 2: Quantify volumes and mean attenuation
            data_dict = quant_transform(data_dict)
            metrics = data_dict["metrics"]
            rows.append(metrics)

    # Build DataFrame matching exact 19-column schema
    df = pd.DataFrame(rows)
    for col in MEASUREMENT_CSV_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[MEASUREMENT_CSV_COLUMNS]

    # Validate schema
    validate_measurement_dataframe(df)

    csv_dest = out_root / csv_filename
    df.to_csv(csv_dest, index=False)
    console.print(f"[bold green]Successfully generated {len(df)} row(s):[/bold green] {csv_dest}")

    # Display preview table
    table = Table(title="Cardiac Fat Quantification Summary (Preview)")
    table.add_column("Participant ID", justify="left", style="cyan")
    table.add_column("EAT Total (cm3)", justify="right")
    table.add_column("EAT Total HU", justify="right")
    table.add_column("PVAT Total (cm3)", justify="right")
    table.add_column("PVAT Total HU", justify="right")

    for _, row in df.head(5).iterrows():
        table.add_row(
            str(row["participant_id"]),
            f"{row['EAT Total Volume (cm3)']:.2f}",
            f"{row['EAT Total Mean Attenuation (HU)']:.1f}",
            f"{row['PVAT Total Volume (cm3)']:.2f}",
            f"{row['PVAT Total Mean Attenuation (HU)']:.1f}",
        )
    console.print(table)


@app.command("federated-train")
def federated_train(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Path to directory containing participant scans.",
        exists=True,
        file_okay=False,
    ),
    rounds: int = typer.Option(1, "--rounds", "-r", help="Number of simulated federated training rounds."),
    fedprox_mu: float = typer.Option(0.01, "--mu", help="FedProx proximal parameter mu."),
    fed_bn: bool = typer.Option(True, "--fed-bn/--no-fed-bn", help="Enable FedBN local norm retention."),
) -> None:
    """Run a local federated client training simulation."""
    console.print(f"[bold cyan]Initializing CardiacFatClientAlgo on:[/bold cyan] {data_dir}")
    client = CardiacFatClientAlgo(
        data_dir=data_dir,
        fedprox_mu=fedprox_mu,
        fed_bn=fed_bn,
    )
    client.initialize()

    from monai.fl.utils.exchange_object import ExchangeObject

    for r in range(1, rounds + 1):
        console.print(f"[bold green]Round {r}/{rounds}: Starting local training...[/bold green]")
        client.train(ExchangeObject())
        weights_obj = client.get_weights()
        num_tensors = len(weights_obj.weights) if weights_obj.weights else 0
        console.print(f"Round {r}: Extracted {num_tensors} weight tensors (FedBN={fed_bn}).")

        eval_obj = client.evaluate(ExchangeObject())
        console.print(f"Round {r} Eval Metrics: {eval_obj.metrics}")

    console.print("[bold green]Federated client training step completed successfully.[/bold green]")


@app.command("federated-stats")
def federated_stats(
    csv_path: Path = typer.Option(
        "cardiac_fat_measurements.csv",
        "--csv-path",
        "-c",
        help="Path to cardiac_fat_measurements.csv file.",
        exists=True,
    )
) -> None:
    """Execute federated statistics extraction across local measurements CSV."""
    executor = CardiacFatStatsExecutor(csv_path=csv_path)
    res = executor.compute_local_stats()
    console.print(f"[bold green]Federated Cohort Statistics (N={res['participant_count']}):[/bold green]")
    for metric_name, details in res.get("metrics", {}).items():
        if details["count"] > 0:
            console.print(f"  • {metric_name}: Mean={details['mean']:.2f}, Std={details['std']:.2f}, Range=[{details['min']:.1f}, {details['max']:.1f}]")


@app.command("compare-models")
def compare_models(
    scan_path: Path = typer.Argument(
        ...,
        help="Path to CT volume file (.nrrd, .nii.gz) or participant DICOM series directory.",
        exists=True,
        resolve_path=True,
    ),
    onnx_seg_model: Path = typer.Option(
        Path("models/deepheartseg_heart.onnx"),
        "--onnx-model",
        help="Path to converted DeepHeartSeg ONNX segmentation model.",
    ),
    onnx_loc_model: Path = typer.Option(
        Path("models/deepheartseg_loc.onnx"),
        "--loc-model",
        help="Path to converted DeepHeartSeg ONNX localization model.",
    ),
) -> None:
    """Run side-by-side comparison between Modernized pipeline and frozen legacy DeepHeartSeg ONNX model."""
    import numpy as np
    from cardiac_fat_fl.measurement_schema import REPORTED_REGIONS

    console.print(f"[bold cyan]Running Equivalency Comparison on:[/bold cyan] {scan_path.name}")

    meta_ct = read_nrrd_as_metatensor(scan_path)
    ct_arr = meta_ct.detach().cpu().numpy()
    while ct_arr.ndim > 3:
        ct_arr = ct_arr[0]
    spacing = tuple(meta_ct.meta.get("spacing", (1.0, 1.0, 1.0)))

    mask_tx = ComputeEatPvatMasksd(keys=["image"])
    quant_tx = QuantifyAdiposeMetricsd(keys=["image"])

    # 1. Run Modernized Heuristic Pipeline
    seg_heur = UnifiedAnatomySegmenter(backend="heuristic")
    anat_heur = seg_heur.segment(ct_arr, spacing=spacing)
    res_heur = quant_tx(mask_tx({"image": meta_ct, "anatomy": anat_heur, "participant_id": "Modern"}))["metrics"]

    # 2. Run DeepHeartSeg ONNX Pipeline
    seg_onnx = UnifiedAnatomySegmenter(
        model_path=onnx_seg_model,
        loc_model_path=onnx_loc_model,
        backend="onnx",
    )
    anat_onnx = seg_onnx.segment(ct_arr, spacing=spacing)
    res_onnx = quant_tx(mask_tx({"image": meta_ct, "anatomy": anat_onnx, "participant_id": "ONNX_Legacy"}))["metrics"]

    # Calculate Dice overlap of Pericardium
    inter = np.sum(anat_heur.pericardium & anat_onnx.pericardium)
    dice = (2.0 * inter) / (np.sum(anat_heur.pericardium) + np.sum(anat_onnx.pericardium))

    table = Table(title=f"Clinical Equivalency Analysis (Pericardium Dice: {dice*100:.2f}%)")
    table.add_column("Anatomical Region", style="cyan", justify="left")
    table.add_column("Modern Vol (cm3)", justify="right")
    table.add_column("ONNX Vol (cm3)", justify="right")
    table.add_column("Vol Diff (cm3)", justify="right")
    table.add_column("Modern HU", justify="right")
    table.add_column("ONNX HU", justify="right")
    table.add_column("HU Delta", justify="right")

    for region in REPORTED_REGIONS:
        reg = region.value
        v_h = res_heur[f"{reg} Volume (cm3)"]
        v_o = res_onnx[f"{reg} Volume (cm3)"]
        hu_h = res_heur[f"{reg} Mean Attenuation (HU)"]
        hu_o = res_onnx[f"{reg} Mean Attenuation (HU)"]
        v_diff = abs(v_h - v_o)
        hu_delta = abs(hu_h - hu_o) if not np.isnan(hu_h) and not np.isnan(hu_o) else np.nan

        table.add_row(
            reg,
            f"{v_h:.2f}",
            f"{v_o:.2f}",
            f"{v_diff:.2f}",
            f"{hu_h:.1f}" if not np.isnan(hu_h) else "NaN",
            f"{hu_o:.1f}" if not np.isnan(hu_o) else "NaN",
            f"{hu_delta:.2f}" if not np.isnan(hu_delta) else "-",
        )

    console.print(table)


@app.command("view")
def view_sample(
    sample_dir: Path = typer.Argument(
        ...,
        help="Path to participant directory containing scan and eat_pvat.seg.nrrd.",
        exists=True,
        resolve_path=True,
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the local web viewer."),
    export_only: bool = typer.Option(False, "--export-only", help="Export DICOM & DICOM-SEG without running server."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open web browser automatically."),
) -> None:
    """Launch the OHIF-compatible interactive viewer for a participant sample."""
    from cardiac_fat_fl.ohif_viewer import launch_viewer

    launch_viewer(
        sample_dir=sample_dir,
        port=port,
        open_browser=not no_browser,
        export_only=export_only,
    )


@app.command("export-ohif")
def export_ohif_cmd(
    sample_dir: Path = typer.Argument(
        ...,
        help="Path to participant directory containing scan and eat_pvat.seg.nrrd.",
        exists=True,
        resolve_path=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for OHIF DICOM and DICOM-SEG files.",
    ),
) -> None:
    """Export a participant sample into OHIF-compliant DICOM CT series and DICOM-SEG."""
    from cardiac_fat_fl.ohif_exporter import export_sample_to_ohif

    dest = export_sample_to_ohif(sample_dir=sample_dir, output_dir=output_dir)
    console.print(f"[bold green]Successfully exported OHIF package to:[/bold green] {dest}")


if __name__ == "__main__":
    app()

