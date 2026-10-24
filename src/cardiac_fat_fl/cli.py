"""Command Line Interface for Cardiac Fat Quantification and Federated Learning.

Provides end-to-end processing, anatomy segmentation, EAT/PVAT mask computation,
CSV export, and federated training routines matching the MGH CardiacFatQuant specification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from cardiac_fat_fl.anatomy_segmenter import UnifiedAnatomySegmenter
from cardiac_fat_fl.fl_client_algo import CardiacFatClientAlgo
from cardiac_fat_fl.fl_stats_executor import CardiacFatStatsExecutor
from cardiac_fat_fl.hardware import (
    detect_hardware,
    display_hardware_summary,
    validate_worker_concurrency,
)
from cardiac_fat_fl.io_metatensor import (
    discover_participants,
    read_nrrd_as_metatensor,
    write_eat_pvat_seg_nrrd,
    write_totalsegmentator_seg_nrrd,
    write_monai_wholebody_ct_segmentation_seg_nrrd,
    write_coronary_sulci_seg_nrrd,
)
from cardiac_fat_fl.measurement_schema import (
    DEFAULT_PVAT_RADIAL_DISTANCE_MM,
    MEASUREMENT_CSV_COLUMNS,
    validate_measurement_dataframe,
)
from cardiac_fat_fl.transforms import ComputeEatPvatMasksd, QuantifyAdiposeMetricsd

app = typer.Typer(
    name="cardiac-fat-fl",
    help="MONAI, FLARE & OHIF pipeline for Epicardial and Pericoronary Adipose Tissue (EAT/PVAT) quantification from cardiac CT.",
    add_completion=False,
)
console = Console()


def _process_one_participant(task: Tuple[str, Path, bool, bool, float, Optional[Path], Path]) -> dict:
    """Worker function to process a single participant scan."""
    pid, scan_path, save_seg, save_intermediate, pvat_radius, model_path, source_dir = task
    try:
        meta_ct = read_nrrd_as_metatensor(scan_path)
        data_dict = {
            "image": meta_ct,
            "participant_id": pid,
        }
        mask_transform = ComputeEatPvatMasksd(
            keys=["image"],
            pvat_radial_distance_mm=pvat_radius,
        )
        if model_path is not None:
            mask_transform._segmenter = UnifiedAnatomySegmenter(
                model_path=model_path,
                backend="onnx" if str(model_path).endswith(".onnx") else "heuristic",
            )
        quant_transform = QuantifyAdiposeMetricsd(keys=["image"])

        # Step 1: Compute 8-compartment EAT/PVAT masks
        data_dict = mask_transform(data_dict)
        seg_mask_8ch = data_dict["pred"]
        anatomy = data_dict.get("anatomy_masks")

        if scan_path.is_dir():
            participant_dir = scan_path.parent if (scan_path.parent.name == pid or scan_path.parent == source_dir) else scan_path
        else:
            participant_dir = scan_path.parent

        ref_header = meta_ct.meta.get("nrrd_header")
        spacing = tuple(meta_ct.meta.get("spacing", (1.0, 1.0, 1.0)))
        origin = tuple(meta_ct.meta.get("origin", (0.0, 0.0, 0.0))) if "origin" in meta_ct.meta else None

        # Save eat_pvat.seg.nrrd in participant directory if requested
        if save_seg:
            seg_path = participant_dir / "eat_pvat.seg.nrrd"
            write_eat_pvat_seg_nrrd(
                seg_data=seg_mask_8ch,
                output_path=seg_path,
                reference_header=ref_header,
                spacing=spacing,
                origin=origin,
            )

        # Save MGH-compatible intermediate masks if requested
        if save_intermediate and anatomy is not None:
            # 1. heart.seg.nrrd (pericardial sac binary mask)
            heart_path = participant_dir / "heart.seg.nrrd"
            write_eat_pvat_seg_nrrd(
                seg_data=anatomy.pericardium.astype(np.uint8),
                output_path=heart_path,
                reference_header=ref_header,
                spacing=spacing,
                origin=origin,
            )

            # 2. totalsegmentator_1-5mm_masks.seg.nrrd (multi-label chamber & vessel mask)
            # 2. monai_wholebody_ct_segmentation_1-5mm_masks.seg.nrrd (strictly neural network anatomy)
            tot_seg = np.zeros_like(anatomy.pericardium, dtype=np.uint16)
            if anatomy.aorta is not None:
                tot_seg[anatomy.aorta] = 7
            if anatomy.pulmonary_artery is not None:
                tot_seg[anatomy.pulmonary_artery] = 49
            if anatomy.myocardium is not None:
                tot_seg[anatomy.myocardium] = 44
            tot_seg[anatomy.la] = 45
            tot_seg[anatomy.lv] = 46
            tot_seg[anatomy.ra] = 47
            tot_seg[anatomy.rv] = 48

            monai_tot_path = participant_dir / "monai_wholebody_ct_segmentation_1-5mm_masks.seg.nrrd"
            write_monai_wholebody_ct_segmentation_seg_nrrd(
                seg_data=tot_seg,
                output_path=monai_tot_path,
                reference_header=ref_header,
                spacing=spacing,
                origin=origin,
            )

            # 3. Optional coronary_sulci.seg.nrrd (derived geometric sulci seeds)
            from cardiac_fat_fl.io_metatensor import write_coronary_sulci_seg_nrrd
            sulci_seg = np.zeros_like(anatomy.pericardium, dtype=np.uint16)
            if anatomy.lad is not None:
                sulci_seg[anatomy.lad] = 1
            if anatomy.lcx is not None:
                sulci_seg[anatomy.lcx] = 2
            if anatomy.rca is not None:
                sulci_seg[anatomy.rca] = 3
            if np.any(sulci_seg > 0):
                write_coronary_sulci_seg_nrrd(
                    seg_data=sulci_seg,
                    output_path=participant_dir / "coronary_sulci.seg.nrrd",
                    reference_header=ref_header,
                    spacing=spacing,
                    origin=origin,
                )

        # Step 2: Quantify volumes and mean attenuation
        data_dict = quant_transform(data_dict)
        return {"status": "ok", "participant_id": pid, "metrics": data_dict["metrics"]}
    except Exception as exc:
        return {"status": "error", "participant_id": pid, "error": str(exc)}


def _display_preview(df: pd.DataFrame) -> None:
    """Render preview table of quantified adipose metrics."""
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
    save_intermediate: bool = typer.Option(
        False,
        "--save-intermediate/--no-save-intermediate",
        help="Save intermediate masks: `heart.seg.nrrd` and `monai_wholebody_ct_segmentation_1-5mm_masks.seg.nrrd`.",
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
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help="Number of parallel worker processes. Default 0 (auto-detect safe concurrency based on host CPU, RAM, and GPU). Set > 0 for manual override.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume execution by skipping participant IDs already recorded in destination CSV.",
    ),
) -> None:
    """Execute end-to-end quantification across participants in source_dir."""
    console.print(f"[bold cyan]Scanning directory:[/bold cyan] {source_dir}")

    # Step 0: Dynamic hardware inspection and safe concurrency sizing
    profile = detect_hardware()
    display_hardware_summary(profile, console=console)

    if workers <= 0:
        actual_workers = profile.recommended_workers
        console.print(f"[bold green]Auto-configured safe worker concurrency: {actual_workers} worker(s)[/bold green]")
    else:
        actual_workers = validate_worker_concurrency(workers, profile)
        if actual_workers < workers:
            console.print(
                f"[bold yellow]Safety Throttle:[/bold yellow] Requested {workers} workers exceeds host safe memory budget "
                f"({profile.available_ram_gb:.1f} GB available). Throttling safely to {actual_workers} worker(s) "
                f"to prevent system freeze or out-of-memory crash."
            )
        else:
            console.print(f"[cyan]Using requested concurrency: {actual_workers} worker(s)[/cyan]")

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
    csv_dest = out_root / csv_filename

    # Handle resume logic
    existing_rows: List[dict] = []
    completed_pids = set()
    if resume and csv_dest.exists():
        try:
            df_existing = pd.read_csv(csv_dest)
            if "participant_id" in df_existing.columns:
                completed_pids = set(df_existing["participant_id"].astype(str))
                existing_rows = df_existing.to_dict("records")
                console.print(f"[cyan]Found existing measurements CSV with {len(completed_pids)} participant(s). Resuming...[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Could not read existing CSV for resume: {e}. Starting fresh.[/yellow]")

    if completed_pids:
        remaining_participants = [p for p in participants if str(p[0]) not in completed_pids]
        console.print(f"[green]{len(completed_pids)} participant(s) already completed. {len(remaining_participants)} remaining to process.[/green]")
    else:
        remaining_participants = participants

    def _flush_csv(all_rows: List[dict]) -> pd.DataFrame:
        df_out = pd.DataFrame(all_rows)
        for col in MEASUREMENT_CSV_COLUMNS:
            if col not in df_out.columns:
                df_out[col] = float("nan")
        df_out = df_out[MEASUREMENT_CSV_COLUMNS]
        df_out.to_csv(csv_dest, index=False)
        return df_out

    if not remaining_participants:
        console.print(f"[bold green]All {len(participants)} participant(s) are already quantified in {csv_dest}.[/bold green]")
        df_final = _flush_csv(existing_rows)
        validate_measurement_dataframe(df_final)
        _display_preview(df_final)
        return

    tasks = [
        (pid, scan_path, save_seg, save_intermediate, pvat_radius, model_path, source_dir)
        for pid, scan_path in remaining_participants
    ]

    all_rows = list(existing_rows)
    total_remaining = len(tasks)

    if actual_workers <= 1:
        with console.status("[bold green]Processing cardiac scans..."):
            for idx, task in enumerate(tasks, start=1):
                pid = task[0]
                console.print(f"[{idx}/{total_remaining}] Processing: [bold]{pid}[/bold]")
                res = _process_one_participant(task)
                if res["status"] == "ok":
                    all_rows.append(res["metrics"])
                    if idx % 5 == 0:
                        _flush_csv(all_rows)
                else:
                    console.print(f"  [red]Failed {pid}: {res.get('error')}[/red]")
    else:
        console.print(f"[bold cyan]Launching parallel processing with {actual_workers} workers...[/bold cyan]")
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_to_pid = {
                executor.submit(_process_one_participant, task): task[0]
                for task in tasks
            }
            completed_count = 0
            for future in as_completed(future_to_pid):
                completed_count += 1
                pid = future_to_pid[future]
                try:
                    res = future.result()
                    if res["status"] == "ok":
                        all_rows.append(res["metrics"])
                        eat_vol = res["metrics"].get("EAT Total Volume (cm3)", 0.0)
                        pvat_vol = res["metrics"].get("PVAT Total Volume (cm3)", 0.0)
                        console.print(f"[{completed_count}/{total_remaining}] Completed: [bold]{pid}[/bold] (EAT: {eat_vol:.1f} cm3, PVAT: {pvat_vol:.1f} cm3)")
                    else:
                        console.print(f"[{completed_count}/{total_remaining}] [red]Failed {pid}: {res.get('error')}[/red]")
                except Exception as exc:
                    console.print(f"[{completed_count}/{total_remaining}] [red]Exception {pid}: {exc}[/red]")

                if completed_count % 10 == 0:
                    _flush_csv(all_rows)

    df_final = _flush_csv(all_rows)
    validate_measurement_dataframe(df_final)
    console.print(f"[bold green]Successfully finalized {len(df_final)} participant measurement(s):[/bold green] {csv_dest}")
    _display_preview(df_final)


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


@app.command("hardware")
def hardware_info() -> None:
    """Inspect host CPUs, RAM, and CUDA GPUs to display safe concurrency recommendations."""
    profile = detect_hardware()
    display_hardware_summary(profile, console=console)


@app.command("download-models")
def download_models_cmd(
    repo_tag: str = typer.Option(
        "v1.0.0",
        "--tag",
        "-t",
        help="Release tag version to fetch models from.",
    ),
    repo_url: str = typer.Option(
        "https://github.com/tylerp61391/monai-cardiac-fat-fl",
        "--repo",
        "-r",
        help="GitHub repository URL.",
    ),
    output_dir: Path = typer.Option(
        Path("models"),
        "--output-dir",
        "-o",
        help="Destination directory for downloaded ONNX models.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-download and overwrite existing ONNX files.",
    ),
) -> None:
    """Download validated DeepHeartSeg ONNX models from GitHub Release assets."""
    import urllib.request
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    models = ["deepheartseg_heart.onnx", "deepheartseg_loc.onnx"]
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = repo_url.rstrip("/") + f"/releases/download/{repo_tag}"

    for model_name in models:
        target = output_dir / model_name
        if target.exists() and not force:
            console.print(f"[green]Model already present:[/green] {target} (use --force to re-download)")
            continue

        url = f"{base_url}/{model_name}"
        console.print(f"[cyan]Downloading {model_name} from:[/cyan] {url}")
        try:
            with Progress(
                TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("download", filename=model_name, total=None)

                def reporthook(block_num: int, block_size: int, total_size: int) -> None:
                    if total_size > 0:
                        progress.update(task_id, total=total_size, completed=block_num * block_size)

                urllib.request.urlretrieve(url, str(target), reporthook=reporthook)
            console.print(f"[bold green]Successfully saved:[/bold green] {target}")
        except Exception as e:
            console.print(f"[yellow]Could not download {model_name} from {url}: {e}[/yellow]")
            console.print("[dim]Note: Ensure release assets are published or specify custom --repo / --tag.[/dim]")




@app.command("push")
def push_pacs(
    target_path: Path = typer.Argument(
        ...,
        help="Path to participant directory, ohif_export directory, or eat_pvat_seg.dcm file to push to PACS.",
        exists=True,
    ),
    url: Optional[str] = typer.Option(
        None,
        "--url",
        "-u",
        help="Target DICOMweb STOW-RS URL (e.g., http://localhost:8042/dicom-web/studies or http://localhost:8042/instances).",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        "-h",
        help="Target PACS hostname or IP address for DIMSE C-STORE (e.g., 192.168.1.50).",
    ),
    port: int = typer.Option(
        104,
        "--port",
        "-p",
        help="Target PACS port for DIMSE C-STORE (common: 104, 4242, 11112).",
    ),
    called_aet: str = typer.Option(
        "ANY-SCP",
        "--called-aet",
        help="Target PACS Application Entity Title (AET).",
    ),
    calling_aet: str = typer.Option(
        "CARDIAC_FAT_FL",
        "--calling-aet",
        help="Local Application Entity Title (AET).",
    ),
    include_ct: bool = typer.Option(
        False,
        "--include-ct",
        help="Push full CT slice series along with the DICOM-SEG (default: DICOM-SEG only).",
    ),
    username: Optional[str] = typer.Option(
        None,
        "--username",
        help="HTTP Basic Auth username for DICOMweb STOW-RS.",
    ),
    password: Optional[str] = typer.Option(
        None,
        "--password",
        help="HTTP Basic Auth password for DICOMweb STOW-RS.",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Bearer Authorization token for DICOMweb STOW-RS.",
    ),
    timeout: int = typer.Option(
        120,
        "--timeout",
        help="Network timeout in seconds.",
    ),
) -> None:
    """Store DICOM-SEG and cardiac CT volumes to a PACS archive via DICOMweb (STOW-RS) or DIMSE (C-STORE)."""
    from cardiac_fat_fl.pacs_client import (
        find_dicom_instances_to_push,
        push_to_dicomweb_stow,
        push_to_dimse_cstore,
    )

    if not url and not host:
        console.print("[bold red]Error:[/bold red] You must specify either a DICOMweb endpoint (--url) or a DIMSE PACS host (--host).")
        console.print("[dim]Example DICOMweb: cardiac-fat-fl push ./0 --url http://localhost:8042/dicom-web/studies[/dim]")
        console.print("[dim]Example DIMSE:    cardiac-fat-fl push ./0 --host 192.168.1.50 --port 104 --called-aet PACSSERVER[/dim]")
        raise typer.Exit(code=1)

    try:
        instances = find_dicom_instances_to_push(target_path, include_ct=include_ct)
    except Exception as e:
        console.print(f"[bold red]Discovery Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Found {len(instances)} DICOM instance(s) to store:[/bold cyan]")
    for inst in instances[:5]:
        console.print(f"  • {inst.name} ({inst.stat().st_size / 1024:.1f} KB)")
    if len(instances) > 5:
        console.print(f"  ... and {len(instances) - 5} more files.")

    # 1. DICOMweb STOW-RS
    if url:
        console.print(f"[cyan]Connecting to DICOMweb STOW-RS endpoint:[/cyan] {url}")
        auth = (username, password) if username and password else None
        res = push_to_dicomweb_stow(instances, stow_url=url, auth=auth, token=token, timeout=timeout)
        
        if res["failed_count"] == 0:
            console.print(f"[bold green]Success: Stored {res['success_count']} instance(s) to PACS via STOW-RS.[/bold green]")
        else:
            console.print(f"[bold yellow]Warning: Transmitted with issues:[/bold yellow] {res['success_count']} succeeded, {res['failed_count']} failed.")
            console.print(f"[dim]{res['details']}[/dim]")
            raise typer.Exit(code=1)
        return

    # 2. DIMSE C-STORE
    if host:
        console.print(f"[cyan]Initiating DICOM DIMSE C-STORE association:[/cyan] {calling_aet} -> {called_aet}@{host}:{port}")
        res = push_to_dimse_cstore(
            instances,
            host=host,
            port=port,
            called_aet=called_aet,
            calling_aet=calling_aet,
            timeout=timeout,
        )

        if res["failed_count"] == 0:
            console.print(f"[bold green]Success: Stored {res['success_count']} instance(s) to PACS via C-STORE.[/bold green]")
        else:
            console.print(f"[bold yellow]Warning: C-STORE issues:[/bold yellow] {res['success_count']} succeeded, {res['failed_count']} failed.")
            console.print(f"[dim]{res['details']}[/dim]")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

