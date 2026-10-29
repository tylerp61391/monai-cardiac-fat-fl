"""Unit tests for MONAI Deploy Application Package (MAP) operators and pipeline."""

from pathlib import Path
import json
import numpy as np
import pydicom
import pytest
from typer.testing import CliRunner

from cardiac_fat_fl.cli import app
from cardiac_fat_fl.measurement_schema import (
    MEASUREMENT_CSV_COLUMNS,
    validate_measurement_dataframe,
)
from cardiac_fat_fl.monai_deploy import (
    CardiacCTStudyLoaderOp,
    CardiacFatQuantApplication,
    CardiacFatQuantifierOp,
    DicomSegAndSrWriterOp,
    run_monai_deploy_pipeline,
)

runner = CliRunner()


def test_cardiac_ct_study_loader_op(synthetic_nrrd_scan: Path) -> None:
    """Test CardiacCTStudyLoaderOp ingests volume correctly into MetaTensor."""
    loader = CardiacCTStudyLoaderOp()
    result = loader.compute(input_path=synthetic_nrrd_scan)

    assert "participant_id" in result
    assert "scan_path" in result
    assert "image" in result
    assert result["image"].ndim == 4  # (1, D, H, W)
    assert result["participant_id"] == synthetic_nrrd_scan.parent.name


def test_cardiac_fat_quantifier_op(synthetic_nrrd_scan: Path) -> None:
    """Test CardiacFatQuantifierOp executes segmentation and adipose quantification."""
    loader = CardiacCTStudyLoaderOp()
    study_data = loader.compute(input_path=synthetic_nrrd_scan)

    quantifier = CardiacFatQuantifierOp(models_dir=None, pvat_radius_mm=10.0)
    quant_data = quantifier.compute(study_dict=study_data)

    assert "pred" in quant_data
    assert "metrics" in quant_data
    assert quant_data["pred"].shape[0] == 8  # 8 compartments
    metrics = quant_data["metrics"]
    assert "EAT Total Volume (cm3)" in metrics
    assert "PVAT Total Volume (cm3)" in metrics


def test_dicom_seg_and_sr_writer_op(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test DicomSegAndSrWriterOp emits standard DICOM-SEG, DICOM-SR TID 1500, and metrics."""
    loader = CardiacCTStudyLoaderOp()
    study_data = loader.compute(input_path=synthetic_nrrd_scan)

    quantifier = CardiacFatQuantifierOp(models_dir=None)
    quant_data = quantifier.compute(study_dict=study_data)

    writer = DicomSegAndSrWriterOp()
    output_dir = tmp_path / "map_output"
    artifacts = writer.compute(quant_dict=quant_data, output_dir=output_dir)

    # 1. eat_pvat.seg.nrrd exists
    assert artifacts["eat_pvat_seg_nrrd"].exists()

    # 2. DICOM-SEG exists and has standard SOP Class UID
    seg_dcm = artifacts["dicom_seg"]
    assert seg_dcm.exists()
    ds_seg = pydicom.dcmread(str(seg_dcm))
    assert ds_seg.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
    assert hasattr(ds_seg, "SegmentSequence")
    assert len(ds_seg.SegmentSequence) == 8

    # 3. DICOM-SR TID 1500 exists and has standard SOP Class UID
    sr_dcm = artifacts["dicom_sr"]
    assert sr_dcm.exists()
    ds_sr = pydicom.dcmread(str(sr_dcm))
    assert ds_sr.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.34"

    # 4. JSON & CSV metrics exist and validate
    assert artifacts["metrics_json"].exists()
    json_data = json.loads(artifacts["metrics_json"].read_text(encoding="utf-8"))
    assert "EAT Total Volume (cm3)" in json_data

    assert artifacts["metrics_csv"].exists()
    import pandas as pd

    df = pd.read_csv(artifacts["metrics_csv"])
    assert list(df.columns) == MEASUREMENT_CSV_COLUMNS
    validate_measurement_dataframe(df)


def test_run_monai_deploy_pipeline_end_to_end(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test top-level run_monai_deploy_pipeline function."""
    output_dir = tmp_path / "pipeline_out"
    artifacts = run_monai_deploy_pipeline(
        input_path=synthetic_nrrd_scan,
        output_path=output_dir,
        pvat_radius_mm=10.0,
    )

    assert artifacts["dicom_seg"].exists()
    assert artifacts["dicom_sr"].exists()
    assert artifacts["metrics_csv"].exists()


def test_cli_map_run_command(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test CLI cardiac-fat-fl map-run execution."""
    output_dir = tmp_path / "cli_map_out"
    result = runner.invoke(
        app,
        [
            "map-run",
            "--input",
            str(synthetic_nrrd_scan),
            "--output",
            str(output_dir),
            "--pvat-radius",
            "10.0",
        ],
    )
    assert result.exit_code == 0, f"CLI output: {result.output}"
    assert "MONAI Deploy pipeline completed successfully!" in result.output
    assert (output_dir / "eat_pvat_seg.dcm").exists()
    assert (output_dir / "eat_pvat_sr.dcm").exists()
    assert (output_dir / "cardiac_fat_measurements.csv").exists()

