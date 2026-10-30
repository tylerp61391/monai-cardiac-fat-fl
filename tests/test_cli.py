"""End-to-End Integration Tests for cardiac-fat-fl CLI.

Verifies:
  - CLI `quantify` on synthetic cohort directory produces `eat_pvat.seg.nrrd` and
    `cardiac_fat_measurements.csv` strictly adhering to the 19-column schema.
  - CLI `federated-stats` computes cohort statistics.
  - CLI `federated-train` executes local client training rounds.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from cardiac_fat_fl.cli import app
from cardiac_fat_fl.measurement_schema import (
    MEASUREMENT_CSV_COLUMNS,
    validate_measurement_dataframe,
)

runner = CliRunner()


def test_cli_quantify_end_to_end(synthetic_cohort_dir: Path) -> None:
    """Test full CLI quantify command on synthetic cohort."""
    csv_out = synthetic_cohort_dir / "cardiac_fat_measurements.csv"

    result = runner.invoke(
        app,
        [
            "quantify",
            str(synthetic_cohort_dir),
            "--output-dir",
            str(synthetic_cohort_dir),
            "--save-seg",
            "--csv-filename",
            "cardiac_fat_measurements.csv",
        ],
    )
    assert result.exit_code == 0, f"CLI quantify failed with output:\n{result.stdout}"
    assert csv_out.is_file(), f"Expected CSV file not created at {csv_out}"

    # Verify both participants produced eat_pvat.seg.nrrd
    p1_seg = synthetic_cohort_dir / "PATIENT_001" / "eat_pvat.seg.nrrd"
    p2_seg = synthetic_cohort_dir / "PATIENT_002" / "eat_pvat.seg.nrrd"
    assert p1_seg.is_file(), f"Missing segmentation for PATIENT_001 at {p1_seg}"
    assert p2_seg.is_file(), f"Missing segmentation for PATIENT_002 at {p2_seg}"

    # Verify CSV schema
    df = pd.read_csv(csv_out)
    assert len(df) == 2
    assert validate_measurement_dataframe(df) is True
    assert set(df["participant_id"]) == {"PATIENT_001", "PATIENT_002"}

    # Verify Volume and HU values are clinically valid
    assert (df["EAT Total Volume (cm3)"] > 0.0).all()
    assert (df["PVAT Total Volume (cm3)"] > 0.0).all()

    for _, row in df.iterrows():
        for region in ["EAT Total", "PVAT Total", "PVAT LAD", "PVAT LCX", "PVAT RCA"]:
            vol = row[f"{region} Volume (cm3)"]
            hu = row[f"{region} Mean Attenuation (HU)"]
            assert vol >= 0.0
            if vol > 0.0:
                assert -190.0 <= hu <= -30.0
            else:
                assert math.isnan(hu)


def test_cli_federated_stats(synthetic_cohort_dir: Path) -> None:
    """Test CLI federated-stats command."""
    csv_path = synthetic_cohort_dir / "cardiac_fat_measurements.csv"
    if not csv_path.is_file():
        # Generate CSV first
        runner.invoke(app, ["quantify", str(synthetic_cohort_dir)])

    result = runner.invoke(app, ["federated-stats", "--csv-path", str(csv_path)])
    assert result.exit_code == 0
    assert "Federated Cohort Statistics" in result.stdout


def test_cli_federated_train(synthetic_cohort_dir: Path) -> None:
    """Test CLI federated-train command."""
    result = runner.invoke(
        app,
        [
            "federated-train",
            "--data-dir",
            str(synthetic_cohort_dir),
            "--rounds",
            "1",
            "--mu",
            "0.01",
            "--fed-bn",
        ],
    )
    assert result.exit_code == 0
    assert "Federated client training step completed successfully" in result.stdout


def test_cli_dashboard_help() -> None:
    """Test CLI dashboard command help."""
    result = runner.invoke(app, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "Launch the interactive Cardiac Research Dashboard" in result.stdout


def test_cli_benchmark_command(synthetic_cohort_dir: Path, tmp_path: Path) -> None:
    """Test CLI benchmark command comparing segmentation masks."""
    # Ensure quantify has been run so eat_pvat.seg.nrrd exists
    p1_seg = synthetic_cohort_dir / "PATIENT_001" / "eat_pvat.seg.nrrd"
    if not p1_seg.is_file():
        runner.invoke(app, ["quantify", str(synthetic_cohort_dir), "--save-seg"])

    csv_out = tmp_path / "benchmark_summary.csv"
    json_out = tmp_path / "benchmark_summary.json"

    # Self-comparison should yield 100% Dice
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--cohort-dir",
            str(synthetic_cohort_dir),
            "--candidate",
            "eat_pvat.seg.nrrd",
            "--reference",
            "eat_pvat.seg.nrrd",
            "--output-csv",
            str(csv_out),
            "--output-json",
            str(json_out),
        ],
    )
    assert result.exit_code == 0, f"Benchmark failed: {result.stdout}"
    assert "Multi-Compartment Segmentation Benchmark" in result.stdout
    assert csv_out.is_file()
    assert json_out.is_file()
