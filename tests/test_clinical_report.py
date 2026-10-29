"""Unit and integration tests for Clinical-Grade Imaging Report & MPR Visualization."""

import io
from pathlib import Path
import numpy as np
import pydicom
from pydicom.uid import EncapsulatedPDFStorage
import pytest
from typer.testing import CliRunner
from PIL import Image

from cardiac_fat_fl.cli import app
from cardiac_fat_fl.clinical_report import (
    CRISP_CT_INFLAMMATION_HU_THRESHOLD,
    encapsulate_pdf_to_dicom,
    evaluate_clinical_risk,
    export_clinical_imaging_report,
    generate_clinical_report_html,
    generate_clinical_report_pdf,
    generate_mpr_snapshot,
)
from cardiac_fat_fl.measurement_schema import ReportedRegion

runner = CliRunner()


def test_evaluate_clinical_risk_low() -> None:
    """Test risk evaluation when EAT volume and PVAT attenuation are within normal limits."""
    metrics = {
        f"{ReportedRegion.EAT_TOTAL} Volume (cm3)": 75.0,
        f"{ReportedRegion.EAT_TOTAL} Mean Attenuation (HU)": -82.0,
        f"{ReportedRegion.PVAT_LAD} Volume (cm3)": 6.2,
        f"{ReportedRegion.PVAT_LAD} Mean Attenuation (HU)": -78.4,
        f"{ReportedRegion.PVAT_LCX} Volume (cm3)": 4.1,
        f"{ReportedRegion.PVAT_LCX} Mean Attenuation (HU)": -75.1,
        f"{ReportedRegion.PVAT_RCA} Volume (cm3)": 5.8,
        f"{ReportedRegion.PVAT_RCA} Mean Attenuation (HU)": -79.0,
    }
    risk = evaluate_clinical_risk(metrics)
    assert risk.overall_tier == "LOW RISK"
    assert risk.eat_burden_category == "Normal"
    assert not risk.pvat_inflammation_detected
    assert "within normal limits" in risk.narrative_summary


def test_evaluate_clinical_risk_elevated_pvat_inflammation() -> None:
    """Test risk evaluation triggers ELEVATED RISK when PVAT attenuation crosses -70.1 HU."""
    metrics = {
        f"{ReportedRegion.EAT_TOTAL} Volume (cm3)": 88.0,
        f"{ReportedRegion.EAT_TOTAL} Mean Attenuation (HU)": -75.0,
        f"{ReportedRegion.PVAT_LAD} Volume (cm3)": 7.0,
        f"{ReportedRegion.PVAT_LAD} Mean Attenuation (HU)": -64.2,  # > -70.1 HU (Inflamed)
        f"{ReportedRegion.PVAT_LCX} Volume (cm3)": 4.5,
        f"{ReportedRegion.PVAT_LCX} Mean Attenuation (HU)": -74.0,
        f"{ReportedRegion.PVAT_RCA} Volume (cm3)": 5.0,
        f"{ReportedRegion.PVAT_RCA} Mean Attenuation (HU)": -78.0,
    }
    risk = evaluate_clinical_risk(metrics)
    assert risk.overall_tier == "ELEVATED RISK"
    assert risk.pvat_inflammation_detected
    assert risk.vessel_risks["LAD"]["is_inflamed"]
    assert not risk.vessel_risks["LCX"]["is_inflamed"]
    assert "LAD" in risk.narrative_summary
    assert "CRISP-CT" in risk.structured_findings_text


def test_evaluate_clinical_risk_elevated_eat_volume() -> None:
    """Test risk evaluation flags high EAT burden when volume exceeds 125 cm3."""
    metrics = {
        f"{ReportedRegion.EAT_TOTAL} Volume (cm3)": 142.5,  # > 125 cm3
        f"{ReportedRegion.EAT_TOTAL} Mean Attenuation (HU)": -78.0,
        f"{ReportedRegion.PVAT_LAD} Volume (cm3)": 8.0,
        f"{ReportedRegion.PVAT_LAD} Mean Attenuation (HU)": -75.0,
    }
    risk = evaluate_clinical_risk(metrics)
    assert risk.overall_tier == "ELEVATED RISK"
    assert risk.eat_burden_category == "Elevated"
    assert risk.eat_percentile >= 75


def test_generate_mpr_snapshot_with_synthetic_ct(synthetic_ct_fixture: tuple) -> None:
    """Test generation of 3-view orthogonal MPR snapshot montage with alpha-blended overlay."""
    ct_vol, _, _, fat_mask = synthetic_ct_fixture

    # Construct mock 8-compartment mask
    d, h, w = ct_vol.shape
    mask_8ch = np.zeros((8, d, h, w), dtype=np.uint8)
    mask_8ch[0] = fat_mask.astype(np.uint8)
    mask_8ch[5] = fat_mask.astype(np.uint8)  # PVAT LAD

    mpr_img = generate_mpr_snapshot(image=ct_vol, mask=mask_8ch, target_view_size=(160, 160))
    assert isinstance(mpr_img, Image.Image)
    assert mpr_img.mode == "RGBA"
    # Width should be (160 * 3) + (8 * 2) = 496
    assert mpr_img.size[0] == 496
    assert mpr_img.size[1] == 160


def test_generate_clinical_report_html_and_pdf(tmp_path: Path) -> None:
    """Test HTML and vector PDF generation with full clinical metadata."""
    metrics = {
        f"{ReportedRegion.EAT_TOTAL} Volume (cm3)": 105.0,
        f"{ReportedRegion.EAT_TOTAL} Mean Attenuation (HU)": -76.5,
        f"{ReportedRegion.PVAT_LAD} Volume (cm3)": 7.5,
        f"{ReportedRegion.PVAT_LAD} Mean Attenuation (HU)": -68.0,
        f"{ReportedRegion.PVAT_TOTAL} Volume (cm3)": 18.2,
    }

    dummy_mpr = Image.new("RGBA", (300, 100), (10, 20, 40, 255))

    # 1. HTML generation
    html = generate_clinical_report_html(
        metrics=metrics,
        participant_id="PATIENT_TEST_01",
        mpr_image=dummy_mpr,
        ohif_viewer_url="http://127.0.0.1:8000/patient/PATIENT_TEST_01/",
    )
    assert "PATIENT_TEST_01" in html
    assert "CRISP-CT" in html
    assert "Multi-Planar Reconstruction" in html
    assert "data:image/png;base64," in html

    # 2. PDF generation
    pdf_bytes = generate_clinical_report_pdf(
        metrics=metrics,
        participant_id="PATIENT_TEST_01",
        mpr_image=dummy_mpr,
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 5000  # Valid substantial PDF document


def test_encapsulate_pdf_to_dicom(tmp_path: Path) -> None:
    """Test standard Encapsulated PDF Storage DICOM object creation (SOP .104.1)."""
    dummy_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    dcm_out = tmp_path / "eat_pvat_report.dcm"

    result_path = encapsulate_pdf_to_dicom(
        pdf_bytes=dummy_pdf,
        output_dcm_path=dcm_out,
        participant_id="PID_12345",
        document_title="Cardiac Fat AI Report",
    )

    assert result_path.exists()
    ds = pydicom.dcmread(str(result_path))
    assert ds.SOPClassUID == EncapsulatedPDFStorage
    assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.104.1"
    assert ds.Modality == "DOC"
    assert ds.MIMETypeOfEncapsulatedDocument == "application/pdf"
    assert ds.PatientID == "PID_12345"
    assert ds.EncapsulatedDocument.startswith(dummy_pdf)


def test_export_clinical_imaging_report_end_to_end(synthetic_ct_fixture: tuple, tmp_path: Path) -> None:
    """Test full bundle export (HTML, PDF, DICOM Encapsulated PDF, and MPR PNG)."""
    ct_vol, _, _, fat_mask = synthetic_ct_fixture
    d, h, w = ct_vol.shape
    mask_8ch = np.zeros((8, d, h, w), dtype=np.uint8)
    mask_8ch[0] = fat_mask.astype(np.uint8)

    metrics = {
        f"{ReportedRegion.EAT_TOTAL} Volume (cm3)": 92.4,
        f"{ReportedRegion.EAT_TOTAL} Mean Attenuation (HU)": -79.1,
        f"{ReportedRegion.PVAT_LAD} Volume (cm3)": 5.4,
        f"{ReportedRegion.PVAT_LAD} Mean Attenuation (HU)": -74.2,
    }

    out_dir = tmp_path / "clinical_bundle"
    artifacts = export_clinical_imaging_report(
        output_dir=out_dir,
        metrics=metrics,
        image=ct_vol,
        mask=mask_8ch,
        participant_id="STUDY_999",
    )

    assert artifacts["report_html"].exists()
    assert artifacts["report_pdf"].exists()
    assert artifacts["report_dcm"].exists()
    assert artifacts["mpr_png"].exists()

    # Verify DICOM is valid Encapsulated PDF
    ds = pydicom.dcmread(str(artifacts["report_dcm"]))
    assert ds.SOPClassUID == EncapsulatedPDFStorage


def test_cli_report_command(tmp_path: Path) -> None:
    """Test CLI invocation of cardiac-fat-fl report command."""
    # Setup mock study dir with measurements.json
    study_dir = tmp_path / "study_0506"
    study_dir.mkdir()
    metrics_file = study_dir / "cardiac_fat_measurements.json"
    metrics_file.write_text('{"EAT_Total Volume (cm3)": 85.0, "EAT_Total Mean Attenuation (HU)": -80.0}', encoding="utf-8")

    res = runner.invoke(app, ["report", str(study_dir), "--no-browser"])
    assert res.exit_code == 0, f"CLI report command failed: {res.output}"
    assert "Clinical-Grade Cardiovascular AI Diagnostic Report" in res.output
    assert (study_dir / "cardiac_fat_report.html").exists()
    assert (study_dir / "cardiac_fat_report.pdf").exists()
    assert (study_dir / "eat_pvat_report.dcm").exists()
