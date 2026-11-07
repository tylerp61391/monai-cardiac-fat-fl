"""Unit tests for DICOM De-identification, Pre-Flight PHI Audit & Dual-Mode Operations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
import pytest
from typer.testing import CliRunner

from cardiac_fat_fl.anonymizer import (
    audit_study_anonymization,
    anonymize_dicom_series,
    generate_rsna_v19_project_config,
)
from cardiac_fat_fl.cli import app

runner = CliRunner()


def _create_synthetic_dicom_slice(
    path: Path,
    patient_name: str = "DOE^JOHN",
    patient_id: str = "MRN12345",
    institution: str = "STANFORD_HOSPITAL",
    physician: str = "DR^SMITH",
    include_geometry: bool = True,
) -> Path:
    """Helper to create a synthetic DICOM CT slice with controlled PHI and geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Identifiers
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19600101"
    ds.InstitutionName = institution
    ds.ReferringPhysicianName = physician
    ds.StudyDate = "20240115"
    ds.SeriesDate = "20240115"
    ds.AccessionNumber = "ACC998877"

    # Geometry & HU Calibration
    if include_geometry:
        ds.PixelSpacing = [0.47, 0.47]
        ds.SliceThickness = 3.0
        ds.ImagePositionPatient = [-120.0, -120.0, 50.0]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.RescaleIntercept = -1024.0
        ds.RescaleSlope = 1.0
        ds.KVP = 120.0
        ds.ConvolutionKernel = "STANDARD"
        ds.PatientSex = "M"
        ds.PatientAge = "064Y"

    # Minimal 16x16 pixel array
    pixels = np.zeros((16, 16), dtype=np.uint16)
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixels.tobytes()

    ds.save_as(str(path), enforce_file_format=True)
    return path


def test_audit_identified_study_fails_research_mode(tmp_path: Path) -> None:
    """An identified study with real patient names and MRNs should fail research mode audit."""
    study_dir = tmp_path / "study_raw"
    _create_synthetic_dicom_slice(study_dir / "slice_0001.dcm", patient_name="DOE^JANE", patient_id="HOSP999")

    report = audit_study_anonymization(study_dir, mode="research")

    assert not report.is_anonymized
    assert report.phi_leaks_found >= 1
    assert any(f.tag_name == "PatientName" for f in report.findings)
    assert any(f.tag_name == "InstitutionName" for f in report.findings)
    assert report.geometry_defects_found == 0


def test_audit_identified_study_passes_clinical_mode(tmp_path: Path) -> None:
    """In clinical mode, identified data is expected; only geometry defects cause failure."""
    study_dir = tmp_path / "study_clinical"
    _create_synthetic_dicom_slice(study_dir / "slice_0001.dcm", patient_name="DOE^JOHN", patient_id="MRN123")

    report = audit_study_anonymization(study_dir, mode="clinical")

    assert report.is_anonymized
    assert report.geometry_defects_found == 0


def test_audit_detects_geometry_defects(tmp_path: Path) -> None:
    """A scan stripped of PixelSpacing or HU RescaleIntercept should be flagged."""
    study_dir = tmp_path / "study_broken_geom"
    _create_synthetic_dicom_slice(
        study_dir / "slice_0001.dcm",
        patient_name="Participant 01",
        patient_id="01",
        include_geometry=False,
    )

    report = audit_study_anonymization(study_dir, mode="research")

    assert not report.is_anonymized
    assert report.geometry_defects_found >= 1
    assert any(f.tag_name == "PixelSpacing" for f in report.findings)
    assert any(f.tag_name == "RescaleIntercept" for f in report.findings)


def test_anonymize_dicom_series_native_and_audit(tmp_path: Path) -> None:
    """Native anonymization should scrub all PHI while strictly preserving 3D geometry."""
    raw_dir = tmp_path / "raw_study"
    out_dir = tmp_path / "anon_study"

    _create_synthetic_dicom_slice(
        raw_dir / "slice_01.dcm",
        patient_name="SMITH^ALICE",
        patient_id="MRN_98765",
        institution="MAYO_CLINIC",
        physician="DR^HOUSE",
    )

    # Execute native de-identification
    anonymize_dicom_series(
        input_dir=raw_dir,
        output_dir=out_dir,
        pseudonym_id="P506",
        date_shift_days=-100,
    )

    # Verify output file exists
    anon_files = list(out_dir.glob("*.dcm"))
    assert len(anon_files) == 1

    ds = pydicom.dcmread(str(anon_files[0]))
    assert ds.PatientID == "P506"
    assert ds.PatientName == "Participant P506"
    assert ds.PatientBirthDate == ""
    assert ds.InstitutionName == "RESEARCH_COHORT"
    assert ds.ReferringPhysicianName == ""

    # Verify 3D geometry and HU calibration are 100% intact!
    assert ds.PixelSpacing == [0.47, 0.47]
    assert ds.SliceThickness == 3.0
    assert ds.RescaleIntercept == -1024.0
    assert ds.RescaleSlope == 1.0
    assert ds.KVP == 120.0

    # Audit the output directory: MUST PASS in research mode!
    report = audit_study_anonymization(out_dir, mode="research")
    assert report.is_anonymized
    assert report.phi_leaks_found == 0
    assert report.geometry_defects_found == 0


def test_generate_rsna_v19_project_config(tmp_path: Path) -> None:
    """Verify RSNA V19 ProjectModel.json generation and configuration rules."""
    cfg_path = tmp_path / "ProjectModel.json"
    generate_rsna_v19_project_config(
        project_name="StanfordCOCA_Research",
        site_id="STANFORD",
        storage_dir=tmp_path / "storage",
        output_config_path=cfg_path,
    )

    assert cfg_path.exists()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["version"] == "19.0.10"
    assert data["project_name"] == "StanfordCOCA_Research"
    assert data["site_id"] == "STANFORD"
    assert data["rules"]["retain_patient_characteristics"] is True
    assert data["rules"]["retain_device_identity"] is True
    assert data["rules"]["retain_longitudinal_temporal_information"] is True
    assert "(0028,0030)" in data["rules"]["mandatory_preserved_elements"]
    assert "(0028,1052)" in data["rules"]["mandatory_preserved_elements"]
    assert data["ai_batch_config"]["enable_pixel_ocr"] is False


def test_cli_deid_audit_and_anonymize(tmp_path: Path) -> None:
    """Test CLI commands for deid-audit and anonymize."""
    raw_dir = tmp_path / "cli_raw"
    anon_dir = tmp_path / "cli_anon"
    _create_synthetic_dicom_slice(raw_dir / "slice_01.dcm", patient_name="BROWN^CHARLIE", patient_id="PEANUTS01")

    # 1. deid-audit in research mode (fails due to PHI leak)
    result = runner.invoke(app, ["deid-audit", str(raw_dir), "--mode", "research"])
    assert result.exit_code != 0
    assert "FAIL" in result.stdout

    # 2. deid-audit in clinical mode (passes because clinical allows identified data)
    result_clin = runner.invoke(app, ["deid-audit", str(raw_dir), "--mode", "clinical"])
    assert result_clin.exit_code == 0
    assert "PASS" in result_clin.stdout

    # 3. anonymize command
    res_anon = runner.invoke(app, ["anonymize", str(raw_dir), "-o", str(anon_dir), "--id", "COCA_001"])
    assert res_anon.exit_code == 0
    assert "De-identification complete" in res_anon.stdout

    # 4. deid-audit on anonymized directory (passes in research mode!)
    res_audit_clean = runner.invoke(app, ["deid-audit", str(anon_dir), "--mode", "research"])
    assert res_audit_clean.exit_code == 0
    assert "PASS" in res_audit_clean.stdout
