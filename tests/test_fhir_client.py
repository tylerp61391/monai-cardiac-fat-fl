"""Unit and integration tests for HL7 FHIR R4 Client & Interoperability Bundle Generator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cardiac_fat_fl.cli import app
from cardiac_fat_fl.dashboard import create_dashboard_app
from cardiac_fat_fl.fhir_client import (
    CRISP_CT_INFLAMMATION_HU_THRESHOLD,
    build_cardiac_fat_diagnostic_report,
    build_cardiac_fat_document_reference,
    build_cardiac_fat_fhir_bundle,
    build_eat_volume_observation,
    build_pvat_attenuation_observation,
    create_bundle_for_study,
    export_fhir_bundle,
    push_to_fhir_server,
)
from cardiac_fat_fl.measurement_schema import ReportedRegion

runner = CliRunner()


def test_build_eat_volume_observation():
    # Normal volume (< 100 cm3)
    obs_norm = build_eat_volume_observation(85.5, patient_id="PATIENT_001")
    assert obs_norm["resourceType"] == "Observation"
    assert obs_norm["status"] == "final"
    assert obs_norm["code"]["coding"][0]["code"] == "84245-0"
    assert obs_norm["code"]["coding"][0]["system"] == "http://loinc.org"
    assert obs_norm["valueQuantity"]["value"] == 85.5
    assert obs_norm["valueQuantity"]["code"] == "cm3"
    assert obs_norm["subject"]["reference"] == "Patient/PATIENT_001"
    assert obs_norm["interpretation"][0]["coding"][0]["code"] == "N"

    # High risk volume (> 125 cm3)
    obs_high = build_eat_volume_observation(145.0, patient_id="PATIENT_002")
    assert obs_high["interpretation"][0]["coding"][0]["code"] == "H"


def test_build_pvat_attenuation_observation():
    # Normal attenuation (<= -70.1 HU)
    obs_norm = build_pvat_attenuation_observation(
        region_name=ReportedRegion.PVAT_LAD.value,
        pvat_mean_hu=-74.5,
        patient_id="PATIENT_001",
    )
    assert obs_norm["resourceType"] == "Observation"
    assert obs_norm["code"]["coding"][0]["code"] == "RID49533"
    assert obs_norm["bodySite"]["coding"][0]["code"] == "91724006"  # LAD SNOMED
    assert obs_norm["valueQuantity"]["value"] == -74.5
    assert obs_norm["valueQuantity"]["code"] == "[hnsf'U]"
    assert obs_norm["interpretation"][0]["coding"][0]["code"] == "N"

    # Inflamed attenuation (> -70.1 HU)
    obs_inflamed = build_pvat_attenuation_observation(
        region_name=ReportedRegion.PVAT_RCA.value,
        pvat_mean_hu=-65.2,
        patient_id="PATIENT_001",
    )
    assert obs_inflamed["bodySite"]["coding"][0]["code"] == "362070004"  # RCA SNOMED
    assert obs_inflamed["interpretation"][0]["coding"][0]["code"] == "H"
    assert "Active Vascular Inflammation" in obs_inflamed["interpretation"][0]["coding"][0]["display"]


def test_build_cardiac_fat_diagnostic_report():
    obs_refs = ["Observation/obs-1", "Observation/obs-2"]
    metrics = {
        f"{ReportedRegion.EAT_TOTAL.value} Volume (cm3)": 92.0,
        f"{ReportedRegion.PVAT_LAD.value} Mean Attenuation (HU)": -66.5,  # Inflamed
        f"{ReportedRegion.PVAT_LCX.value} Mean Attenuation (HU)": -67.0,  # Inflamed
    }
    report = build_cardiac_fat_diagnostic_report(
        metrics=metrics,
        patient_id="PATIENT_001",
        observation_references=obs_refs,
        pdf_bytes=b"%PDF-1.4 mock pdf content",
    )
    assert report["resourceType"] == "DiagnosticReport"
    assert report["status"] == "final"
    assert report["category"][0]["coding"][0]["code"] == "RAD"
    assert report["code"]["coding"][0]["code"] == "79070-9"
    assert len(report["result"]) == 2
    assert "CRISP-CT Active Vascular Inflammation detected in: LAD" in report["conclusion"]
    assert len(report["presentedForm"]) == 1
    assert report["presentedForm"][0]["contentType"] == "application/pdf"


def test_build_cardiac_fat_document_reference():
    doc_ref = build_cardiac_fat_document_reference(
        patient_id="PATIENT_001",
        pdf_bytes=b"%PDF-1.4 test report",
    )
    assert doc_ref["resourceType"] == "DocumentReference"
    assert doc_ref["status"] == "current"
    assert doc_ref["type"]["coding"][0]["code"] == "11526-1"
    assert doc_ref["content"][0]["attachment"]["contentType"] == "application/pdf"
    assert len(doc_ref["content"][0]["attachment"]["data"]) > 0


def test_build_cardiac_fat_fhir_bundle():
    metrics = {
        "EAT Total Volume (cm3)": 110.5,
        "PVAT LAD Mean Attenuation (HU)": -68.0,  # Inflamed
        "PVAT LCX Mean Attenuation (HU)": -73.0,  # Normal
        "PVAT RCA Mean Attenuation (HU)": -72.0,  # Normal
        "PVAT Total Mean Attenuation (HU)": -71.0,
    }
    bundle = build_cardiac_fat_fhir_bundle(
        metrics=metrics,
        patient_id="TEST_PID",
        bundle_type="transaction",
    )
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    entries = bundle["entry"]
    assert len(entries) >= 5  # 1 EAT obs + 4 PVAT obs + 1 DiagnosticReport

    # In transaction bundle, each entry must have a request block
    for entry in entries:
        assert "request" in entry
        assert entry["request"]["method"] == "POST"

    # Check DiagnosticReport references all observations
    report_entry = next(e for e in entries if e["resource"]["resourceType"] == "DiagnosticReport")
    res = report_entry["resource"]
    assert len(res["result"]) >= 4
    assert "CRISP-CT Active Vascular Inflammation detected in: LAD" in res["conclusion"]


def test_export_fhir_bundle(tmp_path: Path):
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": {"resourceType": "Observation", "id": "test-1"}}],
    }
    out_file = tmp_path / "subdir" / "test_bundle.json"
    saved = export_fhir_bundle(bundle, out_file)
    assert saved.exists()

    with open(saved, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["resourceType"] == "Bundle"
    assert len(loaded["entry"]) == 1


def test_push_to_fhir_server():
    bundle = {"resourceType": "Bundle", "type": "transaction", "entry": []}

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"resourceType": "Bundle", "type": "transaction-response"}
        mock_post.return_value = mock_resp

        res = push_to_fhir_server(
            bundle=bundle,
            fhir_base_url="https://fhir.hospital.org/r4",
            auth_token="test_bearer_token",
        )
        assert res["success"] is True
        assert res["status_code"] == 201
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test_bearer_token"
        assert kwargs["headers"]["Content-Type"] == "application/fhir+json; charset=utf-8"


def test_create_bundle_for_study(tmp_path: Path):
    study_dir = tmp_path / "study_001"
    study_dir.mkdir(parents=True)

    # Write dummy cardiac_fat_measurements.csv
    csv_path = study_dir / "cardiac_fat_measurements.csv"
    csv_path.write_text(
        "participant_id,EAT Total Volume (cm3),PVAT LAD Mean Attenuation (HU)\n"
        "study_001,88.4,-73.2\n"
    )

    # Write dummy PDF report
    pdf_path = study_dir / "cardiac_fat_report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test report pdf")

    bundle = create_bundle_for_study(study_dir)
    assert bundle["resourceType"] == "Bundle"
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Observation" in types
    assert "DiagnosticReport" in types
    assert "DocumentReference" in types


def test_cli_fhir_export(tmp_path: Path):
    study_dir = tmp_path / "case_abc"
    study_dir.mkdir(parents=True)
    csv_path = study_dir / "cardiac_fat_measurements.csv"
    csv_path.write_text(
        "participant_id,EAT Total Volume (cm3)\n"
        "case_abc,95.0\n"
    )
    dest_json = tmp_path / "out.json"

    result = runner.invoke(app, ["fhir-export", str(study_dir), "-o", str(dest_json)])
    assert result.exit_code == 0
    assert dest_json.is_file()
    with open(dest_json) as f:
        data = json.load(f)
    assert data["resourceType"] == "Bundle"


def test_cli_push_with_fhir(tmp_path: Path):
    study_dir = tmp_path / "case_xyz"
    study_dir.mkdir(parents=True)
    csv_path = study_dir / "cardiac_fat_measurements.csv"
    csv_path.write_text("participant_id,EAT Total Volume (cm3)\ncase_xyz,90.0\n")

    with patch("cardiac_fat_fl.fhir_client.push_to_fhir_server") as mock_push:
        mock_push.return_value = {"success": True, "status_code": 200, "entries_count": 2, "response": {}}
        result = runner.invoke(app, [
            "push",
            str(study_dir),
            "--fhir-url",
            "https://fhir.hospital.org/r4",
            "--fhir-token",
            "secret_token_123",
        ])
        assert result.exit_code == 0
        assert mock_push.called


def test_dashboard_case_fhir_endpoint(synthetic_cohort_dir):
    app_flask = create_dashboard_app(cohort_dir=synthetic_cohort_dir)
    client = app_flask.test_client()

    # Query FHIR JSON for synthetic participant 'PATIENT_001'
    res = client.get("/api/case/PATIENT_001/fhir")
    assert res.status_code == 200
    data = res.get_json()
    assert data["resourceType"] == "Bundle"
    assert data["type"] == "transaction"

    # Query FHIR download attachment
    dl_res = client.get("/api/case/PATIENT_001/fhir?download=true")
    assert dl_res.status_code == 200
    assert "application/fhir+json" in dl_res.content_type
    assert "attachment" in dl_res.headers.get("Content-Disposition", "")
