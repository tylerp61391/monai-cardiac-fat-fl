"""Unit and Integration Tests for HL7 FHIR R4 Mock Server & Sandbox."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from cardiac_fat_fl.cli import app as cli_app
from cardiac_fat_fl.dashboard import create_dashboard_app
from cardiac_fat_fl.fhir_client import (
    build_cardiac_fat_fhir_bundle,
    push_to_fhir_server,
)
from cardiac_fat_fl.fhir_server import (
    FHIRStore,
    create_fhir_app,
    get_default_fhir_store,
)


@pytest.fixture
def fhir_client(tmp_path: Path):
    """Fixture providing a test client for the standalone FHIR app with isolated store."""
    app = create_fhir_app(storage_dir=tmp_path / "fhir_storage")
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def sample_metrics():
    """19-column clinical metrics dictionary fixture."""
    return {
        "participant_id": "0506",
        "EAT Total Volume (cm3)": 112.5,
        "EAT Total Mean Attenuation (HU)": -82.4,
        "EAT LA Volume (cm3)": 24.1,
        "EAT LA Mean Attenuation (HU)": -79.3,
        "EAT LV Volume (cm3)": 38.6,
        "EAT LV Mean Attenuation (HU)": -84.2,
        "EAT RA Volume (cm3)": 18.2,
        "EAT RA Mean Attenuation (HU)": -80.1,
        "EAT RV Volume (cm3)": 31.6,
        "EAT RV Mean Attenuation (HU)": -83.5,
        "PVAT LAD Volume (cm3)": 14.8,
        "PVAT LAD Mean Attenuation (HU)": -66.4,  # Inflamed (> -70.1)
        "PVAT LCX Volume (cm3)": 10.2,
        "PVAT LCX Mean Attenuation (HU)": -74.1,
        "PVAT RCA Volume (cm3)": 16.5,
        "PVAT RCA Mean Attenuation (HU)": -71.8,
        "PVAT Total Volume (cm3)": 41.5,
        "PVAT Total Mean Attenuation (HU)": -69.2,
    }


def test_fhir_capability_statement(fhir_client):
    """Verify GET /r4/metadata returns valid CapabilityStatement conforming to US Core."""
    res = fhir_client.get("/r4/metadata")
    assert res.status_code == 200
    assert "application/fhir+json" in res.content_type
    data = res.get_json()
    assert data["resourceType"] == "CapabilityStatement"
    assert data["fhirVersion"] == "4.0.1"
    assert data["status"] == "active"

    rest = data["rest"][0]
    res_types = [r["type"] for r in rest["resource"]]
    assert "Observation" in res_types
    assert "DiagnosticReport" in res_types
    assert "DocumentReference" in res_types
    assert "Patient" in res_types


def test_transaction_bundle_ingestion(fhir_client, sample_metrics):
    """Verify POST /r4 and /r4/Bundle ingests atomic transaction bundle and rewrites references."""
    bundle = build_cardiac_fat_fhir_bundle(
        metrics=sample_metrics,
        patient_id="0506",
        bundle_type="transaction",
    )

    res = fhir_client.post(
        "/r4",
        json=bundle,
        headers={"Content-Type": "application/fhir+json"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["resourceType"] == "Bundle"
    assert data["type"] == "transaction-response"
    assert len(data["entry"]) == len(bundle["entry"])

    for entry in data["entry"]:
        resp = entry.get("response", {})
        assert resp.get("status") == "201 Created"
        assert "location" in resp


def test_search_observations_by_patient_and_code(fhir_client, sample_metrics):
    """Verify search for ingested observations by patient and LOINC code."""
    bundle = build_cardiac_fat_fhir_bundle(
        metrics=sample_metrics,
        patient_id="0506",
        bundle_type="transaction",
    )
    res_ingest = fhir_client.post("/r4", json=bundle)
    assert res_ingest.status_code == 200

    # Search by patient ID
    res_search = fhir_client.get("/r4/Observation?patient=0506")
    assert res_search.status_code == 200
    searchset = res_search.get_json()
    assert searchset["resourceType"] == "Bundle"
    assert searchset["type"] == "searchset"
    assert searchset["total"] >= 2

    # Search by LOINC 84245-0 (EAT volume)
    res_code = fhir_client.get("/r4/Observation?code=84245-0")
    assert res_code.status_code == 200
    code_set = res_code.get_json()
    assert code_set["total"] >= 1
    eat_obs = code_set["entry"][0]["resource"]
    assert eat_obs["valueQuantity"]["value"] == 112.5
    assert eat_obs["valueQuantity"]["unit"] == "cm3"


def test_search_diagnostic_report_and_reference_resolution(fhir_client, sample_metrics):
    """Verify DiagnosticReport is retrievable and urn:uuid references were resolved."""
    bundle = build_cardiac_fat_fhir_bundle(
        metrics=sample_metrics,
        patient_id="0506",
        bundle_type="transaction",
    )
    fhir_client.post("/r4", json=bundle)

    res = fhir_client.get("/r4/DiagnosticReport?patient=0506")
    assert res.status_code == 200
    reports = res.get_json()
    assert reports["total"] >= 1
    rep = reports["entry"][0]["resource"]
    assert rep["resourceType"] == "DiagnosticReport"
    assert rep["status"] == "final"
    assert "Active Vascular Inflammation" in rep["conclusion"]

    # Verify result references no longer contain urn:uuid
    for result_ref in rep.get("result", []):
        ref_str = result_ref.get("reference", "")
        assert not ref_str.startswith("urn:uuid:")
        assert ref_str.startswith("Observation/")


def test_discrete_resource_crud_lifecycle(fhir_client):
    """Verify RESTful create, read, and delete on discrete Observation resources."""
    obs = {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "84245-0",
                    "display": "Epicardial adipose tissue volume",
                }
            ]
        },
        "subject": {"reference": "Patient/TEST_PATIENT"},
        "valueQuantity": {"value": 88.0, "unit": "cm3"},
    }

    # 1. Create
    res_post = fhir_client.post("/r4/Observation", json=obs)
    assert res_post.status_code == 201
    created = res_post.get_json()
    obs_id = created["id"]
    assert "Location" in res_post.headers
    assert res_post.headers["Location"] == f"Observation/{obs_id}"

    # 2. Read
    res_get = fhir_client.get(f"/r4/Observation/{obs_id}")
    assert res_get.status_code == 200
    assert res_get.get_json()["valueQuantity"]["value"] == 88.0

    # 3. Delete
    res_del = fhir_client.delete(f"/r4/Observation/{obs_id}")
    assert res_del.status_code == 204

    # 4. Verify Not Found (404 with OperationOutcome)
    res_del_get = fhir_client.get(f"/r4/Observation/{obs_id}")
    assert res_del_get.status_code == 404
    outcome = res_del_get.get_json()
    assert outcome["resourceType"] == "OperationOutcome"
    assert outcome["issue"][0]["code"] == "not-found"


def test_invalid_resource_operation_outcome(fhir_client):
    """Verify OperationOutcome response on malformed payloads."""
    # Bad payload (not a dict)
    res_bad = fhir_client.post("/r4/Observation", data="not json")
    assert res_bad.status_code == 400
    data = res_bad.get_json()
    assert data["resourceType"] == "OperationOutcome"

    # Unsupported resource endpoint
    res_unsupported = fhir_client.post("/r4/UnsupportedType", json={"foo": "bar"})
    assert res_unsupported.status_code == 400


def test_sandbox_summary_and_reset(fhir_client, sample_metrics):
    """Verify /r4/_sandbox inspection and /r4/_reset endpoints."""
    bundle = build_cardiac_fat_fhir_bundle(
        metrics=sample_metrics,
        patient_id="0506",
        bundle_type="transaction",
    )
    fhir_client.post("/r4", json=bundle)

    # Inspect summary
    res_sum = fhir_client.get("/r4/_sandbox")
    assert res_sum.status_code == 200
    summary = res_sum.get_json()
    assert summary["status"] == "active"
    assert summary["total_resources"] > 0
    assert "0506" in summary["patients"]

    # Reset
    res_reset = fhir_client.post("/r4/_reset")
    assert res_reset.status_code == 200
    assert res_reset.get_json()["status"] == "reset"

    # Inspect summary after reset
    res_sum2 = fhir_client.get("/r4/_sandbox")
    assert res_sum2.get_json()["total_resources"] == 0
    assert len(res_sum2.get_json()["patients"]) == 0


def test_dashboard_mounted_fhir_sandbox(synthetic_cohort_dir):
    """Verify FHIR sandbox blueprint is mounted directly in dashboard app at /r4."""
    app_dash = create_dashboard_app(cohort_dir=synthetic_cohort_dir)
    client = app_dash.test_client()

    # Capability statement on dashboard
    res_meta = client.get("/r4/metadata")
    assert res_meta.status_code == 200
    assert res_meta.get_json()["resourceType"] == "CapabilityStatement"

    # Ingest bundle into dashboard FHIR sandbox
    metrics = {
        "participant_id": "PATIENT_001",
        "EAT Total Volume (cm3)": 95.0,
        "PVAT LAD Mean Attenuation (HU)": -68.5,
    }
    b = build_cardiac_fat_fhir_bundle(metrics, "PATIENT_001")
    res_ingest = client.post("/r4", json=b)
    assert res_ingest.status_code == 200

    # Query participant FHIR sandbox endpoint
    res_box = client.get("/api/case/PATIENT_001/fhir_sandbox")
    assert res_box.status_code == 200
    box_data = res_box.get_json()
    assert box_data["participant_id"] == "PATIENT_001"
    assert box_data["total_ingested"] >= 2


def test_cli_fhir_server_help():
    """Verify cardiac-fat-fl fhir-server --help CLI invocation."""
    runner = CliRunner()
    result = runner.invoke(cli_app, ["fhir-server", "--help"])
    assert result.exit_code == 0
    assert "Launch the self-contained HL7 FHIR R4 Mock Server" in result.output
