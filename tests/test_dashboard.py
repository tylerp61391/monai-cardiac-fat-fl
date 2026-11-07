"""Unit and Integration Tests for Cardiac Research Dashboard Server."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from cardiac_fat_fl.dashboard import create_dashboard_app


@pytest.fixture
def test_app():
    cohort_dir = Path("example_source_dir")
    app = create_dashboard_app(cohort_dir=cohort_dir)
    app.config["TESTING"] = True
    return app


def test_dashboard_index_route(test_app):
    """Verify dashboard index serves HTML template with 200 OK."""
    client = test_app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    assert b"Cardiac Fat Research Dashboard" in res.data
    assert b"Chart.js" in res.data or b"chart.js" in res.data


def test_dashboard_api_cohort_summary(test_app):
    """Verify /api/cohort/summary endpoint returns valid JSON."""
    client = test_app.test_client()
    res = client.get("/api/cohort/summary")
    assert res.status_code == 200
    data = res.get_json()
    assert "total_count" in data
    assert "segmented_count" in data
    assert data["total_count"] >= 3


def test_dashboard_api_cohort_scatter(test_app):
    """Verify /api/cohort/scatter endpoint."""
    client = test_app.test_client()
    res = client.get("/api/cohort/scatter")
    assert res.status_code == 200
    pts = res.get_json()
    assert isinstance(pts, list)
    assert len(pts) >= 3


def test_dashboard_api_cohort_cases_and_filters(test_app):
    """Verify /api/cohort/cases endpoint with filters."""
    client = test_app.test_client()
    res = client.get("/api/cohort/cases")
    assert res.status_code == 200
    cases = res.get_json()
    assert len(cases) >= 3

    # Filter by search
    res_search = client.get("/api/cohort/cases?search=0506")
    cases_search = res_search.get_json()
    assert len(cases_search) == 1
    assert cases_search[0]["participant_id"] == "0506"


def test_dashboard_api_case_details(test_app):
    """Verify /api/case/<pid> returns detailed participant metrics."""
    client = test_app.test_client()
    res = client.get("/api/case/0506")
    assert res.status_code == 200
    data = res.get_json()
    assert data["participant_id"] == "0506"
    assert "metrics" in data
    assert "EAT Total Volume (cm3)" in data["metrics"]

    # 404 for non-existent case
    res_404 = client.get("/api/case/NON_EXISTENT")
    assert res_404.status_code == 404


def test_dashboard_set_directory(test_app, tmp_path: Path):
    """Verify switching cohort directory dynamically."""
    client = test_app.test_client()
    res = client.post(
        "/api/cohort/set_directory",
        json={"cohort_dir": str(tmp_path)},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["total_count"] == 0

def test_dashboard_ohif_viewer_routes(test_app):
    """Verify /patient/<pid>/ and /viewer/<pid>/ routes serve OHIF workstation."""
    client = test_app.test_client()
    # Test redirect without trailing slash
    res_redir = client.get("/patient/0506")
    assert res_redir.status_code == 302
    assert res_redir.headers["Location"].endswith("/patient/0506/")

    # Test viewer index
    res_index = client.get("/patient/0506/")
    assert res_index.status_code == 200
    assert b"<!DOCTYPE html>" in res_index.data or b"OHIF" in res_index.data

    # Test viewer subfile (studies.json)
    res_studies = client.get("/patient/0506/studies.json")
    assert res_studies.status_code == 200

    # 404 for non-existent participant
    res_404 = client.get("/patient/NON_EXISTENT/")
    assert res_404.status_code == 404


def test_dashboard_api_statistics(test_app):
    """Verify /api/cohort/statistics returns ANOVA, Kruskal-Wallis, and bivariate regression."""
    client = test_app.test_client()
    res = client.get("/api/cohort/statistics")
    assert res.status_code == 200
    data = res.get_json()
    assert "tier_group_comparisons" in data
    assert "bivariate_regression" in data
    assert "CAC_TIERS" in data["tier_group_comparisons"]
    assert "eat_volume" in data["tier_group_comparisons"]
    assert "pvat_attenuation" in data["tier_group_comparisons"]


def test_dashboard_api_publication_figure(test_app):
    """Verify /api/cohort/publication-figure returns valid 300 DPI PNG attachment."""
    client = test_app.test_client()
    res = client.get("/api/cohort/publication-figure?dpi=150")
    assert res.status_code == 200
    assert res.content_type == "image/png"
    assert len(res.data) > 1000
    # Check PNG magic bytes: \x89PNG\r\n\x1a\n
    assert res.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_dashboard_federated_html_elements(test_app):
    """Verify federated learning UI tab, section, and controls are rendered in HTML."""
    client = test_app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert 'id="tab-federated"' in html
    assert 'id="section-federated"' in html
    assert 'id="flConvergenceChart"' in html
    assert 'id="flSitesGrid"' in html
    assert 'id="flModelsTableBody"' in html
    assert 'id="flConsoleLog"' in html


def test_dashboard_federated_status_api(test_app):
    """Verify /api/federated/status endpoint returns valid state schema."""
    client = test_app.test_client()
    res = client.get("/api/federated/status")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert "config" in data
    assert "client_states" in data
    assert "metrics" in data
    assert "persisted_models" in data
    assert "logs" in data
    assert data["status"] in ("idle", "running", "completed", "failed")


def test_dashboard_federated_models_api(test_app):
    """Verify /api/federated/models endpoint returns list of models."""
    client = test_app.test_client()
    res = client.get("/api/federated/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_dashboard_federated_reset_api(test_app):
    """Verify /api/federated/reset endpoint resets status to idle."""
    client = test_app.test_client()
    res = client.post("/api/federated/reset")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "idle"


def test_dashboard_federated_download_security(test_app):
    """Verify security controls on /api/federated/download/<filename>."""
    client = test_app.test_client()
    # Bad extension
    res_bad_ext = client.get("/api/federated/download/exploit.exe")
    assert res_bad_ext.status_code == 400

    # Path traversal attempt
    res_traversal = client.get("/api/federated/download/..%2F..%2Fsecret.pt")
    assert res_traversal.status_code in (400, 404)

    # Non-existent valid file
    res_404 = client.get("/api/federated/download/nonexistent_model.pt")
    assert res_404.status_code == 404


def test_dashboard_federated_download_success(test_app):
    """Verify downloading a valid federated model checkpoint."""
    models_dir = Path("models/federated")
    models_dir.mkdir(parents=True, exist_ok=True)
    test_pt = models_dir / "test_checkpoint.pt"
    test_pt.write_bytes(b"dummy_pt_weights")
    try:
        client = test_app.test_client()
        res = client.get("/api/federated/download/test_checkpoint.pt")
        assert res.status_code == 200
        assert res.data == b"dummy_pt_weights"
        assert "test_checkpoint.pt" in res.headers.get("Content-Disposition", "")
        res.close()
    finally:
        if test_pt.exists():
            test_pt.unlink()


def test_dashboard_federated_run_trigger(test_app, monkeypatch):
    """Verify /api/federated/run triggers background simulation."""
    import cardiac_fat_fl.dashboard as dash_mod

    called = {}

    def mock_execute(config):
        called["config"] = config

    monkeypatch.setattr(dash_mod, "_execute_fl_simulation", mock_execute)

    client = test_app.test_client()

    # Post simulation launch
    res = client.post(
        "/api/federated/run",
        json={"n_clients": 2, "num_rounds": 1, "threads": 2, "fedprox_mu": 0.01},
    )
    assert res.status_code == 202
    data = res.get_json()
    assert data["status"] == "started"
    assert "config" in data
    assert data["config"]["n_clients"] == 2
    assert called["config"]["n_clients"] == 2


def test_dashboard_api_case_vlm_narrative(test_app, monkeypatch):
    """Verify /api/case/<pid>/vlm_narrative endpoint."""
    import cardiac_fat_fl.nv_reason_reporter as reporter_mod

    client = test_app.test_client()

    # 1. 404 for unknown participant
    res_404 = client.post("/api/case/NON_EXISTENT/vlm_narrative")
    assert res_404.status_code == 404

    # 2. 503 if NV-Reason is not available
    monkeypatch.setattr(reporter_mod, "is_nv_reason_available", lambda: False)
    res_503 = client.post("/api/case/0506/vlm_narrative")
    assert res_503.status_code == 503
    assert "not available" in res_503.get_json()["error"]

    # 3. 200 with mocked narrative
    monkeypatch.setattr(reporter_mod, "is_nv_reason_available", lambda: True)
    monkeypatch.setattr(
        reporter_mod,
        "generate_nv_reason_narrative",
        lambda ct_path, metrics, mode: f"IMPRESSION: Mild plaque in mode {mode}",
    )
    res_200 = client.post("/api/case/0506/vlm_narrative?mode=structured")
    assert res_200.status_code == 200
    data = res_200.get_json()
    assert data["participant_id"] == "0506"
    assert data["mode"] == "structured"
    assert "Mild plaque in mode structured" in data["narrative"]


def test_dashboard_api_case_vlm_narrative_cached(test_app):
    """Verify GET /api/case/<pid>/vlm_narrative retrieves cached narrative."""
    client = test_app.test_client()
    res = client.get("/api/case/0506/vlm_narrative")
    assert res.status_code == 200
    data = res.get_json()
    assert data["participant_id"] == "0506"
    assert data["cached"] is True
    assert data["narrative"] is not None


def test_dashboard_report_routes(test_app):
    """Verify PDF, HTML, and DICOM report serving routes."""
    client = test_app.test_client()

    # 1. 200 OK for existing case 0506
    res_pdf = client.get("/report/0506/pdf")
    assert res_pdf.status_code == 200
    assert res_pdf.mimetype == "application/pdf"

    res_html = client.get("/report/0506/html")
    assert res_html.status_code == 200
    assert "text/html" in res_html.mimetype

    res_dcm = client.get("/report/0506/dicom")
    assert res_dcm.status_code == 200
    assert res_dcm.mimetype == "application/dicom"

    # 2. 404 for unknown participant
    res_404 = client.get("/report/UNKNOWN_PID/pdf")
    assert res_404.status_code == 404


def test_dashboard_privacy_and_regulatory_routes(test_app):
    """Verify Privacy Gate PHI/Geometry audit, RSNA V19 config download, and SaMD blueprint routes."""
    client = test_app.test_client()

    # 1. Privacy Audit in research mode
    res_audit = client.post("/api/privacy/audit", json={"mode": "research", "max_files": 50})
    assert res_audit.status_code == 200
    data = res_audit.get_json()
    assert data["mode_evaluated"] == "research"
    assert data["is_anonymized"] is True
    assert data["phi_leaks_found"] == 0
    assert data["geometry_defects_found"] == 0
    assert len(data["mandatory_geometry_tags"]) == 6

    # 2. RSNA V19 Config Download
    res_cfg = client.get("/api/privacy/rsna-config?project_name=TestCohort&site_id=MGH01")
    assert res_cfg.status_code == 200
    assert res_cfg.mimetype == "application/json"
    cfg_json = res_cfg.get_json()
    assert cfg_json["version"] == "19.0.10"
    assert cfg_json["project_name"] == "TestCohort"
    assert cfg_json["site_id"] == "MGH01"

    # 3. SaMD Regulatory Blueprint routes
    res_bp_html = client.get("/api/regulatory/blueprint?format=html")
    assert res_bp_html.status_code == 200
    res_bp_pdf = client.get("/api/regulatory/blueprint?format=pdf")
    assert res_bp_pdf.status_code == 200
    assert res_bp_pdf.mimetype == "application/pdf"
    res_bp_png = client.get("/api/regulatory/blueprint?format=png")
    assert res_bp_png.status_code == 200
    assert res_bp_png.mimetype == "image/png"

    # 4. FDA 510(k) DHF Evidence Dossier routes (PDF & JSON)
    res_dhf_json = client.get("/api/regulatory/dhf?format=json")
    assert res_dhf_json.status_code == 200
    assert res_dhf_json.mimetype == "application/json"
    dhf_data = res_dhf_json.get_json()
    assert len(dhf_data["iec_62304_traceability_matrix"]) == 13
    assert len(dhf_data["iso_14971_risk_controls"]) == 6

    res_dhf_pdf = client.get("/api/regulatory/dhf?format=pdf")
    assert res_dhf_pdf.status_code == 200
    assert res_dhf_pdf.mimetype == "application/pdf"





