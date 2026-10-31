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

