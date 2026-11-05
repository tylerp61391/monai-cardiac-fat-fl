"""Comprehensive unit and integration tests for the Epic on FHIR Model Context Protocol (MCP) Server."""

import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from cardiac_fat_fl.epic_mcp_server import EpicMCPServer, MCP_PROTOCOL_VERSION, SERVER_NAME
from cardiac_fat_fl.fhir_server import create_fhir_app, FHIRStore


@pytest.fixture
def test_fhir_server(tmp_path: Path):
    """Spin up a test Flask app with FHIRStore and return test client and server instance."""
    store = FHIRStore()
    app = create_fhir_app(store)
    app.config["TESTING"] = True
    client = app.test_client()
    return client, store


@pytest.fixture
def mcp_server():
    """Create an EpicMCPServer instance for testing."""
    return EpicMCPServer(fhir_base_url="http://mock-fhir.local/r4")


def test_mcp_initialize_and_ping(mcp_server: EpicMCPServer) -> None:
    """Test MCP protocol initialization handshake and ping."""
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = mcp_server.handle_message(init_req)
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == SERVER_NAME
    assert "tools" in result["capabilities"]

    # Initialized notification (must return None)
    notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert mcp_server.handle_message(notif) is None

    # Ping
    ping_req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    ping_resp = mcp_server.handle_message(ping_req)
    assert ping_resp is not None
    assert ping_resp["id"] == 2
    assert ping_resp["result"] == {}


def test_mcp_tools_list(mcp_server: EpicMCPServer) -> None:
    """Test tools/list returns all required clinical tools with valid schemas."""
    list_req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    resp = mcp_server.handle_message(list_req)
    assert resp is not None
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]

    expected = [
        "epic_get_patient_observations",
        "epic_get_diagnostic_report",
        "epic_create_diagnostic_report",
        "epic_create_document_reference",
        "cardiac_quantify_study",
        "cardiac_get_cohort_summary",
        "open_epic_lookup_code",
    ]
    for exp in expected:
        assert exp in tool_names, f"Tool '{exp}' must be declared in tools/list"

    # Verify tool schemas
    for tool in tools:
        assert "description" in tool
        assert "inputSchema" in tool
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema


def test_mcp_open_epic_lookup_code(mcp_server: EpicMCPServer) -> None:
    """Test open_epic_lookup_code tool for EAT and PVAT terminology."""
    call_req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "open_epic_lookup_code",
            "arguments": {"concept": "eat"},
        },
    }
    resp = mcp_server.handle_message(call_req)
    assert resp is not None
    assert resp["result"]["isError"] is False
    text = resp["result"]["content"][0]["text"]
    assert "84245-0" in text
    assert "Epicardial Adipose Tissue" in text
    assert "cm3" in text

    # PVAT lookup
    call_req["params"]["arguments"] = {"concept": "pvat"}
    resp = mcp_server.handle_message(call_req)
    text = resp["result"]["content"][0]["text"]
    assert "RID49533" in text
    assert "-70.1" in text

    # All concepts
    call_req["params"]["arguments"] = {"concept": "all"}
    resp = mcp_server.handle_message(call_req)
    text = resp["result"]["content"][0]["text"]
    assert "DiagnosticReport" in text


def test_mcp_patient_observations_and_diagnostic_reports(mcp_server: EpicMCPServer) -> None:
    """Test observation retrieval with mocked FHIR responses."""
    mock_bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{"code": "84245-0", "display": "Total EAT Volume"}],
                        "text": "Total EAT Volume",
                    },
                    "valueQuantity": {"value": 75.4, "unit": "cm3"},
                    "interpretation": [{"text": "Normal EAT Burden"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{"code": "RID49533", "display": "PVAT LAD Attenuation"}],
                        "text": "PVAT LAD Attenuation",
                    },
                    "valueQuantity": {"value": -62.5, "unit": "HU"},
                }
            },
        ],
    }

    with patch.object(mcp_server, "_query_fhir", return_value=(200, mock_bundle)):
        call_req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "epic_get_patient_observations",
                "arguments": {"patient_id": "0506"},
            },
        }
        resp = mcp_server.handle_message(call_req)
        assert resp is not None
        assert resp["result"]["isError"] is False
        text = resp["result"]["content"][0]["text"]
        assert "Total EAT Volume" in text
        assert "75.40 cm3" in text
        assert "ACTIVE INFLAMMATION" in text  # -62.5 HU > -70.1 HU

    # Test DiagnosticReport lookup
    mock_report_bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "rep-001",
                    "status": "final",
                    "effectiveDateTime": "2026-11-01T10:00:00Z",
                    "conclusion": "CRISP-CT Active Vascular Inflammation in LAD.",
                    "result": [{"reference": "Observation/obs-1"}],
                }
            }
        ],
    }
    with patch.object(mcp_server, "_query_fhir", return_value=(200, mock_report_bundle)):
        call_req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "epic_get_diagnostic_report",
                "arguments": {"patient_id": "0506"},
            },
        }
        resp = mcp_server.handle_message(call_req)
        assert resp is not None
        assert resp["result"]["isError"] is False
        text = resp["result"]["content"][0]["text"]
        assert "CRISP-CT Active Vascular Inflammation in LAD" in text
        assert "rep-001" in text


def test_mcp_create_report_and_document_reference(mcp_server: EpicMCPServer, tmp_path: Path) -> None:
    """Test creating DiagnosticReport and DocumentReference via MCP."""
    with patch.object(mcp_server, "_post_fhir", return_value=(201, {"id": "dr-999"})):
        call_req = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "epic_create_diagnostic_report",
                "arguments": {
                    "patient_id": "0506",
                    "conclusion": "Patient exhibits normal cardiac fat metrics.",
                    "status": "final",
                },
            },
        }
        resp = mcp_server.handle_message(call_req)
        assert resp is not None
        assert resp["result"]["isError"] is False
        text = resp["result"]["content"][0]["text"]
        assert "Successfully filed DiagnosticReport `dr-999`" in text

    # DocumentReference creation
    fake_pdf = tmp_path / "report.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 Mock PDF Content")

    with patch.object(mcp_server, "_post_fhir", return_value=(201, {"id": "doc-888"})):
        call_req = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "epic_create_document_reference",
                "arguments": {
                    "patient_id": "0506",
                    "file_path": str(fake_pdf),
                    "title": "Cardiac Fat Clinical Imaging Report",
                },
            },
        }
        resp = mcp_server.handle_message(call_req)
        assert resp is not None
        assert resp["result"]["isError"] is False
        text = resp["result"]["content"][0]["text"]
        assert "Successfully attached DocumentReference `doc-888`" in text


def test_mcp_cohort_summary_tool(mcp_server: EpicMCPServer) -> None:
    """Test cardiac_get_cohort_summary tool."""
    call_req = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "cardiac_get_cohort_summary",
            "arguments": {"cohort_dir": "example_source_dir"},
        },
    }
    resp = mcp_server.handle_message(call_req)
    assert resp is not None
    assert resp["result"]["isError"] is False
    text = resp["result"]["content"][0]["text"]
    assert "Cohort Research Summary" in text
    assert "Total Participants" in text


def test_mcp_cardiac_quantify_study_tool(mcp_server: EpicMCPServer, synthetic_cohort_dir: Path) -> None:
    """Test cardiac_quantify_study tool on a synthetic patient directory."""
    patient_dirs = sorted([d for d in synthetic_cohort_dir.iterdir() if d.is_dir()])
    assert len(patient_dirs) > 0
    patient_dir = patient_dirs[0]
    assert patient_dir.exists()

    call_req = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "cardiac_quantify_study",
            "arguments": {
                "study_path": str(patient_dir),
                "save_seg": True,
            },
        },
    }
    resp = mcp_server.handle_message(call_req)
    assert resp is not None
    assert resp["result"]["isError"] is False
    text = resp["result"]["content"][0]["text"]
    assert "Automated 3D Quantification Complete" in text
    assert "Total EAT Volume" in text
    assert (patient_dir / "eat_pvat.seg.nrrd").exists()


def test_mcp_unknown_method_and_error_handling(mcp_server: EpicMCPServer) -> None:
    """Test handling of unknown JSON-RPC methods and tool execution errors."""
    bad_req = {"jsonrpc": "2.0", "id": 10, "method": "non_existent_method"}
    resp = mcp_server.handle_message(bad_req)
    assert resp is not None
    assert "error" in resp
    assert resp["error"]["code"] == -32601

    # Call unknown tool
    call_req = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "unknown_tool",
            "arguments": {},
        },
    }
    resp = mcp_server.handle_message(call_req)
    assert resp is not None
    assert resp["result"]["isError"] is True
    assert "Unknown tool" in resp["result"]["content"][0]["text"]
