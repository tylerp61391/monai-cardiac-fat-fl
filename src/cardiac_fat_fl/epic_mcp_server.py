"""Native open.epic & Cardiac AI Model Context Protocol (MCP) Server.

Implements the standard Anthropic / Model Context Protocol (MCP) specification (2024-11-05)
over JSON-RPC 2.0 stdio, exposing high-value cardiology imaging tools, discrete HL7 FHIR R4
EHR queries, diagnostic reporting, and automated 3D CT adipose quantification to AI assistants
(Antigravity, Claude Desktop, Cursor, and clinical agents).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from cardiac_fat_fl.cohort_research import CohortDataset
from cardiac_fat_fl.fhir_client import (
    CRISP_CT_INFLAMMATION_HU_THRESHOLD,
    EAT_VOLUME_MODERATE_MAX_CM3,
    EAT_VOLUME_NORMAL_MAX_CM3,
    SNOMED_CORONARY_BODY_SITES,
)
from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    MEASUREMENT_CSV_COLUMNS,
    ReportedRegion,
)

logger = logging.getLogger("cardiac_fat_fl.epic_mcp_server")

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "cardiac-fat-epic-mcp"
SERVER_VERSION = "1.0.0"

# open.epic Terminology Knowledge Base
EPIC_TERMINOLOGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "eat": {
        "concept": "Epicardial Adipose Tissue (EAT)",
        "loinc_code": "84245-0",
        "loinc_display": "Epicardial adipose tissue volume [Volume] by Computed tomography",
        "ucum_unit": "cm3",
        "resource": "Observation",
        "category": "imaging",
        "normal_range": f"< {EAT_VOLUME_NORMAL_MAX_CM3} cm3 (Normal)",
        "moderate_range": f"{EAT_VOLUME_NORMAL_MAX_CM3} - {EAT_VOLUME_MODERATE_MAX_CM3} cm3 (Moderate risk)",
        "high_risk_range": f"> {EAT_VOLUME_MODERATE_MAX_CM3} cm3 (High cardiovascular risk)",
        "epic_profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-observation-imaging",
    },
    "pvat": {
        "concept": "Pericoronary Adipose Tissue (PVAT) Attenuation",
        "radlex_id": "RID49533",
        "radlex_display": "perivascular fat attenuation",
        "ucum_unit": "[HU]",
        "resource": "Observation",
        "category": "imaging",
        "crisp_ct_cutoff": f"> {CRISP_CT_INFLAMMATION_HU_THRESHOLD} HU indicates active vascular inflammation",
        "coronary_sites": {
            "LAD": "91724006 (Left anterior descending coronary artery)",
            "LCX": "244251006 (Left circumflex coronary artery)",
            "RCA": "362070004 (Right coronary artery)",
            "PVAT_Total": "4147007 (Coronary artery structure)",
        },
        "epic_profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-observation-imaging",
    },
    "diagnostic_report": {
        "concept": "Cardiac Fat Diagnostic Report",
        "loinc_code": "79070-9",
        "loinc_display": "CT Heart for calcium scoring and anatomical evaluation",
        "resource": "DiagnosticReport",
        "category": "RAD (Radiology)",
        "epic_profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-diagnosticreport-note",
    },
    "document_reference": {
        "concept": "Clinical Imaging Report (Vector PDF)",
        "loinc_code": "11506-3",
        "loinc_display": "Provider-unspecified Progress note",
        "resource": "DocumentReference",
        "category": "clinical-note",
        "epic_profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-documentreference",
    },
}


class EpicMCPServer:
    """Model Context Protocol (MCP) Server for Cardiac Fat AI and Epic EHR Interoperability."""

    def __init__(
        self,
        fhir_base_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        default_cohort_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the Epic MCP Server.

        Args:
            fhir_base_url: Base URL for HL7 FHIR R4 API (default: http://127.0.0.1:8050/r4).
            bearer_token: Optional OAuth 2.0 Bearer token for SMART on FHIR authorization.
            default_cohort_dir: Optional path to default cohort directory.
        """
        self.fhir_base_url = (fhir_base_url or os.environ.get("CARDIAC_FHIR_URL", "http://127.0.0.1:8050/r4")).rstrip("/")
        self.bearer_token = bearer_token or os.environ.get("CARDIAC_FHIR_TOKEN")
        self.default_cohort_dir = Path(default_cohort_dir) if default_cohort_dir else None

    # ==========================================================================
    # HTTP / FHIR Helper Methods
    # ==========================================================================

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/fhir+json, application/json",
            "Content-Type": "application/fhir+json",
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _query_fhir(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
        url = f"{self.fhir_base_url}/{path.lstrip('/')}"
        try:
            resp = requests.get(url, params=params, headers=self._get_headers(), timeout=15)
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, resp.text
        except requests.RequestException as e:
            return 503, {"resourceType": "OperationOutcome", "issue": [{"diagnostics": str(e)}]}

    def _post_fhir(self, path: str, resource: Dict[str, Any]) -> Tuple[int, Any]:
        url = f"{self.fhir_base_url}/{path.lstrip('/')}"
        try:
            resp = requests.post(url, json=resource, headers=self._get_headers(), timeout=15)
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, resp.text
        except requests.RequestException as e:
            return 503, {"resourceType": "OperationOutcome", "issue": [{"diagnostics": str(e)}]}

    # ==========================================================================
    # MCP Protocol Specification & Tool Declarations
    # ==========================================================================

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the list of tool definitions with JSON Schema parameter contracts."""
        return [
            {
                "name": "epic_get_patient_observations",
                "description": (
                    "Query discrete cardiac fat biomarkers (Total EAT volume in cm3 and vessel-specific PVAT "
                    "attenuation in HU) for a given patient from Epic on FHIR / local sandbox. Formats LOINC "
                    "84245-0 and RadLex RID49533 values with CRISP-CT vascular inflammation analysis."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "Patient MRN or study ID (e.g., '0506', '0001').",
                        },
                        "code": {
                            "type": "string",
                            "description": "Optional LOINC/RadLex code filter (e.g., '84245-0' for EAT, 'RID49533' for PVAT).",
                        },
                    },
                    "required": ["patient_id"],
                },
            },
            {
                "name": "epic_get_diagnostic_report",
                "description": (
                    "Retrieve finalized cardiology and imaging DiagnosticReports for a patient from Epic on FHIR. "
                    "Returns cardiology narrative impressions, risk stratifications, and referenced observation IDs."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "Patient MRN or study ID (e.g., '0506').",
                        },
                    },
                    "required": ["patient_id"],
                },
            },
            {
                "name": "epic_create_diagnostic_report",
                "description": (
                    "File a new clinical DiagnosticReport directly into Epic on FHIR / local sandbox. "
                    "Publishes structured cardiology conclusions, impressions, and findings."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "Patient MRN or study ID.",
                        },
                        "conclusion": {
                            "type": "string",
                            "description": "Clinical narrative impression and diagnostic conclusions.",
                        },
                        "status": {
                            "type": "string",
                            "description": "FHIR status ('final', 'preliminary', 'amended'). Defaults to 'final'.",
                            "enum": ["final", "preliminary", "amended"],
                        },
                    },
                    "required": ["patient_id", "conclusion"],
                },
            },
            {
                "name": "epic_create_document_reference",
                "description": (
                    "File an imaging report document (such as cardiac_fat_report.pdf) as an HL7 FHIR "
                    "DocumentReference with base64 attachment, making it viewable in Epic Chart Review."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "Patient MRN or study ID.",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to the PDF file or clinical report on disk.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Document title (e.g., 'Cardiac Fat Clinical Diagnostic Report').",
                        },
                    },
                    "required": ["patient_id", "file_path"],
                },
            },
            {
                "name": "cardiac_quantify_study",
                "description": (
                    "Execute the 3D MONAI SegResNet & DeepHeartSeg adipose quantification pipeline on a cardiac CT "
                    "study directory or NRRD scan. Automatically extracts EAT and PVAT volume & attenuation metrics, "
                    "and optionally transmits to PACS (DICOMweb STOW-RS) and EHR (HL7 FHIR R4)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "study_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the participant directory or CT scan file.",
                        },
                        "save_seg": {
                            "type": "boolean",
                            "description": "Whether to save 8-channel eat_pvat.seg.nrrd mask (default: true).",
                        },
                        "push_pacs_url": {
                            "type": "string",
                            "description": "Optional DICOMweb STOW-RS endpoint URL to archive DICOM-SEG and DICOM-SR.",
                        },
                        "push_fhir_url": {
                            "type": "string",
                            "description": "Optional HL7 FHIR R4 base URL to file Observation and DiagnosticReport resources.",
                        },
                    },
                    "required": ["study_path"],
                },
            },
            {
                "name": "cardiac_get_cohort_summary",
                "description": (
                    "Query cohort-level cardiac adipose and Agatston calcium score analytics. Returns distribution "
                    "breakdown across 5 CAC tiers (Zero, Minimal, Mild, Moderate, Severe), mean EAT volume, and "
                    "CRISP-CT pericoronary vascular inflammation prevalence."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "cohort_dir": {
                            "type": "string",
                            "description": "Path to the cohort directory containing participant scans and measurements.",
                        },
                    },
                },
            },
            {
                "name": "open_epic_lookup_code",
                "description": (
                    "Look up official open.epic and US Core terminology mappings for cardiac fat biomarkers, "
                    "including LOINC 84245-0, RadLex RID49533, SNOMED-CT body sites, and Epic FHIR profiles."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "Biomarker or concept name: 'eat', 'pvat', 'diagnostic_report', 'document_reference', or 'all'.",
                            "enum": ["eat", "pvat", "diagnostic_report", "document_reference", "all"],
                        },
                    },
                    "required": ["concept"],
                },
            },
        ]

    # ==========================================================================
    # Tool Execution Handlers
    # ==========================================================================

    def execute_epic_get_patient_observations(self, patient_id: str, code: Optional[str] = None) -> str:
        """Execute observation lookup."""
        params: Dict[str, Any] = {"patient": patient_id}
        if code:
            params["code"] = code

        status, resp = self._query_fhir("Observation", params=params)
        if status != 200 or not isinstance(resp, dict):
            return f"FHIR Observation query failed (HTTP {status}): {json.dumps(resp)}"

        entries = resp.get("entry", [])
        if not entries:
            return f"No discrete cardiac fat observations found for patient '{patient_id}' at {self.fhir_base_url}."

        lines = [f"### Discrete Cardiac Fat Biomarkers for Patient {patient_id}:"]
        lines.append(f"- **FHIR Server**: {self.fhir_base_url}")
        lines.append(f"- **Total Observations Found**: {len(entries)}\n")

        for entry in entries:
            res = entry.get("resource", {})
            obs_code = res.get("code", {}).get("text") or res.get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
            code_val = res.get("code", {}).get("coding", [{}])[0].get("code", "")
            val_qty = res.get("valueQuantity", {})
            val = val_qty.get("value")
            unit = val_qty.get("unit", "")
            interp = res.get("interpretation", [{}])[0].get("text", "")

            # Check if PVAT and evaluate inflammation
            inf_badge = ""
            if "PVAT" in obs_code and val is not None:
                if val > CRISP_CT_INFLAMMATION_HU_THRESHOLD:
                    inf_badge = f" ⚠️ **ACTIVE INFLAMMATION** (>{CRISP_CT_INFLAMMATION_HU_THRESHOLD} HU)"
                else:
                    inf_badge = f" ✅ Normal attenuation (≤{CRISP_CT_INFLAMMATION_HU_THRESHOLD} HU)"

            lines.append(f"• **{obs_code}** (`{code_val}`): **{val:.2f} {unit}**{inf_badge}")
            if interp:
                lines.append(f"   *Interpretation*: {interp}")

        return "\n".join(lines)

    def execute_epic_get_diagnostic_report(self, patient_id: str) -> str:
        """Execute DiagnosticReport query."""
        status, resp = self._query_fhir("DiagnosticReport", params={"patient": patient_id})
        if status != 200 or not isinstance(resp, dict):
            return f"FHIR DiagnosticReport query failed (HTTP {status}): {json.dumps(resp)}"

        entries = resp.get("entry", [])
        if not entries:
            return f"No DiagnosticReport records found for patient '{patient_id}' at {self.fhir_base_url}."

        lines = [f"### Cardiac Imaging Diagnostic Reports for Patient {patient_id}:"]
        for idx, entry in enumerate(entries, 1):
            rep = entry.get("resource", {})
            rep_id = rep.get("id", "Unknown")
            rep_status = rep.get("status", "final")
            issued = rep.get("effectiveDateTime") or rep.get("issued", "N/A")
            conclusion = rep.get("conclusion", "No conclusion narrative.")
            results = rep.get("result", [])
            has_pdf = bool(rep.get("presentedForm"))

            lines.append(f"\n#### Report #{idx} (ID: `{rep_id}` - Status: `{rep_status}` - Date: `{issued}`)")
            lines.append(f"**Conclusion**: {conclusion}")
            lines.append(f"**Referenced Observations**: {len(results)} discrete measurements")
            lines.append(f"**Vector PDF Attached**: {'Yes (base64 embedded)' if has_pdf else 'No'}")

        return "\n".join(lines)

    def execute_epic_create_diagnostic_report(self, patient_id: str, conclusion: str, status: str = "final") -> str:
        """File a new DiagnosticReport."""
        report = {
            "resourceType": "DiagnosticReport",
            "status": status,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "RAD",
                            "display": "Radiology",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "79070-9",
                        "display": "CT Heart for calcium scoring and anatomical evaluation",
                    }
                ],
                "text": "Automated Cardiac Adipose (EAT/PVAT) Diagnostic Report",
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
                "display": f"Patient {patient_id}",
            },
            "conclusion": conclusion,
        }

        http_status, resp = self._post_fhir("DiagnosticReport", report)
        if http_status in (200, 201):
            created_id = resp.get("id", "Unknown") if isinstance(resp, dict) else "Created"
            return (
                f"Successfully filed DiagnosticReport `{created_id}` for Patient '{patient_id}' into EHR "
                f"(Status: HTTP {http_status}).\nConclusion: {conclusion}"
            )
        return f"Failed to file DiagnosticReport (HTTP {http_status}): {json.dumps(resp)}"

    def execute_epic_create_document_reference(
        self, patient_id: str, file_path: str, title: Optional[str] = None
    ) -> str:
        """File a DocumentReference."""
        path = Path(file_path)
        if not path.is_file():
            return f"Error: File not found at path '{file_path}'"

        pdf_bytes = path.read_bytes()
        doc_ref = {
            "resourceType": "DocumentReference",
            "status": "current",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11506-3",
                        "display": "Provider-unspecified Progress note",
                    }
                ],
                "text": "Cardiology AI Imaging Diagnostic Report",
            },
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category",
                            "code": "clinical-note",
                            "display": "Clinical Note",
                        }
                    ]
                }
            ],
            "subject": {
                "reference": f"Patient/{patient_id}",
                "display": f"Patient {patient_id}",
            },
            "content": [
                {
                    "attachment": {
                        "contentType": "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain",
                        "title": title or path.name,
                        "data": base64.b64encode(pdf_bytes).decode("ascii"),
                    }
                }
            ],
        }

        http_status, resp = self._post_fhir("DocumentReference", doc_ref)
        if http_status in (200, 201):
            created_id = resp.get("id", "Unknown") if isinstance(resp, dict) else "Created"
            return (
                f"Successfully attached DocumentReference `{created_id}` for Patient '{patient_id}' "
                f"({len(pdf_bytes):,} bytes, title: '{title or path.name}')."
            )
        return f"Failed to create DocumentReference (HTTP {http_status}): {json.dumps(resp)}"

    def execute_cardiac_quantify_study(
        self,
        study_path: str,
        save_seg: bool = True,
        push_pacs_url: Optional[str] = None,
        push_fhir_url: Optional[str] = None,
    ) -> str:
        """Run study quantification and optional push."""
        path = Path(study_path)
        if not path.exists():
            return f"Error: Study path does not exist: '{study_path}'"

        from cardiac_fat_fl.cli import _process_one_participant
        from cardiac_fat_fl.io_metatensor import discover_participants

        pid = path.name if path.is_dir() else path.stem
        # Discover CT scan
        if path.is_dir():
            participants = discover_participants(path)
            if pid in participants:
                scan_file = participants[pid]
            else:
                scan_files = list(path.glob("*.nrrd")) + list(path.glob("*.nii*"))
                if scan_files:
                    scan_file = scan_files[0]
                else:
                    return f"Error: No CT scan (.nrrd or .nii) found in directory '{study_path}'"
        else:
            scan_file = path

        task = (pid, scan_file, save_seg, False, 10.0, None, path.parent)
        result = _process_one_participant(task)

        if not result or result.get("error"):
            return f"Quantification failed: {result.get('error', 'Unknown error')}"

        metrics = result.get("metrics", {})
        eat_vol = metrics.get(f"{ReportedRegion.EAT_TOTAL.value} Volume (cm3)", 0.0)
        pvat_lad_hu = metrics.get(f"{ReportedRegion.PVAT_LAD.value} Mean Attenuation (HU)", float("nan"))
        pvat_total_vol = metrics.get(f"{ReportedRegion.PVAT_TOTAL.value} Volume (cm3)", 0.0)

        lines = [f"### Automated 3D Quantification Complete for Study: `{pid}`"]
        lines.append(f"- **Scan Processed**: `{scan_file.name}`")
        lines.append(f"- **Total EAT Volume**: **{eat_vol:.2f} cm³**")
        lines.append(f"- **Total PVAT Volume**: **{pvat_total_vol:.2f} cm³**")
        lines.append(f"- **LAD PVAT Attenuation**: **{pvat_lad_hu:.2f} HU**" if pvat_lad_hu == pvat_lad_hu else "- **LAD PVAT Attenuation**: N/A")

        # Evaluate inflammation
        if pvat_lad_hu == pvat_lad_hu and pvat_lad_hu > CRISP_CT_INFLAMMATION_HU_THRESHOLD:
            lines.append(f"- **CRISP-CT Status**: ⚠️ **Active Pericoronary Inflammation Detected** in LAD ({pvat_lad_hu:.1f} HU > -70.1 HU)")
        else:
            lines.append("- **CRISP-CT Status**: ✅ Normal pericoronary vascular attenuation")

        # Optional FHIR push
        target_fhir = push_fhir_url or self.fhir_base_url
        if target_fhir:
            from cardiac_fat_fl.fhir_client import create_bundle_for_study, push_to_fhir_server
            try:
                bundle = create_bundle_for_study(path if path.is_dir() else path.parent, pid)
                push_res = push_to_fhir_server(bundle, target_fhir, token=self.bearer_token)
                lines.append(f"- **HL7 FHIR Push**: {'Success' if push_res.get('success') else 'Failed'} ({target_fhir})")
            except Exception as e:
                lines.append(f"- **HL7 FHIR Push Warning**: {e}")

        # Optional PACS push
        if push_pacs_url:
            from cardiac_fat_fl.pacs_client import find_dicom_instances_to_push, push_to_dicomweb_stow
            study_dir = path if path.is_dir() else path.parent
            instances = find_dicom_instances_to_push(study_dir, include_ct=False)
            if instances:
                pacs_res = push_to_dicomweb_stow(push_pacs_url, instances)
                lines.append(f"- **PACS DICOMweb STOW-RS**: {'Success' if pacs_res.get('success') else 'Failed'} ({len(instances)} instances)")

        return "\n".join(lines)

    def execute_cardiac_get_cohort_summary(self, cohort_dir: Optional[str] = None) -> str:
        """Execute cohort summary analytics."""
        target_dir = Path(cohort_dir) if cohort_dir else (self.default_cohort_dir or Path("."))
        if not target_dir.exists():
            return f"Error: Cohort directory does not exist: '{target_dir}'"

        try:
            ds = CohortDataset(target_dir)
            total = len(ds.participants)
            if total == 0:
                return f"No participants discovered in cohort directory '{target_dir}'."

            # Calculate metrics
            eat_volumes: List[float] = []
            cac_counts: Dict[str, int] = {}
            inflamed_count = 0

            eat_key = f"{ReportedRegion.EAT_TOTAL.value} Volume (cm3)"
            for p in ds.participants.values():
                val = p.metrics.get(eat_key)
                if val is not None and not (isinstance(val, float) and (val != val or val < 0)):
                    eat_volumes.append(float(val))

                tier = p.calcium_risk_tier or "Unknown"
                cac_counts[tier] = cac_counts.get(tier, 0) + 1

                if "ACTIVE" in p.crisp_ct_status.upper() or "INFLAM" in p.crisp_ct_status.upper():
                    inflamed_count += 1

            mean_eat = sum(eat_volumes) / len(eat_volumes) if eat_volumes else 0.0

            lines = [f"### Cohort Research Summary: `{target_dir.name}`"]
            lines.append(f"- **Total Participants**: {total}")
            lines.append(f"- **Patients with EAT Measurements**: {len(eat_volumes)}")
            lines.append(f"- **Mean Cohort EAT Volume**: **{mean_eat:.2f} cm³**")
            lines.append(
                f"- **Active CRISP-CT PVAT Inflammation**: **{inflamed_count}/{total}** "
                f"({(inflamed_count / total * 100):.1f}% of cohort)"
            )
            lines.append("\n**Agatston Coronary Artery Calcium (CAC) Stratification**:")
            for tier, count in sorted(cac_counts.items()):
                pct = (count / total) * 100
                lines.append(f"• **{tier}**: {count} patients ({pct:.1f}%)")

            return "\n".join(lines)
        except Exception as e:
            return f"Error computing cohort summary: {e}"

    def execute_open_epic_lookup_code(self, concept: str) -> str:
        """Return terminology and profile mapping for open.epic."""
        key = concept.lower().strip()
        if key in ("all", "*"):
            return "### open.epic Terminology Knowledge Base:\n\n" + json.dumps(EPIC_TERMINOLOGY_REGISTRY, indent=2)

        data = EPIC_TERMINOLOGY_REGISTRY.get(key)
        if not data:
            return (
                f"Concept '{concept}' not recognized. Available concepts: "
                f"{list(EPIC_TERMINOLOGY_REGISTRY.keys())} or 'all'."
            )
        return f"### open.epic Mapping for '{concept.upper()}':\n\n" + json.dumps(data, indent=2)

    # ==========================================================================
    # JSON-RPC 2.0 Message Dispatcher
    # ==========================================================================

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[str, bool]:
        """Dispatch a tool call by name and arguments.

        Returns:
            Tuple of (output_text, is_error).
        """
        try:
            if tool_name == "epic_get_patient_observations":
                return self.execute_epic_get_patient_observations(
                    patient_id=arguments["patient_id"],
                    code=arguments.get("code"),
                ), False
            elif tool_name == "epic_get_diagnostic_report":
                return self.execute_epic_get_diagnostic_report(
                    patient_id=arguments["patient_id"],
                ), False
            elif tool_name == "epic_create_diagnostic_report":
                return self.execute_epic_create_diagnostic_report(
                    patient_id=arguments["patient_id"],
                    conclusion=arguments["conclusion"],
                    status=arguments.get("status", "final"),
                ), False
            elif tool_name == "epic_create_document_reference":
                return self.execute_epic_create_document_reference(
                    patient_id=arguments["patient_id"],
                    file_path=arguments["file_path"],
                    title=arguments.get("title"),
                ), False
            elif tool_name == "cardiac_quantify_study":
                return self.execute_cardiac_quantify_study(
                    study_path=arguments["study_path"],
                    save_seg=arguments.get("save_seg", True),
                    push_pacs_url=arguments.get("push_pacs_url"),
                    push_fhir_url=arguments.get("push_fhir_url"),
                ), False
            elif tool_name == "cardiac_get_cohort_summary":
                return self.execute_cardiac_get_cohort_summary(
                    cohort_dir=arguments.get("cohort_dir"),
                ), False
            elif tool_name == "open_epic_lookup_code":
                return self.execute_open_epic_lookup_code(
                    concept=arguments["concept"],
                ), False
            else:
                return f"Unknown tool: '{tool_name}'", True
        except Exception as e:
            logger.exception("Error executing tool %s", tool_name)
            return f"Error executing tool '{tool_name}': {e}", True

    def handle_message(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC 2.0 request or notification dictionary."""
        method = request.get("method")
        req_id = request.get("id")

        # 1. MCP Initialization Handshake
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            }

        # 2. Initialized Notification (no response needed)
        elif method == "notifications/initialized":
            return None

        # 3. Ping
        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }

        # 4. List Tools
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.get_tool_definitions(),
                },
            }

        # 5. Call Tool
        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            output_text, is_error = self.call_tool(tool_name, arguments)

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": output_text,
                        }
                    ],
                    "isError": is_error,
                },
            }

        # Unknown method error
        else:
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            return None

    def run_stdio(self) -> None:
        """Run the MCP server reading JSON-RPC 2.0 lines from sys.stdin and writing to sys.stdout."""
        logger.info("Starting Epic MCP Server on stdio (protocol %s)", MCP_PROTOCOL_VERSION)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e}",
                    },
                }
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle_message(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()


def launch_mcp_server(
    fhir_base_url: Optional[str] = None,
    bearer_token: Optional[str] = None,
    default_cohort_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Launch the Epic MCP Server in stdio mode."""
    server = EpicMCPServer(
        fhir_base_url=fhir_base_url,
        bearer_token=bearer_token,
        default_cohort_dir=default_cohort_dir,
    )
    server.run_stdio()
