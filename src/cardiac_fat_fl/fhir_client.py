"""HL7 FHIR R4 Client & Bundle Generator for Hospital EHR Interoperability.

Translates 3D cardiac adipose tissue quantification metrics, CRISP-CT inflammation
classifications, and vector PDF reports into standard HL7 FHIR R4 resources:
  - Observation: Discrete numeric Total EAT Volume (LOINC 84245-0, cm3, UCUM)
  - Observation: Per-vessel PVAT Attenuation (LOINC / RadLex RID49533, SNOMED-CT branches, HU)
  - DiagnosticReport: Structured cardiology imaging report with CRISP-CT inflammation summary
  - DocumentReference: EHR Chart Review attachment with Base64 vector diagnostic PDF

Fully compliant with Epic on FHIR US Core specifications and the open.epic Model Context
Protocol (MCP) server: https://fhir.epic.com/Documentation?docId=mcp
"""

from __future__ import annotations

import base64
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import pandas as pd

from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    REPORTED_REGIONS,
    ReportedRegion,
)

logger = logging.getLogger("cardiac_fat_fl.fhir_client")

# Clinical Biomarker Invariants
CRISP_CT_INFLAMMATION_HU_THRESHOLD = -70.1  # PVAT HU > -70.1 indicates active vascular inflammation
EAT_VOLUME_NORMAL_MAX_CM3 = 100.0           # < 100 cm3: Normal
EAT_VOLUME_MODERATE_MAX_CM3 = 125.0         # 100 - 125 cm3: Moderate; > 125 cm3: High Risk

# SNOMED-CT Anatomical Body Site Concepts for Coronary Arteries
SNOMED_CORONARY_BODY_SITES: Dict[str, Tuple[str, str]] = {
    ReportedRegion.PVAT_LAD.value: ("91724006", "Left anterior descending coronary artery"),
    ReportedRegion.PVAT_LCX.value: ("244251006", "Left circumflex coronary artery"),
    ReportedRegion.PVAT_RCA.value: ("362070004", "Right coronary artery"),
    ReportedRegion.PVAT_TOTAL.value: ("4147007", "Coronary artery structure"),
}


def build_eat_volume_observation(
    eat_volume_cm3: float,
    patient_id: str,
    effective_datetime: Optional[str] = None,
    obs_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct an HL7 FHIR R4 Observation resource for Total EAT Volume (LOINC 84245-0).

    Args:
        eat_volume_cm3: Total EAT volume in cubic centimeters (cm3).
        patient_id: Institutional patient MRN / identifier.
        effective_datetime: ISO-8601 study timestamp (defaults to current UTC).
        obs_id: Optional resource ID or UUID.

    Returns:
        FHIR R4 Observation resource dictionary.
    """
    dt_str = effective_datetime or datetime.now(timezone.utc).isoformat()
    oid = obs_id or str(uuid.uuid4())

    is_elevated = eat_volume_cm3 > EAT_VOLUME_MODERATE_MAX_CM3
    is_moderate = EAT_VOLUME_NORMAL_MAX_CM3 < eat_volume_cm3 <= EAT_VOLUME_MODERATE_MAX_CM3

    interpretation_code = "H" if is_elevated else ("A" if is_moderate else "N")
    interpretation_display = "High" if is_elevated else ("Abnormal" if is_moderate else "Normal")

    return {
        "resourceType": "Observation",
        "id": oid,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "84245-0",
                    "display": "Epicardial adipose tissue volume [Volume] by Computed tomography",
                }
            ],
            "text": "Total Epicardial Adipose Tissue (EAT) Volume",
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
            "display": f"Patient {patient_id}",
        },
        "effectiveDateTime": dt_str,
        "valueQuantity": {
            "value": round(float(eat_volume_cm3), 2),
            "unit": "cm3",
            "system": "http://unitsofmeasure.org",
            "code": "cm3",
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": interpretation_code,
                        "display": interpretation_display,
                    }
                ]
            }
        ],
        "referenceRange": [
            {
                "high": {
                    "value": EAT_VOLUME_NORMAL_MAX_CM3,
                    "unit": "cm3",
                    "system": "http://unitsofmeasure.org",
                    "code": "cm3",
                },
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/referencerange-meaning",
                            "code": "normal",
                            "display": "Normal Range (< 100 cm3)",
                        }
                    ]
                },
            },
            {
                "low": {
                    "value": EAT_VOLUME_MODERATE_MAX_CM3,
                    "unit": "cm3",
                    "system": "http://unitsofmeasure.org",
                    "code": "cm3",
                },
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/referencerange-meaning",
                            "code": "elevated",
                            "display": "High Risk (> 125 cm3)",
                        }
                    ]
                },
            },
        ],
    }


def build_pvat_attenuation_observation(
    region_name: str,
    pvat_mean_hu: float,
    patient_id: str,
    effective_datetime: Optional[str] = None,
    obs_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct an HL7 FHIR R4 Observation resource for Pericoronary Adipose Attenuation.

    Args:
        region_name: Region key (e.g. 'PVAT_LAD', 'PVAT_LCX', 'PVAT_RCA', 'PVAT_Total').
        pvat_mean_hu: Mean attenuation in Hounsfield Units (HU).
        patient_id: Institutional patient identifier.
        effective_datetime: ISO-8601 study timestamp.
        obs_id: Resource UUID.

    Returns:
        FHIR R4 Observation resource dictionary.
    """
    dt_str = effective_datetime or datetime.now(timezone.utc).isoformat()
    oid = obs_id or str(uuid.uuid4())

    is_inflamed = pvat_mean_hu > CRISP_CT_INFLAMMATION_HU_THRESHOLD
    interpretation_code = "H" if is_inflamed else "N"
    interpretation_display = "High (Active Vascular Inflammation)" if is_inflamed else "Normal Attenuation"

    body_site_tuple = SNOMED_CORONARY_BODY_SITES.get(region_name)
    body_site_dict = None
    if body_site_tuple:
        body_site_dict = {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": body_site_tuple[0],
                    "display": body_site_tuple[1],
                }
            ],
            "text": body_site_tuple[1],
        }

    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": oid,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://radlex.org",
                    "code": "RID49533",
                    "display": "Pericoronary adipose tissue mean attenuation",
                }
            ],
            "text": f"{region_name} Mean Attenuation",
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
            "display": f"Patient {patient_id}",
        },
        "effectiveDateTime": dt_str,
        "valueQuantity": {
            "value": round(float(pvat_mean_hu), 1),
            "unit": "HU",
            "system": "http://unitsofmeasure.org",
            "code": "[hnsf'U]",
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": interpretation_code,
                        "display": interpretation_display,
                    }
                ]
            }
        ],
        "referenceRange": [
            {
                "high": {
                    "value": CRISP_CT_INFLAMMATION_HU_THRESHOLD,
                    "unit": "HU",
                    "system": "http://unitsofmeasure.org",
                    "code": "[hnsf'U]",
                },
                "type": {
                    "text": f"CRISP-CT Active Vascular Inflammation Cutoff (> {CRISP_CT_INFLAMMATION_HU_THRESHOLD} HU)",
                },
            }
        ],
    }

    if body_site_dict:
        obs["bodySite"] = body_site_dict

    return obs


def build_cardiac_fat_diagnostic_report(
    patient_id: str,
    metrics: Dict[str, float],
    observation_references: List[str],
    pdf_bytes: Optional[bytes] = None,
    effective_datetime: Optional[str] = None,
    report_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct an HL7 FHIR R4 DiagnosticReport resource for cardiology/radiology filing.

    Args:
        patient_id: Patient MRN / identifier.
        metrics: 19-column clinical metrics dictionary.
        observation_references: List of Observation resource references (e.g. ['urn:uuid:...']).
        pdf_bytes: Optional raw PDF bytes for embedded presentation.
        effective_datetime: Study timestamp.
        report_id: Unique report ID.

    Returns:
        FHIR R4 DiagnosticReport dictionary.
    """
    dt_str = effective_datetime or datetime.now(timezone.utc).isoformat()
    rid = report_id or str(uuid.uuid4())

    eat_vol = metrics.get(f"{ReportedRegion.EAT_TOTAL.value} Volume (cm3)", 0.0)
    metrics.get(f"{ReportedRegion.PVAT_TOTAL.value} Mean Attenuation (HU)", float("nan"))

    # Determine vessel-specific inflammation
    inflamed_vessels = []
    for v_name, v_key in [("LAD", ReportedRegion.PVAT_LAD.value), ("LCX", ReportedRegion.PVAT_LCX.value), ("RCA", ReportedRegion.PVAT_RCA.value)]:
        hu = metrics.get(f"{v_key} Mean Attenuation (HU)")
        if hu is not None and not math.isnan(hu) and hu > CRISP_CT_INFLAMMATION_HU_THRESHOLD:
            inflamed_vessels.append(f"{v_name} ({hu:.1f} HU)")

    if inflamed_vessels:
        inflam_summary = f"CRISP-CT Active Vascular Inflammation detected in: {', '.join(inflamed_vessels)} (> {CRISP_CT_INFLAMMATION_HU_THRESHOLD} HU cutoff)."
        conclusion_code = {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "443502000",
                    "display": "Coronary artery inflammation",
                }
            ]
        }
    else:
        inflam_summary = f"No active pericoronary inflammation detected (all vessels <= {CRISP_CT_INFLAMMATION_HU_THRESHOLD} HU)."
        conclusion_code = {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "17621005",
                    "display": "Normal cardiac structure",
                }
            ]
        }

    conclusion_narrative = (
        f"Automated Cardiac Adipose Quantification: Total EAT Volume = {eat_vol:.1f} cm3. "
        f"{inflam_summary} "
        f"Multi-compartment 3D SegResNet segmentation evaluated."
    )

    report: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": rid,
        "status": "final",
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
            "text": "Automated Cardiac Adipose Tissue (EAT/PVAT) Imaging Diagnostic Report",
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
            "display": f"Patient {patient_id}",
        },
        "effectiveDateTime": dt_str,
        "issued": dt_str,
        "result": [{"reference": ref} for ref in observation_references],
        "conclusion": conclusion_narrative,
        "conclusionCode": [conclusion_code],
    }

    if pdf_bytes:
        report["presentedForm"] = [
            {
                "contentType": "application/pdf",
                "language": "en",
                "title": "Cardiac Fat Clinical Diagnostic Report (PDF)",
                "data": base64.b64encode(pdf_bytes).decode("ascii"),
            }
        ]

    return report


def build_cardiac_fat_document_reference(
    patient_id: str,
    pdf_bytes: bytes,
    effective_datetime: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct an HL7 FHIR R4 DocumentReference for Epic Hyperspace chart attachments.

    Args:
        patient_id: Institutional patient identifier.
        pdf_bytes: Raw bytes of the vector clinical PDF.
        effective_datetime: Study timestamp.
        doc_id: Unique document UUID.

    Returns:
        FHIR R4 DocumentReference resource dictionary.
    """
    dt_str = effective_datetime or datetime.now(timezone.utc).isoformat()
    did = doc_id or str(uuid.uuid4())

    return {
        "resourceType": "DocumentReference",
        "id": did,
        "status": "current",
        "docStatus": "final",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11526-1",
                    "display": "Cardiology Diagnostic imaging study report",
                }
            ],
            "text": "Cardiac Fat Clinical Imaging Report",
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
        "date": dt_str,
        "content": [
            {
                "attachment": {
                    "contentType": "application/pdf",
                    "language": "en",
                    "title": "Cardiac Fat Clinical Imaging Report",
                    "data": base64.b64encode(pdf_bytes).decode("ascii"),
                }
            }
        ],
    }


def build_cardiac_fat_fhir_bundle(
    metrics: Dict[str, float],
    patient_id: str,
    pdf_path_or_bytes: Optional[Union[str, Path, bytes]] = None,
    bundle_type: str = "transaction",
    effective_datetime: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a complete, production-ready HL7 FHIR R4 Bundle from study quantification results.

    Creates an atomic transaction or collection bundle containing:
      1. Observation for Total EAT Volume (LOINC 84245-0)
      2. Observations for PVAT Attenuation across LAD, LCX, RCA, and Total (RadLex/LOINC)
      3. DiagnosticReport summarizing findings and referencing observations
      4. DocumentReference attaching the base64-encoded vector PDF for Epic Chart Review

    Args:
        metrics: 19-column clinical metrics dictionary.
        patient_id: Patient MRN / identifier.
        pdf_path_or_bytes: Optional path to `cardiac_fat_report.pdf` or raw PDF bytes.
        bundle_type: "transaction" (default, atomic ingest) or "collection".
        effective_datetime: Optional ISO-8601 timestamp.

    Returns:
        Standard HL7 FHIR R4 Bundle dictionary.
    """
    dt_str = effective_datetime or datetime.now(timezone.utc).isoformat()
    bundle_id = str(uuid.uuid4())

    # Read PDF bytes if provided as path
    pdf_bytes: Optional[bytes] = None
    if pdf_path_or_bytes is not None:
        if isinstance(pdf_path_or_bytes, (str, Path)):
            p = Path(pdf_path_or_bytes)
            if p.is_file():
                pdf_bytes = p.read_bytes()
        elif isinstance(pdf_path_or_bytes, bytes):
            pdf_bytes = pdf_path_or_bytes

    entries: List[Dict[str, Any]] = []
    obs_references: List[str] = []

    # 1. Total EAT Volume Observation
    eat_vol = metrics.get(f"{ReportedRegion.EAT_TOTAL.value} Volume (cm3)")
    if eat_vol is not None and not math.isnan(eat_vol):
        eat_obs_id = str(uuid.uuid4())
        full_url = f"urn:uuid:{eat_obs_id}"
        obs_res = build_eat_volume_observation(
            eat_volume_cm3=eat_vol,
            patient_id=patient_id,
            effective_datetime=dt_str,
            obs_id=eat_obs_id,
        )
        entry: Dict[str, Any] = {"fullUrl": full_url, "resource": obs_res}
        if bundle_type == "transaction":
            entry["request"] = {"method": "POST", "url": "Observation"}
        entries.append(entry)
        obs_references.append(full_url)

    # 2. PVAT Attenuation Observations (LAD, LCX, RCA, Total)
    pvat_targets = [
        ReportedRegion.PVAT_LAD.value,
        ReportedRegion.PVAT_LCX.value,
        ReportedRegion.PVAT_RCA.value,
        ReportedRegion.PVAT_TOTAL.value,
    ]
    for region_name in pvat_targets:
        hu_val = metrics.get(f"{region_name} Mean Attenuation (HU)")
        if hu_val is not None and not math.isnan(hu_val):
            pvat_obs_id = str(uuid.uuid4())
            full_url = f"urn:uuid:{pvat_obs_id}"
            obs_res = build_pvat_attenuation_observation(
                region_name=region_name,
                pvat_mean_hu=hu_val,
                patient_id=patient_id,
                effective_datetime=dt_str,
                obs_id=pvat_obs_id,
            )
            entry = {"fullUrl": full_url, "resource": obs_res}
            if bundle_type == "transaction":
                entry["request"] = {"method": "POST", "url": "Observation"}
            entries.append(entry)
            obs_references.append(full_url)

    # 3. DiagnosticReport Resource
    report_id = str(uuid.uuid4())
    report_full_url = f"urn:uuid:{report_id}"
    diag_report = build_cardiac_fat_diagnostic_report(
        patient_id=patient_id,
        metrics=metrics,
        observation_references=obs_references,
        pdf_bytes=pdf_bytes,
        effective_datetime=dt_str,
        report_id=report_id,
    )
    report_entry: Dict[str, Any] = {"fullUrl": report_full_url, "resource": diag_report}
    if bundle_type == "transaction":
        report_entry["request"] = {"method": "POST", "url": "DiagnosticReport"}
    entries.append(report_entry)

    # 4. DocumentReference Resource (if PDF is available)
    if pdf_bytes:
        doc_id = str(uuid.uuid4())
        doc_full_url = f"urn:uuid:{doc_id}"
        doc_ref = build_cardiac_fat_document_reference(
            patient_id=patient_id,
            pdf_bytes=pdf_bytes,
            effective_datetime=dt_str,
            doc_id=doc_id,
        )
        doc_entry: Dict[str, Any] = {"fullUrl": doc_full_url, "resource": doc_ref}
        if bundle_type == "transaction":
            doc_entry["request"] = {"method": "POST", "url": "DocumentReference"}
        entries.append(doc_entry)

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": bundle_type,
        "timestamp": dt_str,
        "entry": entries,
    }


def export_fhir_bundle(
    bundle: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Save an HL7 FHIR R4 Bundle dictionary to a JSON file on disk.

    Args:
        bundle: Valid FHIR R4 bundle.
        output_path: Target destination path.

    Returns:
        Path to the saved JSON file.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    logger.info("Exported HL7 FHIR R4 bundle to %s (%d entries)", out_p, len(bundle.get("entry", [])))
    return out_p


def push_to_fhir_server(
    bundle: Dict[str, Any],
    fhir_base_url: str,
    auth_token: Optional[str] = None,
    timeout: int = 60,
    ca_cert: Optional[Union[str, Path, bool]] = None,
) -> Dict[str, Any]:
    """Transmit an HL7 FHIR R4 Bundle to an institutional FHIR server (Epic, Cerner, HAPI FHIR).

    Args:
        bundle: Valid FHIR R4 Bundle dictionary.
        fhir_base_url: Base URL of the FHIR endpoint (e.g. `https://fhir.hospital.org/r4`).
        auth_token: Optional OAuth 2.0 Bearer access token.
        timeout: HTTP timeout in seconds.
        ca_cert: Custom CA certificate bundle or False for unverified TLS.

    Returns:
        Dictionary summarizing transmission result (`status_code`, `response`, `success`).
    """
    target_url = fhir_base_url.rstrip("/")
    # If target is base URL, transaction bundles are posted directly to base or /Bundle
    if not (target_url.endswith("/Bundle") or bundle.get("type") == "transaction"):
        target_url = f"{target_url}/Bundle"

    headers = {
        "Content-Type": "application/fhir+json; charset=utf-8",
        "Accept": "application/fhir+json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    verify_param: Union[bool, str] = True
    if ca_cert is not None:
        verify_param = ca_cert

    logger.info("Transmitting FHIR R4 %s bundle to %s...", bundle.get("type"), target_url)

    resp = requests.post(
        target_url,
        json=bundle,
        headers=headers,
        timeout=timeout,
        verify=verify_param,
    )

    is_success = resp.status_code in [200, 201, 202]
    res_data: Any = None
    try:
        res_data = resp.json()
    except Exception:
        res_data = resp.text

    if not is_success:
        logger.error("FHIR transmission failed with status %d: %s", resp.status_code, resp.text[:300])

    return {
        "success": is_success,
        "status_code": resp.status_code,
        "response": res_data,
        "entries_count": len(bundle.get("entry", [])),
    }


def create_bundle_for_study(
    study_dir: Union[str, Path],
    bundle_type: str = "transaction",
    metrics_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Helper to discover measurements, PDF report, and assemble an FHIR R4 bundle from a study directory.

    Args:
        study_dir: Path to study/participant folder.
        bundle_type: "transaction" or "collection".
        metrics_override: Optional explicit metrics dict.

    Returns:
        FHIR R4 Bundle dictionary.
    """
    s_path = Path(study_dir).resolve()
    patient_id = s_path.name

    metrics: Dict[str, float] = {}
    if metrics_override:
        metrics = metrics_override
    else:
        # Search for cardiac_fat_measurements.csv in s_path and parent hierarchy
        curr = s_path
        while curr != curr.parent:
            csv_p = curr / "cardiac_fat_measurements.csv"
            if csv_p.is_file():
                try:
                    df = pd.read_csv(csv_p)
                    # Filter for patient_id if present
                    if "participant_id" in df.columns:
                        matched = df[df["participant_id"].astype(str) == str(patient_id)]
                        row = matched.iloc[0] if len(matched) > 0 else df.iloc[0]
                    else:
                        row = df.iloc[0]

                    for col in df.columns:
                        if col != "participant_id":
                            try:
                                metrics[col] = float(row[col])
                            except (ValueError, TypeError):
                                pass
                    break
                except Exception:
                    pass
            curr = curr.parent

    # Discover clinical report PDF if available
    pdf_candidates = [
        s_path / "cardiac_fat_report.pdf",
        s_path / "report" / "cardiac_fat_report.pdf",
        s_path / "ohif_export" / "cardiac_fat_report.pdf",
    ]
    pdf_path = next((p for p in pdf_candidates if p.is_file()), None)

    return build_cardiac_fat_fhir_bundle(
        metrics=metrics,
        patient_id=patient_id,
        pdf_path_or_bytes=pdf_path,
        bundle_type=bundle_type,
    )
