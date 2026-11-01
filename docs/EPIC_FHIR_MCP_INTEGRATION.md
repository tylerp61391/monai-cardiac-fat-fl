# Epic on FHIR & Model Context Protocol (MCP) Integration Guide

This guide details the architecture for integrating `monai-cardiac-fat-fl` directly with **Epic Systems EHR** via standard **HL7 FHIR R4** and Epic's official **open.epic Model Context Protocol (MCP) Server** ([https://fhir.epic.com/Documentation?docId=mcp](https://fhir.epic.com/Documentation?docId=mcp)).

---

## 1. Executive Summary

`monai-cardiac-fat-fl` bridges the gap between advanced 3D cardiovascular AI and hospital EHR systems. Through FHIR R4 interoperability:

- **Discrete AI Biomarkers** (EAT volume in $\text{cm}^3$, vessel-specific PVAT attenuation in HU, and CRISP-CT inflammation status) flow directly into Epic Flowsheets, In Basket alerts, and Clinical Decision Support (CDS) rules.
- **Diagnostic Reports & Vector PDFs** are automatically filed into Epic Chart Review (Radiology/Cardiology tabs) via `DiagnosticReport` and `DocumentReference`.
- **open.epic MCP Server** empowers AI agents and developers to query Epic's live FHIR data dictionary, schema constraints, and OAuth 2.0 SMART on FHIR specifications dynamically.

---

## 2. open.epic MCP Server Architecture

Epic provides an official **Model Context Protocol (MCP)** server for developers to connect AI coding assistants (Antigravity, Claude Desktop, Cursor) directly to `open.epic`.

### Connecting open.epic MCP in Antigravity / AI Environments

Add the open.epic server to your client's MCP configuration:

```json
{
  "mcpServers": {
    "open-epic": {
      "command": "npx",
      "args": ["-y", "@epic/open-epic-mcp"]
    }
  }
}
```

### AI Agent Capabilities Powered by open.epic MCP

1. **Dynamic Schema Validation**: AI agents inspect Epic's US Core FHIR profiles (mandatory vs. optional elements, Must Support flags, cardinality).
2. **Coding Terminology Lookup**: Verifies LOINC, RadLex, SNOMED-CT, and UCUM units against Epic's supported terminology tables.
3. **OAuth 2.0 Backend Services Verification**: Inspects Epic's asymmetric JWT signature requirements, JWKS endpoint formats, and token endpoints (`/oauth2/token`).
4. **Data Sharing Playbooks**: Retrieves Epic's best-practice workflows for cardiology imaging and AI report ingestion.

---

## 3. Epic on FHIR R4 Resource Mapping

`monai-cardiac-fat-fl` outputs map to standard Epic US Core R4 resources:

```text
                               ┌────────────────────────────────────────┐
                               │   Cardiac CT Study & AI Segmentation   │
                               │      (DICOM-SEG, DICOM-SR, PDF)        │
                               └──────────────────┬─────────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  ▼                                                               ▼
     ┌─────────────────────────┐                                     ┌─────────────────────────┐
     │  Discrete Observations  │                                     │  Diagnostic Deliverable │
     └────────────┬────────────┘                                     └────────────┬────────────┘
                  │                                                               │
     ┌────────────┴──────────────────────┐                          ┌─────────────┴─────────────┐
     ▼                                   ▼                          ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│ Observation: EAT Volume   │ │ Observation: PVAT HU      │ │ DiagnosticReport        │ │ DocumentReference       │
│ • LOINC 84245-0           │ │ • LOINC / RadLex          │ │ • Category: Radiology   │ │ • Category: ClinicalDoc │
│ • Unit: cm3 (UCUM)        │ │ • BodySite: LAD/LCX/RCA   │ │ • Status: final         │ │ • Type: Cardiology Rpt  │
│ • Epic Flowsheet Filing   │ │ • CRISP-CT Threshold:     │ │ • Narrative Conclusion  │ │ • Attachment: Base64    │
│                           │ │   > -70.1 HU Inflamed     │ │ • Result References     │ │   Vector PDF            │
└───────────────────────────┘ └───────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
```

### 1. `Observation` — Total EAT Volume (`LOINC 84245-0`)

Filed as a discrete numeric clinical result in Epic:

```json
{
  "resourceType": "Observation",
  "status": "final",
  "category": [
    {
      "coding": [
        {
          "system": "http://terminology.hl7.org/CodeSystem/observation-category",
          "code": "imaging",
          "display": "Imaging"
        }
      ]
    }
  ],
  "code": {
    "coding": [
      {
        "system": "http://loinc.org",
        "code": "84245-0",
        "display": "Epicardial adipose tissue volume [Volume] by Computed tomography"
      }
    ],
    "text": "Total Epicardial Adipose Tissue (EAT) Volume"
  },
  "subject": {
    "reference": "Patient/e.PATIENT_ID"
  },
  "effectiveDateTime": "2026-10-31T16:00:00Z",
  "valueQuantity": {
    "value": 42.70,
    "unit": "cm3",
    "system": "http://unitsofmeasure.org",
    "code": "cm3"
  },
  "referenceRange": [
    {
      "high": { "value": 100.0, "unit": "cm3", "system": "http://unitsofmeasure.org", "code": "cm3" },
      "type": { "coding": [{ "system": "http://terminology.hl7.org/CodeSystem/referencerange-meaning", "code": "normal" }] }
    },
    {
      "low": { "value": 125.0, "unit": "cm3", "system": "http://unitsofmeasure.org", "code": "cm3" },
      "type": { "coding": [{ "system": "http://terminology.hl7.org/CodeSystem/referencerange-meaning", "code": "elevated" }] }
    }
  ],
  "interpretation": [
    {
      "coding": [
        {
          "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
          "code": "N",
          "display": "Normal"
        }
      ]
    }
  ]
}
```

---

### 2. `Observation` — Pericoronary Adipose Attenuation (`PVAT HU`)

Includes branch anatomy codings (LAD, LCX, RCA) and the CRISP-CT inflammation cutoff ($-70.1\text{ HU}$):

```json
{
  "resourceType": "Observation",
  "status": "final",
  "category": [
    {
      "coding": [
        {
          "system": "http://terminology.hl7.org/CodeSystem/observation-category",
          "code": "imaging"
        }
      ]
    }
  ],
  "code": {
    "coding": [
      {
        "system": "http://radlex.org",
        "code": "RID49533",
        "display": "Pericoronary adipose tissue mean attenuation"
      }
    ],
    "text": "LAD Pericoronary Adipose Tissue Attenuation"
  },
  "subject": {
    "reference": "Patient/e.PATIENT_ID"
  },
  "bodySite": {
    "coding": [
      {
        "system": "http://snomed.info/sct",
        "code": "91724006",
        "display": "Left anterior descending coronary artery"
      }
    ]
  },
  "valueQuantity": {
    "value": -63.67,
    "unit": "HU",
    "system": "http://unitsofmeasure.org",
    "code": "[hnsf'U]"
  },
  "referenceRange": [
    {
      "high": { "value": -70.1, "unit": "HU", "system": "http://unitsofmeasure.org", "code": "[hnsf'U]" },
      "type": {
        "text": "CRISP-CT Active Vascular Inflammation Threshold (> -70.1 HU)"
      }
    }
  ],
  "interpretation": [
    {
      "coding": [
        {
          "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
          "code": "H",
          "display": "High (Active Inflammation)"
        }
      ]
    }
  ]
}
```

---

### 3. `DiagnosticReport` — Cardiology AI Diagnostic Summary

Aggregates findings, references discrete observations, and attaches the base64 report:

```json
{
  "resourceType": "DiagnosticReport",
  "status": "final",
  "category": [
    {
      "coding": [
        {
          "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
          "code": "RAD",
          "display": "Radiology"
        }
      ]
    }
  ],
  "code": {
    "coding": [
      {
        "system": "http://loinc.org",
        "code": "79070-9",
        "display": "CT Heart for calcium scoring and anatomical evaluation"
      }
    ],
    "text": "Automated Cardiac Adipose Tissue (EAT/PVAT) Imaging Report"
  },
  "subject": { "reference": "Patient/e.PATIENT_ID" },
  "effectiveDateTime": "2026-10-31T16:00:00Z",
  "result": [
    { "reference": "Observation/eat-volume-obs" },
    { "reference": "Observation/pvat-lad-obs" },
    { "reference": "Observation/pvat-lcx-obs" },
    { "reference": "Observation/pvat-rca-obs" }
  ],
  "conclusion": "CRISP-CT Active Vascular Inflammation detected in LAD (PVAT -63.7 HU > -70.1 HU). Normal total EAT volume (16.3 cm3). High residual cardiovascular risk.",
  "conclusionCode": [
    {
      "coding": [
        {
          "system": "http://snomed.info/sct",
          "code": "443502000",
          "display": "Coronary artery inflammation"
        }
      ]
    }
  ],
  "presentedForm": [
    {
      "contentType": "application/pdf",
      "language": "en",
      "title": "Cardiac Fat Clinical Imaging Report (PDF)",
      "data": "<BASE64_ENCODED_PDF>"
    }
  ]
}
```

---

### 4. `DocumentReference` — Chart Attachment for Epic Hyperspace

Enables clinicians to open the vector diagnostic report directly from the patient's chart:

```json
{
  "resourceType": "DocumentReference",
  "status": "current",
  "docStatus": "final",
  "type": {
    "coding": [
      {
        "system": "http://loinc.org",
        "code": "11526-1",
        "display": "Cardiology Diagnostic imaging study report"
      }
    ]
  },
  "category": [
    {
      "coding": [
        {
          "system": "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category",
          "code": "clinical-note"
        }
      ]
    }
  ],
  "subject": { "reference": "Patient/e.PATIENT_ID" },
  "date": "2026-10-31T16:00:00Z",
  "content": [
    {
      "attachment": {
        "contentType": "application/pdf",
        "language": "en",
        "title": "Cardiac Fat Clinical Imaging Report",
        "data": "<BASE64_ENCODED_PDF>"
      }
    }
  ]
}
```

---

## 4. SMART on FHIR Backend Services Authentication

Epic uses OAuth 2.0 with asymmetric private-key JWT assertions for automated system-to-system integrations:

1. **Client Assertion JWT**:
   - Alg: `RS384` or `RS512`.
   - Claims: `iss` (Client ID), `sub` (Client ID), `aud` (`https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token`), `jti` (unique UUID), `exp` (valid 5 min).
2. **Access Token Request**:
   - `grant_type=client_credentials`
   - `client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer`
   - `client_assertion=<SIGNED_JWT>`
3. **Scopes Required**:
   - `system/Observation.write`
   - `system/DiagnosticReport.write`
   - `system/DocumentReference.write`

---

## 5. Self-Contained HL7 FHIR R4 Mock Server & Sandbox (`fhir_server.py`)

`monai-cardiac-fat-fl` provides an in-process, zero-dependency, standards-compliant HL7 FHIR R4 mock server and sandbox (`src/cardiac_fat_fl/fhir_server.py`) allowing offline validation of all FHIR transactions and open.epic MCP tools without requiring external network connectivity or live Epic test environment credentials:

### Features & Capabilities

- **CapabilityStatement (`GET /r4/metadata`)**: Formatted for US Core 3.1.1 and SMART-on-FHIR conformance.
- **Transaction & Batch Ingestion (`POST /r4` / `POST /r4/Bundle`)**: Atomically processes transaction bundles, resolves internal `urn:uuid:` references, and returns standard `transaction-response` bundles (`201 Created`).
- **Discrete Resource CRUD & Search**: Full search support for `Observation` (by patient, LOINC `84245-0`, RadLex `RID49533`), `DiagnosticReport`, `DocumentReference`, and `Patient`.
- **Reference Resolution**: Automatically translates temporary bundle UUIDs (`urn:uuid:<uuid>`) into canonical REST references (`Observation/<id>`) in diagnostic report results.
- **Dual Deployment Modes**:
  1. **Embedded in Research Dashboard**: Automatically active at `http://127.0.0.1:8050/r4`.
  2. **Standalone Server**: `cardiac-fat-fl fhir-server --port 8080 --storage-dir "fhir_storage"`.

### Testing & Validation with open.epic MCP

AI agents running with the open.epic MCP server can connect directly to this local endpoint:

```bash
# Push study results directly to local mock server
cardiac-fat-fl push "example_source_dir/0506" --fhir-url "http://127.0.0.1:8050/r4"

# Query sandbox state
curl "http://127.0.0.1:8050/r4/_sandbox"
curl "http://127.0.0.1:8050/r4/Observation?patient=0506"
curl "http://127.0.0.1:8050/r4/DiagnosticReport?patient=0506"
```
