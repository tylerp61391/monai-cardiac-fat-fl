# DICOM De-Identification, RSNA Anonymizer (V19) Integration & Dual-Mode Governance

This document establishes the clinical, engineering, and regulatory guide for de-identifying cardiac CT datasets in `monai-cardiac-fat-fl`. It covers the official **RSNA Anonymizer (V19)** integration, HIPAA Safe Harbor (45 CFR § 164.514(b)(2)) compliance, DICOM PS 3.15 Annex E Basic Confidentiality Profile conformance, pre-flight PHI audit gating, and the **Dual-Mode Operational Boundary** (Clinical vs. Research).

---

## 1. Regulatory & Standards Alignment

Processing cardiac CT studies across academic medical centers and clinical production environments requires strict adherence to international healthcare privacy frameworks:

- **HIPAA Privacy Rule (45 CFR § 164.514(b)(2))**: The Safe Harbor de-identification standard mandates the removal or transformation of 18 specific categories of Protected Health Information (PHI).
- **DICOM PS 3.15 Annex E**: Defines the *Basic Application Level Confidentiality Profile* and associated Clean Pixel Data Options, specifying exact tag-level actions (delete, empty, replace with dummy values, or hash).
- **GDPR (Regulation (EU) 2016/679)**: Recital 26 and Article 4(5) regarding pseudonymization, preventing re-identification without separately stored attribution keys.

### The 18 HIPAA Safe Harbor Statutory PHI Categories

The table below details how `monai-cardiac-fat-fl` handles all 18 HIPAA statutory categories under Research Mode (`--mode research`):

| Statutory Category | DICOM Tag & Keyword | Research Mode Treatment (`--mode research`) |
| :--- | :--- | :--- |
| **1. Names** | `(0010,0010)` `PatientName` | Deterministically replaced: `Participant {pid}` (or `PT{pid}`) |
| **2. Geographic subdivisions** | `(0008,0080)` `InstitutionName`, `(0008,0081)` `InstitutionAddress` | Normalized to `RESEARCH_COHORT` / scrubbed |
| **3. Dates** | `(0010,0030)` `PatientBirthDate`, `(0008,0020)` `StudyDate`, `(0008,0021)` `SeriesDate`, `(0008,0022)` `AcquisitionDate`, `(0008,0023)` `ContentDate` | Shifted by deterministic participant hash offset (e.g., 30 days) or normalized to `19000101` |
| **4. Telephone numbers** | `(0010,2154)` `PatientTelephoneNumbers` | Cleared / removed |
| **5. Fax numbers** | `(0010,2155)` `PatientFaxNumbers` | Cleared / removed |
| **6. Email addresses** | `(0010,2156)` `PatientEmailAddresses` | Cleared / removed |
| **7. Social Security numbers** | Private Vendor Tags / `(0010,1000)` `OtherPatientIDs` | Purged completely |
| **8. Medical Record Numbers (MRN)** | `(0010,0020)` `PatientID` | Replaced with synthetic participant pseudonym `Participant {pid}` |
| **9. Health plan beneficiary numbers** | Private elements | Purged |
| **10. Account numbers** | Private elements | Purged |
| **11. Certificate / license numbers** | Private elements | Purged |
| **12. Vehicle identifiers** | Private elements | Purged |
| **13. Device identifiers & serials** | `(0018,1000)` `DeviceSerialNumber`, `(0008,1010)` `StationName` | Replaced with `ANONYMIZED_STATION` / cleared |
| **14. Web URLs** | Private elements | Purged |
| **15. IP addresses** | Private elements | Purged |
| **16. Biometric identifiers** | Biometric sequence tags | Purged |
| **17. Full-face photographs** | N/A (3D Cardiac CT volumes do not encompass facial surface meshes) | Face-blurring disabled to protect chest geometry |
| **18. Unique identifying numbers / UIDs** | `(0020,000D)` `StudyInstanceUID`, `(0020,000E)` `SeriesInstanceUID`, `(0008,0018)` `SOPInstanceUID`, `(0008,0050)` `AccessionNumber` | Cryptographically derived via deterministic SHA-256 HMAC rooted under prefix `1.2.826.0.1.3680043.8.498.`; accession set to `ACC_{pid}` |

---

## 2. Dual-Mode Operational Boundary: Clinical vs. Research

In clinical healthcare institutions, AI applications operate at the intersection of two opposing requirements:

1. **Hospital Production Ingestion**: Radiologists and clinicians require AI-generated DICOM-SEG overlays (`.66.4`), DICOM-SR TID 1500 reports (`.88.34`), DICOM Encapsulated PDFs (`.104.1`), and HL7 FHIR `DiagnosticReport` / `DocumentReference` records to index directly into the patient's existing Electronic Health Record (Epic, Cerner) and enterprise PACS. Stripping patient identifiers in this context breaks clinical care by misfiling results into an anonymous study bucket.
2. **Research & Federated Learning**: Sharing imaging data across institutions, uploading cohorts to public repositories, or participating in multi-center NVIDIA FLARE federated model training strictly forbids raw PHI.

To resolve this dichotomy, `monai-cardiac-fat-fl` implements a strict **Dual-Mode Operational Boundary**:

```text
                               ┌──────────────────────────────────────────────┐
                               │       Raw Cardiac CT Study / Cohort          │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
         [--mode clinical]                                             [--mode research]
 ┌──────────────────────────────────────────────┐              ┌──────────────────────────────────────────────┐
 │             CLINICAL MODE                    │              │              RESEARCH MODE                   │
 │  • Retains PatientName, MRN (PatientID),     │              │  • Replaces all PHI with pseudonyms          │
 │    AccessionNumber, StudyDate, Physician     │              │    ("Participant {pid}", "ACC_{pid}")        │
 │  • DICOM-SEG & DICOM-SR index to patient MRN │              │  • Deterministic UID SHA-256 HMAC remapping  │
 │  • HL7 FHIR links to real EHR patient chart  │              │  • 30-day date shifting / 1900-01-01 norm    │
 │  • STRICT REQUIREMENT: Locked within local   │              │  • Pre-flight PHI audit gate (fail-closed)   │
 │    hospital intranet / PACS trust boundary   │              │  • Authorized for NVIDIA FLARE multi-center  │
 │  • Prohibited from crossing external firewalls│             │    federated training & public benchmarks    │
 └──────────────────────────────────────────────┘              └──────────────────────────────────────────────┘
```

### CLI Flag Integration Across All Pipelines

The `--mode clinical` and `--mode research` options are unified across all CLI entrypoints:

- `cardiac-fat-fl quantify <cohort> --mode research` (default: audits/enforces de-identification before generating `cardiac_fat_measurements.csv`).
- `cardiac-fat-fl quantify <study> --mode clinical` (processes with full clinical demographics for hospital PACS).
- `cardiac-fat-fl report <study> --mode clinical` (encapsulates real patient metadata into `eat_pvat_report.dcm` and FHIR bundle).
- `cardiac-fat-fl daemon <inbox> --mode clinical` (autonomous PACS router operating inside hospital intranet).
- `cardiac-fat-fl deid-audit <path> --mode research` (audits pre-flight compliance before external sharing).

---

## 3. RSNA Anonymizer (V19) Integration

The **Radiological Society of North America (RSNA)** has modernized its clinical trial de-identification engine into the Python-based [RSNA Anonymizer](https://github.com/RSNA/anonymizer) (V19).

### Dependency Isolation via `uv tool run`

RSNA Anonymizer pins legacy dependencies (`pydicom < 2.5`, `numpy < 2.0`). Because `monai-cardiac-fat-fl` requires modern `numpy >= 2.0` and PyTorch 2.x for accelerated GPU computing, installing `rsna-anonymizer` directly into the project virtual environment would cause package conflicts.

To solve this cleanly, the pipeline executes RSNA Anonymizer through Astral's isolated tool runner (`uv tool run`):

```bash
uv tool run --from rsna-anonymizer rsna-anonymizer \
  --project deploy/anonymization/rsna_cardiac_ct_project.json \
  --input "/path/to/raw_hospital_cohort" \
  --output "/path/to/deidentified_research_cohort"
```

This isolates the RSNA CLI and its legacy dependencies in an ephemeral virtual environment while processing thousands of files natively at maximum speed.

### Project Configuration Specification

The project specification file is located at [`deploy/anonymization/rsna_cardiac_ct_project.json`](../deploy/anonymization/rsna_cardiac_ct_project.json):

```json
{
  "profile": "basic",
  "project_name": "CardiacFatQuant_Research",
  "date_shift_days": 30,
  "keep_uids": false,
  "hash_uids": true,
  "replace_names": true,
  "patient_name_prefix": "Participant_",
  "patient_id_prefix": "PT_",
  "remove_private_tags": true,
  "clean_descriptors": true,
  "keep_descriptors": [
    "SeriesDescription",
    "StudyDescription",
    "ProtocolName"
  ],
  "keep_tags": [
    "PixelSpacing",
    "SliceThickness",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "Rows",
    "Columns",
    "RescaleIntercept",
    "RescaleSlope",
    "KVP",
    "XRayTubeCurrent",
    "ConvolutionKernel",
    "GantryDetectorTilt"
  ],
  "remove_tags": [
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "PatientBirthDate",
    "PatientAge",
    "PatientSex",
    "PatientAddress",
    "PatientTelephoneNumbers"
  ],
  "options": {
    "clean_pixel_data": false,
    "enable_ocr": false,
    "enable_face_blurring": false
  }
}
```

---

## 4. Preserving 3D Volumetric Geometry and HU Invariants

> [!CRITICAL]
> Standard DICOM anonymizers frequently strip or zero out geometry or rescale tags by default, which corrupts 3D volumetric reconstruction and renders epicardial / pericoronary fat quantification invalid.

`monai-cardiac-fat-fl` enforces **zero tolerance** for geometric or radiometric tag modification:

1. **In-Plane Spacing (`PixelSpacing` `0028,0030`)**: Physical millimeter dimensions of each pixel $(dx, dy)$. Stripping this makes voxel volume calculation ($\text{cm}^3$) impossible.
2. **Slice Spacing (`SliceThickness` `0018,0050` / `SpacingBetweenSlices` `0018,0088`)**: Axial millimeter displacement ($dz$). Must be preserved for 3D Voronoi Euclidean Distance Transforms (EDT) and 10 mm radial dilation.
3. **Physical Origin (`ImagePositionPatient` `0020,0032`)**: Physical coordinates $(x_0, y_0, z_0)$ of the upper-left voxel in patient coordinates. Required for natural slice sorting and 3D affine matrix construction.
4. **Direction Cosines (`ImageOrientationPatient` `0020,0037`)**: Row and column directional unit vectors. Essential for orthogonal MPR reconstruction (Axial, Coronal, Sagittal) and 3D Slicer alignment.
5. **HU Calibration (`RescaleIntercept` `0028,1052` and `RescaleSlope` `0028,1053`)**: Converts raw 12-bit detector values into physical Hounsfield Units:
   $$\text{HU} = \text{Raw} \times \text{RescaleSlope} + \text{RescaleIntercept}$$
   Stripping these tags shifts adipose tissue completely outside the $[-190, -30\text{ HU}]$ quantification window, resulting in $0\text{ cm}^3$ volume detection.
6. **Pixel OCR De-Identification Bypass**: Secondary capture or ultrasound scans often burn patient names into pixels, requiring Optical Character Recognition (OCR). In 3D volumetric cardiac CT, pixels represent calibrated X-ray attenuation. Enabling pixel OCR de-identification wastes massive CPU cycles and risks corrupting tissue values. The profile explicitly sets `"clean_pixel_data": false`.

---

## 5. Automated Pre-Flight De-Identification Audit (`cardiac-fat-fl deid-audit`)

Before exporting imaging studies, sharing cohorts with academic partners, or launching federated learning simulation rounds, datasets should pass the automated pre-flight de-identification audit.

### CLI Usage

```bash
# Audit a single study or full cohort directory (e.g., 1,000 cases)
cardiac-fat-fl deid-audit "path/to/cohort" --mode research

# Audit with automatic sanitization of any detected residual PHI leaks
cardiac-fat-fl deid-audit "path/to/cohort" --mode research --sanitize --output audit_report.json
```

### Audit Logic & Fail-Closed Gate

The audit engine inspects every DICOM slice in the target directory across all 18 HIPAA Safe Harbor attributes:

- **Patient Name**: Flags any real names (e.g. `DOE^JOHN`) that do not match the authorized pseudonymous pattern (`Participant {pid}`, `PT{pid}`, `COCA_{pid}`).
- **Patient ID / MRN**: Detects unscrubbed clinical MRNs and validates pseudonymous formats.
- **Accession Number**: Flags active hospital radiology order numbers.
- **Dates**: Detects un-shifted study dates or birth dates that expose patient age $> 89$.
- **Institutional Metadata**: Flags un-sanitized hospital names and facility addresses.
- **Fail-Closed Security**: In `--mode research`, the command prints an itemized violation ledger and exits with code `1`, halting automated CI/CD pipelines, container jobs, or federated training runs.

### Programmatic Audit API

The auditor is available directly in Python via `cardiac_fat_fl.anonymizer`:

```python
from pathlib import Path
from cardiac_fat_fl.anonymizer import audit_study_anonymization

report = audit_study_anonymization(Path("cohort/participant_001"), mode="research")

if not report.is_compliant:
    print(f"Compliance Failure: {len(report.violations)} violations detected.")
    for v in report.violations:
        print(f"  [{v.tag}] {v.keyword}: {v.description}")
else:
    print("Study passed pre-flight PHI de-identification audit.")
```

---

## 6. Native Batch De-Identification (`cardiac-fat-fl anonymize`)

For high-throughput execution without external tool dependencies, `monai-cardiac-fat-fl` includes a native batch de-identification command:

```bash
cardiac-fat-fl anonymize "/path/to/raw_hospital_cohort" -o "/path/to/deidentified_cohort" --mode research
```

### Execution Characteristics

1. **Parallel Multi-Core Processing**: Leverages host CPU cores via multiprocessing to process high-resolution cardiac CT volumes at over 50–100 slices/sec.
2. **Deterministic Pseudonymization**: Derives repeatable participant IDs and cryptographic SHA-256 UIDs so longitudinal studies retain cross-series consistency.
3. **Automatic Date Shifting**: Applies a consistent participant-level pseudo-random date offset, preserving temporal intervals between multi-phase or follow-up scans while obscuring the true calendar date.
4. **Automatic Geometry Verification**: Validates that all output DICOM files preserve physical coordinate tags and HU calibration.

---

## 7. Integration with PACS Watchdog Daemon & NVIDIA FLARE

### PACS Watchdog Daemon (`cardiac-fat-fl daemon`)

The clinical PACS watchdog daemon operates under dual-mode governance:

```bash
# Hospital Intranet Auto-Routing (Clinical Mode):
cardiac-fat-fl daemon "/var/monai/inbox" \
  --outbox "/var/monai/outbox" \
  --archive "/var/monai/archive" \
  --pacs-url "https://pacs.hospital.org/dicom-web/studies" \
  --mode clinical

# Research Staging Auto-Routing (De-Identified Mode):
cardiac-fat-fl daemon "/var/monai/research_inbox" \
  --outbox "/var/monai/research_outbox" \
  --pacs-url "https://research-pacs.consortium.org/dicom-web/studies" \
  --mode research
```

- When configured with `--mode clinical`, incoming studies are processed and routed directly to the hospital PACS with patient demographics preserved.
- When configured with `--mode research`, the daemon automatically triggers pre-flight de-identification, scrubs all 18 HIPAA Safe Harbor tags, hashes UIDs, and audits the result before dispatching to the research archive.

### Multi-Center Federated Learning (`cardiac-fat-fl federated-train` / `federated-simulate`)

In federated learning with NVIDIA FLARE 2.5+:

1. **Local Isolation**: Raw CT data never leaves the hospital site.
2. **Client-Side Pre-Flight Gate**: Prior to model distillation or Welford moment extraction, the client algorithm executes `audit_study_anonymization(mode="research")` to verify that local participant data is strictly de-identified.
3. **Consortium Confidence**: Guarantees zero accidental leakage of patient identifiers in local experiment logs, tensor summaries, or DICOM metadata.
