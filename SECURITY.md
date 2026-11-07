# Security and Healthcare Compliance Policy

This policy outlines the security standards, regulatory research boundaries, vulnerability reporting protocols, and HIPAA/GDPR privacy-preserving safeguards for the `monai-cardiac-fat-fl` framework.

---

## 1. Supported Versions

Security updates, vulnerability patches, and integrity fixes are actively applied to the following release series:

| Version | Supported | Security Maintenance Status |
| :--- | :---: | :--- |
| `0.1.x` / `1.0.x` | Yes | Active development and security maintenance |
| `< 0.1.0` | No | Deprecated development prototypes |

---

## 2. Reporting a Vulnerability

The project maintainers take software security and patient privacy seriously. If you identify a security vulnerability, data leakage risk, or potential exploit in this repository:

1. **Do Not Open a Public Issue**: Do not file public GitHub issues for undisclosed vulnerabilities or zero-day exploits.
2. **Contact the Security Lead**: Send a report directly to:
   - **Primary Contact**: `tcpatay123@gmail.com`
   - **Subject Line**: `[SECURITY VULNERABILITY] monai-cardiac-fat-fl`
3. **Report Contents**: Please include:
   - Component or CLI command affected (`cardiac-fat-fl push`, `fl_client_algo`, OHIF server, etc.).
   - Reproducible proof-of-concept steps or minimal sample script.
   - Potential impact (e.g., PHI disclosure, privilege escalation, denial of service).
   - Any proposed remediation.
4. **Response Service Level Agreement (SLA)**:
   - **Initial Acknowledgment**: Within 48 hours of receipt.
   - **Triage and Status Updates**: Within 72 hours with an estimated remediation timeline.
   - **Coordinated Disclosure**: Fixes will be published via GitHub Security Advisories and release tags before public disclosure.

---

## 3. HIPAA and GDPR Healthcare Compliance Architecture

`monai-cardiac-fat-fl` is designed for privacy-preserving clinical research and multi-center federated learning across independent hospital systems and academic medical centers.

### Technical Safeguards (45 CFR § 164.312)

- **Zero DICOM / Image Transfer Across Firewalls**: Raw CT volumes, slices, and individual patient segmentation masks remain isolated inside each hospital site's local storage. No Protected Health Information (PHI) or Electronic Protected Health Information (ePHI) crosses institutional boundaries.
- **Federated Model Distillation (FedProx + FedBN)**:
  - During federated training, only convolutional model weights are transmitted to the central NVIDIA FLARE server.
  - In accordance with Federated Batch Normalization (FedBN) principles, all batch and instance normalization layers ($(\gamma, \beta)$ parameters and running statistics) remain local to the site, mitigating model inversion and membership inference attacks.
- **Federated Cohort Analytics (Welford Moments)**:
  - Multi-site cohort summary statistics are aggregated exclusively using parallel Welford summary moments ($N_k, \sum x, \sum x^2, \min, \max$).
  - Individual subject measurements are never transmitted to the central server, eliminating patient re-identification risk while preserving exact mathematical equivalence to pooled statistics.

### HIPAA Safe Harbor (45 CFR § 164.514(b)(2)) & DICOM PS 3.15 Annex E

The platform enforces full compliance with the 18 statutory de-identification categories under the HIPAA Privacy Rule Safe Harbor method and the DICOM PS 3.15 Annex E Basic Application Level Confidentiality Profile:

| Statutory PHI Category | Target DICOM Attribute | Research Mode Treatment (`--mode research`) |
| :--- | :--- | :--- |
| **Names** | `(0010,0010)` `PatientName` | Replaced with deterministic pseudonym: `Participant {pid}` |
| **Geographic Subdivisions** | `(0008,0080)` `InstitutionName`, `(0008,0081)` `InstitutionAddress` | Scrubbed or normalized to `RESEARCH_COHORT` |
| **Dates (Birth, Study, Content)** | `(0010,0030)` `PatientBirthDate`, `(0008,0020)` `StudyDate`, etc. | Normalized (`19000101`) or shifted by deterministic participant hash offset |
| **Contact (Phone, Fax, Email)** | `(0010,2154)` `PatientTelephoneNumbers`, etc. | Fully scrubbed (empty or removed) |
| **Social Security Numbers** | Private tags / `(0010,1000)` `OtherPatientIDs` | Stripped and purged |
| **Medical Record Numbers (MRN)** | `(0010,0020)` `PatientID` | Replaced with research participant identifier: `Participant {pid}` |
| **Accession / Order Numbers** | `(0008,0050)` `AccessionNumber` | Pseudonymized as `ACC_{pid}` or generated hash |
| **Device & Station Identifiers** | `(0018,1000)` `DeviceSerialNumber`, `(0008,1010)` `StationName` | Replaced with `ANONYMIZED_STATION` / scrubbed |
| **Unique Identifiers (UIDs)** | `(0020,000D)` `StudyInstanceUID`, `(0020,000E)` `SeriesInstanceUID`, `(0008,0018)` `SOPInstanceUID` | Cryptographically derived via deterministic SHA-256 HMAC rooted under standard DICOM prefix (`1.2.826.0.1.3680043.8.498.`) |
| **Radiologist / Physician Identifiers** | `(0008,0090)` `ReferringPhysicianName`, `(0008,1050)` `PerformingPhysicianName` | Stripped / empty |

### Dual-Mode Operational Boundary: Clinical vs. Research

To satisfy both hospital production deployment and multi-institutional academic collaboration without compromising patient privacy or EHR functionality, the platform implements a strict **Dual-Mode Operational Boundary**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLINICAL MODE                                        │
│  [Hospital Trust Boundary] -- Trusted Intranet / Local PACS / Internal Holoscan DAG    │
│  • Retains PatientName, MRN (PatientID), AccessionNumber, StudyDate                    │
│  • Allows DICOM-SEG, DICOM-SR, Encapsulated PDF & FHIR to match patient EHR chart      │
│  • Prohibited from crossing external network firewalls or joining FL consortiums       │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           ▲
                                           │ cardiac-fat-fl anonymize
                                           │ (or RSNA Anonymizer V19)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RESEARCH MODE                                        │
│  [Zero-Trust Research Boundary] -- Multi-Center FL / Open Datasets / Public Repositories│
│  • Deterministic pseudonymization: "Participant {pid}", synthetic UIDs, shifted dates  │
│  • Pre-flight PHI audit gate: `cardiac-fat-fl deid-audit` (fails if raw PHI is leaked) │
│  • 100% Geometry Invariant: Preserves PixelSpacing, Position, Orientation, Rescale HU  │
│  • Authorized for NVIDIA FLARE multi-hospital federated training and cohort analytics  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Clinical Mode (`--mode clinical`)**:
   - **Intended Scope**: Automated on-premise inference inside hospital networks via MONAI Deploy Application Package (MAP) or PACS watchdog daemon.
   - **Identifier Retention**: Preserves official patient demographics (`PatientName`, `PatientID`, `AccessionNumber`, `StudyDate`, `ReferringPhysicianName`) to guarantee that generated DICOM-SEG (`.66.4`), DICOM-SR TID 1500 (`.88.34`), DICOM Encapsulated PDF (`.104.1`), and HL7 FHIR R4 resources (`DiagnosticReport`, `DocumentReference`, `Observation`) file correctly into the patient's existing Electronic Health Record (Epic, Cerner) and PACS archive.
   - **Boundary Enforcement**: Clinical-mode artifacts are locked within the institutional trust zone. They are strictly blocked from being submitted to central federated learning aggregation servers.

2. **Research Mode (`--mode research`)**:
   - **Intended Scope**: Multi-center clinical trials, public benchmark cohorts (e.g., Stanford COCA), and cross-hospital NVIDIA FLARE federated model training.
   - **Zero-PHI Enforcement**: All 18 HIPAA Safe Harbor identifiers are replaced with pseudonyms or shifted.
   - **Pre-Flight PHI Gate**: The pipeline includes a mandatory de-identification audit gate (`audit_study_anonymization()`) that halts processing with an exit code 1 if unscrubbed patient names, real MRNs, or clinical institution identifiers are discovered.

### RSNA Anonymizer (V19) Integration & Sandboxed Batch Execution

For large-scale institutional cohorts (1,000s of CT scans), `monai-cardiac-fat-fl` integrates with the official **RSNA Anonymizer (V19)** standard:

- **Standard Profile**: Located at [`deploy/anonymization/rsna_cardiac_ct_project.json`](deploy/anonymization/rsna_cardiac_ct_project.json), configuring DICOM PS 3.15 Annex E de-identification rules, custom deterministic UID hashing, and 30-day date shifting.
- **Dependency Isolation**: RSNA Anonymizer pins legacy dependencies (`pydicom < 2.5`, `numpy < 2.0`). To prevent corrupting the modern PyTorch / NumPy 2.x environment, the tool is executed via isolated ephemeral virtualization:

  ```bash
  uv tool run --from rsna-anonymizer rsna-anonymizer \
    --project deploy/anonymization/rsna_cardiac_ct_project.json \
    --input "raw_hospital_cohort/" \
    --output "research_cohort_deid/"
  ```

- **Radiometric & Volumetric Integrity Invariants**:
  The anonymization profile strictly whitelists 3D geometry and HU calibration tags to ensure zero distortion of epicardial and pericoronary fat quantification:
  - `(0028,0030)` `PixelSpacing` (in-plane mm spacing)
  - `(0018,0050)` `SliceThickness` (axial mm thickness)
  - `(0020,0032)` `ImagePositionPatient` (3D physical origin)
  - `(0020,0037)` `ImageOrientationPatient` (directional cosines)
  - `(0028,1052)` `RescaleIntercept` & `(0028,1053)` `RescaleSlope` (Hounsfield Unit calibration)
  - Bypasses pixel OCR modification for 3D volumetric CT, avoiding slice corruption and maximizing throughput (> 100 slices/sec).

### Pre-Flight De-Identification Audit Engine (`cardiac-fat-fl deid-audit`)

To ensure fail-closed security before sharing data or initiating federated learning rounds, the platform includes a native DICOM de-identification auditor:

```bash
# Audit a cohort or single participant study
cardiac-fat-fl deid-audit "path/to/cohort" --mode research

# Audit with automatic sanitization of any detected residual PHI leaks
cardiac-fat-fl deid-audit "path/to/cohort" --mode research --sanitize --output audit_report.json
```

- **Audit Findings**: Scans every DICOM instance for real patient names, non-pseudonymous IDs, un-redacted accession numbers, real birth dates, and unapproved institutional headers.
- **Fail-Closed Gate**: Returns exit code `1` if any violation is detected, preventing accidental data exfiltration in CI/CD pipelines and automated batch scripts.

---

## 4. Regulatory Status and Research Use Disclaimer

### Research Use Only (RUO) Notice

> [!IMPORTANT]
> `monai-cardiac-fat-fl` is developed strictly for **Investigational and Scientific Research Use Only (RUO)** in accordance with 21 CFR § 812.5.
> It has **not** been evaluated, cleared, or approved by the United States Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other national or international regulatory authority for primary diagnostic, prognostic, or therapeutic clinical decision-making.

- **Non-Diagnostic Software**: This software is not intended to serve as a standalone Software as a Medical Device (SaMD) or to replace the clinical judgment of board-certified radiologists, cardiologists, or licensed physicians.
- **Clinical Validation**: Any clinical research use involving human subjects must be conducted under an Institutional Review Board (IRB) or equivalent Ethics Committee approved protocol with appropriate patient consent or waiver of authorization.

---

## 5. PACS Transmission Security (DICOMweb and DIMSE)

When exporting cardiac fat segmentations and structured reports to institutional PACS archives:

- **Enterprise TLS Transport Security**: All DICOMweb STOW-RS communications should use HTTPS/TLS 1.3.
- **Custom Enterprise Root CA Bundles**: Support for private institutional Certificate Authorities (CAs) is provided via the `--ca-bundle` flag in `cardiac-fat-fl push` and the `ca_cert` parameter in `push_to_dicomweb_stow()`.
- **Insecure Mode Warning**: The `--insecure` (`-k`) flag is provided solely for isolated local test environments (e.g., local Orthanc Docker containers without SSL) and must **never** be enabled in production clinical networks.
- **Least-Privilege PACS Access**: STOW-RS authentication supports scoped Bearer JWT tokens and HTTP Basic Auth credentials with write-only permissions (`STOW-RS` / `instances`) rather than administrative privileges.

---

## 6. Secure Software Development Lifecycle (SSDLC)

The repository implements automated safeguards to maintain supply chain security and software integrity:

1. **Cryptographic Checksums (SHA-256)**:
   - Pretrained ONNX model files and NVIDIA NGC / MONAI Model Zoo bundle checkpoints are verified against hardcoded SHA-256 cryptographic digests before deserialization or execution:
     - `deepheartseg_heart.onnx`: `4015c53fa3fcdaa8b93e9d528f2410b8d34cdba07286e69b4f09a6e3c9644553`
     - `deepheartseg_loc.onnx`: `94fbafee767d872f3fb0bae364d5922387963034bf59f75e8ea5971ba0306c93`
     - `monai_wholebody_ct_segmentation` (NVIDIA NGC SegResNet `model.pt`): `80b429fb4b080df11c9ed0b0bdaa8a615ff083921bb213a512cf285afbc4e3fe`
     - `monai_wholebody_ct_segmentation` (`model_lowres.pt`): `c3ab55eb979785fdcb30690872c210bbeee73d79a170c32fdaa1eca117779f90`
2. **Automated Vulnerability Scanning**:
   - Continuous dependency vulnerability audits using `pip-audit`.
   - Static Application Security Testing (SAST) for common Python vulnerabilities using `bandit`.
   - Automated container image vulnerability scanning with Trivy.
3. **Secret Hygiene**:
   - Zero hardcoded passwords, tokens, private keys, or credentials in codebase or commit history.
