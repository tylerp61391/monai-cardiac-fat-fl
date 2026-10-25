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
- **De-Identification & DICOM Confidentiality**:
  - The standalone OHIF exporter and PACS push clients operate with standard de-identification tooling compatible with DICOM PS 3.15 Annex E Basic Application Level Confidentiality Profile.

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
