# FDA SaMD Regulatory Dossier & IEC 62304 / ISO 14971 Traceability Blueprint

This document provides the foundational **Design History File (DHF)** architecture, **IEC 62304 Traceability Matrix**, **ISO 14971 Risk Management / FMEA**, and **FDA Predetermined Change Control Plan (PCCP)** for transitioning `monai-cardiac-fat-fl` from an Investigational Research Use Only (RUO) platform to a Class II 510(k) Software as a Medical Device (SaMD) submission.

---

## 1. Regulatory Classification & Device Profile

| Regulatory Attribute | Specification |
| :--- | :--- |
| **Regulatory Pathway** | FDA Class II Premarket Notification — **510(k)** |
| **Regulation Number** | **21 CFR § 892.2050** (*Medical Image Management and Processing System*) |
| **Product Codes** | **LLZ** (Radiological Image Processing) / **QIH** (Automated Quantitative Image Processing) |
| **FDA Documentation Level** | **Basic Documentation Level** (Non-autonomous quantitative post-processing with clinician review) |
| **IEC 62304 Safety Class** | **Class B** (Non-serious injury possible if unverified; mitigated by human-in-the-loop DICOM-SEG review) |
| **Primary Standards** | IEC 62304:2006+A1:2015, ISO 14971:2019, IEC 62366-1:2015, DICOM PS3.3 / PS3.15 / PS3.16 (TID 1500) |
| **Reference Benchmark** | Harvard / MGH `CardiacFatQuant` & `DeepHeartSeg` ($\Delta = 0.00\,\text{cm}^3$, $\Delta = 0.00\,\text{HU}$, $r = 1.000$) |

### Intended Use Statement (Template for 510(k) Indications for Use)

> `monai-cardiac-fat-fl` is an automated radiological post-processing software package intended for the three-dimensional segmentation and volumetric/attenuation quantification of Epicardial Adipose Tissue (EAT) and Pericoronary Adipose Tissue (PVAT) from gated and non-gated thoracic and cardiac Computed Tomography (CT) scans. The software outputs standard multi-frame DICOM Segmentation (`1.2.840.10008.5.1.4.1.1.66.4`) overlays, DICOM Structured Reports (`1.2.840.10008.5.1.4.1.1.88.34`, TID 1500), and interactive multi-planar reconstructions (MPR) for visual verification by qualified physicians. Results are intended to supplement, and not replace, standard clinical interpretation.

---

## 2. IEC 62304 Requirements & Verification Traceability Matrix

Every clinical, mathematical, and cybersecurity requirement is traced directly to its source module and automated verification test in the 34-test `pytest` suite:

| Req ID | Requirement Description | Source Implementation | Automated Verification Test (`pytest`) | Risk ID |
| :--- | :--- | :--- | :--- | :--- |
| **REQ-001** | **Adipose Attenuation Window**: Voxel inclusion strictly bounded to $[-190.0, -30.0]\,\text{HU}$ inclusive. | `measurement_schema.py` (`ADIPOSE_HU_MIN`, `ADIPOSE_HU_MAX`) | `test_transforms.py::test_compute_eat_pvat_masksd_attenuation_window` | `HAZ-001` |
| **REQ-002** | **8-Compartment Taxonomy**: Output segmentation ordered as `EAT_Total`, `EAT_LA`, `EAT_LV`, `EAT_RA`, `EAT_RV`, `PVAT_LAD`, `PVAT_LCX`, `PVAT_RCA`. | `measurement_schema.py` (`SegmentationCompartment`), `transforms.py` | `test_transforms.py::test_schema_columns_count_and_order` | `HAZ-002` |
| **REQ-003** | **Anisotropic Physical Volume & EDT**: Voxel volume $V = N \cdot \frac{\Delta x \Delta y \Delta z}{1000}\,\text{cm}^3$ and $10.0\,\text{mm}$ radial PVAT dilation must account for anisotropic spacing. | `measurement_schema.py` (`calculate_volume_cm3`), `transforms.py` | `test_transforms.py::test_calculate_volume_cm3`, `test_anisotropic_edt_sampling_and_aliases` | `HAZ-003` |
| **REQ-004** | **Empty Compartment Handling**: Regions with $0$ voxels must report `0.0` $\text{cm}^3$ volume and `NaN` (omitted NUM in DICOM-SR) for mean HU. | `transforms.py` (`QuantifyAdiposeMetricsd`), `dicom_sr.py` | `test_transforms.py::test_empty_mask_handling`, `test_dicom_sr.py::test_create_cardiac_fat_sr_with_nan` | `HAZ-004` |
| **REQ-005** | **Spatial Coordinate Preservation**: Origin, spacing, and direction cosines preserved identically across NRRD, NIfTI, and DICOM I/O. | `io_metatensor.py` (`read_nrrd_as_metatensor`, `write_eat_pvat_seg_nrrd`) | `test_transforms.py::test_header_preserving_nrrd_io` | `HAZ-003` |
| **REQ-006** | **Reference Equivalency**: Zero numerical drift ($\Delta = 0.00$) between converted ONNX `DeepHeartSeg` weights and foundational MGH pipeline. | `anatomy_segmenter.py` (`UnifiedAnatomySegmenter`) | `test_equivalency.py::test_deepheartseg_onnx_equivalency` | `HAZ-005` |
| **REQ-007** | **DICOM-SEG & TID 1500 Compliance**: Valid SOP Classes `.66.4` and `.88.34` with SNOMED-CT, LOINC, and UCUM codes. | `ohif_exporter.py`, `dicom_sr.py` (`create_cardiac_fat_sr`) | `test_ohif.py::test_export_sample_to_ohif_synthetic`, `test_dicom_sr.py::test_export_cardiac_fat_sr_roundtrip` | `HAZ-002` |
| **REQ-008** | **Cryptographic Model Integrity**: SHA-256 verification of all ONNX and NGC PyTorch weights before deserialization or inference. | `cli.py` (`_verify_model_file_sha256`), `anatomy_segmenter.py` | `test_equivalency.py::test_model_checksum_verification` | `HAZ-006` |
| **REQ-009** | **Secure PACS Transmission**: Enterprise TLS 1.3 with custom hospital Root CA bundle verification for DICOMweb STOW-RS. | `pacs_client.py` (`push_to_dicomweb_stow`), `cli.py` | `test_pacs_client.py::test_push_to_dicomweb_stow_custom_ca_bundle` | `HAZ-007` |
| **REQ-010** | **Privacy-Preserving Federated Learning**: Zero patient image transfer; local `FedBN` isolation; exact Welford moment aggregation. | `fl_client_algo.py`, `fl_stats_executor.py`, `model.py` | `test_fl_client.py::test_nvflare_stats_executor_and_multi_site_aggregation` | `HAZ-008` |
| **REQ-011** | **Host Resource & OOM Protection**: Automatic hardware profiling and worker concurrency clamping to prevent workstation crashes. | `hardware.py` (`detect_hardware`, `validate_worker_concurrency`) | `test_hardware.py::test_validate_worker_concurrency_clamps_unsafe_requests` | `HAZ-009` |

---

## 3. ISO 14971 Clinical & Cybersecurity Hazard Analysis (FMEA)

| Risk ID | Failure Mode / Clinical Hazard | Potential Clinical Harm | Engineered Risk Control (Design Mitigation) | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **HAZ-001** | **Iodinated Contrast Blooming**: High-attenuation lumen contrast ($>150\,\text{HU}$) or calcified plaque ($>130\,\text{HU}$) included in PVAT volume. | Artificial inflation of mean attenuation (HU) mischaracterizing pericoronary inflammation. | Strict upper threshold at $-30.0\,\text{HU}$ deterministically excludes contrast, calcium, and myocardium (`measurement_schema.py`). | Verified (`test_transforms.py`) |
| **HAZ-002** | **Unverified Anatomical Boundary**: Pericardial effusion or severe motion artifact causes inaccurate pericardial sac contour. | Incorrect EAT/PVAT volume reported to ordering clinician. | Exports multi-frame `DICOM-SEG` (`eat_pvat_seg.dcm`) and interactive 2D/3D MPR OHIF viewer for mandatory clinician contour review. | Verified (`test_ohif.py`) |
| **HAZ-003** | **Anisotropic Voxel Distortion**: Non-isotropic CT slices (e.g., $0.6 \times 0.6 \times 3.0\,\text{mm}$) treated as cubic voxels during $10\,\text{mm}$ radial dilation. | Over-dilation along the Z-axis capturing non-pericoronary fat. | `scipy.ndimage.distance_transform_edt` explicitly parameterized with physical `sampling=spacing` in millimeters (`transforms.py`). | Verified (`test_transforms.py`) |
| **HAZ-004** | **Division-by-Zero on Empty Mask**: Small FOV scan missing a coronary territory yields `0` voxels, causing `NaN`/`Inf` crash in PACS report. | Pipeline crash or corrupted DICOM-SR transmitted to PACS. | Empty masks return `0.0` $\text{cm}^3$ volume and omit the optional `NUM` sequence item in DICOM-SR TID 1500 (`dicom_sr.py`). | Verified (`test_dicom_sr.py`) |
| **HAZ-005** | **Silent Fallback on Clinical Scans**: Missing neural network weights cause software to run a synthetic geometric approximation on a patient scan. | Grossly inaccurate clinical segmentation. | Clinical CT guardrail ($\max(D,H,W) \ge 128$) strictly raises `RuntimeError` if ONNX / NGC weights are absent (`anatomy_segmenter.py`). | Verified (`anatomy_segmenter.py`) |
| **HAZ-006** | **Supply-Chain Model Tampering**: Maliciously modified `.onnx` or `.pt` checkpoint injected into `models/` directory (CWE-502). | Arbitrary code execution during `torch.load` or corrupted clinical predictions. | Hardcoded SHA-256 verification before download completion and in-memory prior to `torch.load(..., weights_only=True)`. | Verified (`test_equivalency.py`) |
| **HAZ-007** | **Man-in-the-Middle (MitM) on PACS Push**: Unencrypted or unverified TLS connection when pushing DICOM-SEG/SR to hospital archive. | Interception or alteration of patient DICOM instances in transit. | Enforces TLS verification by default with `--ca-bundle` support for institutional Root CAs (`pacs_client.py`). | Verified (`test_pacs_client.py`) |
| **HAZ-008** | **Cross-Site PHI Leakage / Model Inversion**: Federated training leaks site-specific CT intensity distributions or patient identity. | HIPAA / GDPR breach across participating hospitals. | Zero DICOM transfer; `FedBN` keeps `BatchNorm3d`/`InstanceNorm3d` strictly local; cohort stats use aggregate Welford moments only. | Verified (`test_fl_client.py`) |
| **HAZ-009** | **Workstation Out-of-Memory (OOM) Crash**: Excessive parallel workers exhaust RAM/VRAM on clinical workstation. | Denial of service for co-located clinical software. | `hardware.py` reserves $\ge 2$ CPU cores and $\ge 4\,\text{GB}$ host RAM, clamping `--workers` automatically. | Verified (`test_hardware.py`) |

---

## 4. FDA Predetermined Change Control Plan (PCCP) for Federated AI/ML

In accordance with the FDA's finalized guidance on *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions* (December 2024), `monai-cardiac-fat-fl` incorporates a built-in architecture for safe, post-market multi-center model updates without requiring a new 510(k) submission for each re-training cycle:

1. **Description of Planned Modifications**:
   - Periodic federated weight updates to the 3D `SegResNet` distillation student model (`configs/monai_bundle/train.yaml`) across participating hospital sites to expand scanner vendor and protocol coverage (e.g., photon-counting CT, ultra-low-dose screening CT).
   - **Unchanged Invariants**: The 8-compartment anatomical definition, $[-190.0, -30.0]\,\text{HU}$ adipose window, $10.0\,\text{mm}$ anisotropic PVAT radial distance transform, and 19-column / DICOM-SR TID 1500 output schema remain **strictly frozen**.
2. **Modification Protocol (NVIDIA FLARE `FedProx` + `FedBN`)**:
   - Model training is orchestrated via `CardiacFatClientAlgo` (`src/cardiac_fat_fl/fl_client_algo.py`) using `FedProx` ($\mu > 0$ proximal regularization to prevent catastrophic forgetting) and `FedBN` (local normalization layer isolation).
   - Cohort drift and subgroup coverage are monitored via `CardiacFatStatsExecutor` (`src/cardiac_fat_fl/fl_stats_executor.py`) using exact parallel Welford moments across scanner manufacturers and gating protocols.
3. **Automated Acceptance & Rollback Gates**:
   - Candidate global checkpoints must pass all 34 automated regression tests (`pytest`) and achieve:
     - Volumetric Dice similarity coefficient $\ge 0.90$ across all 8 compartments on locked multi-vendor reference cohorts (MGH benchmark + Stanford AIMI COCA).
     - Pearson correlation $r \ge 0.98$ and mean bias within $\pm 2.0\,\text{cm}^3$ / $\pm 1.5\,\text{HU}$ against the frozen reference ONNX baseline (`cardiac-fat-fl compare-models`).
   - Approved checkpoints are assigned a new cryptographic SHA-256 digest in `KNOWN_MODEL_SHA256` and recorded in `models/checksums.sha256`.

---

## 5. FD&C Act Section 524B Cybersecurity & SBOM Governance

To satisfy mandatory FDA Section 524B "Cyber Device" submission requirements:

- **Continuous Vulnerability Management**: Automated weekly and per-commit scans via `.github/workflows/security.yml` running `pip-audit` (PyPI CVE database), `bandit` (Python SAST), and `trivy` (filesystem and container vulnerability scanning).
- **Software Bill of Materials (SBOM)**: Machine-readable CycloneDX JSON SBOM (`sbom.cyclonedx.json`) generated automatically in CI from `pyproject.toml` dependencies.
- **Coordinated Vulnerability Disclosure**: Documented in [`SECURITY.md`](../../SECURITY.md) with a 48-hour acknowledgment SLA and 72-hour triage timeline.
