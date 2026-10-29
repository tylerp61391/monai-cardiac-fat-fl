# AGENTS.md — Agent & Developer Guide for monai-cardiac-fat-fl

This document is automatically loaded into agent context. It provides the codebase map, token-saving navigation rules, environment setup, clinical invariants, and task recipes for working on `monai-cardiac-fat-fl`.

---

## 1. Project Overview & Architecture

`monai-cardiac-fat-fl` modernizes the foundational Harvard / MGH `CardiacFatQuant` pipeline into a unified **Python 3.11+ MONAI / PyTorch** and **NVIDIA FLARE 2.5+** federated learning system with embedded **OHIF v3.13** zero-footprint web visualization.

- **Core Task**: Automated 3D segmentation and volumetric/attenuation quantification of Epicardial Adipose Tissue (EAT) and Pericoronary Adipose Tissue (PVAT) from cardiac CT scans.
- **Privacy & FL**: Coordinates multi-hospital federated model distillation (FedProx + FedBN) and cohort moment aggregation (Welford) with zero patient DICOM/image transfer.
- **Visualization & PACS**: Converts studies into web-ready DICOM + DICOM-SEG volumes served via a local DCM4CHEE-compatible static JSON-DICOM server for interactive OHIF 2D MPR and 3D Slicer rendering.
- **PACS Archive Push**: Natively exports and transmits standard multi-frame DICOM-SEG (`1.2.840.10008.5.1.4.1.1.66.4`), DICOM-SR TID 1500 (`1.2.840.10008.5.1.4.1.1.88.34`), and CT volumes to institutional PACS via DICOMweb (STOW-RS) and DIMSE (C-STORE).

---

## 2. Codebase Map & Token Efficiency Guide

To minimize token usage, **do not read `README.md` (583 lines) or full large files** unless explicitly editing them. Jump directly to the relevant module and line range below:

### Source Modules (`src/cardiac_fat_fl/`)

| File | Lines | Key Symbols & Line Ranges | Responsibility |
| :--- | :---: | :--- | :--- |
| `measurement_schema.py` | 148 | `ADIPOSE_HU_MIN/MAX`, `SegmentationCompartment`, `ReportedRegion`, `MEASUREMENT_CSV_COLUMNS`, `calculate_volume_cm3` | **Read first** for data contracts, enums, and 19-column CSV validation. |
| `anatomy_segmenter.py` | 577 | `AnatomyMasks`, `UnifiedAnatomySegmenter` | 3D anatomy engine: NGC `monai_wholebody_ct_segmentation` (1.5mm SegResNet), DeepHeartSeg ONNX pericardium, and geometric coronary sulci derivation. |
| `transforms.py` | 439 | `ComputeEatPvatMasksd`, `QuantifyAdiposeMetricsd` | MONAI dictionary transforms for 8-compartment fat masking, 10mm anisotropic EDT radial dilation, and volume/HU calculation. |
| `io_metatensor.py` | 551 | `discover_participants`, `read_nrrd_as_metatensor`, `write_eat_pvat_seg_nrrd` | Header-preserving 3D I/O for NRRD, NIfTI, and DICOM series. |
| `dicom_sr.py` | 345 | `create_cardiac_fat_sr`, `export_cardiac_fat_sr` | DICOM-SR TID 1500 (`1.2.840.10008.5.1.4.1.1.88.34`) structured report builder using SNOMED-CT / LOINC / UCUM codes. |
| `pacs_client.py` | 320 | `find_dicom_instances_to_push`, `push_to_dicomweb_stow`, `push_to_dimse_cstore` | PACS archive push via DICOMweb STOW-RS (multipart/related) and DIMSE C-STORE (`pynetdicom`). |
| `ohif_exporter.py` | 2299 | **L1–603**: `export_ct_to_dicom_series`, `export_segmentation_to_dicom_seg`, `generate_ohif_dicom_json`; **L604–2021**: *Embedded HTML/CSS/JS templates (SKIP unless editing UI)*; **L2022–2299**: `export_sample_to_ohif` | Exports CT slices, multi-frame DICOM-SEG, DICOM-SR, `studies.json`, and standalone OHIF v3.13 workstation assets. |
| `ohif_viewer.py` | 244 | `launch_ohif_viewer` | Local multi-threaded HTTP server (`127.0.0.1:8000`) serving OHIF studies. |
| `cli.py` | 749 | **L51**: `_process_one_participant`; **L176**: `quantify`; **L361**: `federated-train`; **L399**: `federated-stats`; **L418**: `compare-models`; **L501**: `view`; **L524**: `export-ohif`; **L546**: `hardware`; **L553**: `download-models`; **L627**: `push` | Typer CLI entrypoints (`cardiac-fat-fl`). |
| `fl_client_algo.py` | 409 | `CardiacFatClientAlgo` | NVFLARE client training/evaluation executor (FedProx + FedBN). |
| `fl_stats_executor.py` | 203 | `CardiacFatStatsExecutor`, `aggregate_site_moments` | NVFLARE federated Welford statistical moments ($N, \sum x, \sum x^2, \min, \max$). |
| `model.py` | 128 | `create_eat_pvat_segresnet`, `filter_fedbn_state_dict` | MONAI 3D SegResNet architecture and FedBN normalization layer filtering. |
| `hardware.py` | 219 | `detect_hardware`, `validate_worker_concurrency` | Host CPU/RAM/GPU profiler and OOM-safe worker concurrency throttling. |
| `monai_deploy.py` | 390 | `CardiacCTStudyLoaderOp`, `CardiacFatQuantifierOp`, `DicomSegAndSrWriterOp`, `CardiacFatQuantApplication` | MONAI Deploy Application Package (MAP) DAG for clinical Holoscan / Informatics Gateway deployment. |

### Test Suite (`tests/`)

- `conftest.py` (224L): Synthetic 3D cardiac CT phantom fixtures (`synthetic_ct_fixture`, `synthetic_nrrd_scan`, `synthetic_cohort_dir`).
- Unit/Integration tests: `test_transforms.py` (274L), `test_fl_client.py` (180L), `test_pacs_client.py` (148L), `test_dicom_sr.py` (119L), `test_cli.py` (101L), `test_monai_deploy.py` (130L), `test_ohif.py` (73L), `test_equivalency.py` (71L), `test_hardware.py` (60L).

---

## 3. Environment, Testing & Task Recipes

### Virtual Environment & Test Execution

- **Windows Python**: `.venv\Scripts\python.exe`
- **Linux/macOS Python**: `.venv/bin/python`
- **Run Full Test Suite** (39 tests; run before committing):

  ```bash
  .venv\Scripts\python.exe -m pytest -v
  ```

### Recipe 1: Process a CT Scan and Launch OHIF Viewer

When a user provides a CT scan or participant directory and asks to process and visualize it:

1. **Check for Existing Segmentations**: Inspect if `<patient_dir>/eat_pvat.seg.nrrd` already exists.
2. **Quantify First (if missing)**: Run quantification on the parent cohort directory (add `--save-intermediate` for `heart.seg.nrrd`, `monai_wholebody_ct_segmentation_1-5mm_masks.seg.nrrd`, and `coronary_sulci.seg.nrrd`):

   ```bash
   python -m cardiac_fat_fl.cli quantify "<cohort_or_study_dir>" --save-seg
   ```

3. **Launch OHIF Viewer (Background)**:

   ```bash
   python -m cardiac_fat_fl.cli view "<patient_dir>" --port 8000 --no-browser
   ```

4. **Respond to User**: Provide `http://127.0.0.1:8000/patient/<patient_id>/` and summarize computed EAT/PVAT volumes ($\text{cm}^3$) and mean attenuation (HU).

### Recipe 2: Anatomy Segmentation Provenance (`monai_wholebody_ct_segmentation`)

- Neural network chamber/vessel labels from NGC `monai_wholebody_ct_segmentation` (1.5 mm isotropic SegResNet):
  - Aorta (`7`), Myocardium (`44`), Left Atrium (`45`), Left Ventricle (`46`), Right Atrium (`47`), Right Ventricle (`48`), Pulmonary Artery (`49`).
- Pericardium is segmented via converted zero-drift DeepHeartSeg ONNX weights (`heart.seg.nrrd`).
- Coronary sulci centerlines (LAD `1`, LCX `2`, RCA `3`) are derived geometrically from chamber boundary grooves and exported separately as `coronary_sulci.seg.nrrd`.

### Recipe 3: Push Study Results to a PACS Archive

1. **Verify Exports**: Ensure `eat_pvat_seg.dcm` and `eat_pvat_sr.dcm` exist in `<study_dir>/ohif_export` (or run `cardiac-fat-fl export-ohif "<study_dir>"` if missing).
2. **DICOMweb STOW-RS (Orthanc, dcm4chee-arc, Cloud PACS)**:

   ```bash
   # Add --ca-bundle <path> for custom hospital Root CAs, or --insecure for local testbeds
   cardiac-fat-fl push "<study_dir>" --url "https://pacs.hospital.org/dicom-web/studies" --ca-bundle /path/to/root_ca.pem
   ```

3. **Traditional DICOM DIMSE C-STORE (Hospital PACS)** (add `--include-ct` if the destination archive also requires the underlying CT series):

   ```bash
   cardiac-fat-fl push "<study_dir>" --host 192.168.1.50 --port 104 --called-aet CENTRAL_PACS
   ```

---

## 4. Strict Clinical & Mathematical Invariants

Whenever modifying preprocessing, segmentation, transforms, or reporting:

1. **Adipose Attenuation Window**:
   - Strictly `[-190.0, -30.0]` Hounsfield Units (`-190.0 <= HU <= -30.0`), defined in `measurement_schema.py`.
2. **8 Segmentation Compartments** (`eat_pvat.seg.nrrd` / `eat_pvat_seg.dcm`):
   - Ch `0`: `EAT_Total` | Ch `1`: `EAT_LA` | Ch `2`: `EAT_LV` | Ch `3`: `EAT_RA` | Ch `4`: `EAT_RV`
   - Ch `5`: `PVAT_LAD` (10 mm radial dilation) | Ch `6`: `PVAT_LCX` (10 mm) | Ch `7`: `PVAT_RCA` (10 mm)
3. **19-Column Schema** (`cardiac_fat_measurements.csv`):
   - `participant_id` + 9 regions (`EAT_Total`, `EAT_LA`, `EAT_LV`, `EAT_RA`, `EAT_RV`, `PVAT_LAD`, `PVAT_LCX`, `PVAT_RCA`, `PVAT_Total`) × 2 metrics (`Volume (cm3)`, `Mean Attenuation (HU)`).
   - `PVAT_Total` is the boolean union of LAD, LCX, and RCA masks.
   - Empty masks MUST return `0.0` for volume and `NaN` (blank in CSV; omitted in DICOM-SR NUM sequence) for mean HU.
4. **Coordinate & DICOM SOP Compliance**:
   - Origin, spacing, and direction cosines must be preserved identically across input and output volumes.
   - DICOM-SEG SOP Class UID: `1.2.840.10008.5.1.4.1.1.66.4`.
   - DICOM-SR Comprehensive 3D SR SOP Class UID: `1.2.840.10008.5.1.4.1.1.88.34` (TID 1500).

---

## 5. Federated Learning & OHIF UI Conventions

- **Zero DICOM Transfer**: No raw scans, patient identifiers, or segmentation masks cross client firewalls.
- **FedBN**: Normalization layers (`BatchNorm3d`, `InstanceNorm3d`) remain strictly local to each hospital site; only convolutional/linear weights are aggregated.
- **Federated Analytics**: Uses parallel Welford aggregation ($N_k, \sum x, \sum x^2, \min, \max$) for 100% mathematical equivalence to pooled data.
- **OHIF Workstation**: Autospin MPR defaults to `OFF` (`autoSpinMPR: false`). 2x2 layout follows 3D Slicer convention (Top-Left: Axial, Top-Right: 3D Volume, Bottom-Left: Coronal, Bottom-Right: Sagittal).

---

## 6. Git & Commit Guidelines

- **Commit Author**: All commits must be authored as:

  ```bash
  git commit --author="Tyler Patay <tcpatay123@gmail.com>" -m "<message>"
  ```

- **Branch & Target**: Primary branch is `main`.
- **Documentation Math**: Use GitHub Flavored Markdown math blocks (`$$ ... $$`) with named TeX tokens (`\lbrace`, `\rbrace`) inside `\left` and `\right`.
