# AGENTS.md — Agent & Developer Guide for monai-cardiac-fat-fl

This document provides concise instructions, environment setup, clinical invariants, and architectural rules for AI coding assistants and human developers working on this repository.

---

## 1. Project Overview & Architecture
`monai-cardiac-fat-fl` modernizes the legacy Harvard / MGH `CardiacFatQuant` pipeline into a unified **Python 3.11+ MONAI / PyTorch** and **NVIDIA FLARE 2.5+** federated learning system with embedded **OHIF v3.13** zero-footprint web visualization.

- **Core Task**: Automated 3D segmentation and volumetric/attenuation quantification of Epicardial Adipose Tissue (EAT) and Pericoronary Adipose Tissue (PVAT) from cardiac CT scans.
- **Privacy & FL**: Coordinates multi-hospital federated model distillation (FedProx + FedBN) and cohort moment aggregation (Welford) with zero patient DICOM/image transfer.
- **Visualization & PACS**: Converts studies into web-ready DICOM + DICOM-SEG volumes served via a local DCM4CHEE-compatible static JSON-DICOM server for interactive OHIF 2D MPR and 3D Slicer rendering.
- **PACS Archive Push**: Natively exports and transmits standard multi-frame DICOM-SEG (`1.2.840.10008.5.1.4.1.1.66.4`) and CT volumes to institutional PACS via DICOMweb (STOW-RS) and DIMSE (C-STORE).

---

## 2. Environment & Execution Commands

### Virtual Environment
- **Windows**: `.venv\Scripts\python.exe`
- **Linux/macOS**: `.venv/bin/python`
- Managed via `uv` or `pip`. Project is installed in editable mode (`pip install -e .` or `uv pip install -e .`).

### Verification & Testing
Run all unit and integration tests before committing:
```bash
pytest -v
```

### CLI Quick Reference
```bash
# End-to-end quantification across a cohort directory:
cardiac-fat-fl quantify /path/to/cohort --output-dir /path/to/cohort --save-seg

# With intermediate anatomical masks (heart.seg.nrrd + monai_wholebody_ct_segmentation_1-5mm_masks.seg.nrrd + coronary_sulci.seg.nrrd):
cardiac-fat-fl quantify /path/to/cohort --save-seg --save-intermediate

# Run quantification on included benchmark scans:
cardiac-fat-fl quantify example_source_dir --save-seg

# Launch local OHIF MPR viewer for a patient:
cardiac-fat-fl view "example_source_dir/0506" --port 8000 --no-browser

# Push DICOM-SEG to PACS archive (DICOMweb STOW-RS or DIMSE C-STORE):
cardiac-fat-fl push "example_source_dir/0506" --url "http://localhost:8042/dicom-web/studies"
cardiac-fat-fl push "example_source_dir/0506" --host 192.168.1.50 --port 104 --called-aet PACS_AET

# Run federated client mock step:
cardiac-fat-fl federated-train --data-dir "example_source_dir"
```

### Common Agent Workflows & Task Recipes

#### Recipe 1: "I have a CT scan at <path> and want the app to run and bring it up in OHIF so I can see results"
When a user provides a CT scan or participant directory and asks to process and visualize it:
1. **Check for Existing Segmentations**: Inspect if `<patient_dir>/eat_pvat.seg.nrrd` already exists.
2. **Quantify First (if missing)**: If segmentation does not exist, run the quantification pipeline to generate the 8-channel masks and measurement CSV:
   ```bash
   python -m cardiac_fat_fl.cli quantify "<cohort_or_study_dir>" --save-seg
   ```
   *(Note: `quantify` expects a cohort directory containing participant subfolders like `cohort/<patient_id>/scan.nrrd` or DICOM slices).*
3. **Launch OHIF Viewer (Background)**:
   ```bash
   python -m cardiac_fat_fl.cli view "<patient_dir>" --port 8000 --no-browser
   ```
4. **Respond to User**: Provide the interactive local URL: `http://127.0.0.1:8000/patient/<patient_id>/` and summarize the computed EAT/PVAT volumes ($cm^3$) and mean attenuation values (HU).

#### Recipe 2: "Compute fat measurements across a cohort"
```bash
# Standard high-throughput run:
cardiac-fat-fl quantify "/path/to/cohort" --output-dir "/path/to/cohort" --save-seg

# With full intermediate anatomical files (heart.seg.nrrd + monai_wholebody_ct_segmentation_1-5mm_masks.seg.nrrd + coronary_sulci.seg.nrrd):
cardiac-fat-fl quantify "/path/to/cohort" --output-dir "/path/to/cohort" --save-seg --save-intermediate
```
Verify that `cardiac_fat_measurements.csv` contains the exact 19-column schema.

#### Recipe 3: "Use NVIDIA NGC monai_wholebody_ct_segmentation for Cardiac Chambers"
- The anatomy segmentation engine uses NVIDIA NGC's official `monai_wholebody_ct_segmentation` bundle (1.5 mm isotropic SegResNet).
- Relevant macro-vascular and cardiac chamber labels predicted directly by neural network:
  - Aorta (`7`), Pulmonary Artery (`49`), Myocardium (`44`), Left Atrium (`45`), Left Ventricle (`46`), Right Atrium (`47`), Right Ventricle (`48`).
- Pericardium is segmented via converted zero-drift DeepHeartSeg ONNX weights (`heart.seg.nrrd`).
- Coronary sulci centerlines (LAD `1`, LCX `2`, RCA `3`) are derived downstream from chamber boundary grooves and exported separately as `coronary_sulci.seg.nrrd` to preserve pristine model provenance.


---

#### Recipe 4: "Push study results to a PACS archive"
When a user asks to store or transmit results to a hospital or research PACS:
1. **Verify DICOM-SEG exists**: Ensure `eat_pvat_seg.dcm` exists in the participant directory (or run `export-ohif` first if missing).
2. **DICOMweb STOW-RS (Orthanc, dcm4chee-arc, Cloud PACS)**:
   ```bash
   cardiac-fat-fl push "<study_or_export_dir>" --url "http://localhost:8042/dicom-web/studies"
   ```
3. **Traditional DICOM DIMSE C-STORE (Hospital PACS)**:
   ```bash
   cardiac-fat-fl push "<study_or_export_dir>" --host 192.168.1.50 --port 104 --called-aet CENTRAL_PACS
   ```
4. **Include Full CT Series**: Add `--include-ct` if the destination archive requires both the CT volume and segmentation overlay together:
   ```bash
   cardiac-fat-fl push "<study_or_export_dir>" --include-ct --url "http://localhost:8042/dicom-web/studies"
   ```


## 3. Strict Clinical & Mathematical Invariants

Whenever modifying preprocessing, segmentation, transforms, or reporting:

1. **Adipose Attenuation Window**:
   - Strictly `[-190.0, -30.0]` Hounsfield Units (`-190.0 <= HU <= -30.0`).
   - Defined in `cardiac_fat_fl.measurement_schema` (`ADIPOSE_HU_MIN = -190.0`, `ADIPOSE_HU_MAX = -30.0`).

2. **8 Segmentation Compartments** (saved to `eat_pvat.seg.nrrd` / `eat_pvat_seg.dcm`):
   - Channel 0: `EAT_Total` (all fat within pericardium)
   - Channel 1: `EAT_LA` (Left Atrium epicardial fat)
   - Channel 2: `EAT_LV` (Left Ventricle epicardial fat)
   - Channel 3: `EAT_RA` (Right Atrium epicardial fat)
   - Channel 4: `EAT_RV` (Right Ventricle epicardial fat)
   - Channel 5: `PVAT_LAD` (Left Anterior Descending pericoronary fat, 10 mm radial dilation)
   - Channel 6: `PVAT_LCX` (Left Circumflex pericoronary fat, 10 mm radial dilation)
   - Channel 7: `PVAT_RCA` (Right Coronary Artery pericoronary fat, 10 mm radial dilation)

3. **19-Column Schema** (`cardiac_fat_measurements.csv`):
   - Exactly: `participant_id` + 9 regions (`EAT_Total`, `EAT_LA`, `EAT_LV`, `EAT_RA`, `EAT_RV`, `PVAT_LAD`, `PVAT_LCX`, `PVAT_RCA`, `PVAT_Total`) × 2 metrics (`volume_cm3`, `mean_attenuation_hu`).
   - `PVAT_Total` is the union of LAD, LCX, and RCA masks.
   - Empty masks MUST return `0.0` for volume and `NaN` (represented as blank/null in CSV) for mean HU.

4. **Coordinate & Header Integrity**:
   - Origin, spacing, direction cosines, and coordinate systems must be preserved exactly between input NRRD/DICOM, intermediate tensors, and output segmentation files.
   - DICOM-SEG exports must adhere to SOP Class UID `1.2.840.10008.5.1.4.1.1.66.4`.

---

## 4. Federated Learning (NVIDIA FLARE) Rules

- **Zero DICOM Transfer**: No raw scans, patient identifiers, or segmentation masks cross client firewalls.
- **FedBN (Domain Adaptation)**: Feature normalization layers (`BatchNorm3d`, `InstanceNorm3d`) remain strictly local to each hospital site to prevent inter-scanner calibration drift; only convolutional and linear weights are globally aggregated.
- **Federated Analytics (Mode 2)**: Distributed statistical moments are computed using parallel Welford aggregation ($N_k, \sum x, \sum x^2, \min, \max$), guaranteeing 100% mathematical equivalency to pooled data without sharing individual patient rows.

---

## 5. UI & OHIF Workstation Conventions

- **Autospin MPR**: Defaults to `OFF` (`autoSpinMPR: false` in OHIF configuration) to ensure stable diagnostic presentation upon viewer launch.
- **Viewport Layout**: Uses standard 3D Slicer clinical convention:
  - Top-Left: Axial CT (Primary view)
  - Top-Right: 3D Volume Rendering / Segmentations
  - Bottom-Left: Coronal CT (MPR)
  - Bottom-Right: Sagittal CT (MPR)

---

## 6. Git & Commit Guidelines

- **Commit Author**: All commits must be authored as:
  ```bash
  git commit --author="Tyler Patay <tcpatay123@gmail.com>" -m "<message>"
  ```
- **Branch & Target**: Primary branch is `main`.
- **Documentation Math**: Use standard GitHub Flavored Markdown math blocks (`$$ ... $$`). To avoid delimiter-stripping parsing bugs, use named TeX tokens like `\lbrace` and `\rbrace` instead of `\{` and `\}` inside `\left` and `\right`.

