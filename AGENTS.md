# AGENTS.md — Agent & Developer Guide for monai-cardiac-fat-fl

This document provides concise instructions, environment setup, clinical invariants, and architectural rules for AI coding assistants and human developers working on this repository.

---

## 1. Project Overview & Architecture
`monai-cardiac-fat-fl` modernizes the legacy Harvard / MGH `CardiacFatQuant` pipeline into a unified **Python 3.11+ MONAI / PyTorch** and **NVIDIA FLARE 2.5+** federated learning system with embedded **OHIF v3.13** zero-footprint web visualization.

- **Core Task**: Automated 3D segmentation and volumetric/attenuation quantification of Epicardial Adipose Tissue (EAT) and Pericoronary Adipose Tissue (PVAT) from cardiac CT scans.
- **Privacy & FL**: Coordinates multi-hospital federated model distillation (FedProx + FedBN) and cohort moment aggregation (Welford) with zero patient DICOM/image transfer.
- **Visualization**: Converts studies into web-ready DICOM + DICOM-SEG volumes served via a local DCM4CHEE-compatible static JSON-DICOM server for interactive OHIF 2D MPR and 3D Slicer rendering.

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
python -m cardiac_fat_fl.cli quantify /path/to/cohort --output-csv measurements.csv

# Launch local OHIF MPR viewer for a patient:
python -m cardiac_fat_fl.cli view "path/to/patient" --port 8000 --no-browser

# Run federated client mock step:
python -m cardiac_fat_fl.cli fl-step --data-dir "path/to/patient"
```

---

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
  - Top-Right: Sagittal CT (MPR)
  - Bottom-Left: Coronal CT (MPR)
  - Bottom-Right: 3D Volume Rendering / Segmentations

---

## 6. Git & Commit Guidelines

- **Commit Author**: All commits must be authored as:
  ```bash
  git commit --author="Tyler Patay <tcpatay123@gmail.com>" -m "<message>"
  ```
- **Branch & Target**: Primary branch is `main`.
- **Documentation Math**: Use standard GitHub Flavored Markdown math blocks (`$$ ... $$`). To avoid delimiter-stripping parsing bugs, use named TeX tokens like `\lbrace` and `\rbrace` instead of `\{` and `\}` inside `\left` and `\right`.

