# Research Dashboard & OHIF v3.13 Medical Workstation Integration

## 1. Executive Summary & Overview

The `monai-cardiac-fat-fl` platform integrates an interactive, research-grade web dashboard (`cardiac-fat-fl dashboard`) paired natively with the **Open Health Imaging Foundation (OHIF) Viewer v3.13 Next** zero-footprint medical workstation. 

This environment is engineered to provide cardiovascular researchers, imaging core laboratories, and cardiologists with instant multi-cohort exploration, clinical correlation analytics, and high-fidelity 3D volume/MPR visualization without requiring external PACS servers, heavy desktop installations, or cloud data transfers.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       CARDIAC FAT RESEARCH DASHBOARD (Port 8050)                       │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│     COHORT RESEARCH KPIS      │      INTERACTIVE PLOTS        │     CASE EXPLORER      │
│  - Total Cohort N: 1,000      │  - EAT Volume vs. CAC Score   │  - Natural Human Sort  │
│  - Quantified N: 996 (99.6%)  │  - PVAT HU vs. CAC (CRISP-CT) │  - Multi-Column Sort   │
│  - CAC Scored N: 449 (44.9%)  │  - Adipose Volume by Region   │  - Filter & Deep Dive  │
│  - Vascular Inflam: 520 (52%) │  - Plaque Burden Breakdown    │  - 1-Click OHIF Launch │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
                                               │
                         Click [Segmented] / [View OHIF]
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   OHIF v3.13 NEXT MEDICAL WORKSTATION (/viewer/<pid>/)                 │
├────────────────────┬───────────────────────────────────┬───────────────────────────────┤
│  STUDIES DRAWER    │         AXIAL VIEWPORT            │         3D MPR VOLUME         │
│  (Thumbnails)      │   (Top-Left, Smart Scrollbar)     │   (Top-Right, 3D Slicer Grid) │
├────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│  SEGMENTATION MASK │        CORONAL VIEWPORT           │       SAGITTAL VIEWPORT       │
│  (8-Ch DICOM-SEG)  │   (Bottom-Left MPR Orthogonal)    │   (Bottom-Right MPR Orthogonal│
└────────────────────┴───────────────────────────────────┴───────────────────────────────┘
```

---

## 2. Dashboard Architecture & REST API Contracts

The research dashboard is powered by a multi-threaded Python/Flask server (`src/cardiac_fat_fl/dashboard.py`) and a responsive frontend utilizing Tailwind CSS, Chart.js, and Feather Icons.

### REST API Endpoints

| Endpoint | Method | Response | Description |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | HTML | Serves the single-page responsive dashboard web application. |
| `/api/cohort/summary` | `GET` | JSON | Aggregated cohort metrics: sample counts, EAT/PVAT volume and HU statistics (mean, median, SD, IQR, skew), CAC tier breakdown, and Pearson/Spearman correlation coefficients. |
| `/api/cohort/cases` | `GET` | JSON | Full cohort participant list sorted naturally (`0, 1, 2, ... 10, ... 100`) with scan type, CAC score, risk tier, EAT/PVAT metrics, and 3D segmentation mask availability (`has_seg`). |
| `/api/case/<participant_id>` | `GET` | JSON | Detailed single-case profile containing 19-metric quantification, chamber and coronary vessel breakdown, Agatston CAC tier, and CRISP-CT inflammation status. |
| `/viewer/<participant_id>/` | `GET` | HTML/Assets | Serves the standalone OHIF v3.13 Medical Workstation directly for the specified participant. Triggers dynamic on-the-fly export if DICOM-SEG / OHIF assets are not yet cached on disk. |
| `/patient/<participant_id>/` | `GET` | HTML/Assets | Alias for OHIF viewer route ensuring full backward compatibility with `ohif_viewer.py`. |
| `/api/cohort/switch` | `POST` | JSON | Dynamically switches the active cohort directory on-the-fly without requiring server restart. Expects `{"cohort_dir": "/new/path"}`. |
| `/api/benchmark/run` | `POST` | JSON | Executes the multi-compartment DICE, IoU, RVE%, and Hausdorff surface distance benchmark engine across the cohort. |

---

## 3. Natural Human Numerical Sorting & Interactive Case Explorer

### Alphanumeric vs. Lexicographical Sorting

Standard file systems and basic JavaScript sort algorithms perform lexicographical string sorting, resulting in disjointed ordering such as:

$$\text{Lexicographical}: \quad [0, 1, 10, 100, 101, 102, \dots, 11, 110, \dots, 2, 20, \dots]$$

`monai-cardiac-fat-fl` enforces **Natural Human Numerical Sorting** end-to-end:

$$\text{Natural Sorting}: \quad [0, 1, 2, 3, 4, \dots, 9, 10, 11, \dots, 99, 100, 101, \dots]$$

### Implementation Layers

1. **Cohort Discovery (`src/cardiac_fat_fl/io_metatensor.py`)**:
   ```python
   def _natural_sort_key(name: str):
       return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', name)]
   ```
2. **Dataset Model (`src/cardiac_fat_fl/cohort_research.py`)**:
   Participants discovered across flat or nested folders are stored in natural alphanumeric order.
3. **Backend API (`src/cardiac_fat_fl/dashboard.py`)**:
   The `/api/cohort/cases` endpoint applies regex tokenization across alphanumeric identifiers (e.g., `nongated_1`, `nongated_2`, `nongated_10`).
4. **Interactive Table Frontend**:
   The Case Explorer table implements a bi-directional `naturalSort` comparator on all columns, accompanied by a dedicated `#` consecutive row counter column (`1, 2, 3, ...`) for absolute row indexing independent of participant ID.

### Multi-Column Sortable Table Headers

Users can click any column header in the Case Explorer to sort ascending or descending:
- **`#`**: Row counter index.
- **`PARTICIPANT ID`**: Natural numerical sort.
- **`SCAN TYPE`**: Gated vs. non-gated acquisition.
- **`AGATSTON CAC`**: Numerical Agatston plaque burden.
- **`EAT VOLUME`**: Total Epicardial Adipose Tissue volume ($\text{cm}^3$).
- **`PVAT VOLUME`**: Total Pericoronary Adipose Tissue volume ($\text{cm}^3$).
- **`PVAT MEAN HU`**: CRISP-CT vascular attenuation index.
- **`CRISP-CT STATUS`**: Active inflammation ($> -70.1\text{ HU}$) vs. normal attenuation.
- **`ARTIFACTS`**: Groups cases by segmentation presence (`Segmented` vs. `Unsegmented`), allowing core labs to instantly isolate cases ready for OHIF review or identify unquantified scans.

---

## 4. Embedded OHIF v3.13 Medical Workstation Integration

The dashboard embeds the complete OHIF v3.13 Medical Workstation directly within the Flask web process (`:8050`):

### 1-Click Launch Workflows

1. **Clickable `Segmented` Badge Pill**:
   In the Case Explorer table, any participant with 3D masks on disk renders a bright green `Segmented` badge pill. Clicking this badge opens the OHIF viewer (`/viewer/<participant_id>/`) in a new tab.
2. **Dedicated `View OHIF` Action Button**:
   The table's `ACTIONS` column provides a direct `View OHIF` button alongside the `Deep Dive` modal trigger.
3. **Modal Header Launcher**:
   Inside the Case Deep Dive modal, the header features a prominent `Open in OHIF` action button with direct target link.

### Dynamic On-The-Fly OHIF Generation

If a participant's CT scan and segmentation mask (`eat_pvat.seg.nrrd`) exist but OHIF-ready DICOM files have not yet been pre-exported, navigating to `/viewer/<participant_id>/` triggers an on-the-fly export:
- Slices the 3D CT volume into standard DICOM CT slices.
- Generates standard multi-frame DICOM-SEG (`1.2.840.10008.5.1.4.1.1.66.4`) with 8 clinical segments.
- Generates Comprehensive 3D DICOM-SR TID 1500 (`1.2.840.10008.5.1.4.1.1.88.34`).
- Writes DCM4CHEE-compatible `studies.json` manifest.
- Delivers the study seamlessly to the browser with zero terminal intervention.

### Layout & Orientation
- **Viewport Layout**: 2x2 multi-planar reconstruction (MPR) following 3D Slicer / radiology convention:
  - **Top-Left**: Axial MPR with smart scrollbar.
  - **Top-Right**: 3D Volume Rendering grid.
  - **Bottom-Left**: Coronal MPR.
  - **Bottom-Right**: Sagittal MPR.
- **Autospin MPR**: Set to `false` (`autoSpinMPR: false`) to avoid distracting rotations during clinical review.

---

## 5. Canonical 4-Way Color Palette Synchronization

To ensure immediate visual familiarity across analytical and diagnostic modalities, colors are hardcoded identically across all 4 visualization interfaces:

| Compartment | Channel | Hex Code | RGB Triplet | Anatomical Role |
| :--- | :---: | :---: | :---: | :--- |
| **EAT Total** | `Ch 0` | `#9333ea` | `(147, 51, 234)` | Total Epicardial Adipose Tissue |
| **EAT LA** | `Ch 1` | `#3b82f6` | `(59, 130, 246)` | Left Atrium epicardial fat |
| **EAT LV** | `Ch 2` | `#06b6d4` | `(6, 182, 212)` | Left Ventricle epicardial fat |
| **EAT RA** | `Ch 3` | `#10b981` | `(16, 185, 129)` | Right Atrium epicardial fat |
| **EAT RV** | `Ch 4` | `#eab308` | `(234, 179, 8)` | Right Ventricle epicardial fat |
| **PVAT LAD** | `Ch 5` | `#f97316` | `(249, 115, 22)` | Left Anterior Descending pericoronary fat |
| **PVAT LCX** | `Ch 6` | `#ef4444` | `(239, 68, 68)` | Left Circumflex pericoronary fat |
| **PVAT RCA** | `Ch 7` | `#ec4899` | `(236, 72, 153)` | Right Coronary Artery pericoronary fat |

### Synchronized Interfaces
1. **OHIF v3.13 Medical Workstation**: Segment rendering overlays and measurement list labels (`ohif_exporter.py`).
2. **3D Slicer DICOM-SEG**: Multi-frame DICOM-SEG metadata color categories (`export_segmentation_to_dicom_seg`).
3. **Clinical Diagnostic Report**: 3-view orthogonal MPR montage (`mpr_montage.png`), PDF vector bar charts, and HTML summary (`clinical_report.py`).
4. **Research Dashboard**: Adipose Volume Distribution by Compartment bar chart and Case Deep Dive modal compartment indicators (`dashboard.py`).

---

## 6. Multi-Series DICOM Collision Resolution

### The Clinical Problem

In real-world institutional cohorts (such as the Stanford COCA dataset), a single patient study directory frequently contains multiple DICOM series:
- The target 3D cardiac CT series (e.g., 40 to 45 axial slices).
- A secondary single-slice scout view, localizer, or patient positioning scan (1 slice) stored in the same folder.

When reading the directory via SimpleITK:
```python
series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(study_path)
# series_ids may return: ['1.2.840...scout', '1.2.840...axial_volume']
```
Picking the first series ID non-deterministically selected the 1-slice scout view, causing downstream 3D deep learning models and MONAI transforms to throw volume shape errors (`ValueError: Expected 3D volume`).

### The Robust Resolution

`src/cardiac_fat_fl/io_metatensor.py` inspects all detected series IDs and selects the series with the maximum number of file slices:

```python
# Identify series with maximum slice count to avoid 1-slice scout collisions
best_series_id = max(
    series_ids,
    key=lambda sid: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(study_path, sid))
)
```

This ensures 100% deterministic selection of the true 3D cardiac acquisition across complex clinical archives.

---

## 7. Cohort Translation & Edge Case Telemetry

### Translating to New Cohorts

The research dashboard works out-of-the-box on any cardiac imaging repository. Supported directory layouts include:

1. **Standard Flat Layout**:
   ```text
   /path/to/cohort/
     ├── PATIENT_001/
     │     ├── dicom/ (or .nrrd / .nii.gz)
     │     └── eat_pvat.seg.nrrd (optional)
     ├── PATIENT_002/
     └── cardiac_fat_measurements.csv
   ```
2. **Gated / Non-Gated Split (e.g., Stanford COCA)**:
   ```text
   /path/to/cohort/
     ├── 0/ ... 499/ (gated cases)
     └── nongated_0/ ... nongated_499/
   ```
3. **Automated Calcium Score Linking**:
   Drop any CSV or Excel file containing participant IDs and Agatston scores (e.g., `calcium_scores.csv`, `coca_agatston.xlsx`) into the cohort root. The engine automatically indexes and matches scores to participants.

### Root-Cause Diagnostics on Stanford COCA (1,000 Cases)

Across the Stanford COCA 1,000-case cohort, the pipeline achieved a **99.6% automated quantification success rate** (996 / 1,000 cases). A detailed diagnostic root-cause analysis was conducted on the 4 unquantified cases:

| Case ID | Nature of Failure | Root Cause Analysis | Remediation |
| :--- | :--- | :--- | :--- |
| **`135`** | Segmentation failure | Directory contained 45 axial CT slices and 1 stray scout slice. The loader picked the 1-slice series. | **Resolved** by multi-series max-slice selection. |
| **`763`** | Segmentation failure | Directory contained 40 axial CT slices and 1 stray scout slice. | **Resolved** by multi-series max-slice selection. |
| **`nongated_195`** | DICOM I/O error | Corrupt DICOM metadata: tag `(0018, 0088) SpacingBetweenSlices` is explicitly hardcoded to `0.000000` and `SeriesInstanceUID` is `None` in the raw vendor file, causing SimpleITK C++ exception. | Quarantine / vendor re-export. |
| **`nongated_210`** | Anatomical localization failure | Severely truncated acquisition: scan contains only 6 slices ($3.0\text{ cm}$ total $Z$-coverage), failing anatomical chamber localization. | Excluded for insufficient $Z$-axis field of view. |

