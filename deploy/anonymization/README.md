# RSNA Anonymizer (V19) Integration & Cardiac CT Guide

This guide explains how to install, configure, and execute the **RSNA DICOM Anonymizer (V19)** ([github.com/RSNA/anonymizer](https://github.com/RSNA/anonymizer)) alongside `monai-cardiac-fat-fl` for batch de-identification of thousands of cardiac CT scans.

---

## 1. Environment & Setup (Using `uv`)

Because RSNA Anonymizer V19 currently pins `numpy < 2.0.0` and `pydicom < 2.5`, install and run it in an isolated virtual environment using `uv` (which is already installed on the host):

```bash
# Option A: Run directly as an isolated ephemeral tool (Recommended)
uv tool run --from rsna-anonymizer rsna-anonymizer --help

# Option B: Create a persistent standalone virtual environment
uv venv rsna-env --python 3.12
rsna-env\Scripts\activate
uv pip install rsna-anonymizer
```

---

## 2. Cardiac CT De-Identification Invariants

Generic DICOM anonymizers often wipe physical coordinate tags or Hounsfield Unit scaling factors, which causes 3D volume reconstruction and adipose segmentation to fail.

The provided profile `rsna_cardiac_ct_project.json` enforces:

1. **3D Physical Coordinates & Spacing**:
   - `PixelSpacing` `(0028,0030)` and `SliceThickness` `(0018,0050)`
   - `ImagePositionPatient` `(0020,0032)` and `ImageOrientationPatient` `(0020,0037)`
2. **Hounsfield Unit (HU) Calibration**:
   - `RescaleIntercept` `(0028,1052)` and `RescaleSlope` `(0028,1053)` (Strictly required for the $[-190, -30]\text{ HU}$ fat window)
3. **Acquisition & Risk Stratification Covariates**:
   - `KVP` `(0018,0060)`, `ConvolutionKernel` `(0018,1210)`
   - `PatientSex` `(0010,0040)`, `PatientAge` `(0010,1010)`
4. **Deterministic Pseudonyms & Longitudinal Date-Shifting**:
   - Assigns consistent project-scoped IDs (`SITE01_0001`, `SITE01_0002`).
   - Shifts study and series dates by a deterministic offset, preserving interval deltas between follow-up scans while removing real calendar dates.
5. **AI OCR Performance Optimization**:
   - Disables pixel OCR on primary 3D volumetric series (saving days of GPU processing across 500,000 axial slices).

---

## 3. Headless Batch Execution

Run the anonymizer headless against the project configuration:

```bash
rsna-anonymizer -c deploy/anonymization/rsna_cardiac_ct_project.json
```

Once processed, the output folder can be passed directly into the repository pipeline:

```bash
python -m cardiac_fat_fl.cli quantify "anonymized_cohort" --mode research
```

---

## 4. Pre-Flight De-Identification Audit (`cardiac-fat-fl deid-audit`)

To verify that an existing cohort or incoming batch is 100% clean of PHI before launching federated learning or public research benchmarks, run the built-in audit:

```bash
# Audit a single study or full cohort directory
cardiac-fat-fl deid-audit "path/to/cohort" --mode research

# Audit and automatically sanitize any residual PHI leaks
cardiac-fat-fl deid-audit "path/to/cohort" --mode research --sanitize
```
