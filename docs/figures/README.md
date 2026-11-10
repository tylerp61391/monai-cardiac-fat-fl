# monai-cardiac-fat-fl Figures & Visual Asset Catalog

This directory contains publication figures, clinical validation visuals, architectural infographics, and UI workstation captures for **monai-cardiac-fat-fl**. Assets are organized into two primary subdirectories according to their domain:

- **[`coca/`](coca/)**: Scientific publication figures, cohort demographic breakdowns, and multi-model concordance benchmarks derived from the **Stanford AIMI COCA** cohort ($N = 1,000$ cases).
- **[`project/`](project/)**: System architecture diagrams, SaMD regulatory pipeline blueprints, and zero-footprint OHIF v3.13 medical workstation screenshots for the software platform.

---

## Directory Structure

```text
docs/figures/
├── README.md                                  # This catalog
├── coca/                                      # Stanford AIMI COCA cohort & scientific figures
│   ├── coca_figure1_mpr_orthogonal.png        # Fig 1: 3-view orthogonal MPR montage (300 DPI)
│   ├── coca_figure2_5model_concordance.png    # Fig 2: 5-model architectural concordance
│   ├── coca_figure3_cohort_analytics.png      # Fig 3: Agatston CAC box plots & DICE heatmap
│   ├── coca_edge_cases_and_nv_reason_showcase.png # 300 DPI executive clinical edge cases slide
│   ├── coca_edge_cases_and_nv_reason_showcase.pdf # Vector PDF executive clinical slide
│   ├── coca_cac_distribution.png              # Cohort Agatston calcium score tier histogram
│   ├── coca_manufacturer_breakdown.png        # Scanner vendor distribution (GE, Siemens, etc.)
│   ├── coca_slice_thickness_breakdown.png     # Reconstruction slice thickness distribution
│   └── coca_vessel_prevalence.png             # Branch-specific CAC prevalence (LAD, LCX, RCA)
│
└── project/                                   # Platform architecture & deployment visuals
    ├── cardiac_fat_fl_architecture_infographic.jpg # End-to-end modernized pipeline infographic
    ├── ohif_viewer_snapshot.png               # Embedded OHIF v3.13 3D Slicer 4-up workstation
    ├── samd_regulatory_pipeline_big_picture.png   # End-to-end clinical SaMD & FL blueprint
    └── samd_regulatory_pipeline_big_picture.pdf   # Vector PDF SaMD regulatory pipeline blueprint
```

---

## 1. Stanford AIMI COCA Figures (`docs/figures/coca/`)

Figures in this folder illustrate clinical validation, biological correlation, and multi-model architectural benchmarking across the Stanford AIMI COCA cohort ($N = 1,000$ patient CT scans).

| File | Resolution / Format | Primary Purpose & Description | Generation Source |
| :--- | :---: | :--- | :--- |
| **`coca_figure1_mpr_orthogonal.png`** | 300 DPI PNG | **Orthogonal 3-View MPR Montage**: High-resolution axial, coronal, and sagittal cross-sections displaying 8-compartment fat overlays with DeepHeartSeg ONNX pericardial boundary. | `scripts/generate_publication_figures.py` |
| **`coca_figure2_5model_concordance.png`** | 300 DPI PNG | **5-Model Concordance Comparison**: Side-by-side mid-ventricular slice comparing Harvard reference, MONAI SegResNet, TotalSegmentator v1, NV-Segment-CTMR, and NV-Segment-CT. | `scripts/generate_publication_figures.py` |
| **`coca_figure3_cohort_analytics.png`** | 300 DPI PNG | **Cohort Analytics & Heatmap**: Panel A shows EAT volume progression across 5 Agatston CAC tiers ($p < 10^{-15}$); Panel B shows the pairwise 5-model DSC correlation heatmap. | `scripts/generate_publication_figures.py` |
| **`coca_edge_cases_and_nv_reason_showcase.png`** | 300 DPI PNG | **Executive Edge Cases Showcase**: Highlights Discordant Phenotypes ($\text{CAC} = 0$ with active PVAT inflammation), anatomical edge cases, and NV-Reason-CT 3D VLM discoveries. | `scripts/generate_edge_cases_showcase_slide.py` |
| **`coca_edge_cases_and_nv_reason_showcase.pdf`** | Vector PDF | Printable vector-rendered slide deck format of the executive clinical edge case showcase. | `scripts/generate_edge_cases_showcase_slide.py` |
| **`coca_cac_distribution.png`** | High-res PNG | **Agatston CAC Tier Breakdown**: Distribution of coronary calcium severity across the 1,000-scan cohort (Zero, Minimal, Mild, Moderate, Severe). | Cohort demographics pipeline |
| **`coca_manufacturer_breakdown.png`** | High-res PNG | **Scanner Vendor Distribution**: Vendor stratification (GE Healthcare, Siemens Healthineers, Philips, Toshiba/Canon). | Cohort demographics pipeline |
| **`coca_slice_thickness_breakdown.png`** | High-res PNG | **Reconstruction Slice Thickness**: Distribution across 1.25 mm–3.0 mm ECG-gated and routine chest series. | Cohort demographics pipeline |
| **`coca_vessel_prevalence.png`** | High-res PNG | **Vessel Prevalence**: Calcification prevalence across specific coronary branches (LAD, LCX, RCA). | Cohort demographics pipeline |

---

## 2. Project & Platform Figures (`docs/figures/project/`)

Figures in this folder illustrate the software platform architecture, FDA Class II SaMD regulatory submission pipeline, and clinical deployment interfaces.

| File | Resolution / Format | Primary Purpose & Description | Referenced In |
| :--- | :---: | :--- | :--- |
| **`cardiac_fat_fl_architecture_infographic.jpg`** | High-res JPEG | **End-to-End System Infographic**: Comprehensive blueprint illustrating 3D CT ingestion, PyTorch / MONAI transforms, 8-compartment adipose quantification, and NVIDIA FLARE federation. | [`README.md`](../../README.md), [`docs/architecture_infographic.html`](../architecture_infographic.html) |
| **`ohif_viewer_snapshot.png`** | High-res PNG | **Embedded OHIF v3.13 Medical Workstation**: Four-up 2x2 grid layout (Axial, 3D MPR Volume, Coronal, Sagittal) with smart scrollbars and 8-layer DICOM-SEG overlays. | [`README.md`](../../README.md), Research Dashboard |
| **`samd_regulatory_pipeline_big_picture.png`** | 300 DPI PNG | **FDA Class II SaMD & FL Blueprint**: Architecture spanning dual-mode privacy gate (clinical identified vs. research RSNA V19), MONAI MAP DAG, and multi-hospital consortium. | [`README.md`](../../README.md), [`docs/regulatory/FDA_SAMD_REGULATORY_DOSSIER.md`](../regulatory/FDA_SAMD_REGULATORY_DOSSIER.md) |
| **`samd_regulatory_pipeline_big_picture.pdf`** | Vector PDF | Printable vector-rendered regulatory architecture diagram included in the automated FDA 510(k) DHF dossier. | [`src/cardiac_fat_fl/regulatory_dossier.py`](../../src/cardiac_fat_fl/regulatory_dossier.py) |
