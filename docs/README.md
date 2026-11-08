# monai-cardiac-fat-fl Documentation Index

This directory contains architectural specifications, regulatory submissions, clinical validation benchmarks, edge-case case studies, and integration guides for **monai-cardiac-fat-fl**.

---

## Documentation Structure

```text
docs/
├── case_studies/                      # Clinical Edge Cases & NV-Reason-CT 3D VLM Spotlights
│   ├── README.md                      # Catalog of clinical case studies
│   ├── nongated_51_bilateral_lung_masses_case_study.png
│   ├── nongated_151_pacemaker_and_valves_case_study.png
│   ├── participant_442_coronary_inflammation_case_study.png
│   └── assets/                        # Supporting high-resolution viewport assets
│
├── figures/                           # Global Publication & Architecture Figures
│   ├── cardiac_fat_fl_architecture_infographic.jpg
│   ├── samd_regulatory_pipeline_big_picture.png
│   ├── coca_edge_cases_and_nv_reason_showcase.png
│   ├── coca_figure1_mpr_orthogonal.png
│   ├── coca_figure2_5model_concordance.png
│   ├── coca_figure3_cohort_analytics.png
│   └── ohif_viewer_snapshot.png
│
├── regulatory/                        # Regulatory SaMD, FDA 510(k), and DHF Evidence
│   ├── FDA_SAMD_REGULATORY_DOSSIER.md # 4-pillar FDA SaMD submission blueprint
│   └── FDA_510k_SaMD_DHF_Evidence_Dossier.pdf # Generated 3-page vector DHF dossier
│
├── manuscript/                        # Scientific Abstracts & Manuscript Drafts
│   ├── coca_full_manuscript.md        # Complete academic journal manuscript
│   ├── coca_cohort_abstract.md        # Conference abstract preprint
│   ├── coca_cohort_abstract.html      # Responsive interactive preprint
│   └── scoping_nv_reason_ct_report_generation.md
│
├── benchmarks/                        # Multi-Model Validation & Parity Data
│   ├── coca_stratified_pilot_results.csv
│   └── coca_stratified_pilot_summary.json
│
├── architecture_review/               # Slide-by-Slide Visual Walkthrough Gallery
│   └── README.md
│
├── ARCHITECTURE_CODE_REVIEW.md        # Comprehensive technical architecture & design audit
├── Cardiac_Segmentation_Architecture.pdf # High-resolution architecture slide deck
├── DICOM_DEIDENTIFICATION_RSNA_V19_GUIDE.md # HIPAA Safe Harbor & RSNA Anonymizer V19
├── EPIC_FHIR_MCP_INTEGRATION.md       # HL7 FHIR R4 & open.epic Model Context Protocol
├── MONAI_DEPLOY_AND_FLARE_INTEGRATION.md # MAP DAG and NVIDIA FLARE 2.5+ federation
├── PARITY_BENCHMARK_ROADMAP.md        # Harvard/MGH CardiacFatQuant equivalence roadmap
├── presentation_deck.html             # Standalone interactive HTML presentation
├── RESEARCH_DASHBOARD_AND_OHIF_WORKSTATION.md # Research Dashboard & embedded OHIF guide
└── samd_big_picture_infographic.html  # Interactive SVG system blueprint
```

---

## Quick Reference Links

- **Clinical Edge Cases**: [`case_studies/README.md`](case_studies/README.md)
- **Regulatory SaMD 510(k)**: [`regulatory/FDA_SAMD_REGULATORY_DOSSIER.md`](regulatory/FDA_SAMD_REGULATORY_DOSSIER.md)
- **EHR & FHIR Interoperability**: [`EPIC_FHIR_MCP_INTEGRATION.md`](EPIC_FHIR_MCP_INTEGRATION.md)
- **MONAI Deploy & NVIDIA FLARE**: [`MONAI_DEPLOY_AND_FLARE_INTEGRATION.md`](MONAI_DEPLOY_AND_FLARE_INTEGRATION.md)
- **Research Dashboard & OHIF Workstation**: [`RESEARCH_DASHBOARD_AND_OHIF_WORKSTATION.md`](RESEARCH_DASHBOARD_AND_OHIF_WORKSTATION.md)
- **DICOM De-identification & PHI Audit**: [`DICOM_DEIDENTIFICATION_RSNA_V19_GUIDE.md`](DICOM_DEIDENTIFICATION_RSNA_V19_GUIDE.md)
- **Architecture Code Review**: [`ARCHITECTURE_CODE_REVIEW.md`](ARCHITECTURE_CODE_REVIEW.md)
