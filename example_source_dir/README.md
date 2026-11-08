# Example Source Cohort (`example_source_dir`)

This directory contains the reference cardiac CT cohort ($N=3$: `0506`, `0909`, `0987`) from the foundational Harvard / MGH `CardiacFatQuant` dataset, packaged with turnkey outputs for testing, parity verification, and zero-footprint web visualization.

---

## Directory Organization

Each participant directory is organized into a clean root containing the primary study deliverables, with multi-model comparison segmentations cleanly segregated into `segmentation_models/`:

```text
example_source_dir/
├── 0506/                              # Participant 0506
│   ├── img.nrrd                       # Primary 3D cardiac CT volume
│   ├── heart.seg.nrrd                 # DeepHeartSeg ONNX pericardium envelope
│   ├── coronary_sulci.seg.nrrd        # Geometric coronary sulci centerlines (LAD, LCX, RCA)
│   ├── eat_pvat.seg.nrrd              # Primary 8-compartment EAT/PVAT segmentation
│   ├── cardiac_fat_measurements.csv   # Validated 19-column clinical metrics
│   ├── cardiac_fat_report.pdf         # 2-Page clinical report with MPR snapshots & VLM
│   ├── cardiac_fat_report.html        # Interactive responsive report
│   ├── eat_pvat_report.dcm            # DICOM Encapsulated PDF (SOP .104.1)
│   ├── cardiac_fat_fhir.json          # HL7 FHIR R4 Bundle (DiagnosticReport, Observations)
│   ├── mpr_montage.png                # 3-view orthogonal MPR montage
│   ├── nv_reason_clinical_narrative.txt # NV-Reason-CT generative clinical impression
│   ├── ohif_export/                   # Embedded OHIF v3.13 workstation slices & DICOM-SEG
│   │
│   └── segmentation_models/           # [ORGANIZED] Multi-Model Segmentation Benchmarks
│       ├── harvard_cardiacfatquant/   # Harvard / MGH circ-ml/CardiacFatQuant
│       │   ├── eat_pvat_harvard_cfq.seg.nrrd
│       │   ├── harvard_cfq_measurements.csv
│       │   └── eat_pvat_legacy_voronoi.seg.nrrd
│       ├── monai_wholebody/           # NVIDIA NGC monai_wholebody_ct_segmentation (1.5mm)
│       │   ├── eat_pvat_monai_wholebody.seg.nrrd
│       │   ├── monai_cardiac_fat_measurements.csv
│       │   └── monai_wholebody_ct_segmentation_1-5mm_masks.seg.nrrd
│       ├── totalsegmentator/          # TotalSegmentator (1.5mm nnU-Net Task253)
│       │   ├── eat_pvat_totalseg.seg.nrrd
│       │   └── totalseg_measurements.csv
│       ├── nvsegment_ctmr/            # NVIDIA NV-Segment-CTMR (VISTA-3D 346-class foundation)
│       │   ├── eat_pvat_nvsegment.seg.nrrd
│       │   ├── nvsegment_measurements.csv
│       │   └── nv_segment_ctmr_masks.seg.nrrd
│       └── nvsegment_ct/              # NVIDIA NV-Segment-CT (VISTA-3D 133-class CT foundation)
│           ├── eat_pvat_nvsegment_ct.seg.nrrd
│           ├── nvsegment_ct_measurements.csv
│           └── nv_segment_ct_masks.seg.nrrd
│
├── 0909/                              # Participant 0909 (same clean layout)
├── 0987/                              # Participant 0987 (same clean layout)
│
├── cardiac_fat_measurements.csv       # Cohort-wide aggregated measurements table
├── multi_model_smoke_test_results.csv # 5-model cross-comparison results
├── multi_model_smoke_test_summary.json# Statistical metrics across all 5 models
└── parity_verification_*/             # Automated Harvard-vs-MONAI parity audit cache
```
