# Clinical Edge Cases & NV-Reason-CT 3D VLM Spotlights

This directory curates peer-reviewed, publication-grade visual case studies (300 DPI composite figures and vector PDFs) demonstrating the multi-modal synergy between **monai-cardiac-fat-fl** automated 3D cardiac fat quantification and **NVIDIA NV-Reason-CT** (3D ViT + Qwen3.5-4B) zero-shot clinical vision-language dictation on the Stanford COCA cohort.

---

## Catalog of Case Studies

| Case Study | Participant | Clinical Discovery / Pathology | Key AI Findings & Synergies | Deliverables |
| :--- | :---: | :--- | :--- | :--- |
| **Bilateral Lung Masses & Left Diaphragm Elevation** | `nongated_51` | Incidental bilateral lower lobe lung masses directly abutting pericardium; left hemidiaphragm elevation. | • DeepHeartSeg ONNX pericardial mesh + HU adipose window (`[-190, -30] HU`) excludes 100% of masses (zero false-positive bleed).<br>• NV-Reason-CT independently dictates *"Bilateral lower lobe lung masses. Left hemidiaphragm elevation."* | [PNG](nongated_51_bilateral_lung_masses_case_study.png)<br>[PDF](nongated_51_bilateral_lung_masses_case_study.pdf) |
| **Pacemaker, Leads & Valve Replacements** | `nongated_151` | Post-operative cardiomegaly with left chest wall pacemaker pulse generator, transvenous RA/RV leads, and AVR/MVR. | • Adipose HU thresholding rejects metallic beam hardening (>3,000 HU) and dark streak bands without corrupting EAT/PVAT.<br>• NV-Reason-CT autonomously catalogs all hardware, leads, and cardiomegaly. | [PNG](nongated_151_pacemaker_and_valves_case_study.png)<br>[PDF](nongated_151_pacemaker_and_valves_case_study.pdf) |
| **Severe Coronary Microvascular Inflammation** | `442` | Dissociation of normal EAT volume (`89.3 cm³`) with critical LCX (`-57.1 HU`, `+13.0 HU` shift) & RCA (`-64.3 HU`) inflammation. | • Demonstrates why volume-alone tools miss high-risk patients.<br>• NV-Reason-CT independently detects *"Marked increase in vessels adjacent to coronaries consistent with inflammatory infiltrates."* | [PNG](participant_442_coronary_inflammation_case_study.png)<br>[PDF](participant_442_coronary_inflammation_case_study.pdf) |
| **Zero CAC Paradox & Invisible Coronary Risk** | `nongated_140` | Zero Agatston score (`CAC = 0.0`, Zero Tier) dissociated from high EAT burden (`103.3 cm³`) and diffuse multi-vessel pericoronary inflammation (`PVAT: -51.1 HU`, `+19.0 HU` shift; RCA `-44.9 HU`, `+25.2 HU` shift). | • Directly exposes the critical diagnostic blind spot of calcium scoring (blind to non-calcified lipid-rich plaque and active vascular inflammation).<br>• 10mm radial EDT PVAT identifies severe multi-vessel perivascular edema (LAD `-53.6 HU`, LCX `-52.2 HU`, RCA `-44.9 HU`).<br>• NV-Reason-CT confirms cardiomegaly and postoperative changes concordant with elevated cardiovascular risk. | [PNG](nongated_140_zero_cac_invisible_risk_case_study.png)<br>[PDF](nongated_140_zero_cac_invisible_risk_case_study.pdf) |

---

## Directory Layout

```text
docs/case_studies/
├── README.md                                                   # This catalog and clinical guide
├── nongated_51_bilateral_lung_masses_case_study.png            # 300 DPI 4-panel publication composite
├── nongated_51_bilateral_lung_masses_case_study.pdf            # Vector PDF for print / manuscript submission
├── nongated_140_zero_cac_invisible_risk_case_study.png         # 300 DPI 4-panel publication composite
├── nongated_140_zero_cac_invisible_risk_case_study.pdf         # Vector PDF for print / manuscript submission
├── nongated_151_pacemaker_and_valves_case_study.png           # 300 DPI 4-panel publication composite
├── nongated_151_pacemaker_and_valves_case_study.pdf           # Vector PDF for print / manuscript submission
├── participant_442_coronary_inflammation_case_study.png       # 300 DPI 4-panel publication composite
├── participant_442_coronary_inflammation_case_study.pdf       # Vector PDF for print / manuscript submission
└── assets/                                                     # Raw cropped OHIF viewport snapshots
    ├── nongated_51_lll_mass_z25.jpg
    └── nongated_51_rll_mass_z28.jpg
```

---

## Generating & Reproducing Figures

To regenerate any of the case study figures at 300 DPI:

```bash
# Participant nongated_51
python scripts/generate_nongated_51_case_study.py

# Participant nongated_140 (Zero CAC Paradox)
python scripts/generate_nongated_140_case_study.py

# Participant nongated_151
python scripts/generate_nongated_151_case_study.py

# Participant 442
python scripts/generate_participant_442_case_study.py
```
