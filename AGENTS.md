# AGENTS.md — Agent & Developer Guide for monai-cardiac-fat-fl

This document is automatically loaded into agent context. It provides the codebase map, token-saving navigation rules, environment setup, clinical invariants, and task recipes for working on `monai-cardiac-fat-fl`.

---

## 1. Project Overview & Architecture

`monai-cardiac-fat-fl` modernizes the foundational Harvard / MGH `CardiacFatQuant` pipeline into a unified **Python 3.11+ MONAI / PyTorch** and **NVIDIA FLARE 2.5+** federated learning system with embedded **OHIF v3.13** zero-footprint web visualization.

- **Core Task**: Automated 3D segmentation and volumetric/attenuation quantification of Epicardial Adipose Tissue (EAT) and Pericoronary Adipose Tissue (PVAT) from cardiac CT scans.
- **Privacy & FL**: Coordinates multi-hospital federated model distillation (FedProx + FedBN) and cohort moment aggregation (Welford) with zero patient DICOM/image transfer.
- **Visualization & PACS**: Converts studies into web-ready DICOM + DICOM-SEG volumes served via a local DCM4CHEE-compatible static JSON-DICOM server for interactive OHIF 2D MPR and 3D Slicer rendering.
- **PACS Archive Push & HL7 FHIR Bridge**: Natively exports and transmits standard multi-frame DICOM-SEG (`1.2.840.10008.5.1.4.1.1.66.4`), DICOM-SR TID 1500 (`1.2.840.10008.5.1.4.1.1.88.34`), DICOM Encapsulated PDF (`1.2.840.10008.5.1.4.1.1.104.1`), and CT volumes to institutional PACS via DICOMweb (STOW-RS) and DIMSE (C-STORE), integrating directly with hospital HL7 FHIR R4/R5 interoperability buses (DiagnosticReport, DocumentReference, Observation).
- **MONAI Deploy & Kubernetes**: Turnkey MONAI Deploy Application Package (MAP) with 3-operator DAG (`monai_deploy.py`), Docker Compose PACS testbed (`deploy/`), and Kubernetes hospital cluster manifests (`k8s/`).

---

## 2. Codebase Map & Token Efficiency Guide

To minimize token usage, **do not read `README.md` (800+ lines) or full large files** unless explicitly editing them. Jump directly to the relevant module and line range below:

### Source Modules (`src/cardiac_fat_fl/`)

| File | Lines | Key Symbols & Line Ranges | Responsibility |
| :--- | :---: | :--- | :--- |
| `measurement_schema.py` | 151 | `ADIPOSE_HU_MIN/MAX`, `SegmentationCompartment`, `ReportedRegion`, `MEASUREMENT_CSV_COLUMNS`, `calculate_volume_cm3` | **Read first** for data contracts, enums, and 19-column CSV validation. |
| `anatomy_segmenter.py` | 726 | `AnatomyMasks`, `UnifiedAnatomySegmenter` | 3D anatomy engine: NGC `monai_wholebody_ct_segmentation` (1.5mm SegResNet), DeepHeartSeg ONNX pericardium, and geometric coronary sulci derivation. |
| `transforms.py` | 439 | `ComputeEatPvatMasksd`, `QuantifyAdiposeMetricsd` | MONAI dictionary transforms for 8-compartment fat masking, 10mm anisotropic EDT radial dilation, and volume/HU calculation. |
| `io_metatensor.py` | 576 | `discover_participants`, `read_dicom_series_as_metatensor`, `read_nrrd_as_metatensor`, `write_eat_pvat_seg_nrrd` | Header-preserving 3D I/O for NRRD, NIfTI, and DICOM series with natural sorting and multi-series max-slice collision resolution. |
| `clinical_report.py` | 819 | `evaluate_clinical_risk`, `generate_mpr_snapshot`, `generate_clinical_report_html`, `generate_clinical_report_pdf`, `encapsulate_pdf_to_dicom`, `export_clinical_imaging_report` | SOTA Cleerly/HeartFlow-style clinical imaging report (risk stratification, 3-view orthogonal MPR snapshots with 8-compartment overlays, vector PDF, and DICOM Encapsulated PDF SOP `.104.1`). |
| `dicom_sr.py` | 345 | `create_cardiac_fat_sr`, `export_cardiac_fat_sr` | DICOM-SR TID 1500 (`1.2.840.10008.5.1.4.1.1.88.34`) structured report builder using SNOMED-CT / LOINC / UCUM codes. |
| `fhir_client.py` | 714 | `build_eat_volume_observation`, `build_pvat_attenuation_observation`, `build_cardiac_fat_diagnostic_report`, `build_cardiac_fat_document_reference`, `build_cardiac_fat_fhir_bundle`, `push_to_fhir_server`, `export_fhir_bundle`, `create_bundle_for_study` | HL7 FHIR R4 interoperability client and bundle generator (Epic US Core DiagnosticReport, DocumentReference, LOINC 84245-0 discrete EAT observation, RadLex RID49533 PVAT attenuation, open.epic MCP compatibility). |
| `fhir_server.py` | 650 | `FHIRStore`, `create_fhir_blueprint`, `create_fhir_app`, `launch_fhir_server`, `process_transaction_bundle` | Self-contained HL7 FHIR R4 mock server & sandbox (US Core / Epic CapabilityStatement, atomic transaction bundle ingestion, reference rewriting, discrete CRUD/search, and dashboard :8050/r4 embedding). |
| `epic_mcp_server.py` | 460 | `EpicMCPServer`, `launch_mcp_server`, `call_tool`, `handle_message`, `EPIC_TERMINOLOGY_REGISTRY` | Native Model Context Protocol (MCP) JSON-RPC 2.0 server exposing clinical FHIR tools, diagnostic reports, and 3D quantification to AI assistants. |
| `pacs_client.py` | 349 | `find_dicom_instances_to_push`, `push_to_dicomweb_stow`, `push_to_dimse_cstore` | PACS archive push via DICOMweb STOW-RS (multipart/related) and DIMSE C-STORE (`pynetdicom`). |
| `pacs_daemon.py` | 407 | `DaemonConfig`, `PACSWatchdogDaemon`, `is_study_settled`, `find_pending_studies`, `process_study` | Automated clinical PACS watchdog and auto-router daemon (zero-click processing, file settling check, dry-run mode, and fault-tolerant error quarantine). |
| `anonymizer.py` | 440 | `audit_study_anonymization`, `anonymize_dicom_series`, `generate_rsna_v19_project_config` | DICOM PS 3.15 Annex E & HIPAA Safe Harbor de-identification engine, dual-mode (clinical vs. research), pre-flight PHI auditor, and RSNA Anonymizer V19 profile generator. |
| `ohif_exporter.py` | 2299 | **L1–603**: `export_ct_to_dicom_series`, `export_segmentation_to_dicom_seg`, `generate_ohif_dicom_json`; **L604–2021**: *Embedded HTML/CSS/JS templates (SKIP unless editing UI)*; **L2022–2299**: `export_sample_to_ohif` | Exports CT slices, multi-frame DICOM-SEG, DICOM-SR, `studies.json`, and standalone OHIF v3.13 workstation assets. |
| `ohif_viewer.py` | 244 | `launch_ohif_viewer` | Local multi-threaded HTTP server (`127.0.0.1:8000`) serving OHIF studies. |
| `cohort_research.py` | 620 | `CohortDataset`, `CohortParticipant`, `categorize_calcium_score`, `CAC_TIER_ORDER`, `compute_tier_group_comparisons`, `compute_bivariate_regression` | Smart cohort discovery (flat, nested, COCA gated/non-gated with natural numerical sorting), Agatston CAC linking across all 5 tiers, ANOVA / Kruskal-Wallis omnibus testing, pairwise post-hoc tests, and bivariate regression modeling. |
| `publication_figure.py` | 420 | `generate_publication_figure` | 300 DPI publication-grade multi-panel composite figure generator formatted for peer-reviewed cardiovascular imaging journals (*JACC*, *Radiology*, *Circulation*). |
| `dashboard.py` | 2260 | `create_dashboard_app`, `launch_dashboard` | Interactive research dashboard & embedded OHIF workstation server (Flask, Chart.js, KPI scorecards, interactive SVG box plots across 5 CAC tiers with ANOVA/Kruskal scorecards, bivariate regression model, 300 DPI publication figure export endpoint `/api/cohort/publication-figure`, case explorer, `/viewer/<pid>/` launcher, DICE benchmark runner). |
| `segmentation_evaluator.py` | 480 | `SegmentationBenchmarkEngine`, `CohortBenchmarkReport`, `compute_compartment_dice`, `compute_volume_metrics`, `compute_surface_distance_metrics` | Extensible multi-compartment segmentation evaluation, DICE metrics, IoU, RVE, MASD/HD95. |
| `cli.py` | 1714 | **L51**: `_process_one_participant`; **L176**: `quantify`; **L361**: `federated-train`; **L399**: `federated-stats`; **L418**: `federated-simulate`; **L480**: `compare-models`; **L501**: `view`; **L524**: `export-ohif`; **L546**: `hardware`; **L553**: `download-models`; **L627**: `push`; **L934**: `map-run`; **L990**: `daemon`; **L1100**: `report`; **L1310**: `dashboard`; **L1340**: `benchmark`; **L1425**: `fhir-export`; **L1582**: `fhir-server`; **L1620**: `mcp-server`; **L1650**: `parity-audit` | Typer CLI entrypoints (`cardiac-fat-fl`). |
| `coca_parity.py` | 1810 | `run_architectural_parity_audit`, `compute_cardiacfatquant_eat_pvat_masks`, `rasterize_coca_calcium_xml`, `align_eat_pvat_channels_from_cardiacfatquant_header`, `HARVARD_7_RULES` | 4-checkpoint end-to-end architectural parity engine and direct execution bridge for Harvard/MGH `circ-ml/CardiacFatQuant` vs. `monai-cardiac-fat-fl` on Stanford COCA. |
| `nv_reason_reporter.py` | 225 | `is_nv_reason_available`, `clean_vlm_narrative`, `build_clinical_prompt`, `generate_nv_reason_narrative` | NVIDIA NV-Reason-CT 3D Vision-Language AI Radiologist (3D ViT + Qwen3.5-4B), BF16 / SDPA, ML tag sanitization (`KEY FINDINGS:`), and structured narrative report generator. |
| `federated_simulator.py` | 185 | `run_federated_simulation`, `ensure_nvflare_windows_compatibility` | Turnkey SOTA NVIDIA FLARE 2.5+ multi-hospital simulation runner with automated Windows OS compatibility patching. |
| `fl_client_algo.py` | 450 | `CardiacFatClientAlgo`, `CardiacFatTrainer` | NVFLARE client training/evaluation executor and Model Trainer bridging DXO weights to MONAI (FedProx + FedBN). |
| `fl_stats_executor.py` | 203 | `CardiacFatStatsExecutor`, `aggregate_site_moments` | NVFLARE federated Welford statistical moments ($N, \sum x, \sum x^2, \min, \max$). |
| `model.py` | 128 | `create_eat_pvat_segresnet`, `filter_fedbn_state_dict` | MONAI 3D SegResNet architecture and FedBN normalization layer filtering. |
| `hardware.py` | 219 | `detect_hardware`, `validate_worker_concurrency` | Host CPU/RAM/GPU profiler and OOM-safe worker concurrency throttling. |
| `monai_deploy.py` | 412 | `CardiacCTStudyLoaderOp`, `CardiacFatQuantifierOp`, `DicomSegAndSrWriterOp`, `CardiacFatQuantApplication` | MONAI Deploy Application Package (MAP) DAG for clinical Holoscan / Informatics Gateway deployment. |

### Deployment & Infrastructure Layout

| Directory / File | Description | Responsibility |
| :--- | :--- | :--- |
| `k8s/hospital-clinical-testbed.yaml` | Kubernetes manifests | Provisions `clinical-ai` namespace, Orthanc PACS Deployment/Service (8042/4242), and MAP batch Job. |
| `k8s/README.md` | Kubernetes deployment guide | Quickstart instructions for deploying, port-forwarding, pushing studies, and tearing down on clusters. |
| `deploy/docker-compose.monai-deploy.yml` | Turnkey Docker testbed | Spins up local Orthanc PACS on `:8042` and `:4242` for DICOMweb/DIMSE testing. |
| `deploy/app.yaml` | MAP container manifest | OCI image specification obeying MONAI Deploy / NVIDIA Holoscan SDK resource contracts. |

### Test Suite (`tests/`)

- `conftest.py` (224L): Synthetic 3D cardiac CT phantom fixtures (`synthetic_ct_fixture`, `synthetic_nrrd_scan`, `synthetic_cohort_dir`).
- Unit/Integration tests (123 tests): `test_clinical_report.py` (171L), `test_transforms.py` (274L), `test_fl_client.py` (245L), `test_pacs_client.py` (148L), `test_pacs_daemon.py` (160L), `test_dicom_sr.py` (119L), `test_fhir_client.py` (250L), `test_fhir_server.py` (175L), `test_epic_mcp.py` (220L), `test_cli.py` (141L), `test_monai_deploy.py` (189L), `test_ohif.py` (73L), `test_equivalency.py` (71L), `test_hardware.py` (60L), `test_benchmark.py` (110L), `test_cohort_research.py` (170L), `test_dashboard.py` (135L), `test_ci_workflows.py` (95L), `test_coca_parity.py` (172L).

---

## 3. Environment, Testing & Task Recipes

### Virtual Environment & Test Execution

- **Windows Python**: `.venv\Scripts\python.exe`
- **Linux/macOS Python**: `.venv/bin/python`
- **Run Full Test Suite** (123 tests; run before committing):

  ```bash
  .venv\Scripts\python.exe -m pytest -v
  ```

- **Run Documentation Linter (Zero Markdownlint Regressions)**:

  ```bash
  npx markdownlint-cli "**/*.md" --ignore-path .markdownlintignore
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

### Recipe 2: Anatomy Segmentation Provenance & Chamber Backend

- Chamber backends selectable via `--chamber-backend monai|totalsegmentator|nvsegment|nvsegment_ct` (default: `monai`):
  - **`monai`**: NGC `monai_wholebody_ct_segmentation` (1.5 mm isotropic SegResNet). License-free, 100% open-source native PyTorch GPU execution without subprocess overhead.
  - **`totalsegmentator`**: TotalSegmentator (1.5 mm nnU-Net). Replicates original Harvard manuscript segmentation using official open-source `Task253_part3_cardiac` checkpoint (`models/totalseg_v1_cardiac`, SHA-256 `6fcd6e71...`) or TotalSegmentator v2 licensed tasks.
  - **`nvsegment` / `nvsegment_ctmr`**: NVIDIA-MedTech `NV-Segment-CTMR` (VISTA-3D 346-class foundation model). Native 4-chamber CT+MR dual foundation model prompting classes `[6, 149, 151, 152, 153, 154]`.
  - **`nvsegment_ct`**: NVIDIA-MedTech `NV-Segment-CT` (VISTA-3D 133-class CT-dedicated foundation model). Fast, low-VRAM CT foundation model prompting aorta `6` and whole heart envelope `115`, combined with 4-quadrant anatomical chamber partitioning.
  - All backends map to standard anatomical labels: Aorta (`7`), Myocardium (`44`), Left Atrium (`45`), Left Ventricle (`46`), Right Atrium (`47`), Right Ventricle (`48`), Pulmonary Artery (`49`).
- Across ALL tracks, the outer pericardium envelope is segmented via converted zero-drift DeepHeartSeg ONNX weights with physical millimeter spacing curation (`heart.seg.nrrd`), ensuring clinical envelope invariance.
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

### Recipe 4: Execute MONAI Deploy Application Package (MAP) Pipeline

1. **Standalone CLI Execution**:

   ```bash
   python -m cardiac_fat_fl.cli map-run -i "<study_dir_or_scan>" -o "<output_dir>"
   ```

2. **Containerized Execution via MONAI Deploy / Docker**:

   ```bash
   docker run --gpus all --rm \
     -v "/path/to/study:/var/monai/input:ro" \
     -v "/path/to/output:/var/monai/output:rw" \
     ghcr.io/tylerp61391/monai-cardiac-fat-map:1.0.0
   ```

3. **Outputs Written to Output Directory**:
   - `eat_pvat_seg.dcm` (multi-frame DICOM-SEG `1.2.840.10008.5.1.4.1.1.66.4`, 8 segments)
   - `eat_pvat_sr.dcm` (Comprehensive 3D DICOM-SR TID 1500 `1.2.840.10008.5.1.4.1.1.88.34`)
   - `cardiac_fat_measurements.csv` (19-column clinical metrics)
   - `eat_pvat.seg.nrrd` (8-channel NRRD)

### Recipe 5: Deploy & Test on Local / Hospital Kubernetes (`k8s/`)

1. **Deploy Clinical Testbed**:

   ```bash
   kubectl apply -f k8s/hospital-clinical-testbed.yaml
   ```

2. **Access Web UI via Port-Forward**:

   ```bash
   kubectl port-forward svc/orthanc-pacs -n clinical-ai 8043:8042
   ```

   Navigate to `http://localhost:8043` in browser.
3. **Push Patient Study to Kubernetes PACS**:

   ```bash
   cardiac-fat-fl push "example_source_dir/0506" --url "http://localhost:8043/dicom-web/studies" --include-ct
   ```

4. **Tear Down**:

   ```bash
   kubectl delete -f k8s/hospital-clinical-testbed.yaml --ignore-not-found=true
   ```

### Recipe 6: Run Automated PACS Watchdog & Auto-Router Daemon

To run the autonomous daemon monitoring an incoming folder for CT scans:

1. **Safe Dry-Run Mode (Testing & Validation without PACS push)**:

   ```bash
   cardiac-fat-fl daemon "path/to/inbox" --outbox "path/to/outbox" --dry-run
   ```

2. **Continuous Auto-Routing to Institutional PACS via DICOMweb**:

   ```bash
   cardiac-fat-fl daemon "/var/monai/inbox" --outbox "/var/monai/outbox" --archive "/var/monai/archive" --pacs-url "https://pacs.hospital.org/dicom-web/studies"
   ```

3. **Single-Pass (Batch / Cron Mode)**:

   ```bash
   cardiac-fat-fl daemon "staging/incoming" --single-pass --dry-run
   ```

### Recipe 7: Generate SOTA Clinical Imaging Report with MPR Snapshots

To generate a Cleerly/HeartFlow-style clinical diagnostic report with 3-view orthogonal MPR snapshots (Axial, Coronal, Sagittal) and 8-compartment color overlays:

1. **CLI Execution**:

   ```bash
   # Single study execution
   cardiac-fat-fl report "path/to/study_dir" --output-dir "path/to/study_dir/report"

   # Full cohort batch mode (e.g. Stanford COCA, 998 participants in ~60s)
   cardiac-fat-fl report "path/to/cohort_dir" --limit 100 --overwrite
   ```

2. **Generated Deliverables**:
   - `cardiac_fat_report.html` (interactive, responsive, printable HTML with embedded base64 MPR montage)
   - `cardiac_fat_report.pdf` (vector-rendered Letter-sized PDF for EHR chart upload)
   - `eat_pvat_report.dcm` (DICOM Encapsulated PDF SOP `1.2.840.10008.5.1.4.1.1.104.1` for PACS indexing)
   - `mpr_montage.png` (high-res 3-view orthogonal montage with soft-tissue CT windowing & alpha overlays)
3. **HL7 FHIR Interoperability (EHR Integration)**:
   - `eat_pvat_report.dcm` maps automatically to FHIR `DocumentReference` (`application/pdf`) and `DiagnosticReport` (CRISP-CT inflammation summary and risk tier).
   - `eat_pvat_sr.dcm` maps to discrete numeric FHIR `Observation` resources (LOINC `84245-0` for EAT volume in cm3, RadLex/LOINC for vessel PVAT HU attenuation).

---

### Recipe 8: Launch Interactive Cardiac Research Dashboard & Embedded OHIF

To launch the web dashboard for exploring cohorts, clinical metrics, Agatston CAC scores, and correlation analytics:

1. **CLI Execution**:

   ```bash
   cardiac-fat-fl dashboard --cohort-dir "/path/to/cohort" --port 8050
   ```

   *(Or set environment variable `CARDIAC_COHORT_DIR=/path/to/cohort`)*.
2. **Dashboard Features**:
   - **Embedded OHIF v3.13 Medical Workstation**: Direct routes `/viewer/<pid>/` and `/patient/<pid>/` serving full zero-footprint workstation.
   - **1-Click Launchers**: Clickable `Segmented` badge pill in the Case Explorer table, `View OHIF` button in Actions column, and `Open in OHIF` in Deep Dive modal.
   - **On-the-Fly Export**: Automatically generates multi-frame DICOM-SEG and slice instances if not already pre-exported.
   - **Natural Numerical Sorting**: Alphanumeric sorting across participant IDs (`0, 1, 2, ... 10, ... 100`) and dedicated `#` consecutive row counter column.
   - **Multi-Column Sortable Table**: Interactive sort for Participant ID, Scan Type, CAC, EAT Vol, PVAT Vol, PVAT HU, CRISP-CT Status, and `Artifacts` (`has_seg`).
   - **Canonical 4-Way Color Palette**: Identical colors across OHIF, 3D Slicer, Clinical Report, and Research Dashboard.
   - **Multi-Series DICOM Collision Resilience**: Automatic selection of primary 3D volume (`max(series_ids, key=len)`), preventing 1-slice scout collisions.
   - **Dynamic Cohort Switcher**: Change active cohort directory on-the-fly without server restart.
   - **Federated Learning Control & Monitoring Panel (`#tab-federated`)**: Turnkey in-browser launcher and real-time dashboard for NVIDIA FLARE 2.5+ multi-hospital simulation. Features live convergence curves (Chart.js dual-axis validation DICE & loss), participating hospital site cards (`site-1`, `site-2`), checkpoint weights table with 1-click `.pt` downloads, and live streaming monospace audit log.

### Recipe 9: Run Extensible Multi-Compartment Segmentation DICE Benchmark

To benchmark a candidate segmentation method against ground truth or alternative algorithms across a cohort:

1. **CLI Execution**:

   ```bash
   cardiac-fat-fl benchmark --cohort-dir "/path/to/cohort" --candidate "eat_pvat.seg.nrrd" --reference "ground_truth.seg.nrrd" -o "benchmark_results.csv"
   ```

2. **Computed Multi-Compartment Metrics**:
   - Dice Similarity Coefficient (DSC) & IoU for all 8 compartments + PVAT Total.
   - Absolute Volume Difference ($\text{cm}^3$) & Relative Volume Error (RVE %).
   - Mean Absolute Surface Distance (MASD) & 95th Percentile Hausdorff Distance (HD95).
   - Export to structured CSV and JSON.

---

### Recipe 10: Export & Transmit HL7 FHIR R4 Bundles to Hospital EHR (Epic on FHIR)

To export or push clinical biomarkers and diagnostic reports directly into hospital EHR systems:

1. **CLI Export to File**:

   ```bash
   cardiac-fat-fl fhir-export "path/to/study_dir" -o "path/to/study_dir/cardiac_fat_fhir.json"
   ```

2. **Push to Institutional FHIR R4 Server (Epic, Cerner, HAPI FHIR)**:

   ```bash
   cardiac-fat-fl push "path/to/study_dir" --fhir-url "https://fhir.hospital.org/r4" --fhir-token "$OAUTH_BEARER_TOKEN"
   ```

3. **Simultaneous PACS + EHR Submission**:

   ```bash
   cardiac-fat-fl push "path/to/study_dir" --url "http://pacs.hospital.org:8042/dicom-web/studies" --fhir-url "https://fhir.hospital.org/r4" --fhir-token "$OAUTH_TOKEN"
   ```

4. **Self-Contained Mock FHIR Server & Sandbox**:
   - Built-in on the Research Dashboard (`http://127.0.0.1:8050/r4`):

     ```bash
     cardiac-fat-fl push "example_source_dir/0506" --fhir-url "http://127.0.0.1:8050/r4"
     ```

   - Or launch standalone server on custom port:

     ```bash
     cardiac-fat-fl fhir-server --port 8080 --storage-dir "fhir_storage"
     ```

   - Query ingested observations: `http://127.0.0.1:8050/r4/Observation?patient=0506`
   - CapabilityStatement: `http://127.0.0.1:8050/r4/metadata`
   - Sandbox status summary: `http://127.0.0.1:8050/r4/_sandbox`
5. **Interactive Dashboard 1-Click Export**:
   In the Research Dashboard (`:8050`), open any participant via **Deep Dive** in the Case Explorer and click **FHIR R4 Bundle** to download a standards-compliant JSON bundle on-demand, or query `http://127.0.0.1:8050/api/case/<pid>/fhir`.
   See [`docs/EPIC_FHIR_MCP_INTEGRATION.md`](docs/EPIC_FHIR_MCP_INTEGRATION.md) for full Epic on FHIR and open.epic Model Context Protocol documentation.

---

### Recipe 11: Execute Turnkey NVIDIA FLARE 2.5+ Federated Simulation

To simulate multi-hospital federated learning with zero DICOM transfer (FedProx proximal loss, FedBN local batch norm retention, and CrossSiteModelEval cross-site validation):

1. **CLI Execution**:

   ```bash
   cardiac-fat-fl federated-simulate --n-clients 2 --threads 2 --rounds 2
   ```

2. **Options**:
   - `--job-dir "jobs/cardiac_fat_fl"`: Custom NVFLARE job specification folder (default: `jobs/cardiac_fat_fl`).
   - `--n-clients 2`: Number of participating hospital client sites.
   - `--threads 2`: Parallel client execution threads.
   - `--rounds 2`: Aggregation round count.
   - `--workspace "path/to/ws"`: Custom workspace directory (persists `SRV_FL_global_model.pt` consensus weights).
   - `--gpu "0"`: GPU assignment or CPU mode.
3. **Architecture & Deliverables**:
   - `CardiacFatTrainer(Executor)`: DXO-based MONAI SegResNet weight unpack/pack, FedProx loss, and local norm weight filtering.
   - `PTFileModelPersistor` & `AccumulateWeightedAggregator`: Server-side model checkpointing and weighted parameter consensus.
   - `PTFileModelLocator` & `CrossSiteModelEval`: Automated post-training cross-site validation of global consensus model.
   - Self-healing Windows runtime patching: Handles NTFS file-locking and POSIX `setsid` gracefully on Windows.

---

### Recipe 12: Launch Native Model Context Protocol (MCP) Server for AI Assistants

To connect AI assistants (Antigravity, Claude Desktop, Cursor) directly to the cardiac fat quantification engine and HL7 FHIR EHR repository via stdio JSON-RPC 2.0:

1. **CLI Execution**:

   ```bash
   cardiac-fat-fl mcp-server --fhir-url "http://127.0.0.1:8050/r4"
   ```

2. **Connect to Institutional Epic on FHIR**:

   ```bash
   cardiac-fat-fl mcp-server --fhir-url "https://fhir.hospital.org/r4" --token "$SMART_OAUTH_TOKEN"
   ```

3. **Exposed Tools**:
   - `epic_get_patient_observations`: Retrieves LOINC 84245-0 EAT volume and RadLex PVAT attenuation with CRISP-CT inflammation analysis.
   - `epic_get_diagnostic_report`: Retrieves cardiology report impression and attachments.
   - `epic_create_diagnostic_report`: Direct filing of structured cardiology notes into EHR.
   - `epic_create_document_reference`: Attaches PDF reports into Epic Chart Review.
   - `cardiac_quantify_study`: Triggers automated 3D CT AI inference and optional PACS/EHR push.
   - `cardiac_get_cohort_summary`: Queries Agatston CAC tier distributions and cardiovascular risk statistics.
   - `open_epic_lookup_code`: Inspects open.epic terminology standards and US Core FHIR profiles.

---

### Recipe 13: Run Harvard/MGH Architectural Parity & Equivalence Audit (`parity-audit`)

To verify end-to-end mathematical, anatomical, and raw CT pipeline parity between `monai-cardiac-fat-fl` and the reference Harvard/MGH `circ-ml/CardiacFatQuant` pipeline:

1. **CLI Execution**:

   ```bash
   cardiac-fat-fl parity-audit "/path/to/cohort" --mode controlled_stage3 --output parity_report.json
   ```

2. **The 3-Goal Scientific Parity & Benchmark Framework**:
   - **Goal 1 (Controlled Mathematical Parity)**: Evaluates identical 7-rule morphology and island filtering against direct executions of cloned Harvard routines (`generate_eat_pvat_masks.py` and `generate_measurements.py`). Verified across all **998 Stanford COCA cases**: `Mean Dice = 1.000000` (100.00%) across all 8 compartments and PVAT Total, $\text{Volume MAE} = 0.000\text{ cm}^3$, $\text{Attenuation MAE} = 0.000\text{ HU}$, and $100.0\%$ CRISP-CT agreement ($\kappa = 1.0000$).
   - **Goal 2 (Sensitivity Benchmark & Variance Root Cause)**:
     - *Algorithmic Drift*: Legacy Voronoi radial dilation method yielded `0.2965 Mean Dice` (~30%), $+54.71\text{ cm}^3$ volume bias, and completely missed the anterior interventricular groove (PVAT LAD Dice = 0.0430).
     - *Boundary Contour Sensitivity*: Single 1-voxel ($1.5\text{ mm}$) myocardial contour shifts result in $5.9\%$ to $8.8\%$ PVAT volume shifts, explaining the observed compound ~20–30% raw CT variation between nnU-Net and SegResNet backends.
   - **Goal 3 (Full Drop-in Pipeline Equivalence on Raw CT)**:
     - End-to-end raw CT execution comparing Tyler's platform configured with `--chamber-backend totalsegmentator` vs Harvard's `circ-ml/CardiacFatQuant` repository achieves `Mean Dice = 1.000000` across all 8 compartments and $0.0000\text{ cm}^3$ metric discrepancy on raw CT scans.
     - Verified TotalSegmentator v1 manuscript checkpoint (`Task253`, SHA-256 `6fcd6e71...`) in `models/totalseg_v1_cardiac`.
3. **Automated Verification Deliverables** (`parity_verification_998/` / `parity_verification_100/`):
   - `repo_to_repo_parity_controlled_stage3.csv`: Per-patient 8-compartment DICE scores, absolute volume differences, and mean attenuation differences.
   - `monai_cardiac_fat_measurements.csv` vs `harvard_cardiacfatquant_measurements.csv`: Exact 19-column clinical metrics verification.
   - Complete technical roadmap and derivation in [`docs/PARITY_BENCHMARK_ROADMAP.md`](docs/PARITY_BENCHMARK_ROADMAP.md).

---

### Recipe 14: Acquire & Cryptographically Verify Pretrained Foundation Models (`download-models`)

To download and cryptographically verify model checkpoints across all tracks (DeepHeartSeg ONNX, NGC WholeBody, NV-Segment foundation models, TotalSegmentator, NV-Reason-CT):

1. **One-Command Download & Verification of All Tracks**:

   ```bash
   cardiac-fat-fl download-models --include-all
   ```

2. **Track-Specific Flags**:
   - DeepHeartSeg ONNX (pericardium): `cardiac-fat-fl download-models`
   - NVIDIA NGC WholeBody SegResNet: `cardiac-fat-fl download-models --include-wholebody`
   - NVIDIA-MedTech VISTA-3D Foundation Models (`NV-Segment-CT` & `NV-Segment-CTMR`): `cardiac-fat-fl download-models --include-nvsegment`
   - TotalSegmentator v1 Task253 Checkpoint: `cardiac-fat-fl download-models --include-totalseg`
   - NVIDIA NV-Reason-CT 3D VLM: `cardiac-fat-fl download-models --include-nvreason --nvreason-dir "E:\models\NV-Reason-CT"` (or `python scripts/download_nv_reason_ct.py`)
3. **Cryptographic Verification**:
   All files are validated against official SHA-256 digests in [`models/checksums.sha256`](models/checksums.sha256) and `KNOWN_MODEL_SHA256` in `cli.py`. Full details in [`MODEL_RELEASE_INSTRUCTIONS.md`](MODEL_RELEASE_INSTRUCTIONS.md).

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
4. **Canonical 8-Compartment Color Palette**:
   - Synchronized identically across OHIF v3.13, 3D Slicer DICOM-SEG, Clinical Reports (PDF/HTML/PNG), and Research Dashboard:
     - `EAT_Total`: `#9333ea` (RGB `147, 51, 234`)
     - `EAT_LA`: `#3b82f6` (RGB `59, 130, 246`)
     - `EAT_LV`: `#06b6d4` (RGB `6, 182, 212`)
     - `EAT_RA`: `#10b981` (RGB `16, 185, 129`)
     - `EAT_RV`: `#eab308` (RGB `234, 179, 8`)
     - `PVAT_LAD`: `#f97316` (RGB `249, 115, 22`)
     - `PVAT_LCX`: `#ef4444` (RGB `239, 68, 68`)
     - `PVAT_RCA`: `#ec4899` (RGB `236, 72, 153`)
5. **Coordinate, DICOM & FHIR Standards Compliance**:
   - Origin, spacing, and direction cosines must be preserved identically across input and output volumes.
   - DICOM-SEG SOP Class UID: `1.2.840.10008.5.1.4.1.1.66.4`.
   - DICOM-SR Comprehensive 3D SR SOP Class UID: `1.2.840.10008.5.1.4.1.1.88.34` (TID 1500).
   - DICOM Encapsulated PDF SOP Class UID: `1.2.840.10008.5.1.4.1.1.104.1`.
   - CRISP-CT Active Vascular Inflammation Cutoff: PVAT Mean Attenuation $> -70.1\text{ HU}$.
   - HL7 FHIR Interoperability Resources: `DiagnosticReport` (cardiology conclusion), `DocumentReference` (PDF chart attachment), and `Observation` (LOINC `84245-0` discrete EAT volume).

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
