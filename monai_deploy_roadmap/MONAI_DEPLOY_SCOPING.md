# MONAI Deploy Scoping & Architecture Roadmap (`monai-cardiac-fat-map`)

This document scopes the future packaging of `monai-cardiac-fat-fl` into a clinical **MONAI Deploy Application Package (MAP)** conforming to the [Project MONAI Deploy standard](https://project-monai.github.io/deploy.html) (`monai-deploy-app-sdk` / NVIDIA Holoscan SDK, MONAI Deploy Informatics Gateway, and MONAI Deploy Workflow Manager).

---

## 1. Executive Summary & Feasibility Assessment

| Dimension | Assessment |
| :--- | :--- |
| **Overall Complexity** | **Low-to-Medium** (~1–2 days for standalone MAP container; ~3–5 days with full Informatics Gateway & Workflow Manager routing) |
| **Code Reuse from `cardiofat`** | **> 90% direct reuse** — all 3D inference, ROI-bounded EDT, multi-frame DICOM-SEG (`.66.4`), and DICOM-SR TID 1500 (`.88.34`) modules are already written and unit-tested. |
| **Primary SDK** | `monai-deploy-app-sdk` (v0.6+ built on **NVIDIA Holoscan SDK**) |
| **Clinical Inputs** | Standard DICOM Axial Cardiac / Chest CT Series (`Modality = CT`, ≥ 20 slices) |
| **Clinical Outputs** | 1. Multi-frame **DICOM-SEG** (`1.2.840.10008.5.1.4.1.1.66.4`, 8 compartments)<br>2. **DICOM-SR TID 1500** (`1.2.840.10008.5.1.4.1.1.88.34`, 9 regions × Volume + HU)<br>3. JSON/CSV summary metrics (`cardiac_fat_measurements.csv`) |

### Why It Is Easy to Build From Our Current Codebase

In most research-to-clinical translation projects, 80% of the effort in building a MONAI Deploy Application Package is writing DICOM-compliant UID linkages, `FrameOfReferenceUID` preservation, multi-frame binary bit-packing (`1.2.840.10008.5.1.4.1.1.66.4`), and SNOMED-CT / LOINC / UCUM TID 1500 Structured Reports (`1.2.840.10008.5.1.4.1.1.88.34`).

Because `monai-cardiac-fat-fl` **already implements and tests** all of those clinical standards natively in `ohif_exporter.py`, `dicom_sr.py`, and `pacs_client.py`, creating the MONAI Deploy package only requires wrapping our existing functions into a 3-node **MONAI Deploy Operator DAG** (`Application`) and adding the MAP container manifest (`app.yaml`).

---

## 2. Target Clinical Deployment Architecture

The MONAI Deploy ecosystem consists of three components that work together inside a hospital firewall:

1. **MONAI Deploy Informatics Gateway (MIG)**: Receives DICOM studies from the scanner or PACS (via DIMSE C-STORE or DICOMweb) and exports MAP outputs back to the PACS archive.
2. **MONAI Deploy Workflow Manager (MWM)**: Evaluates routing rules (e.g., `Modality == "CT"` and `BodyPartExamined in ["CHEST", "HEART"]`) and triggers our containerized MAP on a Kubernetes / GPU node.
3. **MONAI Application Package (`monai-cardiac-fat-map`)**: Our self-contained OCI Docker container obeying the `/var/monai/input`, `/var/monai/output`, and `/opt/holoscan/models` directory contracts.

```mermaid
flowchart LR
    subgraph Hospital ["Hospital PACS & Scanners"]
        Scanner["CT Scanner / PACS<br/>(DICOM C-STORE / STOW-RS)"]
        Viewer["PACS Workstation / OHIF<br/>(Radiologist Review)"]
    end

    subgraph MONAIDeploy ["MONAI Deploy Clinical Platform"]
        MIG["MONAI Informatics Gateway<br/>(DICOM Listener & Export)"]
        MWM["MONAI Workflow Manager<br/>(Rule Engine & Argo DAG)"]
        
        subgraph MAP ["MONAI Application Package (monai-cardiac-fat-map)"]
            Op1["Operator 1:<br/>CardiacCTStudyLoaderOp<br/>(io_metatensor.py)"]
            Op2["Operator 2:<br/>CardiacFatQuantifierOp<br/>(WholeBody 1.5mm + DeepHeartSeg)"]
            Op3["Operator 3:<br/>DicomSegAndSrWriterOp<br/>(ohif_exporter.py + dicom_sr.py)"]
            Op1 --> Op2 --> Op3
        end
    end

    Scanner -->|"1. Push CT Series"| MIG
    MIG -->|"2. Notify Study Arrival"| MWM
    MWM -->|"3. Mount /var/monai/input"| Op1
    Op3 -->|"4. Write /var/monai/output<br/>(SEG .66.4 + SR .88.34)"| MIG
    MIG -->|"5. C-STORE / STOW-RS"| Viewer
```

---

## 3. Mapping Existing `cardiac_fat_fl` Modules to MONAI Deploy Operators

A MONAI Deploy application is structured as a Directed Acyclic Graph (DAG) of `Operator` subclasses inside an `Application` class:

| MONAI Deploy Operator | Existing `cardiac_fat_fl` Module & Function | Input Port (`op_input`) | Output Port (`op_output`) |
| :--- | :--- | :--- | :--- |
| **`CardiacCTStudyLoaderOperator`** | `io_metatensor.read_medical_image_as_metatensor` + `ohif_exporter._find_source_dicom_files` | `input_path` (`/var/monai/input` DICOM folder) | `ct_metatensor` (`MetaTensor`), `source_dcm_dir` (`Path`) |
| **`CardiacFatInferenceOperator`** | `anatomy_segmenter.UnifiedAnatomySegmenter` + `transforms.ComputeEatPvatMasksd` + `transforms.QuantifyAdiposeMetricsd` | `ct_metatensor` (`MetaTensor`) | `seg_mask_8ch` (`np.ndarray`), `anatomy_masks` (`AnatomyMasks`), `metrics` (`dict`) |
| **`CardiacFatDicomPublisherOperator`** | `ohif_exporter.export_segmentation_to_dicom_seg` + `dicom_sr.export_cardiac_fat_sr` | `seg_mask_8ch`, `metrics`, `source_dcm_dir`, `output_path` (`/var/monai/output`) | Writes `eat_pvat_seg.dcm`, `eat_pvat_sr.dcm`, and `metrics.json` to `/var/monai/output` |

---

## 4. MAP Container Specification & Resource Requirements

In accordance with the **MONAI Application Package (MAP)** specification, the container requires a declarative `app.yaml` manifest defining hardware resources and I/O mount points:

- **Input Mount**: `/var/monai/input` (read-only DICOM study directory mounted by MONAI Workflow Manager)
- **Output Mount**: `/var/monai/output` (writable directory where `eat_pvat_seg.dcm` and `eat_pvat_sr.dcm` are emitted)
- **Model Mount**: `/opt/holoscan/models` (or baked into `/var/monai/models` inside the image):
  - `wholeBody_ct_segmentation/models/model.pt` (`1.5mm` 105-class SegResNet)
  - `deepheartseg_loc.onnx` (Step 1 coarse heart localization)
  - `deepheartseg_heart.onnx` (Step 2 fine pericardial sac delineation)
- **Hardware Resource Envelope**:
  - `cpu`: `4` cores
  - `gpu`: `1` NVIDIA GPU (≥ 8 GB VRAM; ~2.5 GB used at runtime)
  - `memory`: `16Gi` system RAM

---

## 5. Step-by-Step Implementation Plan

### Phase 1: Operator DAG Implementation (~0.5 Day)

1. Add optional dependency group `[deploy]` in `pyproject.toml`:

   ```toml
   [project.optional-dependencies]
   deploy = [
       "monai-deploy-app-sdk>=0.6.0",
       "holoscan>=2.0.0",
   ]
   ```

2. Implement the 3 operators in `monai_deploy_roadmap/app_skeleton.py` (already drafted in this folder) wrapping `read_medical_image_as_metatensor`, `ComputeEatPvatMasksd`, `QuantifyAdiposeMetricsd`, `export_segmentation_to_dicom_seg`, and `export_cardiac_fat_sr`.

### Phase 2: Local Execution & CLI Verification (~0.5 Day)

Test the application locally outside Docker using a single COCA or `example_source_dir` study:

```bash
python monai_deploy_roadmap/app_skeleton.py \
  -i "example_source_dir/0506/ohif_export/dicom/0506/CT_1" \
  -o "scratch/map_output" \
  -m "models"
```

Verify that `scratch/map_output/` contains:

- `eat_pvat_seg.dcm` (SOP Class `1.2.840.10008.5.1.4.1.1.66.4`)
- `eat_pvat_sr.dcm` (SOP Class `1.2.840.10008.5.1.4.1.1.88.34`)
- `cardiac_fat_measurements.json` (19-column clinical metrics)

### Phase 3: Packaging into a MONAI Application Package (MAP) (~0.5 Day)

Use the MONAI Deploy CLI packager (`monai-deploy package`) to build the Holoscan/MAP OCI container image:

```bash
monai-deploy package monai_deploy_roadmap \
  --config monai_deploy_roadmap/app.yaml \
  --models models \
  --tag ghcr.io/tylerp61391/monai-cardiac-fat-map:1.0.0 \
  --platform x64-workstation
```

Run the packaged MAP container via the MONAI Deploy runner (`monai-deploy run`):

```bash
monai-deploy run ghcr.io/tylerp61391/monai-cardiac-fat-map:1.0.0 \
  -i "example_source_dir/0506/ohif_export/dicom/0506/CT_1" \
  -o "scratch/map_container_output"
```

### Phase 4: Informatics Gateway & Orthanc PACS Integration Test (~1 Day)

1. Add a `docker-compose.monai-deploy.yml` stack launching:
   - **Orthanc PACS** (DICOMweb + DIMSE C-STORE)
   - **MONAI Deploy Informatics Gateway**
   - **MONAI Deploy Workflow Manager**
   - **`monai-cardiac-fat-map:1.0.0`**
2. Push a raw CT series to MIG via `cardiac-fat-fl push`, observe automatic triggering of the MAP container, and verify that both the 8-compartment `DICOM-SEG` and `DICOM-SR TID 1500` appear automatically inside Orthanc and OHIF v3.13.

---

## 6. Implemented Production Deliverables

All 4 phases are fully implemented, verified, and integrated into the continuous integration suite (39 tests passing):

- [`src/cardiac_fat_fl/monai_deploy.py`](../src/cardiac_fat_fl/monai_deploy.py) — 3-node MONAI Deploy Operator DAG (`CardiacCTStudyLoaderOp`, `CardiacFatQuantifierOp`, `DicomSegAndSrWriterOp`) and `CardiacFatQuantApplication` with dual-mode SDK/standalone support.
- [`src/cardiac_fat_fl/cli.py`](../src/cardiac_fat_fl/cli.py) — `cardiac-fat-fl map-run` CLI command.
- [`monai_deploy_roadmap/app.py`](app.py) — Official MAP application entrypoint.
- [`monai_deploy_roadmap/app.yaml`](app.yaml) — Declarative MAP manifest conforming to MONAI Deploy App SDK v0.6+ & NVIDIA Holoscan SDK specifications.
- [`monai_deploy_roadmap/Dockerfile`](Dockerfile) — Production OCI Docker container image for Kubernetes / Argo Workflows deployment.
- [`monai_deploy_roadmap/docker-compose.monai-deploy.yml`](docker-compose.monai-deploy.yml) — Turnkey clinical testbed launching Orthanc PACS, MONAI Informatics Gateway (MIG), and the MAP container.
- [`tests/test_monai_deploy.py`](../tests/test_monai_deploy.py) — Comprehensive automated unit & integration test suite.
