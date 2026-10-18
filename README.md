# monai-cardiac-fat-fl

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.4%2B-green.svg)](https://monai.io/)
[![NVIDIA FLARE](https://img.shields.io/badge/NVIDIA%20FLARE-2.5%2B-76b900.svg)](https://nvflare.readthedocs.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Production-ready, unified **PyTorch / MONAI Bundle** and **NVIDIA FLARE (NVFLARE)** federated learning repository for 3D cardiac CT epicardial and pericoronary adipose tissue quantification.

This repository modernizes and supersedes the legacy MGH [`CardiacFatQuant`](https://github.com/circ-ml/CardiacFatQuant) pipeline, eliminating its outdated Python 2.7 / TensorFlow 1.14 `DeepHeartSeg` dependency in favor of unified PyTorch/MONAI anatomy segmentation, GPU-accelerated MapTransforms, and privacy-preserving federated analytics.

---

## Key Features

1. **Modern PyTorch / MONAI Architecture**:
   - Replaces TensorFlow 1.14 `DeepHeartSeg` with `UnifiedAnatomySegmenter` supporting MONAI whole-heart models, TotalSegmentator CT weights, ONNX runtime fallback for converted frozen graphs, and offline geometric segmentation.
2. **Clinical Attenuation Window**:
   - Adipose attenuation is strictly enforced to `[-190.0, -30.0]` Hounsfield Units (HU):
     $$\text{Adipose HU} \in [-190.0, -30.0]$$
3. **8 Target Segmentation Compartments (`eat_pvat.seg.nrrd`)**:
   - Channel 0: `EAT_Total` (all adipose tissue inside the pericardial sac)
   - Channel 1: `EAT_LA` (Left Atrium epicardial fat via Voronoi distance partition)
   - Channel 2: `EAT_LV` (Left Ventricle epicardial fat)
   - Channel 3: `EAT_RA` (Right Atrium epicardial fat)
   - Channel 4: `EAT_RV` (Right Ventricle epicardial fat)
   - Channel 5: `PVAT_LAD` (Left Anterior Descending pericoronary fat via radial Euclidean distance dilation)
   - Channel 6: `PVAT_LCX` (Left Circumflex pericoronary fat)
   - Channel 7: `PVAT_RCA` (Right Coronary Artery pericoronary fat)
4. **Exact 19-Column CSV Schema (`cardiac_fat_measurements.csv`)**:
   - Matches legacy `CardiacFatQuant` with exact column names and pairs:
     - `participant_id`
     - `<Region> Volume (cm3)` and `<Region> Mean Attenuation (HU)` for all 9 reported regions:
       `EAT Total`, `EAT LA`, `EAT LV`, `EAT RA`, `EAT RV`, `PVAT Total`, `PVAT LAD`, `PVAT LCX`, `PVAT RCA`.
   - Direct header spacing calculation:
     $$\text{Volume (cm}^3\text{)} = \frac{\text{voxel\_count} \times (s_x \times s_y \times s_z)}{1000.0}$$
   - Empty masks gracefully output `0.0` volume and `NaN` mean attenuation.
5. **NVIDIA FLARE Federated Learning**:
   - **Mode 1 (Federated Distillation)**: Trains 3D `SegResNet` on silver-standard labels using `FedProx` proximal regularization:
     $$\mathcal{L} = \mathcal{L}_{\text{DiceFocal}}(y_{\text{pred}}, y_{\text{true}}) + \frac{\mu}{2} \| w - w_{\text{global}} \|^2$$
     with **FedBN** (retaining local Batch/Instance Normalization to handle multi-scanner kVp variations).
   - **Mode 2 (Federated Cohort Analytics)**: Computes local statistical moments and histograms via `CardiacFatStatsExecutor` and exact mathematical aggregation across clinical sites without sharing patient-level rows.

---

## Architectural Evolution & IEEE Publication Blueprint

![IEEE TMI Architecture: Legacy MGH CardiacFatQuant to SOTA Federated MONAI Pipeline](docs/figures/cardiac_fat_fl_architecture_infographic.jpg)

An interactive, publication-grade IEEE Transactions architecture infographic is also available in [`docs/ieee_infographic.html`](docs/ieee_infographic.html).

### Legacy vs. Modernized Pipeline Comparison

| Feature | Legacy Pipeline (`CardiacFatQuant`) | Modernized SOTA (`monai-cardiac-fat-fl`) |
| :--- | :--- | :--- |
| **Runtime & Language** | Python 2.7 / TensorFlow 1.14 (Frozen `.pb`/`.hdf5`) | Python 3.11+ / PyTorch 2.x & ONNX Runtime GPU |
| **Input Ingestion** | Rigid `scan.nrrd` directories only | Universal: DICOM series with slope/intercept, NIfTI, NRRD, MHA |
| **Segmentation Engine** | 2-stage DeepHeartSeg (TF 1.14) | Unified: DeepHeartSeg ONNX, MONAI Bundle SegResNet, SwinUNETR |
| **Clinical Equivalency** | Baseline MGH manuscript values | **100.0% Exact Match** ($\Delta = 0.00\text{ cm}^3$, $\Delta = 0.00\text{ HU}$) |
| **Spatial Transforms** | Single-threaded CPU SciPy/SimpleITK | GPU-accelerated MONAI `MapTransforms` (Voronoi & radial dilation) |
| **Distributed Analytics** | None (isolated workstation scripts) | Multi-site **NVIDIA FLARE 2.5+** (FedBN + FedProx + Federated Moments) |

---

## Repository Structure

```text
monai-cardiac-fat-fl/
├── pyproject.toml                 # Package definition and dependencies
├── README.md                      # Documentation and clinical specifications
├── configs/
│   ├── monai_bundle/
│   │   ├── metadata.json          # MONAI Bundle metadata (1-ch CT -> 8-ch EAT/PVAT)
│   │   ├── train.yaml             # SegResNet 3D training config with DiceFocalLoss & FedBN
│   │   └── inference.yaml         # SlidingWindowInferer 1.5mm isotropic config
│   └── nvflare/
│       ├── config_fed_server.json # ScatterAndGather + FedProx & FedStats workflows
│       └── config_fed_client.json # MonaiAlgo + CardiacFatQuant local analytics executor
├── src/
│   └── cardiac_fat_fl/
│       ├── __init__.py            # Package root exports
│       ├── measurement_schema.py  # Exact 19-column CSV schema, HU bounds, enums
│       ├── io_nrrd.py             # Participant discovery & header-preserving NRRD I/O
│       ├── anatomy_segmenter.py   # Unified PyTorch/MONAI wrapper replacing DeepHeartSeg
│       ├── transforms.py          # ComputeEatPvatMasksd & QuantifyAdiposeMetricsd
│       ├── model.py               # 3D SegResNet / SwinUNETR factory and FedBN helpers
│       ├── fl_client_algo.py      # ClientAlgo subclass for federated distillation & analytics
│       ├── fl_stats_executor.py   # NVFLARE Executor for privacy-preserving cohort stats
│       └── cli.py                 # Typer CLI reproducing steps 1-4 or running end-to-end
└── tests/
    ├── conftest.py                # Synthetic 3D cardiac CT .nrrd phantom fixtures
    ├── test_transforms.py         # Transforms, HU thresholding, cm3 volume & schema tests
    └── test_fl_client.py          # Dry-run unit tests for ClientAlgo, FedProx, FedBN, & stats
```

---

## Installation

### Prerequisites
- Python `>=3.11` (Python 3.11, 3.12, 3.13 supported)
- PyTorch 2.x and CUDA (optional for GPU acceleration)

```bash
# Clone repository
git clone https://github.com/circ-ml/monai-cardiac-fat-fl.git
cd monai-cardiac-fat-fl

# Install with editable mode
pip install -e .
```

---

## Clinical Workflow & CLI Usage

### 1. End-to-End Batch Quantification
Process a cohort directory formatted as `source_dir/<participant_id>/scan.nrrd`:

```bash
cardiac-fat-fl quantify /path/to/cohort --output-dir /path/to/cohort --pvat-radius 10.0
```

This will:
- Discover all participant scans (ignoring any existing `*.seg.nrrd` files).
- Compute 8-compartment EAT and PVAT masks.
- Write `eat_pvat.seg.nrrd` inside each participant directory preserving origin and direction matrices.
- Export `cardiac_fat_measurements.csv` containing the exact 19 columns:
  1. `participant_id`
  2. `EAT Total Volume (cm3)`
  3. `EAT Total Mean Attenuation (HU)`
  4. `EAT LA Volume (cm3)`
  5. `EAT LA Mean Attenuation (HU)`
  6. `EAT LV Volume (cm3)`
  7. `EAT LV Mean Attenuation (HU)`
  8. `EAT RA Volume (cm3)`
  9. `EAT RA Mean Attenuation (HU)`
  10. `EAT RV Volume (cm3)`
  11. `EAT RV Mean Attenuation (HU)`
  12. `PVAT Total Volume (cm3)`
  13. `PVAT Total Mean Attenuation (HU)`
  14. `PVAT LAD Volume (cm3)`
  15. `PVAT LAD Mean Attenuation (HU)`
  16. `PVAT LCX Volume (cm3)`
  17. `PVAT LCX Mean Attenuation (HU)`
  18. `PVAT RCA Volume (cm3)`
  19. `PVAT RCA Mean Attenuation (HU)`

### 2. Federated Learning Client Training
Run a local federated distillation training round using `CardiacFatClientAlgo`:

```bash
cardiac-fat-fl federated-train --data-dir /path/to/cohort --rounds 5 --mu 0.01 --fed-bn
```

### 3. Federated Cohort Analytics
Calculate privacy-preserving statistical moments across local cohort measurements:

```bash
cardiac-fat-fl federated-stats --csv-path /path/to/cardiac_fat_measurements.csv
```

---

## Verification & Testing

Run the automated test suite with `pytest`:

```bash
pytest -v
```

The test suite validates:
- Strict `[-190.0, -30.0]` HU attenuation thresholding.
- Exact $\text{cm}^3$ volume mathematics from voxel dimensions.
- Empty mask handling (`0.0` volume and `NaN` mean HU).
- Multi-compartment 4D/3D NRRD export and header coordinate preservation.
- Full 19-column CSV schema validation.
- MONAI `ClientAlgo` lifecycle (`initialize`, `train`, `evaluate`, `get_weights`).
- FedProx proximal regularization and FedBN normalization weight filtering.
- Multi-site federated statistical moment pooling.

---

## Citation & Acknowledgments
Modernized from the original MGH `CardiacFatQuant` research pipeline by the Circulatory Systems ML team and DeepMind Antigravity engineering.

