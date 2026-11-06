# Pretrained Anatomy Segmentation & Generative Foundation Models

This directory stores the deep learning model weights used for cardiac anatomy, fat compartment segmentation, and AI clinical report generation.

The repository supports complementary deep learning model tracks:

1. **DeepHeartSeg Converted ONNX Models**: `deepheartseg_loc.onnx` & `deepheartseg_heart.onnx` (100% mathematical equivalency to reference MGH pericardial sac delineation).
2. **NVIDIA NGC / MONAI Model Zoo**: `monai_wholebody_ct_segmentation` (104 whole-body organ and cardiac chamber structures at 1.5 mm isotropic resolution).
3. **NVIDIA-MedTech VISTA-3D Foundation Models**:
   - `NV-Segment-CT`: High-throughput CT-dedicated foundation model (133 classes, 7.22 s inference, 6.5 GB peak VRAM, DSC 0.948).
   - `NV-Segment-CTMR`: Dual CT + Cardiac MR 4-chamber foundation model (346 classes, 7.0 GB peak VRAM, DSC 0.941).
4. **TotalSegmentator Cardiac Models**: Open-source v1 manuscript cardiac checkpoint (`Task253`) and TotalSegmentator v2 checkpoints for exact Harvard baseline replication.
5. **NVIDIA NV-Reason-CT 3D Vision-Language Foundation Model**: 3D ViT + Qwen3.5-4B generative radiology findings, CRISP-CT inflammation reasoning, and automated EHR narrative reporting.
6. **NVIDIA FLARE Federated Checkpoints**: Global consensus SegResNet model (`models/federated/FL_global_model.pt`) from multi-hospital federated distillation.

---

## 1. Automated Model Acquisition via CLI

Download and verify all models using the built-in CLI:

```bash
# Core DeepHeartSeg ONNX models (GitHub Releases v1.0.0):
cardiac-fat-fl download-models

# Include NVIDIA NGC WholeBody SegResNet bundle:
cardiac-fat-fl download-models --include-wholebody

# Include NVIDIA-MedTech NV-Segment foundation models (Hugging Face):
cardiac-fat-fl download-models --include-nvsegment

# Include TotalSegmentator v1 Task253 checkpoint:
cardiac-fat-fl download-models --include-totalseg

# Include NVIDIA NV-Reason-CT 3D VLM:
cardiac-fat-fl download-models --include-nvreason --nvreason-dir "E:\models\NV-Reason-CT"

# Download and verify all supported tracks:
cardiac-fat-fl download-models --include-all
```

---

## 2. NVIDIA NGC / MONAI Model Zoo: `monai_wholebody_ct_segmentation`

NVIDIA's official **[`monai_wholebody_ct_segmentation`](https://catalog.ngc.nvidia.com/orgs/nvidia/monaitoolkit/models/monai_wholebody_ct_segmentation)** bundle is a high-resolution 1.5 mm isotropic SegResNet trained on whole-body CT scans to segment 104 anatomical classes. It replaces external command-line TotalSegmentator calls with a native in-memory MONAI pipeline.

- Label 45: Left Atrium (`heart_atrium_left`)
- Label 46: Left Ventricle (`heart_ventricle_left`)
- Label 47: Right Atrium (`heart_atrium_right`)
- Label 48: Right Ventricle (`heart_ventricle_right`)
- Label 7: Aorta (`aorta`)
- Label 49: Pulmonary Artery (`pulmonary_artery`)

**Python Acquisition**:

```python
import monai.bundle
monai.bundle.download(name="monai_wholebody_ct_segmentation", bundle_dir="models")
```

---

## 3. DeepHeartSeg ONNX Models (MGH Reference Equivalency)

Used to segment the binary pericardial sac envelope with 100.0% zero-drift match ($\Delta = 0.00\,\text{cm}^3$, $\Delta = 0.00\,\text{HU}$) to the foundational Harvard/MGH research pipeline:

1. `deepheartseg_loc.onnx` (~104 MB): Step 1 coarse 3D localization network ($64 \times 128 \times 128$).
2. `deepheartseg_heart.onnx` (~104 MB): Step 2 high-resolution pericardial sac segmentation ($112 \times 128 \times 128$).

**Download via `curl`**:

```bash
curl -L -o models/deepheartseg_heart.onnx https://github.com/tylerp61391/monai-cardiac-fat-fl/releases/download/v1.0.0/deepheartseg_heart.onnx
curl -L -o models/deepheartseg_loc.onnx https://github.com/tylerp61391/monai-cardiac-fat-fl/releases/download/v1.0.0/deepheartseg_loc.onnx
```

---

## 4. NVIDIA-MedTech VISTA-3D Foundation Models

- **`NV-Segment-CT`**: Dedicated CT foundation model for whole-body anatomy and cardiac chamber extraction.
  - Weights: `models/nv_segment_ct/vista3d_pretrained_model/model.pt` (871.9 MB, SHA-256: `c92bab26d00b4a5d89fa8a383900cdeb88302fd318e5e816df0bbec7106d9a1b`).
  - Hugging Face: [`nvidia/NV-Segment-CT`](https://huggingface.co/nvidia/NV-Segment-CT).
- **`NV-Segment-CTMR`**: Dual CT + Cardiac MR foundation model with discrete 4-chamber segmentation (`[6, 149, 151, 152, 153, 154]`).
  - Weights: `models/nv_segment_ctmr/vista3d_pretrained_model/model.pt` (871.9 MB, SHA-256: `6c6d86a48a7a9d377ddb41d663d487cfedd6456df3b828ecd931e5cba5e71f99`).
  - GitHub / Hugging Face: [`nvidia/NV-Segment-CTMR`](https://huggingface.co/nvidia/NV-Segment-CTMR).

---

## 5. TotalSegmentator Cardiac Reference Checkpoints

To replicate the original Harvard / MGH manuscript baseline with 100% bit-exact equivalence, the platform supports TotalSegmentator nnU-Net models via `--chamber-backend totalsegmentator`:

- **TotalSegmentator v1 Official Manuscript Checkpoint (`Task253_part3_cardiac`)**:
  - Open-source cardiac model from GitHub Releases `v1.5.6-weights`.
  - Directory: `models/totalseg_v1_cardiac/Task253_TotalSegmentator_part3_cardiac_1139subj/nnUNetTrainerV2_ep4000_nomirror__nnUNetPlansv2.1/fold_0/`
  - File: `model_final_checkpoint.model` (250,464,714 bytes)
  - Verified SHA-256: `6fcd6e711a6eeb2241590a3f51b52fc96479cc75807b59ca6de0eb4fb506d944`
- **TotalSegmentator v2 Part 3 Checkpoint (`Dataset293_part3_cardiac`)**:
  - nnUNetv2 model in `~/.totalsegmentator/nnunet/results/Dataset293_TotalSegmentator_part3_cardiac_1559subj/`
  - File: `checkpoint_final.pth` (251,268,859 bytes)
  - Verified SHA-256: `189010ac67fec3f1704df7547306d0067eabaaf620e0a46a1bdd183afa0800e4`

---

## 6. NVIDIA NV-Reason-CT 3D Vision-Language AI Radiologist

- Model: `nvidia/NV-Reason-CT` (3D ViT + Qwen3.5-4B).
- Storage: `E:\models\NV-Reason-CT` (or configured via `NV_REASON_MODEL_DIR`).
- Size: ~10.64 GB (`model.safetensors`).
- Generates 2-page clinical diagnostic reports with orthogonal 3-view MPR montage, discrete findings, and sanitized `KEY FINDINGS:` under `IMPRESSION`.
- Download via Python: `python scripts/download_nv_reason_ct.py`.

---

## 7. Cryptographic Verification

All local model weights are verified against `models/checksums.sha256`:

```bash
# Linux / macOS
cd models && sha256sum -c checksums.sha256

# Windows PowerShell
Get-FileHash models/*.onnx, models/wholeBody_ct_segmentation/models/*.pt, models/nv_segment_ct/vista3d_pretrained_model/*.pt, models/nv_segment_ctmr/vista3d_pretrained_model/*.pt, models/federated/*.pt -Algorithm SHA256
```
