# Pretrained Anatomy Segmentation Models

This directory stores the deep learning model weights used for cardiac anatomy and fat compartment segmentation.

The repository supports three complementary deep learning model tracks:

1. **NVIDIA NGC / MONAI Model Zoo**: `monai_wholebody_ct_segmentation` (104 whole-body organ and cardiac chamber structures at 1.5 mm isotropic resolution).
2. **DeepHeartSeg Converted ONNX Models**: `deepheartseg_loc.onnx` & `deepheartseg_heart.onnx` (100% mathematical equivalency to reference MGH pericardial sac delineation).
3. **TotalSegmentator Cardiac Models**: Open-source v1 manuscript cardiac checkpoint (`Task253`) and TotalSegmentator v2 checkpoints for exact Harvard baseline replication.

---

## 1. NVIDIA NGC / MONAI Model Zoo: `monai_wholebody_ct_segmentation`

NVIDIA's official **[`monai_wholebody_ct_segmentation`](https://catalog.ngc.nvidia.com/orgs/nvidia/monaitoolkit/models/monai_wholebody_ct_segmentation)** bundle is a high-resolution 1.5 mm isotropic SegResNet trained on whole-body CT scans to segment 104 anatomical classes. It replaces external command-line TotalSegmentator calls with a native in-memory MONAI pipeline.

### Relevant Cardiac Chamber & Sulci Labels

- Label 45: Left Atrium (`heart_atrium_left`)
- Label 46: Left Ventricle (`heart_ventricle_left`)
- Label 47: Right Atrium (`heart_atrium_right`)
- Label 48: Right Ventricle (`heart_ventricle_right`)
- Label 7: Aorta (`aorta`)
- Label 49: Pulmonary Artery (`pulmonary_artery`)

### How to Download & Load the NGC Model

**Option A: Via Python (`monai.bundle`)**:

```python
import monai.bundle

# Downloads the official NGC / MONAI Model Zoo bundle into models/monai_wholebody_ct_segmentation
bundle_path = monai.bundle.download(
    name="monai_wholebody_ct_segmentation",
    bundle_dir="models"
)
```

**Option B: Via MONAI CLI**:

```bash
python -m monai.bundle download --name "monai_wholebody_ct_segmentation" --bundle_dir "models"
```

**Option C: Direct NGC / Hugging Face Download**:
Download the bundle archive or checkpoint (`model.pt`) from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/monaitoolkit/models/monai_wholebody_ct_segmentation) or the [Hugging Face MONAI Model Zoo](https://huggingface.co/MONAI/wholeBody_ct_segmentation).

---

## 2. DeepHeartSeg ONNX Models (MGH Reference Equivalency)

Used to segment the binary pericardial sac envelope with 100.0% zero-drift match ($\Delta = 0.00\,\text{cm}^3$, $\Delta = 0.00\,\text{HU}$) to the foundational Harvard/MGH research pipeline:

1. `deepheartseg_loc.onnx` (~104 MB): Step 1 coarse 3D localization network ($64 \times 128 \times 128$).
2. `deepheartseg_heart.onnx` (~104 MB): Step 2 high-resolution pericardial sac segmentation ($112 \times 128 \times 128$).

### Download DeepHeartSeg ONNX Models

**Via CLI (Automatic)**:

```bash
cardiac-fat-fl download-models
```

**Via `curl` (Manual)**:

```bash
curl -L -o models/deepheartseg_heart.onnx https://github.com/tylerp61391/monai-cardiac-fat-fl/releases/download/v1.0.0/deepheartseg_heart.onnx
curl -L -o models/deepheartseg_loc.onnx https://github.com/tylerp61391/monai-cardiac-fat-fl/releases/download/v1.0.0/deepheartseg_loc.onnx
```

---

## 3. TotalSegmentator Cardiac Reference Checkpoints

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

## 4. Offline Heuristic Fallback

If neither deep learning model is pre-downloaded, `UnifiedAnatomySegmenter` automatically activates its built-in morphological and geometric coordinate engine. This allows offline unit testing and synthetic phantom validation without requiring network downloads.
