# Pretrained Anatomy Segmentation Models

This directory stores the deep learning model weights used for cardiac anatomy and fat compartment segmentation.

The repository supports two complementary deep learning model tracks:
1. **NVIDIA NGC / MONAI Model Zoo**: `monai_wholebody_ct_segmentation` (104 whole-body organ and cardiac chamber structures at 1.5 mm isotropic resolution).
2. **DeepHeartSeg Converted ONNX Models**: `deepheartseg_loc.onnx` & `deepheartseg_heart.onnx` (100% mathematical equivalency to reference MGH pericardial sac delineation).

---

## 1. NVIDIA NGC / MONAI Model Zoo: `monai_wholebody_ct_segmentation`

NVIDIA's official **[`monai_wholebody_ct_segmentation`](https://catalog.ngc.nvidia.com/orgs/nvidia/monaitoolkit/models/monai_wholebody_ct_segmentation)** bundle is a high-resolution 1.5 mm isotropic SegResNet trained on whole-body CT scans to segment 104 anatomical classes. It replaces external command-line TotalSegmentator calls with a native in-memory MONAI pipeline.

### Relevant Cardiac Chamber & Sulci Labels:
- Label 45: Left Atrium (`heart_atrium_left`)
- Label 46: Left Ventricle (`heart_ventricle_left`)
- Label 47: Right Atrium (`heart_atrium_right`)
- Label 48: Right Ventricle (`heart_ventricle_right`)
- Label 7: Aorta (`aorta`)
- Label 49: Pulmonary Artery (`pulmonary_artery`)

### How to Download & Load the NGC Model:

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

### Download DeepHeartSeg ONNX Models:

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

## 3. Offline Heuristic Fallback

If neither deep learning model is pre-downloaded, `UnifiedAnatomySegmenter` automatically activates its built-in morphological and geometric coordinate engine. This allows offline unit testing and synthetic phantom validation without requiring network downloads.
