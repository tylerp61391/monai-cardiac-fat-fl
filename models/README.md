# DeepHeartSeg Legacy Weights & ONNX Models

This directory stores the pretrained DeepHeartSeg TensorFlow 1.14 / Keras model weights and their converted ONNX models:

1. `deepheartseg_loc.onnx` & `deepheartseg_step1.hdf5` (~104 MB):
   - Step 1 Cardiac Localization model (`64 x 128 x 128` input).
   - Predicts the 3D cardiac bounding box from whole-chest CT.

2. `deepheartseg_heart.onnx` & `deepheartseg_step2.hdf5` (~104 MB):
   - Step 2 Cardiac Chamber & Pericardium Segmentation model (`112 x 128 x 128` input).
   - Segments myocardium and pericardial sac within the cropped cardiac volume.

---

## Distribution via GitHub Release Assets (Option A)

Because each model weight file exceeds GitHub's $100\,\text{MB}$ repository file limit, model binaries are hosted on GitHub Releases rather than tracked in git.

### 1. Download Pretrained Models

You can download the converted ONNX models directly into this folder using either method:

**Via CLI (Automatic)**:
```bash
cardiac-fat-fl download-models
```

**Via `curl` (Manual)**:
```bash
curl -L -o models/deepheartseg_heart.onnx https://github.com/circ-ml/monai-cardiac-fat-fl/releases/download/v1.0.0/deepheartseg_heart.onnx
curl -L -o models/deepheartseg_loc.onnx https://github.com/circ-ml/monai-cardiac-fat-fl/releases/download/v1.0.0/deepheartseg_loc.onnx
```

### 2. Repository Maintainer Instructions
When pushing this repository to GitHub:
1. Push the codebase normally (`git push`). The `.gitignore` keeps large binaries out of git.
2. In the GitHub web UI, create release `v1.0.0`.
3. Upload `deepheartseg_heart.onnx` and `deepheartseg_loc.onnx` as binary assets to the release.

### 3. Offline Heuristic Fallback
If model weights are not present, `UnifiedAnatomySegmenter` automatically falls back to the heuristic anatomical segmentation engine without throwing errors, ensuring immediate out-of-the-box functionality.
