# Pretrained Model Weights Deployment Guide

This checklist provides step-by-step instructions for uploading and distributing the DeepHeartSeg ONNX model weights via GitHub Releases (Option A).

---

## 1. Summary of Weight Files

The two validated ONNX models are located in your local `models/` directory:

| Filename | File Size | Description |
| :--- | :---: | :--- |
| `deepheartseg_loc.onnx` | **104.2 MB** | Step 1: 3D cardiac localization network (`64 x 128 x 128` input) |
| `deepheartseg_heart.onnx` | **104.2 MB** | Step 2: High-resolution pericardial sac segmentation network (`112 x 128 x 128` input) |

> [!NOTE]
> These files are excluded from git via `.gitignore` because GitHub has a hard 100 MB per-file limit for regular git commits. Hosting them on GitHub Releases allows users to download them without repository bloat.

---

## 2. Step-by-Step GitHub Release Instructions

Follow these steps once you push this repository to GitHub:

### Step 1: Navigate to Releases
1. Open your repository on GitHub in your browser (`https://github.com/<your-org>/monai-cardiac-fat-fl`).
2. On the right-hand sidebar, click **Releases** (or navigate to `https://github.com/<your-org>/monai-cardiac-fat-fl/releases`).
3. Click **Draft a new release** (or **Create a new release**).

### Step 2: Configure Tag & Title
- **Choose a tag**: Type `v1.0.0` and click *Create new tag: v1.0.0 on publish*.
- **Target**: `main` (or `master`).
- **Release title**: `v1.0.0 - Production MONAI & ONNX Pretrained Models`.
- **Description**:
  ```markdown
  ### Pretrained DeepHeartSeg ONNX Models
  Validated PyTorch/ONNX converted weights matching legacy MGH CardiacFatQuant with 0.00 delta equivalency.

  - `deepheartseg_loc.onnx`: 3D cardiac localization bounding-box model (104.2 MB)
  - `deepheartseg_heart.onnx`: 3D high-resolution pericardial sac segmentation model (104.2 MB)
  ```

### Step 3: Attach the Binary Files
1. In the **Attach binaries by dropping them here or selecting them** box at the bottom:
2. Drag and drop both files from your local folder:
   - `models/deepheartseg_loc.onnx`
   - `models/deepheartseg_heart.onnx`
3. Wait for the upload progress bar to turn green for both files (GitHub Releases support files up to 2.0 GB each).

### Step 4: Publish
- Click the green **Publish release** button.

---

## 3. How Users Will Download the Models

Once published, any user or autonomous MCP agent cloning your repository can automatically retrieve the weights by running:

```bash
# Automated 1-click download with progress reporting
cardiac-fat-fl download-models
```

Or via standard `curl`:
```bash
curl -L -o models/deepheartseg_loc.onnx https://github.com/<your-org>/monai-cardiac-fat-fl/releases/download/v1.0.0/deepheartseg_loc.onnx
curl -L -o models/deepheartseg_heart.onnx https://github.com/circ-ml/monai-cardiac-fat-fl/releases/download/v1.0.0/deepheartseg_heart.onnx
```

---

## 4. Local Backup Verification

The source weights are also permanently backed up locally on this machine:
- Local Workspace: `models/deepheartseg_*.onnx` and `models/deepheartseg_step*.hdf5`
- Artifact Directory Vault: `C:\Users\Brad\.gemini\antigravity\brain\21a6de61-0bcc-47c6-9433-f8915867eee1\`
