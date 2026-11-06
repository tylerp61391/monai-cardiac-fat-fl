# Pretrained Model Weights Deployment & Cryptographic Verification Guide

This guide provides explicit instructions for both **human developers** and **autonomous AI / Model Context Protocol (MCP) agents** on how to download, upload, distribute, and verify the cryptographic integrity of deep learning model weights in compliance with institutional Healthcare Security and Secure Software Development Lifecycle (SSDLC) standards.

---

## 1. Summary of Weight Files & Cryptographic SHA-256 Checksums

The validated deep learning models are stored across `models/` and high-performance local NVMe storage. All weights are pinned to official cryptographic digests to ensure zero supply-chain tampering and bitwise reproducibility:

| Model Filename / Key | File Size | Cryptographic SHA-256 Digest | Origin & Architecture | Primary Function |
| :--- | :---: | :--- | :--- | :--- |
| `deepheartseg_heart.onnx` | **104.2 MB** | `4015c53fa3fcdaa8b93e9d528f2410b8d34cdba07286e69b4f09a6e3c9644553` | Harvard / MGH CIRC | Step 2 Pericardial Sac Segmentation ($112 \times 128 \times 128$) |
| `deepheartseg_loc.onnx` | **104.2 MB** | `94fbafee767d872f3fb0bae364d5922387963034bf59f75e8ea5971ba0306c93` | Harvard / MGH CIRC | Step 1 Coarse Heart Localization ($64 \times 128 \times 128$) |
| `wholeBody_ct_segmentation/models/model.pt` | **75.2 MB** | `80b429fb4b080df11c9ed0b0bdaa8a615ff083921bb213a512cf285afbc4e3fe` | NVIDIA NGC / MONAI Model Zoo | 104-Class Whole-Body SegResNet (1.5 mm Isotropic) |
| `wholeBody_ct_segmentation/models/model_lowres.pt` | **75.2 MB** | `c3ab55eb979785fdcb30690872c210bbeee73d79a170c32fdaa1eca117779f90` | NVIDIA NGC / MONAI Model Zoo | Low-Resolution Fallback Checkpoint |
| `nv_segment_ct/vista3d_pretrained_model/model.pt` | **871.9 MB** | `c92bab26d00b4a5d89fa8a383900cdeb88302fd318e5e816df0bbec7106d9a1b` | NVIDIA-MedTech (VISTA-3D) | Dedicated CT Foundation Model (133 classes, 7.22 s latency) |
| `nv_segment_ctmr/vista3d_pretrained_model/model.pt` | **871.9 MB** | `6c6d86a48a7a9d377ddb41d663d487cfedd6456df3b828ecd931e5cba5e71f99` | NVIDIA-MedTech (VISTA-3D) | Dual CT + MR Foundation Model (346 classes, 4 discrete chambers) |
| `totalseg_v1_cardiac/.../model_final_checkpoint.model` | **250.5 MB** | `6fcd6e711a6eeb2241590a3f51b52fc96479cc75807b59ca6de0eb4fb506d944` | TotalSegmentator v1 (Task253) | Exact Harvard/MGH Manuscript nnU-Net Baseline Checkpoint |
| `federated/FL_global_model.pt` | **18.8 MB** | `e606fabbc0df835fd739582adde5854483e7916b8309785345bd4cd60ceeeeb3` | NVIDIA FLARE Consensus | Federated SegResNet Distillation Consensus Model |
| `NV-Reason-CT/model.safetensors` | **10.64 GB** | *Direct Hugging Face Snapshot* | NVIDIA-MedTech | 3D Vision-Language AI Radiologist (3D ViT + Qwen3.5-4B) |

> [!NOTE]
> Large weight files (>100 MB) are excluded from git tracking via `.gitignore` to maintain a lean repository. They are distributed via GitHub Releases, NVIDIA NGC, and Hugging Face with end-to-end cryptographic SHA-256 verification.

---

## 2. Downstream Agent Setup & Turnkey Acquisition

When cloning this repository on a new machine or cluster, acquire and verify the necessary models using the integrated CLI:

### Step 2.1: Default Core Pipeline (DeepHeartSeg ONNX)

```bash
# Downloads deepheartseg_heart.onnx and deepheartseg_loc.onnx from GitHub Release v1.0.0
cardiac-fat-fl download-models
```

### Step 2.2: Add NVIDIA NGC WholeBody SegResNet

```bash
# Downloads and installs official monai_wholebody_ct_segmentation bundle into models/
cardiac-fat-fl download-models --include-wholebody
```

### Step 2.3: Add NVIDIA-MedTech Foundation Models (NV-Segment-CT & NV-Segment-CTMR)

```bash
# Downloads VISTA-3D foundation models from Hugging Face (nvidia/NV-Segment-CT and nvidia/NV-Segment-CTMR)
cardiac-fat-fl download-models --include-nvsegment
```

### Step 2.4: Add TotalSegmentator Manuscript Cardiac Checkpoint

```bash
# Downloads official TotalSegmentator v1 Task253 cardiac checkpoint (v1.5.6-weights)
cardiac-fat-fl download-models --include-totalseg
```

### Step 2.5: Add NVIDIA NV-Reason-CT 3D Vision-Language AI Radiologist

```bash
# Downloads 3D ViT + Qwen3.5-4B generative report model from Hugging Face (nvidia/NV-Reason-CT)
cardiac-fat-fl download-models --include-nvreason --nvreason-dir "E:\models\NV-Reason-CT"

# Or execute the standalone Python script (CPU/disk I/O only, 0 MB VRAM):
python scripts/download_nv_reason_ct.py
```

### Step 2.6: One-Command Comprehensive Verification (`--include-all`)

```bash
# Downloads and cryptographically verifies all supported model tracks in one command:
cardiac-fat-fl download-models --include-all
```

---

## 3. Cryptographic Verification & Manifests

### 3.1 Automated Verification Output

Running `cardiac-fat-fl download-models` scans all pre-staged and downloaded weights, validating them against `KNOWN_MODEL_SHA256`:

```text
Model already present and verified: models/deepheartseg_heart.onnx (SHA-256 valid)
Model already present and verified: models/deepheartseg_loc.onnx (SHA-256 valid)
NGC WholeBody model present and verified: models/wholeBody_ct_segmentation/models/model.pt (SHA-256 valid)
NGC WholeBody low-res model verified: models/wholeBody_ct_segmentation/models/model_lowres.pt (SHA-256 valid)
NV-Segment-CT model verified: models/nv_segment_ct/vista3d_pretrained_model/model.pt (SHA-256 valid)
NV-Segment-CTMR model verified: models/nv_segment_ctmr/vista3d_pretrained_model/model.pt (SHA-256 valid)
TotalSegmentator Task253 cardiac model verified: models/totalseg_v1_cardiac/.../model_final_checkpoint.model (SHA-256 valid)
NVIDIA NV-Reason-CT model present: E:\models\NV-Reason-CT\model.safetensors (10.64 GB)
Federated consensus global model verified: models/federated/FL_global_model.pt (SHA-256 valid)
```

### 3.2 Generating the Official `checksums.sha256` Manifest

To update or regenerate the repository manifest across all local weights:

```bash
# Linux / macOS
cd models && sha256sum *.onnx wholeBody_ct_segmentation/models/*.pt nv_segment_ct/vista3d_pretrained_model/*.pt nv_segment_ctmr/vista3d_pretrained_model/*.pt federated/*.pt totalseg_v1_cardiac/*/*/*/*/model_final_checkpoint.model > checksums.sha256 && cd ..

# Windows PowerShell
Get-FileHash models/*.onnx, models/wholeBody_ct_segmentation/models/*.pt, models/nv_segment_ct/vista3d_pretrained_model/*.pt, models/nv_segment_ctmr/vista3d_pretrained_model/*.pt, models/federated/*.pt, models/totalseg_v1_cardiac/*/*/*/*/model_final_checkpoint.model -Algorithm SHA256 | ForEach-Object { "$($_.Hash.ToLower())  $($_.Path.Replace((Get-Location).Path + [System.IO.Path]::DirectorySeparatorChar + 'models' + [System.IO.Path]::DirectorySeparatorChar, '').Replace('\', '/'))" } | Set-Content models/checksums.sha256
```

### 3.3 Manual Integrity Verification

To independently verify checksums outside of Python:

```bash
# Linux / macOS
cd models && sha256sum -c checksums.sha256

# Windows PowerShell
Get-FileHash models/deepheartseg_*.onnx, models/wholeBody_ct_segmentation/models/*.pt, models/nv_segment_ct/vista3d_pretrained_model/*.pt, models/nv_segment_ctmr/vista3d_pretrained_model/*.pt, models/federated/*.pt -Algorithm SHA256
```

---

## 4. Publishing Release Assets to GitHub

When creating a new official GitHub release tag (e.g. `v1.0.0`), attach the binary assets and checksum manifest:

### Method A: Automated via GitHub CLI (`gh`)

```bash
gh release create v1.0.0 \
  models/deepheartseg_loc.onnx \
  models/deepheartseg_heart.onnx \
  models/checksums.sha256 \
  --title "v1.0.0 - Production MONAI & ONNX Pretrained Models" \
  --notes "Production cardiac anatomy and fat quantification checkpoints with cryptographic checksum manifest.

Cryptographic SHA-256 Checksums:
- deepheartseg_heart.onnx: 4015c53fa3fcdaa8b93e9d528f2410b8d34cdba07286e69b4f09a6e3c9644553
- deepheartseg_loc.onnx:   94fbafee767d872f3fb0bae364d5922387963034bf59f75e8ea5971ba0306c93
- wholeBody_ct_segmentation/models/model.pt:        80b429fb4b080df11c9ed0b0bdaa8a615ff083921bb213a512cf285afbc4e3fe
- wholeBody_ct_segmentation/models/model_lowres.pt: c3ab55eb979785fdcb30690872c210bbeee73d79a170c32fdaa1eca117779f90
- nv_segment_ct/vista3d_pretrained_model/model.pt:   c92bab26d00b4a5d89fa8a383900cdeb88302fd318e5e816df0bbec7106d9a1b
- nv_segment_ctmr/vista3d_pretrained_model/model.pt: 6c6d86a48a7a9d377ddb41d663d487cfedd6456df3b828ecd931e5cba5e71f99
- federated/FL_global_model.pt:                      e606fabbc0df835fd739582adde5854483e7916b8309785345bd4cd60ceeeeb3
- totalseg_v1_cardiac/.../model_final_checkpoint.model: 6fcd6e711a6eeb2241590a3f51b52fc96479cc75807b59ca6de0eb4fb506d944"
```

### Method B: Automated Python Publishing Script

```bash
# Requires GITHUB_TOKEN environment variable set
python scripts/publish_release.py v1.0.0
```

---

## 5. Security & Tamper-Resistance Matrix

| Threat / Scenario | Automated Protection Mechanism | Outcome & Compliance Safeguard |
| :--- | :--- | :--- |
| **Tampered / Corrupted Download** | `_verify_model_file_sha256()` validates digest against hardcoded `KNOWN_MODEL_SHA256` | Download aborted immediately; corrupted file unlinked; process exits with code 1. |
| **Poisoned PyTorch Weights (`model.pt`)** | `UnifiedAnatomySegmenter` inspects SHA-256 before `torch.load()` | Deserialization is halted or warned if hash does not match NGC / HF release digest. |
| **Missing Weights on Clinical Scan** | Strict clinical guardrail triggers when $\max(D,H,W) \ge 128$ | Morphological fallback prohibited on clinical scans; raises `RuntimeError` requiring valid weights. |
| **Air-gapped / Offline Hospital Node** | Weights pre-staged manually in `models/` | `cardiac-fat-fl download-models` validates existing local files without network calls. |
| **Oversubscribed Host RAM** | `hardware.py` profiles available RAM and reserved host budget | Concurrency automatically throttled to prevent OOM crash or system freezing. |
| **Enterprise TLS / Custom Root CA** | `cardiac-fat-fl push --ca-bundle <path>` | Transmits DICOM-SEG to institutional PACS over TLS 1.3 with validated internal Root CA. |

---

## 6. Healthcare Compliance and Regulatory Governance

- **[`SECURITY.md`](SECURITY.md)**: Details HIPAA Technical Safeguards (45 CFR § 164.312), GDPR privacy-preserving boundaries, zero patient DICOM transfer guarantee, FedBN site isolation, and responsible vulnerability disclosure SLAs.
- **Regulatory Status**: Investigational / Research Use Only (RUO) in accordance with 21 CFR § 812.5; non-FDA cleared software for clinical research protocols under IRB oversight.
