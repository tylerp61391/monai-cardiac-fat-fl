# Pretrained Model Weights Deployment & Cryptographic Verification Guide

This guide provides explicit instructions for both **human developers** and **autonomous AI / Model Context Protocol (MCP) agents** on how to upload, distribute, and verify the cryptographic integrity of deep learning model weights in compliance with institutional Healthcare Security and Secure Software Development Lifecycle (SSDLC) standards.

---

## 1. Summary of Weight Files & Cryptographic SHA-256 Checksums

The validated deep learning models are stored in the local `models/` directory. All weights are pinned to official cryptographic digests to ensure zero supply-chain tampering and bitwise reproducibility:

| Model Filename | File Size | Cryptographic SHA-256 Digest | Origin & Architecture |
| :--- | :---: | :--- | :--- |
| `deepheartseg_loc.onnx` | **104.2 MB** | `94fbafee767d872f3fb0bae364d5922387963034bf59f75e8ea5971ba0306c93` | Mass General Brigham / CIRC (Step 1 Coarse Localization, `64x128x128`) |
| `deepheartseg_heart.onnx` | **104.2 MB** | `4015c53fa3fcdaa8b93e9d528f2410b8d34cdba07286e69b4f09a6e3c9644553` | Mass General Brigham / CIRC (Step 2 Pericardial Sac, `112x128x128`) |
| `wholeBody_ct_segmentation/models/model.pt` | **75.2 MB** | `80b429fb4b080df11c9ed0b0bdaa8a615ff083921bb213a512cf285afbc4e3fe` | NVIDIA NGC / MONAI Model Zoo (104-Class SegResNet 1.5mm Isotropic) |
| `wholeBody_ct_segmentation/models/model_lowres.pt` | **75.2 MB** | `c3ab55eb979785fdcb30690872c210bbeee73d79a170c32fdaa1eca117779f90` | NVIDIA NGC / MONAI Model Zoo (Low-Resolution Fallback Checkpoint) |

> [!NOTE]
> The ONNX models (>100 MB) are excluded from git via `.gitignore` to adhere to GitHub file size limits. Hosting them via GitHub Releases allows zero-bloat distribution with cryptographically verified downloads.

---

## 2. Publishing the Release with Checksum Provenance

When publishing release `v1.0.0`, always generate and publish the official `checksums.sha256` manifest alongside the binary assets.

### Step 2.1: Generate the Checksum Manifest

Generate the official SHA-256 manifest in the `models/` directory:

```bash
# Linux / macOS
cd models && sha256sum *.onnx > checksums.sha256 && cd ..

# Windows PowerShell
Get-FileHash models/*.onnx -Algorithm SHA256 | ForEach-Object { "$($_.Hash.ToLower())  $($_.Path | Split-Path -Leaf)" } | Set-Content models/checksums.sha256
```

### Step 2.2: Publish Release Assets

Choose either **Method A** (for automated agents / terminal) or **Method B** (for web browser):

#### Method A: Automated via GitHub CLI or Python Script (Recommended for AI Agents)

**Using GitHub CLI (`gh`)**:

```bash
gh release create v1.0.0 \
  models/deepheartseg_loc.onnx \
  models/deepheartseg_heart.onnx \
  models/checksums.sha256 \
  --title "v1.0.0 - Production MONAI & ONNX Pretrained Models" \
  --notes "Pretrained DeepHeartSeg ONNX models and cryptographic checksum manifest matching foundational MGH CardiacFatQuant with 0.00 delta.
SHA-256 Checksums:
- deepheartseg_heart.onnx: 4015c53fa3fcdaa8b93e9d528f2410b8d34cdba07286e69b4f09a6e3c9644553
- deepheartseg_loc.onnx:   94fbafee767d872f3fb0bae364d5922387963034bf59f75e8ea5971ba0306c93"
```

**Using the automated Python publishing script**:

```bash
# Requires GITHUB_TOKEN environment variable set
python scripts/publish_release.py v1.0.0
```

#### Method B: Manual Web UI (Drag & Drop)

1. Navigate to `https://github.com/tylerp61391/monai-cardiac-fat-fl/releases`.
2. Click **Draft a new release**.
3. Set Tag: `v1.0.0`, Title: `v1.0.0 - Production MONAI & ONNX Pretrained Models`.
4. In the release description, paste the official SHA-256 hashes listed in Section 1.
5. In the **Attach binaries** box, drag and drop:
   - `models/deepheartseg_loc.onnx`
   - `models/deepheartseg_heart.onnx`
   - `models/checksums.sha256`
6. Click **Publish release**.

---

## 3. Downstream Agent Setup & Automated Checksum Verification

When an autonomous AI agent or institutional researcher clones this repository, they can execute this 4-step sequence to achieve verified deployment:

```bash
# Step 1: Clone repository
git clone https://github.com/tylerp61391/monai-cardiac-fat-fl.git
cd monai-cardiac-fat-fl

# Step 2: Install package dependencies
pip install -e .

# Step 3: Automatically download release ONNX weights & verify SHA-256 checksums
cardiac-fat-fl download-models

# (Optional): Also download and verify NVIDIA NGC wholebody segmentation bundle
cardiac-fat-fl download-models --include-wholebody

# Step 4: Run hardware probe and safe batch quantification
cardiac-fat-fl hardware
cardiac-fat-fl quantify /path/to/cohort --workers 0
```

### Manual Integrity Verification

To manually re-verify checksums independently of the CLI:

```bash
# Linux / macOS
cd models && sha256sum -c checksums.sha256

# Windows PowerShell
Get-FileHash models/deepheartseg_*.onnx -Algorithm SHA256
```

---

## 4. Failure Mode, Security & Tamper-Resistance Matrix

The runtime implements defense-in-depth against malicious weight substitution and corrupted downloads:

| Threat / Scenario | Automated Protection Mechanism | Outcome & Compliance Safeguard |
| :--- | :--- | :--- |
| **Tampered / Corrupted Download** | `_verify_model_file_sha256()` validates digest against hardcoded `KNOWN_MODEL_SHA256` | Download aborted immediately; corrupted file unlinked; process exits with code 1. |
| **Poisoned PyTorch Weights (`model.pt`)** | `UnifiedAnatomySegmenter` inspects SHA-256 in-memory before `torch.load()` | Deserialization is halted or warned if hash does not match NGC release digest. |
| **Missing Weights on Clinical Scan** | Strict clinical guardrail triggers when $\max(D,H,W) \ge 128$ | Morphological fallback prohibited on clinical scans; raises `RuntimeError` requiring valid weights. |
| **Air-gapped / Offline Node** | Weights pre-staged manually in `models/` | `cardiac-fat-fl download-models` validates existing local files without network calls. |
| **Oversubscribed Host RAM** | `hardware.py` profiles available RAM and reserved host budget | Concurrency automatically throttled to prevent OOM crash or system freezing. |
| **Enterprise TLS / Custom Root CA** | `cardiac-fat-fl push --ca-bundle <path>` | Transmits DICOM-SEG to institutional PACS over TLS 1.3 with validated internal Root CA. |

---

## 5. Healthcare Compliance and Regulatory Governance

For formal institutional compliance documentation, refer to:

- **[`SECURITY.md`](SECURITY.md)**: Details HIPAA Technical Safeguards (45 CFR § 164.312), GDPR privacy-preserving boundaries, zero patient DICOM transfer guarantee, FedBN site isolation, and responsible vulnerability disclosure SLAs.
- **Regulatory Status**: Investigational / Research Use Only (RUO) in accordance with 21 CFR § 812.5; non-FDA cleared software for clinical research protocols under IRB oversight.
