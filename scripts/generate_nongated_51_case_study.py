"""Generate a 300 DPI medical-publication light-theme visual case study for Participant nongated_51.

Formatted in clean clinical journal style (JACC / Radiology / Circulation light theme):
- Panel A: OHIF v3.13 Four-Up MPR at Z=28 (29/54) demonstrating the Right Lower Lobe (RLL) mass
  abutting the right atrial pericardium with zero segmentation bleed.
- Panel B: OHIF v3.13 Four-Up MPR at Z=25 (26/54) demonstrating the Left Lower Lobe (LLL) mass
  and Left Hemidiaphragm Elevation abutting the left ventricular pericardium.
- Panel C: 8-Compartment Quantitative Cardiac Fat & CRISP-CT Coronary Inflammation Profile
  (EAT = 118.4 cm3, LCX PVAT = -60.6 HU [INFLAMED], RCA PVAT = -63.6 HU [INFLAMED]).
- Panel D: Verbatim NVIDIA NV-Reason-CT (3D ViT + Qwen3.5-4B) Zero-Shot Clinical Dictation
  & Anatomical Boundary Verification.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def crop_ohif_viewport(img: Image.Image) -> Image.Image:
    """Crop browser chrome/address bar at the top and Windows taskbar at the bottom."""
    arr = np.asarray(img)
    h, w, _ = arr.shape
    row_brightness = arr.mean(axis=(1, 2))
    top_crop = 0
    for y in range(int(h * 0.03), int(h * 0.18)):
        if row_brightness[y] < 60:
            top_crop = y
            break
    bottom_crop = h
    for y in range(h - 1, int(h * 0.92), -1):
        if row_brightness[y] < 60:
            bottom_crop = min(h, y + 2)
            break
    return img.crop((0, top_crop, w, bottom_crop))


def generate_case_study(
    rll_screenshot: Path,
    lll_screenshot: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    rll_asset = assets_dir / "nongated_51_rll_mass_z28.jpg"
    lll_asset = assets_dir / "nongated_51_lll_mass_z25.jpg"
    if rll_screenshot.exists() and rll_screenshot.resolve() != rll_asset.resolve():
        shutil.copy2(rll_screenshot, rll_asset)
    if lll_screenshot.exists() and lll_screenshot.resolve() != lll_asset.resolve():
        shutil.copy2(lll_screenshot, lll_asset)

    img_rll = crop_ohif_viewport(Image.open(rll_asset).convert("RGB"))
    img_lll = crop_ohif_viewport(Image.open(lll_asset).convert("RGB"))

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.edgecolor"] = "#cbd5e1"

    # Publication Light Theme Canvas (#ffffff)
    fig = plt.figure(figsize=(20, 12.2), dpi=300, facecolor="#ffffff")
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.10, 0.56, 0.34],
        width_ratios=[1.0, 1.0],
        hspace=0.09,
        wspace=0.04,
        left=0.02,
        right=0.98,
        top=0.97,
        bottom=0.025,
    )

    # -------------------------------------------------------------------------
    # Header Banner (Light Clinical Slate Card)
    # -------------------------------------------------------------------------
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor("#ffffff")
    ax_hdr.axis("off")

    hdr_box = mpatches.FancyBboxPatch(
        (0.0, 0.02),
        1.0,
        0.94,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=1.5,
        transform=ax_hdr.transAxes,
    )
    ax_hdr.add_patch(hdr_box)

    ax_hdr.text(
        0.015,
        0.66,
        "STANFORD COCA CASE STUDY — PARTICIPANT nongated_51: 3D FAT QUANTIFICATION & VLM DISCOVERY",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        transform=ax_hdr.transAxes,
        va="center",
    )
    ax_hdr.text(
        0.015,
        0.26,
        "8-Compartment Epicardial/Pericoronary Fat Segmentation (Zero Bleed into Paracardiac Masses) + NV-Reason-CT Bilateral Lower Lobe Mass Detection",
        fontsize=9.8,
        color="#475569",
        transform=ax_hdr.transAxes,
        va="center",
    )

    # Right-side stacked badges in header (light publication style)
    badges = [
        ("EAT: 118.4 cm³  •  LCX PVAT: -60.6 HU (INFLAMED)  •  RCA PVAT: -63.6 HU (INFLAMED)", "#fef2f2", "#b91c1c", "#fca5a5", 0.815, 0.68),
        ("NV-Reason-CT (3D ViT + Qwen3.5-4B): Bilateral Lower Lobe Masses + Left Hemidiaphragm Elevation", "#ecfdf5", "#047857", "#6ee7b7", 0.815, 0.28),
    ]
    for text, bg, fg, border_c, x_pos, y_pos in badges:
        ax_hdr.text(
            x_pos,
            y_pos,
            text,
            fontsize=8.6,
            fontweight="bold",
            color=fg,
            transform=ax_hdr.transAxes,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.34,rounding_size=0.2", facecolor=bg, edgecolor=border_c, linewidth=1.1),
        )

    # -------------------------------------------------------------------------
    # Panel A: Right Lower Lobe Mass (Z = 28 / Slice 29 of 54)
    # -------------------------------------------------------------------------
    ax_rll = fig.add_subplot(gs[1, 0])
    ax_rll.set_facecolor("#f8fafc")
    ax_rll.imshow(img_rll)
    ax_rll.set_xticks([])
    ax_rll.set_yticks([])
    for spine in ax_rll.spines.values():
        spine.set_edgecolor("#0284c7")
        spine.set_linewidth(1.8)

    ax_rll.text(
        0.015,
        0.965,
        "A   RIGHT LOWER LOBE (RLL) MASS — Axial / Coronal / Sagittal Crosshairs at Z: 28 (Slice 29/54)",
        fontsize=10.3,
        fontweight="bold",
        color="#0f172a",
        transform=ax_rll.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#0284c7", linewidth=1.2, alpha=0.95),
    )
    ax_rll.text(
        0.015,
        0.028,
        "Well-circumscribed RLL mass posterior to RA/IVC • DeepHeartSeg Pericardium + HU Gating [-190, -30] excludes 100% of mass",
        fontsize=8.6,
        fontweight="bold",
        color="#0369a1",
        transform=ax_rll.transAxes,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#bae6fd", linewidth=1.0, alpha=0.95),
    )

    # -------------------------------------------------------------------------
    # Panel B: Left Lower Lobe Mass + Left Hemidiaphragm (Z = 25 / Slice 26 of 54)
    # -------------------------------------------------------------------------
    ax_lll = fig.add_subplot(gs[1, 1])
    ax_lll.set_facecolor("#f8fafc")
    ax_lll.imshow(img_lll)
    ax_lll.set_xticks([])
    ax_lll.set_yticks([])
    for spine in ax_lll.spines.values():
        spine.set_edgecolor("#d97706")
        spine.set_linewidth(1.8)

    ax_lll.text(
        0.015,
        0.965,
        "B   LEFT LOWER LOBE (LLL) MASS & ELEVATED LEFT HEMIDIAPHRAGM — Crosshairs at Z: 25 (Slice 26/54)",
        fontsize=10.3,
        fontweight="bold",
        color="#0f172a",
        transform=ax_lll.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#d97706", linewidth=1.2, alpha=0.95),
    )
    ax_lll.text(
        0.015,
        0.028,
        "Contralateral LLL mass directly above elevated left hemidiaphragm • Zero false-positive bleed into LV/LA epicardial fat",
        fontsize=8.6,
        fontweight="bold",
        color="#b45309",
        transform=ax_lll.transAxes,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#fde68a", linewidth=1.0, alpha=0.95),
    )

    # -------------------------------------------------------------------------
    # Panel C: Quantitative 8-Compartment Cardiac Fat & CRISP-CT Profile
    # -------------------------------------------------------------------------
    ax_quant = fig.add_subplot(gs[2, 0])
    ax_quant.set_facecolor("#ffffff")
    ax_quant.axis("off")

    q_box = mpatches.FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=1.5,
        transform=ax_quant.transAxes,
    )
    ax_quant.add_patch(q_box)

    ax_quant.text(
        0.03,
        0.90,
        "C   8-COMPARTMENT CARDIAC ADIPOSE & CRISP-CT INFLAMMATION QUANTIFICATION",
        fontsize=11.5,
        fontweight="bold",
        color="#0369a1",
        transform=ax_quant.transAxes,
    )

    rows = [
        ("Compartment / Biomarker", "Volume (cm³)", "Mean HU", "Clinical Interpretation / Threshold"),
        ("Total Epicardial Fat (EAT)", "118.4 cm³", "-67.6 HU", "Moderate Burden (~68th %ile; Normal < 100 cm³)"),
        ("  • Right Ventricle (EAT RV)", "63.6 cm³", "-67.2 HU", "Dominant anterior/inferior RV fat pad"),
        ("  • Left Ventricle (EAT LV)", "38.4 cm³", "-68.8 HU", "Intact posterior LV boundary (0% LLL mass bleed)"),
        ("  • Left Atrium (EAT LA)", "26.2 cm³", "-66.4 HU", "Periatrial adipose envelope"),
        ("  • Right Atrium (EAT RA)", "26.1 cm³", "-67.9 HU", "Intact posterior RA boundary (0% RLL mass bleed)"),
        ("Total Pericoronary Fat (PVAT)", "30.5 cm³", "-66.5 HU", "10mm 3D Anisotropic EDT Perivascular Cuff"),
        ("  • LAD Pericoronary (PVAT LAD)", "13.2 cm³", "-71.8 HU", "NORMAL (<= -70.1 HU CRISP-CT threshold)"),
        ("  • LCX Pericoronary (PVAT LCX)", "7.1 cm³", "-60.6 HU", "ACTIVE INFLAMMATION (> -70.1 HU; +9.5 HU shift)"),
        ("  • RCA Pericoronary (PVAT RCA)", "10.1 cm³", "-63.6 HU", "ACTIVE INFLAMMATION (> -70.1 HU; +6.5 HU shift)"),
    ]

    y_positions = np.linspace(0.78, 0.08, len(rows))
    col_x = [0.03, 0.36, 0.50, 0.63]

    for r_idx, (c1, c2, c3, c4) in enumerate(rows):
        y = y_positions[r_idx]
        if r_idx == 0:
            for x, val in zip(col_x, (c1, c2, c3, c4)):
                ax_quant.text(x, y, val, fontsize=9.2, fontweight="bold", color="#475569", transform=ax_quant.transAxes)
            ax_quant.plot([0.025, 0.975], [y - 0.03, y - 0.03], color="#cbd5e1", linewidth=1.2, transform=ax_quant.transAxes)
            continue

        is_inflamed = "ACTIVE INFLAMMATION" in c4
        is_total = c1.startswith("Total")
        fg_main = "#0f172a" if is_total else "#1e293b"
        fw = "bold" if (is_total or is_inflamed) else "normal"
        fg_status = "#b91c1c" if is_inflamed else ("#047857" if "NORMAL" in c4 else "#334155")

        ax_quant.text(col_x[0], y, c1, fontsize=9.0, fontweight=fw, color=fg_main, transform=ax_quant.transAxes)
        ax_quant.text(col_x[1], y, c2, fontsize=9.0, fontweight=fw, color="#0f172a", transform=ax_quant.transAxes)
        ax_quant.text(col_x[2], y, c3, fontsize=9.0, fontweight=fw, color=fg_status if is_inflamed else "#0369a1", transform=ax_quant.transAxes)
        ax_quant.text(col_x[3], y, c4, fontsize=8.8, fontweight=fw, color=fg_status, transform=ax_quant.transAxes)

    # -------------------------------------------------------------------------
    # Panel D: Verbatim NV-Reason-CT 3D Vision-Language Report & Architectural Proof
    # -------------------------------------------------------------------------
    ax_vlm = fig.add_subplot(gs[2, 1])
    ax_vlm.set_facecolor("#ffffff")
    ax_vlm.axis("off")

    v_box = mpatches.FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=1.5,
        transform=ax_vlm.transAxes,
    )
    ax_vlm.add_patch(v_box)

    ax_vlm.text(
        0.03,
        0.90,
        "D   NVIDIA NV-REASON-CT (3D ViT + Qwen3.5-4B) — VERBATIM AI RADIOLOGIST REPORT",
        fontsize=11.5,
        fontweight="bold",
        color="#047857",
        transform=ax_vlm.transAxes,
    )

    dictation_box = mpatches.FancyBboxPatch(
        (0.03, 0.36),
        0.94,
        0.48,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="#ffffff",
        edgecolor="#10b981",
        linewidth=1.3,
        transform=ax_vlm.transAxes,
    )
    ax_vlm.add_patch(dictation_box)

    vlm_lines = [
        ("TECHNIQUE:", "Non-contrast CT (COCA Non-Gated Chest/Cardiac CT • 54 slices • 3.0 mm).", "#475569", "normal"),
        ("Lungs and airways:", "Bilateral lower lobe lung masses. Left hemidiaphragm elevation. Central airways are patent.", "#047857", "bold"),
        ("Pleura / Mediastinum:", "No pleural effusion or pneumothorax. No hilar lymphadenopathy or mediastinal mass.", "#334155", "normal"),
        ("Heart & Pericardium:", "Normal heart size. No pericardial effusion. Normal caliber thoracic aorta and main PA.", "#334155", "normal"),
        ("IMPRESSION:", "1. Bilateral lower lobe lung masses.   2. Left hemidiaphragm elevation.", "#b45309", "bold"),
        ("KEY FINDINGS TAG:", "Lung mass", "#0369a1", "bold"),
    ]

    v_y = np.linspace(0.77, 0.41, len(vlm_lines))
    for y, (label, content, color, weight) in zip(v_y, vlm_lines):
        ax_vlm.text(0.045, y, label, fontsize=8.8, fontweight="bold", color="#475569", transform=ax_vlm.transAxes)
        ax_vlm.text(0.23, y, content, fontsize=8.8, fontweight=weight, color=color, transform=ax_vlm.transAxes)

    ax_vlm.text(
        0.03,
        0.26,
        "WHY COMPLEMENTARY 3D SEGMENTATION + 3D VLM REASONING MATTERS IN CLINICAL PRACTICE:",
        fontsize=9.5,
        fontweight="bold",
        color="#0f172a",
        transform=ax_vlm.transAxes,
    )
    takeaways = [
        "1. Orthogonal Physics & Anatomical Gating: DeepHeartSeg ONNX pericardial mesh + adipose HU thresholding [-190, -30 HU] prevents soft-tissue",
        "   tumors (+20 to +50 HU) abutting the heart from corrupting EAT/PVAT volumes, while detecting severe LCX (-60.6 HU) & RCA (-63.6 HU) inflammation.",
        "2. Zero-Shot Whole-Thorax Safety Net: Standard cardiac calcium pipelines ignore extra-cardiac lung parenchyma; NV-Reason-CT inspects the full",
        "   192³ LPS volume and accurately flags both the Right Lower Lobe mass (Z=28), Left Lower Lobe mass (Z=25), and Left Hemidiaphragm Elevation.",
    ]
    for idx, line in enumerate(takeaways):
        ax_vlm.text(
            0.03,
            0.19 - idx * 0.048,
            line,
            fontsize=8.6,
            color="#334155",
            transform=ax_vlm.transAxes,
        )

    out_png = output_dir / "nongated_51_bilateral_lung_masses_case_study.png"
    out_pdf = output_dir / "nongated_51_bilateral_lung_masses_case_study.pdf"
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png, out_pdf


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "docs" / "case_studies"
    rll_src = out_dir / "assets" / "nongated_51_rll_mass_z28.jpg"
    lll_src = out_dir / "assets" / "nongated_51_lll_mass_z25.jpg"
    png_path, pdf_path = generate_case_study(rll_src, lll_src, out_dir)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
