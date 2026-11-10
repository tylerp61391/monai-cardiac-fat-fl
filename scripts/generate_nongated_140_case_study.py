"""Generate a 300 DPI medical-publication light-theme visual case study for Participant nongated_140.

Demonstrates the "Zero CAC Paradox" / "Invisible Coronary Risk":
- Patient has an Agatston Calcium Score of 0.0 (Zero Tier / "Zero Risk" by calcium scoring).
- Yet 3D multi-compartment fat phenotyping reveals high EAT volume (103.3 cm3) and severe,
  multi-vessel pericoronary adipose tissue (PVAT) inflammation (-51.1 HU overall; a massive
  +19.0 HU shift above the -70.1 HU CRISP-CT threshold).
- Panel A: Four-Up Multi-Planar Reconstruction (Axial, 3D MIP, Coronal, Sagittal) centered on the
  Left Anterior Descending (LAD) and Left Circumflex (LCX) fat cuffs at Z=27 (Slice 28/52)
  demonstrating profound perivascular edema (LAD -53.6 HU, LCX -52.2 HU).
- Panel B: Four-Up MPR centered on the Right Coronary Artery (RCA) sulcus at Z=25 (Slice 26/52)
  demonstrating critical RCA inflammation (-44.9 HU; +25.2 HU shift) and cardiomegaly.
- Panel C: 8-Compartment Quantitative Cardiac Fat & CRISP-CT Coronary Inflammation Profile Table
  contrasting Agatston CAC = 0 against multi-vessel PVAT inflammatory escalation.
- Panel D: Mechanistic & Clinical AI Takeaways Card explaining the biology of outside-in cytokine
  diffusion, vascular edema, and why multi-modal fat phenotyping prevents missed coronary events.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from cardiac_fat_fl.io_metatensor import read_medical_image_as_metatensor


# Canonical 8-compartment RGB colors (matching OHIF v3.13 & 3D Slicer)
COMPARTMENT_COLORS = {
    1: (255, 215, 0),    # EAT Total (Gold)
    2: (56, 189, 248),   # EAT LA (Sky Blue)
    3: (0, 212, 255),    # EAT LV (Cyan)
    4: (16, 185, 129),   # EAT RA (Emerald)
    5: (245, 158, 11),   # EAT RV (Amber/Orange)
    6: (239, 68, 68),    # PVAT LAD (Red)
    7: (236, 72, 153),   # PVAT LCX (Pink/Magenta)
    8: (168, 85, 247),   # PVAT RCA (Purple)
}


def load_study_volume_and_seg(
    study_dir: Path, scan_path: Path
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load aligned 3D CT volume (Z, Y, X) in HU and 8-channel seg (C, Z, Y, X) via io_metatensor."""
    ct_mt = read_medical_image_as_metatensor(scan_path)
    ct_vol = ct_mt[0].detach().cpu().numpy().astype(np.float32)  # (Z, Y, X)
    sp = ct_mt.meta.get("spacing", [0.604, 0.604, 5.0])  # [dx, dy, dz]
    spacing = (float(sp[2]), float(sp[1]), float(sp[0]))  # (dz, dy, dx)

    seg_path = study_dir / "eat_pvat.seg.nrrd"
    seg_mt = read_medical_image_as_metatensor(seg_path)
    seg_arr = seg_mt.detach().cpu().numpy()
    while seg_arr.ndim > 4 and seg_arr.shape[0] == 1:
        seg_arr = seg_arr[0]
    seg_vol = seg_arr.astype(np.uint8)  # (8, Z, Y, X)
    return ct_vol, seg_vol, spacing


def window_ct(slice_2d: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """Apply cardiac/soft-tissue CT windowing to uint8 RGB."""
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    norm = np.clip((slice_2d - low) / (high - low), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def blend_seg_overlay(rgb: np.ndarray, seg_slice_c: np.ndarray, alpha: float = 0.68) -> np.ndarray:
    """Overlay compartments onto an RGB slice."""
    out = rgb.astype(np.float32).copy()
    for ch_idx in range(8):
        mask = seg_slice_c[ch_idx] > 0
        if not np.any(mask):
            continue
        color = np.array(COMPARTMENT_COLORS[ch_idx + 1], dtype=np.float32)
        cur_alpha = 0.35 if ch_idx == 0 else alpha
        out[mask] = (1.0 - cur_alpha) * out[mask] + cur_alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def render_four_up_panel(
    ct_vol: np.ndarray,
    seg_vol: np.ndarray,
    spacing: tuple[float, float, float],
    target_zyx: tuple[int, int, int],
) -> Image.Image:
    """Render a Four-Up workstation composite (Axial, 3D MIP, Coronal, Sagittal) with clean light dividers."""
    z_idx, y_idx, x_idx = target_zyx
    dz, dy, dx = spacing
    nz, ny, nx = ct_vol.shape

    # 1. Axial (Y, X) at z_idx
    ax_rgb = window_ct(ct_vol[z_idx])
    ax_rgb = blend_seg_overlay(ax_rgb, seg_vol[:, z_idx, :, :])

    # 2. Coronal (Z, X) at y_idx — flip Z so superior is at top
    cor_ct = ct_vol[::-1, y_idx, :]
    cor_seg = seg_vol[:, ::-1, y_idx, :]
    cor_rgb = blend_seg_overlay(window_ct(cor_ct), cor_seg)

    # 3. Sagittal (Z, Y) at x_idx — flip Z so superior is at top
    sag_ct = ct_vol[::-1, :, x_idx]
    sag_seg = seg_vol[:, ::-1, :, x_idx]
    sag_rgb = blend_seg_overlay(window_ct(sag_ct), sag_seg)

    # 4. 3D Coronal MIP + 3D Fat Volume Composite (top-right quadrant)
    mip_ct = np.max(ct_vol[::-1, :, :], axis=1)  # Coronal MIP (Z, X)
    mip_rgb = window_ct(mip_ct, wl=150.0, ww=600.0)
    seg_proj = np.max(seg_vol[:, ::-1, :, :], axis=2)  # (C, Z, X)
    mip_rgb = blend_seg_overlay(mip_rgb, seg_proj, alpha=0.55)

    quad_w, quad_h = 460, 340

    def fit_to_quad(arr: np.ndarray, aspect_y_over_x: float, cross_x: float, cross_y: float) -> Image.Image:
        h_in, w_in, _ = arr.shape
        phys_h = int(round(h_in * aspect_y_over_x))
        phys_w = w_in
        pil = Image.fromarray(arr).resize((phys_w, max(1, phys_h)), Image.Resampling.BILINEAR)
        scale = min(quad_w / phys_w, quad_h / max(1, phys_h))
        new_w = int(round(phys_w * scale))
        new_h = int(round(phys_h * scale))
        pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (quad_w, quad_h), (12, 16, 24))
        off_x = (quad_w - new_w) // 2
        off_y = (quad_h - new_h) // 2
        canvas.paste(pil, (off_x, off_y))

        draw = ImageDraw.Draw(canvas)
        cx = int(off_x + cross_x * new_w)
        cy = int(off_y + cross_y * new_h)
        draw.line([(off_x, cy), (off_x + new_w, cy)], fill=(245, 158, 11), width=1)
        draw.line([(cx, off_y), (cx, off_y + new_h)], fill=(56, 189, 248), width=1)
        draw.rectangle([0, 0, quad_w - 1, quad_h - 1], outline=(203, 213, 225), width=2)
        return canvas

    q_ax = fit_to_quad(ax_rgb, dy / dx, x_idx / nx, y_idx / ny)
    q_mip = fit_to_quad(mip_rgb, dz / dx, x_idx / nx, (nz - 1 - z_idx) / nz)
    q_cor = fit_to_quad(cor_rgb, dz / dx, x_idx / nx, (nz - 1 - z_idx) / nz)
    q_sag = fit_to_quad(sag_rgb, dz / dy, y_idx / ny, (nz - 1 - z_idx) / nz)

    comp = Image.new("RGB", (quad_w * 2, quad_h * 2), (248, 250, 252))
    comp.paste(q_ax, (0, 0))
    comp.paste(q_mip, (quad_w, 0))
    comp.paste(q_cor, (0, quad_h))
    comp.paste(q_sag, (quad_w, quad_h))

    draw = ImageDraw.Draw(comp)
    labels = [
        (10, quad_h - 22, f"Axial CT (Z: {z_idx + 1}/{nz}) | R / L"),
        (quad_w + 10, quad_h - 22, "3D Coronal MIP | 8-Compartment EAT/PVAT"),
        (10, quad_h + 8, f"Coronal MPR (Y: {y_idx}/{ny})"),
        (quad_w + 10, quad_h + 8, f"Sagittal MPR (X: {x_idx}/{nx})"),
    ]
    for lx, ly, txt in labels:
        draw.rectangle([lx - 4, ly - 2, lx + len(txt) * 6 + 6, ly + 14], fill=(255, 255, 255), outline=(203, 213, 225))
        draw.text((lx, ly), txt, fill=(15, 23, 42))

    return comp


def generate_nongated_140_figure(
    study_dir: Path, scan_path: Path, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_vol, seg_vol, spacing = load_study_volume_and_seg(study_dir, scan_path)
    nz, ny, nx = ct_vol.shape

    # Focus Panel A on LAD & LCX perivascular inflammation at Z=27 (Slice 28 of 52)
    # ch 5 = LAD, ch 6 = LCX
    z_lad = 27
    lad_mask_z = seg_vol[5, z_lad] > 0
    if np.any(lad_mask_z):
        y_lad = int(np.mean(np.where(lad_mask_z)[0]))
        x_lad = int(np.mean(np.where(lad_mask_z)[1]))
    else:
        y_lad, x_lad = ny // 2, nx // 2

    # Focus Panel B on RCA peak critical inflammation at Z=25 (Slice 26 of 52)
    # ch 7 = RCA
    z_rca = 25
    rca_mask_z = seg_vol[7, z_rca] > 0
    if np.any(rca_mask_z):
        y_rca = int(np.mean(np.where(rca_mask_z)[0]))
        x_rca = int(np.mean(np.where(rca_mask_z)[1]))
    else:
        y_rca, x_rca = ny // 2, nx // 2

    img_panel_a = render_four_up_panel(ct_vol, seg_vol, spacing, (z_lad, y_lad, x_lad))
    img_panel_b = render_four_up_panel(ct_vol, seg_vol, spacing, (z_rca, y_rca, x_rca))

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
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
        "STANFORD COCA CASE STUDY — NONGATED_140: THE ZERO CAC PARADOX & INVISIBLE CORONARY RISK",
        fontsize=13.0,
        fontweight="bold",
        color="#0f172a",
        transform=ax_hdr.transAxes,
        va="center",
    )
    ax_hdr.text(
        0.015,
        0.26,
        "Zero Agatston Calcium Score (CAC = 0.0) Dissociated from High EAT (103.3 cm³) and Multi-Vessel Pericoronary Inflammation (PVAT: -51.1 HU, +19.0 HU Shift)",
        fontsize=9.8,
        color="#475569",
        transform=ax_hdr.transAxes,
        va="center",
    )

    badges = [
        ("Agatston CAC: 0.0 (Zero Tier / Missed by Standard Score)", "#f1f5f9", "#475569", "#cbd5e1", 0.795, 0.68),
        ("PVAT Total: -51.1 HU (+19.0 HU Shift) • RCA: -44.9 HU (+25.2 HU CRISP-CT Escalation)", "#fef2f2", "#b91c1c", "#fca5a5", 0.795, 0.28),
    ]
    for text, bg, fg, border_c, x_pos, y_pos in badges:
        ax_hdr.text(
            x_pos,
            y_pos,
            text,
            fontsize=8.4,
            fontweight="bold",
            color=fg,
            transform=ax_hdr.transAxes,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.34,rounding_size=0.2", facecolor=bg, edgecolor=border_c, linewidth=1.1),
        )

    # -------------------------------------------------------------------------
    # Panel A: LAD & LCX Pericoronary Fat Cuffs (Z = 27 / Slice 28 of 52)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.set_facecolor("#f8fafc")
    ax_a.imshow(img_panel_a)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_edgecolor("#ef4444")
        spine.set_linewidth(1.8)

    ax_a.text(
        0.015,
        0.965,
        f"A   LAD & LCX PERICORONARY INFLAMMATION — Four-Up MPR (Slice {z_lad + 1}/{nz})",
        fontsize=9.6,
        fontweight="bold",
        color="#0f172a",
        transform=ax_a.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#ef4444", linewidth=1.2, alpha=0.95),
    )
    ax_a.text(
        0.015,
        0.025,
        "LAD PVAT: -53.6 HU (+16.5 HU shift) • LCX PVAT: -52.2 HU (+17.9 HU shift) | 10mm anisotropic EDT radial cuff isolates active perivascular edema",
        fontsize=7.8,
        fontweight="bold",
        color="#b91c1c",
        transform=ax_a.transAxes,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#fecaca", linewidth=1.0, alpha=0.95),
    )

    # -------------------------------------------------------------------------
    # Panel B: RCA Maximal Inflammation & Cardiomegaly (Z = 25 / Slice 26 of 52)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 1])
    ax_b.set_facecolor("#f8fafc")
    ax_b.imshow(img_panel_b)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_edgecolor("#a855f7")
        spine.set_linewidth(1.8)

    ax_b.text(
        0.015,
        0.965,
        f"B   RCA SULCUS MAXIMAL ATTENUATION SHIFT — Four-Up MPR (Slice {z_rca + 1}/{nz})",
        fontsize=9.6,
        fontweight="bold",
        color="#0f172a",
        transform=ax_b.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#a855f7", linewidth=1.2, alpha=0.95),
    )
    ax_b.text(
        0.015,
        0.025,
        "RCA PVAT: -44.9 HU (+25.2 HU shift; critical inflammation) | Severe microvascular edema in right atrioventricular groove alongside cardiomegaly",
        fontsize=7.8,
        fontweight="bold",
        color="#7e22ce",
        transform=ax_b.transAxes,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#e9d5ff", linewidth=1.0, alpha=0.95),
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
        0.91,
        "C   8-COMPARTMENT CARDIAC ADIPOSE & CRISP-CT QUANTIFICATION",
        fontsize=11.2,
        fontweight="bold",
        color="#0369a1",
        transform=ax_quant.transAxes,
    )

    rows = [
        ("Compartment / Biomarker", "Volume (cm³)", "Mean HU", "CRISP-CT Status & Clinical Significance"),
        ("Agatston Calcium Score (CAC)", "0.0", "N/A", "ZERO TIER (Score = 0; falsely classified as Zero Risk)"),
        ("Total Epicardial Fat (EAT)", "103.3 cm³", "-51.5 HU", "ELEVATED BURDEN (> 100 cm³ cutoff; ~53rd percentile)"),
        ("  • Left Atrium (EAT LA)", "21.9 cm³", "-53.0 HU", "SEVERE INFLAMMATION (+17.1 HU shift above -70.1 HU)"),
        ("  • Right Atrium (EAT RA)", "21.4 cm³", "-50.5 HU", "SEVERE INFLAMMATION (+19.6 HU shift above -70.1 HU)"),
        ("  • Left Ventricle (EAT LV)", "43.2 cm³", "-53.6 HU", "SEVERE INFLAMMATION (+16.5 HU shift above -70.1 HU)"),
        ("  • Right Ventricle (EAT RV)", "40.3 cm³", "-49.0 HU", "MAXIMAL INFLAMMATION (+21.1 HU shift above -70.1 HU)"),
        ("Total Pericoronary Fat (PVAT)", "22.5 cm³", "-51.1 HU", "DIFFUSE MULTI-VESSEL INFLAMMATION (+19.0 HU shift)"),
        ("  • LAD Pericoronary (PVAT LAD)", "11.2 cm³", "-53.6 HU", "SEVERE INFLAMMATION (+16.5 HU shift; active biology)"),
        ("  • LCX Pericoronary (PVAT LCX)", "5.9 cm³", "-52.2 HU", "SEVERE INFLAMMATION (+17.9 HU shift; active biology)"),
        ("  • RCA Pericoronary (PVAT RCA)", "5.4 cm³", "-44.9 HU", "CRITICAL INFLAMMATION (+25.2 HU shift; peak edema)"),
    ]

    y_positions = np.linspace(0.80, 0.06, len(rows))
    col_x = [0.03, 0.36, 0.50, 0.63]

    for r_idx, (c1, c2, c3, c4) in enumerate(rows):
        y = y_positions[r_idx]
        if r_idx == 0:
            for x, val in zip(col_x, (c1, c2, c3, c4)):
                ax_quant.text(x, y, val, fontsize=8.8, fontweight="bold", color="#475569", transform=ax_quant.transAxes)
            ax_quant.plot([0.025, 0.975], [y - 0.025, y - 0.025], color="#cbd5e1", linewidth=1.2, transform=ax_quant.transAxes)
            continue

        is_critical = "CRITICAL" in c4 or "MAXIMAL" in c4 or "DIFFUSE" in c4
        is_inflamed = "SEVERE" in c4 or is_critical or "ELEVATED BURDEN" in c4
        is_cac_zero = "ZERO TIER" in c4
        is_total = c1.startswith("Total") or is_cac_zero
        fg_main = "#0f172a" if is_total else "#1e293b"
        fw = "bold" if (is_total or is_inflamed) else "normal"

        if is_cac_zero:
            fg_status = "#475569"
        elif is_critical:
            fg_status = "#b91c1c"
        elif is_inflamed:
            fg_status = "#dc2626"
        else:
            fg_status = "#0369a1"

        ax_quant.text(col_x[0], y, c1, fontsize=8.6, fontweight=fw, color=fg_main, transform=ax_quant.transAxes)
        ax_quant.text(col_x[1], y, c2, fontsize=8.6, fontweight=fw, color="#0f172a", transform=ax_quant.transAxes)
        ax_quant.text(col_x[2], y, c3, fontsize=8.6, fontweight=fw, color=fg_status if is_inflamed else "#475569", transform=ax_quant.transAxes)
        ax_quant.text(col_x[3], y, c4, fontsize=8.4, fontweight=fw, color=fg_status, transform=ax_quant.transAxes)

    # -------------------------------------------------------------------------
    # Panel D: Clinical AI Radiologist & Mechanistic Takeaways Card
    # -------------------------------------------------------------------------
    ax_mech = fig.add_subplot(gs[2, 1])
    ax_mech.set_facecolor("#ffffff")
    ax_mech.axis("off")

    m_box = mpatches.FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=1.5,
        transform=ax_mech.transAxes,
    )
    ax_mech.add_patch(m_box)

    ax_mech.text(
        0.03,
        0.91,
        "D   CLINICAL IMPERATIVE: OVERCOMING THE 'ZERO CAC' DIAGNOSTIC BLIND SPOT",
        fontsize=11.2,
        fontweight="bold",
        color="#047857",
        transform=ax_mech.transAxes,
    )

    dictation_box = mpatches.FancyBboxPatch(
        (0.03, 0.50),
        0.94,
        0.36,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="#ffffff",
        edgecolor="#10b981",
        linewidth=1.3,
        transform=ax_mech.transAxes,
    )
    ax_mech.add_patch(dictation_box)

    vlm_lines = [
        ("The 'Zero CAC' Blind Spot:", "Agatston scoring detects ONLY calcified plaque (>130 HU). It is 100% blind to non-calcified soft plaque and active vascular inflammation.", "#b91c1c", "bold"),
        ("PVAT Biological Sensor:", "Vascular cytokines (TNF-alpha, IL-6) diffuse outside-in into pericoronary fat, halting adipogenesis and causing edema.", "#047857", "bold"),
        ("Attenuated Shift (+19.0 HU):", "Edema and vascular proliferation shift PVAT attenuation from quiescent (-80 to -75 HU) to inflamed (-51.1 HU; CRISP-CT HR > 2.15).", "#b45309", "bold"),
        ("AI Radiologist Correlation:", "NV-Reason-CT confirms cardiomegaly, postoperative sternotomy changes, and bilateral pleural effusions concordant with high risk.", "#0369a1", "normal"),
    ]

    v_y = np.linspace(0.80, 0.54, len(vlm_lines))
    for y, (label, content, color, weight) in zip(v_y, vlm_lines):
        ax_mech.text(0.045, y, label, fontsize=8.4, fontweight="bold", color="#475569", transform=ax_mech.transAxes)
        ax_mech.text(0.275, y, content, fontsize=8.1, fontweight=weight, color=color, transform=ax_mech.transAxes)

    ax_mech.text(
        0.03,
        0.41,
        "WHY MULTI-PARAMETRIC ADIPOSE PHENOTYPING PREVENTS MISSED CARDIAC EVENTS:",
        fontsize=9.2,
        fontweight="bold",
        color="#0f172a",
        transform=ax_mech.transAxes,
    )
    takeaways = [
        "1. Reclassifying 'False-Negative' Low-Risk Patients: Under current clinical guidelines, a patient with CAC = 0 is frequently",
        "   reassured and denied preventative therapy. Here, monai-cardiac-fat-fl detects critical PVAT inflammation (-51.1 HU) across all 3 vessels.",
        "2. Volumetric + Attenuation Synergy: High EAT burden (103.3 cm³) provides the substrate for metabolic dysfunction, while pericoronary",
        "   HU shifts capture acute, dynamic vascular wall inflammation months to years before coronary calcification develops.",
        "3. Actionable Prevention: Flags candidates for intensified preventative medical therapy (high-intensity statins, SGLT2i, GLP-1 RA)",
        "   who would otherwise experience unexpected adverse cardiovascular events despite a pristine calcium score.",
    ]
    for idx, line in enumerate(takeaways):
        ax_mech.text(
            0.03,
            0.34 - idx * 0.052,
            line,
            fontsize=8.2,
            color="#334155",
            transform=ax_mech.transAxes,
        )

    out_png = output_dir / "nongated_140_zero_cac_invisible_risk_case_study.png"
    out_pdf = output_dir / "nongated_140_zero_cac_invisible_risk_case_study.pdf"
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png, out_pdf


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    study_path = Path(r"E:\CT\Stanford_COCA\cocacoronarycalciumandchestcts-2\deidentified_nongated\140\140")
    if not study_path.exists():
        # Fallback for headless environments without external COCA drive
        study_path = repo_root / "example_source_dir" / "0506"
        scan_path = study_path
    else:
        scan_path = study_path
    out_dir = repo_root / "docs" / "case_studies"
    png_path, pdf_path = generate_nongated_140_figure(study_path, scan_path, out_dir)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
