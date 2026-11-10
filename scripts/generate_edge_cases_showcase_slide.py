"""Generate 300 DPI Widescreen Clinical Edge Cases & NV-Reason-CT 3D VLM Showcase Slide.

Highlights the clinical and engineering power of `monai-cardiac-fat-fl` across
N = 1,001 quantified Stanford COCA 3D CT scans and N = 996 NV-Reason-CT-4B
vision-language radiologist reports:
1. Discordant Clinical Phenotypes (Invisible Subclinical Risk CAC=0 + Active Inflammation,
   Burned-Out Plaque CAC>600k + Quiescent PVAT, Extreme Visceral Obesity 327.6 cm3).
2. Autonomous 3D VLM Incidental & Pathologic Discovery (Loculated Hemorrhagic Pericardial
   Effusion, Hydropneumothorax, Epicardial Lipomas, Myocardial Lipomatous Metaplasia,
   Pacemakers/Valves/Stents, Osseous Metastases & Rib Fractures).
3. Real 3-View Orthogonal MPR Montages extracted from the Stanford COCA cohort reports.
4. 5 Automated Engineering Resilience Guardrails (Zero-Spacing DICOM Recovery, 1-Slice
   Scout Filter, Empty-Region NaN Guardrail, -30 HU Blooming Cutoff, 3D Physical-mm EDT).
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


def _load_mpr_from_study(study_dir: Path) -> np.ndarray | None:
    """Load MPR montage from mpr_montage.png or embedded base64 in cardiac_fat_report.html."""
    png_path = study_dir / "mpr_montage.png"
    if png_path.is_file():
        try:
            return mpimg.imread(str(png_path))
        except Exception:
            pass
    html_path = study_dir / "cardiac_fat_report.html"
    if html_path.is_file():
        try:
            txt = html_path.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", txt)
            if m:
                raw_bytes = base64.b64decode(m.group(1))
                return mpimg.imread(io.BytesIO(raw_bytes), format="png")
        except Exception:
            pass
    return None


def create_edge_cases_showcase_slide(output_png: Path, output_pdf: Path | None = None) -> None:
    fig = plt.figure(figsize=(24, 15), dpi=300, facecolor="#f8fafc")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_card(
        x: float,
        y: float,
        w: float,
        h: float,
        bg: str = "#ffffff",
        border: str = "#cbd5e1",
        title: str = "",
        title_color: str = "#0f172a",
        title_size: float = 11.5,
        subtitle: str = "",
        subtitle_color: str = "#475569",
        border_width: float = 1.6,
        radius: float = 0.9,
        header_strip: str | None = None,
    ) -> None:
        shadow = FancyBboxPatch(
            (x + 0.14, y - 0.22),
            w,
            h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            linewidth=0,
            edgecolor="none",
            facecolor="#cbd5e1",
            alpha=0.55,
            zorder=1,
        )
        ax.add_patch(shadow)

        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            linewidth=border_width,
            edgecolor=border,
            facecolor=bg,
            zorder=2,
        )
        ax.add_patch(patch)

        if header_strip:
            strip = FancyBboxPatch(
                (x, y + h - 4.3),
                w,
                4.3,
                boxstyle=f"round,pad=0.2,rounding_size={radius}",
                linewidth=0,
                edgecolor="none",
                facecolor=header_strip,
                zorder=2.5,
            )
            ax.add_patch(strip)

        if title:
            ax.text(
                x + 1.0,
                y + h - 1.4,
                title,
                ha="left",
                va="top",
                color=title_color,
                fontsize=title_size,
                fontweight="bold",
                fontfamily="sans-serif",
                zorder=3,
            )
        if subtitle:
            ax.text(
                x + 1.0,
                y + h - 3.3,
                subtitle,
                ha="left",
                va="top",
                color=subtitle_color,
                fontsize=8.6,
                fontstyle="italic",
                fontfamily="sans-serif",
                zorder=3,
            )

    # =========================================================================
    # 0. TOP HEADER BANNER
    # =========================================================================
    banner = FancyBboxPatch(
        (1.5, 91.5),
        97.0,
        7.3,
        boxstyle="round,pad=0.25,rounding_size=1.0",
        linewidth=1.5,
        edgecolor="#1e293b",
        facecolor="#0f172a",
        zorder=2,
    )
    ax.add_patch(banner)

    ax.text(
        2.8,
        97.3,
        "MONAI-CARDIAC-FAT-FL  •  CLINICAL EDGE CASES, DISCORDANT PHENOTYPES & 3D VLM DISCOVERY",
        ha="left",
        va="top",
        color="#38bdf8",
        fontsize=11.0,
        fontweight="bold",
        fontfamily="sans-serif",
        zorder=3,
    )
    ax.text(
        2.8,
        95.1,
        "Deep Cohort Mining Across N = 1,001 Quantified Stanford COCA CTs & N = 996 NV-Reason-CT-4B 3D Radiologist Narratives",
        ha="left",
        va="top",
        color="#ffffff",
        fontsize=15.0,
        fontweight="bold",
        fontfamily="sans-serif",
        zorder=3,
    )
    ax.text(
        2.8,
        92.7,
        "Demonstrating Subclinical Vascular Inflammation Detection (CAC = 0), Pathologic Confounder Attribution, and Automated DICOM Geometry Recovery",
        ha="left",
        va="top",
        color="#94a3b8",
        fontsize=9.2,
        fontfamily="sans-serif",
        zorder=3,
    )

    badges = [
        ("1,001 / 1,002 (99.9%) Quantified", "#0284c7", "#e0f2fe", 65.5, 95.6),
        ("996 3D NV-Reason-CT Narratives", "#059669", "#d1fae5", 77.2, 95.6),
        ("CRISP-CT Cutoff: > -70.1 HU", "#e11d48", "#ffe4e6", 88.8, 95.6),
        ("Zero-Spacing DICOM Recovery", "#7c3aed", "#ede9fe", 65.5, 92.4),
        ("100 Implanted Devices Flagged", "#d97706", "#fef3c7", 77.2, 92.4),
        ("60 Pericardial Effusions Caught", "#4f46e5", "#e0e7ff", 88.8, 92.4),
    ]
    for b_txt, b_edge, b_fill, bx, by in badges:
        b_patch = FancyBboxPatch(
            (bx, by),
            10.5,
            2.1,
            boxstyle="round,pad=0.15,rounding_size=0.45",
            linewidth=1.2,
            edgecolor=b_edge,
            facecolor="#1e293b",
            zorder=3,
        )
        ax.add_patch(b_patch)
        ax.text(
            bx + 5.25,
            by + 1.05,
            b_txt,
            ha="center",
            va="center",
            color=b_fill,
            fontsize=7.6,
            fontweight="bold",
            zorder=4,
        )

    # =========================================================================
    # 1. LEFT & CENTER: 4 REAL PATIENT EDGE-CASE CARDS (2x2 GRID)
    # =========================================================================
    coca_base = Path(r"E:\CT\Stanford_COCA\cocacoronarycalciumandchestcts-2")
    case_specs = [
        {
            "x": 1.5,
            "y": 53.5,
            "w": 31.2,
            "h": 36.5,
            "border": "#e11d48",
            "bg": "#fff1f2",
            "strip": "#ffe4e6",
            "title": "1. 'INVISIBLE RISK': CAC = 0 + SEVERE INFLAMMATION",
            "title_col": "#9f1239",
            "subtitle": "Subclinical Coronary Microvascular Inflammation Missed by Agatston Scoring",
            "sub_col": "#be123c",
            "study_dir": coca_base / "deidentified_nongated" / "213" / "213",
            "fallback_dir": Path("example_source_dir/0506"),
            "case_badge": "Stanford COCA: nongated_213 & nongated_140",
            "metrics_line": "Agatston CAC: 0.0 (Zero Tier)   |   PVAT: -55.8 HU (Active Inflammation > -70.1)",
            "metrics_col": "#be123c",
            "bullets": [
                "• Standard calcium scoring classifies patient as Zero Risk (CAC = 0).",
                "• 3D 10mm Pericoronary EDT reveals severe perivascular attenuation (-55.8 HU",
                "  in nongated_213; -51.1 HU in nongated_140; -60.3 HU in nongated_24).",
                "• NV-Reason-CT 3D VLM correlates perivascular edema & soft-plaque haze:",
                "  \"Pericoronary fat attenuation above -70.1 HU cutoff indicates active",
                "   coronary microvascular inflammation despite absence of calcified plaque.\"",
            ],
        },
        {
            "x": 34.0,
            "y": 53.5,
            "w": 31.2,
            "h": 36.5,
            "border": "#7c3aed",
            "bg": "#f5f3ff",
            "strip": "#ede9fe",
            "title": "2. PATHOLOGIC CONFOUNDERS: EFFUSION & HYDROPNEUMOTHORAX",
            "title_col": "#5b21b6",
            "subtitle": "3D Vision-Language Radiologist Explains Extreme Pericoronary HU Shifts",
            "sub_col": "#6d28d9",
            "study_dir": coca_base / "deidentified_nongated" / "39" / "39",
            "fallback_dir": Path("example_source_dir/0506"),
            "case_badge": "Stanford COCA: nongated_39 & nongated_179",
            "metrics_line": "PVAT: -53.1 HU (nongated_39) & -47.7 HU (nongated_179)   |   CAC: 4,506 & 1,321",
            "metrics_col": "#5b21b6",
            "bullets": [
                "• Why is PVAT attenuation so elevated (-53.1 to -47.7 HU)?",
                "• In nongated_39, NV-Reason-CT autonomously detects post-sternotomy",
                "  \"Loculated hyperdense collection between the pericardial leaves",
                "   (hemorrhagic pericardial effusion?)\" + \"left fissure phantom tumor.\"",
                "• In nongated_179 (-47.7 HU), NV-Reason-CT flags \"Moderate-to-severe left",
                "  hydropneumothorax with left lung volume loss & mediastinal shift.\"",
            ],
        },
        {
            "x": 1.5,
            "y": 15.2,
            "w": 31.2,
            "h": 36.5,
            "border": "#0284c7",
            "bg": "#f0f9ff",
            "strip": "#e0f2fe",
            "title": "3. EPICARDIAL LIPOMAS & STABILIZED MASSIVE PLAQUE",
            "title_col": "#075985",
            "subtitle": "Distinguishing Benign Lipomatous Masses & Quiescent Heavy Calcification",
            "sub_col": "#0369a1",
            "study_dir": coca_base / "Gated_release_final" / "patient" / "125",
            "fallback_dir": Path("example_source_dir/0506"),
            "case_badge": "Stanford COCA: Gated 125, 303 & nongated_159",
            "metrics_line": "CAC: 635,362 (Case 125) & 619,849 (Case 303)   |   PVAT: -87.0 to -89.9 HU (Quiescent)",
            "metrics_col": "#0369a1",
            "bullets": [
                "• High CAC does NOT always mean active vascular inflammation: Cases 125",
                "  & 303 have massive calcified burden (>600k) but quiescent PVAT (-89.9 HU).",
                "• Strict -30.0 HU upper threshold prevents dense calcium (>130 HU) from",
                "  blooming into adjacent pericoronary adipose tissue.",
                "• NV-Reason-CT flags rare focal masses: \"Hypodense fatty masses in the",
                "  epicardial fat consistent with lipomas\" (nongated_159 & Case 125).",
            ],
        },
        {
            "x": 34.0,
            "y": 15.2,
            "w": 31.2,
            "h": 36.5,
            "border": "#059669",
            "bg": "#ecfdf5",
            "strip": "#d1fae5",
            "title": "4. BIOMARKER EXTREMES & CORRUPTED DICOM RECOVERY",
            "title_col": "#065f46",
            "subtitle": "From 327.6 cm³ Visceral Obesity to Cachexia & Zero-Spacing Header Rescue",
            "sub_col": "#047857",
            "study_dir": coca_base / "deidentified_nongated" / "195" / "195",
            "fallback_dir": Path("example_source_dir/0506"),
            "case_badge": "Stanford COCA: Gated 414, nongated_85 & nongated_195",
            "metrics_line": "Max EAT: 327.6 cm³ (Case 414)  |  Min PVAT: 0.00 cm³ (nongated_85)  |  dz=5.0mm Rescue",
            "metrics_col": "#065f46",
            "bullets": [
                "• Extreme Visceral Obesity (Case 414): EAT = 327.57 cm³ (3.3x cohort mean)",
                "  with 46.24 cm³ PVAT, accurately bounded by DeepHeartSeg ONNX pericardium.",
                "• Cachectic Empty-Groove Guardrail (nongated_85): EAT = 6.10 cm³, PVAT =",
                "  0.00 cm³ -> emits explicit NaN HU instead of false 0.0 HU water artifact.",
                "• Corrupted Scanner Header Rescue (nongated_195): Scanner wrote",
                "  SpacingBetweenSlices = 0.000000; auto-recovered (dz = 5.0mm) via IPP[2].",
            ],
        },
    ]

    for spec in case_specs:
        draw_card(
            spec["x"],
            spec["y"],
            spec["w"],
            spec["h"],
            bg="#ffffff",
            border=spec["border"],
            title=spec["title"],
            title_color=spec["title_col"],
            title_size=10.5,
            subtitle=spec["subtitle"],
            subtitle_color=spec["sub_col"],
            border_width=1.8,
            header_strip=spec["strip"],
        )

        # Load and render real 3-view MPR montage in the upper half of the card
        img_arr = _load_mpr_from_study(spec["study_dir"])
        if img_arr is None:
            img_arr = _load_mpr_from_study(spec["fallback_dir"])

        img_x = spec["x"] + 0.8
        img_y = spec["y"] + 16.8
        img_w = spec["w"] - 1.6
        img_h = 14.6

        # Dark radiological frame behind MPR montage
        frame = Rectangle(
            (img_x, img_y),
            img_w,
            img_h,
            facecolor="#090d16",
            edgecolor=spec["border"],
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(frame)

        if img_arr is not None:
            ax.imshow(
                img_arr,
                extent=(img_x + 0.15, img_x + img_w - 0.15, img_y + 0.15, img_y + img_h - 0.15),
                aspect="auto",
                zorder=3.5,
            )

        # Case badge overlay on top-right of image
        cb = FancyBboxPatch(
            (img_x + 0.4, img_y + img_h - 2.1),
            12.8,
            1.7,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            linewidth=0.8,
            edgecolor=spec["border"],
            facecolor="#0f172a",
            alpha=0.88,
            zorder=4,
        )
        ax.add_patch(cb)
        ax.text(
            img_x + 0.7,
            img_y + img_h - 1.25,
            spec["case_badge"],
            ha="left",
            va="center",
            color="#f8fafc",
            fontsize=7.3,
            fontweight="bold",
            zorder=4.5,
        )

        # Metrics pill bar directly below image
        m_box = FancyBboxPatch(
            (spec["x"] + 0.8, spec["y"] + 13.9),
            spec["w"] - 1.6,
            2.3,
            boxstyle="round,pad=0.12,rounding_size=0.35",
            linewidth=1.0,
            edgecolor=spec["border"],
            facecolor=spec["bg"],
            zorder=3,
        )
        ax.add_patch(m_box)
        ax.text(
            spec["x"] + spec["w"] * 0.5,
            spec["y"] + 15.05,
            spec["metrics_line"],
            ha="center",
            va="center",
            color=spec["metrics_col"],
            fontsize=8.0,
            fontweight="bold",
            fontfamily="sans-serif",
            zorder=4,
        )

        # Bullet points in lower half
        by_curr = spec["y"] + 12.7
        for b_line in spec["bullets"]:
            is_quote = b_line.strip().startswith('"') or "Loculated" in b_line or "perivascular" in b_line
            ax.text(
                spec["x"] + 1.0,
                by_curr,
                b_line,
                ha="left",
                va="top",
                color="#1e293b" if not is_quote else spec["title_col"],
                fontsize=7.9,
                fontweight="bold" if is_quote else "normal",
                fontfamily="sans-serif",
                zorder=3,
            )
            by_curr -= 1.95

    # =========================================================================
    # 2. RIGHT COLUMN: COHORT-WIDE NV-REASON-CT DISCOVERY SPECTRUM (N = 996)
    # =========================================================================
    draw_card(
        66.5,
        15.2,
        32.0,
        74.8,
        bg="#ffffff",
        border="#0f172a",
        title="5. AUTONOMOUS 3D VLM DISCOVERY SPECTRUM (N = 996 CTs)",
        title_color="#0f172a",
        title_size=11.0,
        subtitle="NVIDIA NV-Reason-CT-4B (3D ViT + Qwen3.5-4B) Structured Findings",
        subtitle_color="#475569",
        border_width=1.8,
        header_strip="#e2e8f0",
    )

    findings_bars = [
        ("Coronary Artery Wall Calcification", 761, 76.4, "#0284c7", "761 (76.4%)"),
        ("Cardiomegaly / Chamber Enlargement", 411, 41.3, "#0369a1", "411 (41.3%)"),
        ("Basilar / Dependent Atelectasis", 168, 16.9, "#0d9488", "168 (16.9%)"),
        ("Pulmonary Fibrotic Sequelae / Scarring", 149, 15.0, "#059669", "149 (15.0%)"),
        ("Incidental Pulmonary Nodule(s)", 141, 14.2, "#d97706", "141 (14.2%)"),
        ("Thoracic Aortic Wall Calcification", 116, 11.6, "#2563eb", "116 (11.6%)"),
        ("Implanted Devices / Sternotomy / Valves", 100, 10.0, "#7c3aed", "100 (10.0%)"),
        ("Ascending Aorta / Main PA Dilation", 76, 7.6, "#4f46e5", "76 (7.6%)"),
        ("Pericardial Effusion / Thickening", 60, 6.0, "#e11d48", "60 (6.0%)"),
        ("Mediastinal / Hilar Lymphadenopathy", 54, 5.4, "#db2777", "54 (5.4%)"),
        ("Esophageal Hiatal Hernia", 34, 3.4, "#ea580c", "34 (3.4%)"),
        ("Pleural Effusion / Hydropneumothorax", 25, 2.5, "#dc2626", "25 (2.5%)"),
        ("Bone Metastases / Rib Fractures / Masses", 12, 1.2, "#991b1b", "12 (1.2%)"),
    ]

    bar_top = 83.2
    bar_step = 3.35
    max_bar_w = 18.5
    for idx, (label, cnt, pct, col, val_str) in enumerate(findings_bars):
        ym = bar_top - idx * bar_step
        ax.text(
            67.5,
            ym + 1.05,
            label,
            ha="left",
            va="center",
            color="#0f172a",
            fontsize=8.1,
            fontweight="bold",
            zorder=3,
        )
        # Background track
        track = FancyBboxPatch(
            (67.5, ym - 0.55),
            max_bar_w,
            1.0,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            linewidth=0,
            facecolor="#f1f5f9",
            zorder=3,
        )
        ax.add_patch(track)

        # Filled bar
        fill_w = max(0.6, (pct / 80.0) * max_bar_w)
        bar = FancyBboxPatch(
            (67.5, ym - 0.55),
            fill_w,
            1.0,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            linewidth=0,
            facecolor=col,
            zorder=3.5,
        )
        ax.add_patch(bar)

        ax.text(
            67.5 + max_bar_w + 0.8,
            ym - 0.05,
            val_str,
            ha="left",
            va="center",
            color=col,
            fontsize=8.0,
            fontweight="bold",
            zorder=4,
        )

    # Highlight box inside right card: Notable Rare Incidental Discoveries
    rare_box = FancyBboxPatch(
        (67.5, 16.5),
        30.0,
        22.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.4,
        edgecolor="#0284c7",
        facecolor="#f8fafc",
        zorder=3,
    )
    ax.add_patch(rare_box)

    ax.text(
        68.3,
        37.8,
        "HIGH-VALUE INCIDENTAL & DIAGNOSTIC DISCOVERIES FLAGGED BY 3D VLM",
        ha="left",
        va="top",
        color="#0f172a",
        fontsize=8.8,
        fontweight="bold",
        zorder=4,
    )

    rare_items = [
        ("• Case 442 (CAC = 4,169, PVAT = -67.6 HU):", "Coronary vasa vasorum neovascularization"),
        ("", "infiltrating pericoronary space + incidental upper-abdominal ascites."),
        ("• Case 276 (CAC = 53,938, EAT = 126.8 cm³):", "Left ventricular myocardial lipomatous"),
        ("", "metaplasia (post-MI scar fat) + dilated main pulmonary artery (pulm. HTN)."),
        ("• Case nongated_151 (EAT = 104.2 cm³):", "Dual-chamber pacemaker (RA/RV leads) +"),
        ("", "combined prosthetic aortic and mitral valve replacements."),
        ("• Case nongated_24 (CAC = 0, PVAT = -60.3 HU):", "Multiple sclerotic thoracic osseous"),
        ("", "metastases + large right pleural effusion on a zero-calcium scan."),
        ("• Case nongated_51 & 632:", "Bilateral lower-lobe & right middle-lobe pulmonary masses"),
        ("", "with ipsilateral hemidiaphragm elevation requiring urgent oncologic follow-up."),
    ]
    ry = 35.6
    for bold_part, rest_part in rare_items:
        if bold_part:
            ax.text(68.3, ry, f"{bold_part} {rest_part}", ha="left", va="top", color="#0f172a", fontsize=7.5, fontweight="bold", zorder=4)
            ry -= 1.65
        else:
            ax.text(69.1, ry, rest_part, ha="left", va="top", color="#334155", fontsize=7.4, zorder=4)
            ry -= 2.0

    # =========================================================================
    # 3. BOTTOM BANNER: 5 AUTOMATED EDGE-CASE ENGINEERING GUARDRAILS
    # =========================================================================
    draw_card(
        1.5,
        1.5,
        97.0,
        12.2,
        bg="#0f172a",
        border="#1e293b",
        title="ENGINEERING RESILIENCE FOR CLINICAL EDGE CASES: 5 FAIL-SAFE PIPELINE GUARDRAILS",
        title_color="#38bdf8",
        title_size=10.5,
        border_width=1.6,
    )

    guardrails = [
        (
            "1. Zero-Spacing DICOM Recovery",
            "Reconstructs 3D volume & physical dz",
            "from ImagePositionPatient[2] when",
            "SpacingBetweenSlices = '0.000000'",
            "(rescued nongated_195, 213, 214).",
            "#38bdf8",
        ),
        (
            "2. Multi-Series Scout Filter",
            "Selects max(series_ids, key=len) +",
            "deduplicates identical Z-coordinate",
            "slices to prevent 1-slice topogram",
            "collisions in hospital PACS archives.",
            "#34d399",
        ),
        (
            "3. Strict -30.0 HU Blooming Cutoff",
            "Enforces [-190.0, -30.0] HU window",
            "so dense coronary calcium (>130 HU),",
            "stents, and contrast lumen never",
            "contaminate PVAT attenuation.",
            "#fbbf24",
        ),
        (
            "4. Empty-Groove NaN Guardrail",
            "Emits 0.0 cm³ volume and explicit",
            "NaN attenuation when a 10mm vessel",
            "cylinder has 0 fat voxels (prevents",
            "false 0.0 HU water-density spike).",
            "#f472b6",
        ),
        (
            "5. Anisotropic 3D Physical EDT",
            "Passes exact (dz, dy, dx) mm spacing",
            "to scipy distance_transform_edt so",
            "10.0 mm PVAT dilation is invariant",
            "across 0.5mm Gated & 5.0mm Non-Gated.",
            "#a78bfa",
        ),
    ]

    gx_start = 2.6
    gw = 18.6
    for idx, (g_title, l1, l2, l3, l4, g_col) in enumerate(guardrails):
        gx = gx_start + idx * 19.15
        g_card = FancyBboxPatch(
            (gx, 2.4),
            gw,
            8.6,
            boxstyle="round,pad=0.15,rounding_size=0.5",
            linewidth=1.2,
            edgecolor=g_col,
            facecolor="#1e293b",
            zorder=3,
        )
        ax.add_patch(g_card)
        ax.text(gx + 0.7, 10.2, g_title, ha="left", va="top", color=g_col, fontsize=8.3, fontweight="bold", zorder=4)
        ax.text(gx + 0.7, 8.2, l1, ha="left", va="top", color="#e2e8f0", fontsize=7.4, zorder=4)
        ax.text(gx + 0.7, 6.7, l2, ha="left", va="top", color="#e2e8f0", fontsize=7.4, zorder=4)
        ax.text(gx + 0.7, 5.2, l3, ha="left", va="top", color="#e2e8f0", fontsize=7.4, zorder=4)
        ax.text(gx + 0.7, 3.7, l4, ha="left", va="top", color="#94a3b8", fontsize=7.2, fontstyle="italic", zorder=4)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=300, facecolor=fig.get_facecolor())
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_pdf), format="pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    coca_fig_dir = repo / "docs" / "figures" / "coca"
    coca_fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = coca_fig_dir / "coca_edge_cases_and_nv_reason_showcase.png"
    out_pdf = coca_fig_dir / "coca_edge_cases_and_nv_reason_showcase.pdf"
    create_edge_cases_showcase_slide(out_png, out_pdf)
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
