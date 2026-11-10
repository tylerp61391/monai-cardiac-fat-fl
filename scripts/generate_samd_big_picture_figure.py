"""Generate 300 DPI Clinical Light-Mode Big Picture Architecture Diagram.

Visualizes the end-to-end SaMD framework in pristine medical light theme:
CT Ingestion -> Dual-Mode Gateway (Clinical vs. Research/RSNA V19) ->
Multi-Model 3D Segmentation (MONAI / TotalSeg / NV-Segment) ->
Local EHR / PACS & (World) Federated Learning ->
Foundation for FDA SaMD 510(k), IEC 62304 & ISO 14971 Regulatory Filing.

Features zero-collision typography, balanced 2-row header badges, and
generates both high-res PNG (300 DPI) and vector PDF formats.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch, Rectangle
import numpy as np


def create_big_picture_figure(output_png: Path, output_pdf: Path | None = None) -> None:
    # 16:10 aspect ratio, ultra-high resolution (300 DPI)
    # Clean hospital crisp background
    fig = plt.figure(figsize=(24, 15), dpi=300, facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Helper for drawing rounded boxes with clinical shadows/borders
    def draw_card(
        x: float, y: float, w: float, h: float,
        bg: str, border: str, title: str,
        title_color: str = "#0f172a",
        title_size: float = 12.0,
        subtitle: str = "",
        subtitle_color: str = "#64748b",
        border_width: float = 1.6,
        radius: float = 1.0,
    ) -> None:
        # Subtle drop shadow
        shadow = FancyBboxPatch(
            (x + 0.15, y - 0.25), w, h,
            boxstyle=f"round,pad=0.25,rounding_size={radius}",
            linewidth=0,
            edgecolor="none",
            facecolor="#e2e8f0",
            alpha=0.6,
            zorder=1,
        )
        ax.add_patch(shadow)

        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.25,rounding_size={radius}",
            linewidth=border_width,
            edgecolor=border,
            facecolor=bg,
            zorder=2,
        )
        ax.add_patch(patch)
        if title:
            ax.text(
                x + w * 0.5, y + h - 1.7, title,
                ha="center", va="top",
                color=title_color, fontsize=title_size,
                fontweight="bold", fontfamily="sans-serif",
                zorder=3,
            )
        if subtitle:
            ax.text(
                x + w * 0.5, y + h - 3.7, subtitle,
                ha="center", va="top",
                color=subtitle_color, fontsize=9.0,
                fontstyle="italic", fontfamily="sans-serif",
                zorder=3,
            )

    # Helper for connector arrows
    def draw_arrow(
        x1: float, y1: float, x2: float, y2: float,
        color: str = "#0284c7",
        width: float = 2.0,
        label: str = "",
        label_color: str = "#0369a1",
        label_y_offset: float = 1.0,
        dashed: bool = False,
    ) -> None:
        style = ArrowStyle("Simple,head_width=5,head_length=8")
        linestyle = "--" if dashed else "-"
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            zorder=4,
        )
        ax.add_patch(arrow)
        if label:
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5 + label_y_offset
            ax.text(
                mx, my, label,
                ha="center", va="center",
                color=label_color, fontsize=8.0,
                fontweight="bold", fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor=color, alpha=0.95, lw=1.0),
                zorder=5,
            )

    # ==========================================
    # 0. HEADER & BANNER (ZERO COLLISION LAYOUT)
    # ==========================================
    # Deep clinical navy rounded header bar
    banner = FancyBboxPatch(
        (1.5, 91.5), 97.0, 7.5,
        boxstyle="round,pad=0.25,rounding_size=1.0",
        linewidth=1.5, edgecolor="#1e293b", facecolor="#0f172a", zorder=1
    )
    ax.add_patch(banner)

    # Left Column: Category Kicker, Title, Subtitle (strictly confined to x: 3.0 to 53.0)
    ax.text(
        3.2, 97.5,
        "MONAI-CARDIAC-FAT-FL  •  CLINICAL AI & FEDERATED LEARNING SaMD ARCHITECTURE",
        ha="left", va="center",
        color="#38bdf8", fontsize=10.0, fontweight="bold", fontfamily="sans-serif", zorder=3,
    )
    ax.text(
        3.2, 95.3,
        "End-to-End Processing, Privacy Governance & Regulatory Pipeline",
        ha="left", va="center",
        color="#ffffff", fontsize=15.0, fontweight="bold", fontfamily="sans-serif", zorder=3,
    )
    ax.text(
        3.2, 93.1,
        "Multi-Vendor CT Ingestion → Dual-Mode Privacy Gate → 3D AI Engine → Local EHR/PACS & Multi-Center FL Consortium",
        ha="left", va="center",
        color="#cbd5e1", fontsize=9.0, fontfamily="sans-serif", zorder=3,
    )

    # Right Column: 6 Compact Badges in 2 Balanced Rows (strictly confined to x: 55.0 to 97.0)
    badges_row1 = [
        ("FDA Class II 510(k)", "#0284c7", "#e0f2fe", 55.5, 95.8, 13.0, 2.3),
        ("IEC 62304 Class B", "#0d9488", "#ccfbf1", 69.8, 95.8, 13.0, 2.3),
        ("ISO 14971 FMEA", "#d97706", "#fef3c7", 84.1, 95.8, 13.0, 2.3),
    ]
    badges_row2 = [
        ("NVIDIA FLARE 2.5+", "#16a34a", "#dcfce7", 55.5, 92.8, 13.0, 2.3),
        ("OHIF v3.13 Next", "#9333ea", "#f3e8ff", 69.8, 92.8, 13.0, 2.3),
        ("HIPAA Safe Harbor", "#4f46e5", "#e0e7ff", 84.1, 92.8, 13.0, 2.3),
    ]

    for label, border, text_col, bx, by, bw, bh in badges_row1 + badges_row2:
        patch = FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.15,rounding_size=0.6",
            linewidth=1.2, edgecolor=border, facecolor="#1e293b", zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            bx + bw * 0.5, by + bh * 0.5, label,
            ha="center", va="center", color=text_col, fontsize=8.0, fontweight="bold", zorder=4
        )

    # ==========================================
    # 1. COLUMN 1: CARDIAC CT INGESTION (X: 2.5 - 18.5)
    # ==========================================
    draw_card(
        x=2.5, y=35.5, w=16.0, h=54.0,
        bg="#ffffff", border="#0284c7",
        title="1. CT INGESTION",
        title_color="#0369a1", title_size=12.5,
        subtitle="Universal 3D MetaTensor Loader",
        border_width=2.0, radius=1.2,
    )

    # Subcard 1: Modalities & Scanners (top: 84.2)
    draw_card(
        x=3.8, y=69.5, w=13.4, h=14.5,
        bg="#f0f9ff", border="#bae6fd",
        title="Multi-Vendor Scanners",
        title_color="#0369a1", title_size=10.0,
        subtitle="Calibrated Volumetric Series",
    )
    scanner_bullets = [
        "• Siemens Healthineers (Force, Flash)",
        "• GE Healthcare (Revolution, HD)",
        "• Philips Healthcare (Brilliance, iCT)",
        "• Canon Medical (Aquilion ONE)",
        "• Non-gated & Gated CCTA / CAC",
    ]
    for idx, b in enumerate(scanner_bullets):
        ax.text(4.5, 77.5 - idx * 1.5, b, color="#1e293b", fontsize=7.8, va="top", zorder=4)

    # Subcard 2: Formats & I/O
    draw_card(
        x=3.8, y=52.5, w=13.4, h=15.5,
        bg="#f0f9ff", border="#bae6fd",
        title="Native 3D I/O Formats",
        title_color="#0369a1", title_size=10.0,
        subtitle="Header & Geometry Preserving",
    )
    format_bullets = [
        "• DICOM Series (Multi-file axial)",
        "• NRRD (scan.nrrd / img.nrrd)",
        "• NIfTI (.nii, .nii.gz) & MHA",
        "• Natural Slice Sorting (IPP Z)",
        "• Multi-series scout collision safe",
        "• Fast os.scandir cohort discovery",
    ]
    for idx, b in enumerate(format_bullets):
        ax.text(4.5, 61.5 - idx * 1.45, b, color="#1e293b", fontsize=7.8, va="top", zorder=4)

    # Subcard 3: Spatial Coordinates
    draw_card(
        x=3.8, y=37.0, w=13.4, h=14.0,
        bg="#f0f9ff", border="#bae6fd",
        title="Coordinate Preservation",
        title_color="#0369a1", title_size=10.0,
        subtitle="Physical Patient Space",
    )
    spatial_bullets = [
        "• PixelSpacing (dx, dy mm)",
        "• SliceThickness / Spacing (dz)",
        "• ImagePositionPatient (x0, y0, z0)",
        "• ImageOrientationPatient (cosines)",
        "• Affine matrix alignment [4x4]",
    ]
    for idx, b in enumerate(spatial_bullets):
        ax.text(4.5, 45.2 - idx * 1.45, b, color="#0f766e", fontsize=7.8, fontweight="bold", va="top", zorder=4)

    # ==========================================
    # 2. COLUMN 2: DUAL-MODE GATEWAY (X: 22.0 - 41.0)
    # ==========================================
    draw_card(
        x=22.0, y=35.5, w=19.0, h=54.0,
        bg="#ffffff", border="#059669",
        title="2. DUAL-MODE PRIVACY GATE",
        title_color="#047857", title_size=12.5,
        subtitle="Regulatory Trust Boundary Enforcement",
        border_width=2.0, radius=1.2,
    )

    # Track A: CLINICAL MODE (Top branch, top: 84.2)
    draw_card(
        x=23.3, y=62.0, w=16.4, h=22.0,
        bg="#f0f9ff", border="#0284c7",
        title="CLINICAL MODE (--mode clinical)",
        title_color="#0369a1", title_size=10.5,
        subtitle="Local Hospital Intranet Trust Zone",
        border_width=1.6,
    )
    clin_bullets = [
        "• Identified Patient Metadata Retained",
        "• PatientName, MRN (PatientID)",
        "• AccessionNumber, StudyDate, Physician",
        "• Files directly to Hospital PACS archive",
        "• Ingests into Patient EHR Chart (Epic)",
        "• RESTRICTION: Blocked from crossing",
        "  hospital enterprise firewall or FL",
    ]
    for idx, b in enumerate(clin_bullets):
        col = "#b91c1c" if "RESTRICTION" in b else "#1e293b"
        weight = "bold" if "RESTRICTION" in b else "normal"
        ax.text(24.2, 76.5 - idx * 1.75, b, color=col, fontsize=7.8, fontweight=weight, va="top", zorder=4)

    # Track B: RESEARCH MODE / RSNA V19 (Bottom branch)
    draw_card(
        x=23.3, y=37.0, w=16.4, h=23.5,
        bg="#f0fdf4", border="#10b981",
        title="RESEARCH MODE (--mode research)",
        title_color="#047857", title_size=10.5,
        subtitle="HIPAA Safe Harbor & RSNA V19 Profile",
        border_width=1.6,
    )
    research_bullets = [
        "• Official RSNA Anonymizer (V19) Integration",
        "  uv tool run --from rsna-anonymizer",
        "• Native CLI: cardiac-fat-fl anonymize",
        "• 18 HIPAA Safe Harbor Statutory PHI Scrubbed",
        "• Pseudonyms: Participant {pid}, ACC_{pid}",
        "• Cryptographic SHA-256 HMAC UID remapping",
        "• Deterministic 30-day date shifting",
        "• Pre-Flight PHI Audit Gate (deid-audit)",
        "  Fail-closed security: Halts on raw PHI leak",
        "• Invariant: 100% Geometry & HU preserved",
    ]
    for idx, b in enumerate(research_bullets):
        col = "#b45309" if "Invariant" in b or "Fail-closed" in b else "#1e293b"
        weight = "bold" if "Invariant" in b or "Fail-closed" in b else "normal"
        ax.text(24.2, 54.6 - idx * 1.6, b, color=col, fontsize=7.6, fontweight=weight, va="top", zorder=4)

    # Connect Ingestion to Dual-Mode
    draw_arrow(18.5, 73.0, 23.3, 73.0, color="#0284c7", width=2.2, label="Clinical", label_color="#0369a1")
    draw_arrow(18.5, 48.0, 23.3, 48.0, color="#059669", width=2.2, label="Research", label_color="#047857")

    # ==========================================
    # 3. COLUMN 3: 3D ANATOMY & SEGMENTATION ENGINE (X: 44.5 - 67.5)
    # ==========================================
    draw_card(
        x=44.5, y=35.5, w=23.0, h=54.0,
        bg="#ffffff", border="#9333ea",
        title="3. 3D ANATOMY & FAT QUANTIFICATION",
        title_color="#7e22ce", title_size=12.5,
        subtitle="Unified Multi-Foundation & Geometric Engine",
        border_width=2.0, radius=1.2,
    )

    # Subcard: Model Selection Track (top: 84.2)
    draw_card(
        x=45.8, y=67.5, w=20.4, h=16.5,
        bg="#faf5ff", border="#d8b4fe",
        title="Selectable Chamber Backend (--chamber-backend)",
        title_color="#7e22ce", title_size=10.0,
        subtitle="Multi-Foundation Model Options",
    )
    model_tracks = [
        ("• monai (Default)", "NGC SegResNet 1.5mm isotropic, 104 labels", "#0284c7"),
        ("• totalsegmentator", "nnU-Net Task253, Bit-exact Harvard match", "#059669"),
        ("• nvsegment / nvsegment_ct", "NVIDIA-MedTech VISTA-3D foundation model", "#d97706"),
        ("• DeepHeartSeg ONNX", "Zero-drift pericardial sac envelope (heart.seg.nrrd)", "#db2777"),
    ]
    for idx, (m_title, m_desc, m_col) in enumerate(model_tracks):
        ax.text(46.5, 77.0 - idx * 2.3, m_title, color=m_col, fontsize=8.0, fontweight="bold", va="top", zorder=4)
        ax.text(47.5, 75.8 - idx * 2.3, m_desc, color="#475569", fontsize=7.4, va="top", zorder=4)

    # Subcard: Spatial Morphology & Invariants
    draw_card(
        x=45.8, y=50.2, w=20.4, h=16.0,
        bg="#faf5ff", border="#d8b4fe",
        title="Clinical Invariants & 3D Spatial Transforms",
        title_color="#7e22ce", title_size=10.0,
        subtitle="Standardized Anatomical Taxonomy",
    )
    invariant_bullets = [
        "• Adipose Attenuation Window: strictly [-190.0, -30.0] HU",
        "• Coronary Sulci Derivation: geometric grooves (LAD, LCX, RCA)",
        "• 10.0 mm Anisotropic Radial PVAT Dilation (Voronoi 3D EDT)",
        "• CRISP-CT Vascular Inflammation Cutoff: Mean PVAT > -70.1 HU",
        "• Empty Region Guardrail: 0.0 cm3 volume & NaN mean HU",
    ]
    for idx, b in enumerate(invariant_bullets):
        ax.text(46.5, 60.0 - idx * 1.75, b, color="#1e293b", fontsize=7.6, va="top", zorder=4)

    # Subcard: 8 Compartment Palette
    draw_card(
        x=45.8, y=36.8, w=20.4, h=12.0,
        bg="#faf5ff", border="#d8b4fe",
        title="8-Compartment Segmentation (eat_pvat.seg.nrrd)",
        title_color="#7e22ce", title_size=9.5,
        subtitle="Canonical 4-Way Synchronized Color Palette",
    )
    compartments = [
        ("EAT LA", "#3b82f6"), ("EAT LV", "#06b6d4"), ("EAT RA", "#10b981"), ("EAT RV", "#eab308"),
        ("PVAT LAD", "#f97316"), ("PVAT LCX", "#ef4444"), ("PVAT RCA", "#ec4899"), ("EAT Total", "#9333ea"),
    ]
    for idx, (c_name, c_color) in enumerate(compartments):
        row = idx // 4
        col = idx % 4
        cx = 46.5 + col * 4.9
        cy = 41.2 - row * 2.5
        p = Rectangle((cx, cy), 1.0, 1.35, facecolor=c_color, edgecolor="#64748b", linewidth=0.6, zorder=4)
        ax.add_patch(p)
        ax.text(cx + 1.3, cy + 0.675, c_name, color="#0f172a", fontsize=7.2, fontweight="bold", va="center", zorder=4)

    # Connect Dual-Mode to Segmentation
    draw_arrow(39.7, 73.0, 44.5, 68.0, color="#0284c7", width=2.0)
    draw_arrow(39.7, 48.0, 44.5, 53.0, color="#059669", width=2.0)

    # ==========================================
    # 4. COLUMN 4: DUAL DOWNSTREAM DESTINATIONS (X: 71.0 - 97.5)
    # ==========================================
    # Destination A: LOCAL HOSPITAL PACS & EHR (Top)
    draw_card(
        x=71.0, y=63.5, w=26.5, h=26.0,
        bg="#ffffff", border="#0284c7",
        title="LOCAL EHR, PACS & WORKSTATION",
        title_color="#0369a1", title_size=12.0,
        subtitle="Hospital Clinical Production Deployment",
        border_width=2.0, radius=1.2,
    )
    # Sub-item 1: PACS
    draw_card(
        x=72.2, y=65.0, w=11.8, h=19.0,
        bg="#f0f9ff", border="#bae6fd",
        title="Enterprise PACS Archive",
        title_color="#0369a1", title_size=9.5,
        subtitle="DICOMweb & DIMSE Services",
    )
    pacs_items = [
        "• DICOM-SEG (SOP .66.4, 8 layers)",
        "• DICOM-SR TID 1500 (SOP .88.34)",
        "• Encapsulated PDF (SOP .104.1)",
        "• DICOMweb STOW-RS (TLS 1.3)",
        "• DIMSE C-STORE (pynetdicom)",
        "• Custom Enterprise Root CAs",
    ]
    for idx, b in enumerate(pacs_items):
        ax.text(72.8, 77.0 - idx * 1.6, b, color="#1e293b", fontsize=7.4, va="top", zorder=4)

    # Sub-item 2: EHR & OHIF
    draw_card(
        x=84.5, y=65.0, w=12.0, h=19.0,
        bg="#f0f9ff", border="#bae6fd",
        title="Hospital EHR & OHIF",
        title_color="#0369a1", title_size=9.5,
        subtitle="HL7 FHIR & Zero-Footprint UI",
    )
    ehr_items = [
        "• HL7 FHIR R4 DiagnosticReport",
        "• DocumentReference Chart PDF",
        "• LOINC 84245-0 EAT volume cm3",
        "• RadLex RID49533 PVAT HU",
        "• Open.epic MCP Server (stdio)",
        "• OHIF v3.13 2x2 Slicer Workstation",
    ]
    for idx, b in enumerate(ehr_items):
        ax.text(85.1, 77.0 - idx * 1.6, b, color="#1e293b", fontsize=7.4, va="top", zorder=4)

    # Destination B: (WORLD) FEDERATED LEARNING CONSORTIUM (Bottom)
    draw_card(
        x=71.0, y=35.5, w=26.5, h=26.0,
        bg="#ffffff", border="#d97706",
        title="(WORLD) FEDERATED LEARNING",
        title_color="#b45309", title_size=12.0,
        subtitle="Privacy-Preserving Multi-Hospital AI Consortium",
        border_width=2.0, radius=1.2,
    )
    # FL Subcard 1: FL Framework
    draw_card(
        x=72.2, y=37.0, w=11.8, h=19.0,
        bg="#fffbeb", border="#fde68a",
        title="NVIDIA FLARE 2.5+",
        title_color="#b45309", title_size=9.5,
        subtitle="Zero DICOM Transfer Across Firewalls",
    )
    fl_items = [
        "• Zero CT Image / Patient Transfer",
        "• FedProx Proximal Regularization",
        "• FedBN: Local Normalization",
        "  (BatchNorm3d remains on-site)",
        "• CrossSiteModelEval Consensus",
        "• Global weights: SRV_FL_model.pt",
        "• Self-healing Windows patching",
    ]
    for idx, b in enumerate(fl_items):
        col = "#b91c1c" if "Zero" in b else "#1e293b"
        weight = "bold" if "Zero" in b else "normal"
        ax.text(72.8, 49.0 - idx * 1.6, b, color=col, fontsize=7.4, fontweight=weight, va="top", zorder=4)

    # FL Subcard 2: Federated Analytics
    draw_card(
        x=84.5, y=37.0, w=12.0, h=19.0,
        bg="#fffbeb", border="#fde68a",
        title="Federated Analytics",
        title_color="#b45309", title_size=9.5,
        subtitle="Welford Scalar Moment Aggregation",
    )
    stat_items = [
        "• Exact Welford Summary Moments",
        "  (N, sum x, sum x^2, min, max)",
        "• 100% Mathematical Equivalence",
        "  to pooled multi-center data",
        "• Zero patient re-identification",
        "• Real-Time FL Monitor Dashboard",
        "• Automated site loss/DICE curves",
    ]
    for idx, b in enumerate(stat_items):
        ax.text(85.1, 49.0 - idx * 1.6, b, color="#1e293b", fontsize=7.4, va="top", zorder=4)

    # Connect Segmentation to Destinations
    draw_arrow(67.5, 68.0, 71.0, 74.0, color="#0284c7", width=2.2, label="Clinical Output", label_color="#0369a1")
    draw_arrow(67.5, 47.0, 71.0, 47.0, color="#d97706", width=2.2, label="De-ID Model Gradients", label_color="#b45309")

    # Feedback loop from FL to Model selection (Dashed arrow)
    draw_arrow(84.0, 35.5, 56.0, 35.5, color="#d97706", width=1.6, dashed=True, label="PCCP Distilled Global Weights", label_color="#b45309", label_y_offset=-1.2)

    # ==========================================
    # 5. FOUNDATION FOR REGULATORY FILING (SaMD) (Bottom Banner: Y: 2.5 - 32.0)
    # ==========================================
    draw_card(
        x=2.5, y=2.5, w=95.0, h=29.5,
        bg="#ffffff", border="#4f46e5",
        title="FOUNDATION FOR REGULATORY FILING: FDA Class II 510(k) Software as a Medical Device (SaMD)",
        title_color="#3730a3", title_size=12.5,
        subtitle="Comprehensive Quality Management System (QMS), IEC 62304 Life Cycle, ISO 14971 Risk Management & AI/ML PCCP",
        border_width=2.2, radius=1.2,
    )

    # 4 Pillar Cards
    pillars = [
        (
            "Pillar 1: FDA 510(k) Device Profile",
            "#818cf8", "#eef2ff", "#312e81",
            [
                "• Pathway: 21 CFR § 892.2050",
                "• Product Codes: LLZ (Image Proc), QIH (Automated Quant)",
                "• Documentation Level: Basic (Radiologist-in-the-loop)",
                "• Reference Predicate: Harvard CardiacFatQuant",
                "  Controlled Parity: r = 1.000, 100% DICE bit-exact",
                "• Indications: 3D EAT/PVAT volumetry from cardiac CT",
            ],
        ),
        (
            "Pillar 2: IEC 62304 Software Safety",
            "#2dd4bf", "#f0fdfa", "#115e59",
            [
                "• Safety Class: Class B (Non-serious injury risk)",
                "• Traceability Matrix: REQ-001 through REQ-013",
                "• Verification Suite: 138 automated pytest tests",
                "• Turnkey MAP Application Package (monai_deploy.py)",
                "• Fail-Closed Execution Architecture",
                "• Header-Preserving Spatial & Coordinate Integrity",
            ],
        ),
        (
            "Pillar 3: ISO 14971 Risk Controls",
            "#fcd34d", "#fffbeb", "#92400e",
            [
                "• FMEA Hazard Analysis: HAZ-001 through HAZ-010",
                "• Blooming Mitigation: Strict -30 HU contrast threshold",
                "• Boundary Guardrail: OHIF mandatory contour review",
                "• Anisotropic Sampling: Physical mm distance EDT",
                "• Model Tampering Defense: SHA-256 cryptographic check",
                "• PHI Leak Defense: RSNA V19 & pre-flight deid-audit",
            ],
        ),
        (
            "Pillar 4: FDA AI/ML PCCP Protocol",
            "#f472b6", "#fdf2f8", "#9d174d",
            [
                "• Predetermined Change Control Plan (FDA 2024)",
                "• Post-Market Federated Learning Retraining Bounds",
                "• Invariants Frozen: 8-ch, [-190, -30] HU, 10mm EDT",
                "• Multi-Site Acceptance: Dice >= 0.90, Pearson r >= 0.98",
                "• Automated Rollback: Reverts on threshold defect",
                "• Section 524B Cybersecurity: CycloneDX SBOM & SAST",
            ],
        ),
    ]

    for idx, (p_title, p_border, p_bg, p_text_col, p_bullets) in enumerate(pillars):
        px = 4.0 + idx * 23.5
        py = 4.2
        pw = 22.5
        ph = 22.0
        draw_card(
            x=px, y=py, w=pw, h=ph,
            bg=p_bg, border=p_border,
            title=p_title, title_color=p_text_col, title_size=9.5,
            border_width=1.4,
        )
        for b_idx, b in enumerate(p_bullets):
            ax.text(px + 0.8, py + ph - 4.4 - b_idx * 2.4, b, color="#1e293b", fontsize=7.4, va="top", zorder=4)

    # Save high-res PNG
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_png,
        dpi=300,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    print(f"Successfully generated 300 DPI Medical Light-Mode Big Picture PNG at: {output_png}")

    # Also save vector PDF if requested
    if output_pdf:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_pdf,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        print(f"Successfully generated Vector PDF at: {output_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path("docs/figures/project")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "samd_regulatory_pipeline_big_picture.png"
    out_pdf = out_dir / "samd_regulatory_pipeline_big_picture.pdf"
    create_big_picture_figure(out_png, out_pdf)
