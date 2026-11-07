"""Generate 300 DPI Publication-Grade Big Picture Architecture Diagram.

Visualizes the end-to-end SaMD framework:
CT Ingestion -> Dual-Mode Gateway (Clinical vs. Research/RSNA V19) ->
Multi-Model 3D Segmentation (MONAI / TotalSeg / NV-Segment) ->
Local EHR / PACS & (World) Federated Learning ->
Foundation for FDA SaMD 510(k), IEC 62304 & ISO 14971 Regulatory Filing.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch, Rectangle
import numpy as np


def create_big_picture_figure(output_path: Path) -> None:
    # 16:10 aspect ratio, ultra-high resolution
    fig = plt.figure(figsize=(24, 15), dpi=300, facecolor="#090d16")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#090d16")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Helper for drawing rounded boxes
    def draw_card(
        x: float, y: float, w: float, h: float,
        bg: str, border: str, title: str,
        title_color: str = "#ffffff",
        title_size: int = 12,
        subtitle: str = "",
        subtitle_color: str = "#94a3b8",
        border_width: float = 1.8,
        radius: float = 1.0,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.3,rounding_size={radius}",
            linewidth=border_width,
            edgecolor=border,
            facecolor=bg,
            zorder=2,
        )
        ax.add_patch(patch)
        if title:
            ax.text(
                x + w * 0.5, y + h - 1.8, title,
                ha="center", va="top",
                color=title_color, fontsize=title_size,
                fontweight="bold", fontfamily="sans-serif",
                zorder=3,
            )
        if subtitle:
            ax.text(
                x + w * 0.5, y + h - 3.8, subtitle,
                ha="center", va="top",
                color=subtitle_color, fontsize=9.5,
                fontstyle="italic", fontfamily="sans-serif",
                zorder=3,
            )

    # Helper for connector arrows
    def draw_arrow(
        x1: float, y1: float, x2: float, y2: float,
        color: str = "#38bdf8",
        width: float = 2.0,
        label: str = "",
        label_color: str = "#38bdf8",
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
                color=label_color, fontsize=8.5,
                fontweight="bold", fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#090d16", edgecolor=color, alpha=0.9, lw=0.8),
                zorder=5,
            )

    # ==========================================
    # 0. HEADER & BANNER
    # ==========================================
    # Top banner bar
    banner = Rectangle((0, 93), 100, 7, facecolor="#0f172a", edgecolor="#1e293b", linewidth=1.5, zorder=1)
    ax.add_patch(banner)

    ax.text(
        3.0, 97.5,
        "MONAI-CARDIAC-FAT-FL  |  END-TO-END CLINICAL & FEDERATED SaMD PIPELINE ARCHITECTURE",
        ha="left", va="center",
        color="#38bdf8", fontsize=16, fontweight="bold", fontfamily="sans-serif",
    )
    ax.text(
        3.0, 94.6,
        "From Multi-Vendor CT Ingestion & Dual-Mode Privacy Gate to 3D Foundation Segmentation, Institutional EHR/PACS & Multi-Center Federated Learning",
        ha="left", va="center",
        color="#cbd5e1", fontsize=11, fontfamily="sans-serif",
    )

    # Badges on top right
    badges = [
        ("FDA Class II 510(k)", "#0284c7", "#e0f2fe"),
        ("IEC 62304 Class B", "#0d9488", "#ccfbf1"),
        ("ISO 14971 FMEA", "#d97706", "#fef3c7"),
        ("NVIDIA FLARE 2.5+", "#16a34a", "#dcfce7"),
        ("OHIF v3.13 Next", "#9333ea", "#f3e8ff"),
    ]
    bx = 97.0
    for label, border, text_col in reversed(badges):
        blen = len(label) * 0.72 + 2.0
        bx -= blen
        patch = FancyBboxPatch(
            (bx, 94.5), blen - 0.8, 3.2,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            linewidth=1.2, edgecolor=border, facecolor="#1e293b", zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            bx + (blen - 0.8) * 0.5, 96.1, label,
            ha="center", va="center", color=text_col, fontsize=8.5, fontweight="bold", zorder=4
        )
        bx -= 0.6

    # ==========================================
    # 1. COLUMN 1: CARDIAC CT INGESTION (X: 2.5 - 18.5)
    # ==========================================
    draw_card(
        x=2.5, y=36.0, w=16.0, h=54.0,
        bg="#0b172a", border="#38bdf8",
        title="1. CT INGESTION",
        title_color="#38bdf8", title_size=13,
        subtitle="Universal 3D MetaTensor Loader",
        border_width=2.0, radius=1.2,
    )

    # Inner CT sub-cards
    # Subcard 1: Modalities & Scanners
    draw_card(
        x=3.8, y=72.0, w=13.4, h=14.0,
        bg="#0f213e", border="#1e3a5f",
        title="Multi-Vendor Scanners",
        title_color="#93c5fd", title_size=10,
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
        ax.text(4.5, 79.5 - idx * 1.5, b, color="#e2e8f0", fontsize=8.0, va="top", zorder=4)

    # Subcard 2: Formats & I/O
    draw_card(
        x=3.8, y=54.5, w=13.4, h=15.5,
        bg="#0f213e", border="#1e3a5f",
        title="Native 3D I/O Formats",
        title_color="#93c5fd", title_size=10,
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
        ax.text(4.5, 63.5 - idx * 1.45, b, color="#cbd5e1", fontsize=8.0, va="top", zorder=4)

    # Subcard 3: Spatial Coordinates
    draw_card(
        x=3.8, y=38.0, w=13.4, h=14.5,
        bg="#0f213e", border="#1e3a5f",
        title="Coordinate Preservation",
        title_color="#93c5fd", title_size=10,
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
        ax.text(4.5, 46.0 - idx * 1.5, b, color="#a5f3fc", fontsize=8.0, va="top", zorder=4)

    # ==========================================
    # 2. COLUMN 2: DUAL-MODE GATEWAY (X: 22.0 - 41.0)
    # ==========================================
    # Stage 2 Outer boundary
    draw_card(
        x=22.0, y=36.0, w=19.0, h=54.0,
        bg="#071822", border="#10b981",
        title="2. DUAL-MODE PRIVACY GATE",
        title_color="#34d399", title_size=13,
        subtitle="Regulatory Trust Boundary Enforcement",
        border_width=2.0, radius=1.2,
    )

    # Track A: CLINICAL MODE (Top branch)
    draw_card(
        x=23.3, y=65.5, w=16.4, h=21.0,
        bg="#092535", border="#0284c7",
        title="CLINICAL MODE (--mode clinical)",
        title_color="#38bdf8", title_size=10.5,
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
        col = "#f87171" if "RESTRICTION" in b else "#e0f2fe"
        ax.text(24.2, 79.5 - idx * 1.7, b, color=col, fontsize=8.0, va="top", zorder=4)

    # Track B: RESEARCH MODE / RSNA V19 (Bottom branch)
    draw_card(
        x=23.3, y=38.0, w=16.4, h=25.5,
        bg="#062e24", border="#059669",
        title="RESEARCH MODE (--mode research)",
        title_color="#34d399", title_size=10.5,
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
        col = "#fde047" if "Invariant" in b or "Fail-closed" in b else "#d1fae5"
        ax.text(24.2, 57.5 - idx * 1.7, b, color=col, fontsize=7.8, va="top", zorder=4)

    # Connect Ingestion to Dual-Mode
    draw_arrow(18.5, 75.0, 23.3, 75.0, color="#38bdf8", width=2.2, label="Clinical", label_color="#38bdf8")
    draw_arrow(18.5, 50.0, 23.3, 50.0, color="#10b981", width=2.2, label="Research", label_color="#10b981")

    # ==========================================
    # 3. COLUMN 3: 3D ANATOMY & SEGMENTATION ENGINE (X: 44.5 - 67.5)
    # ==========================================
    draw_card(
        x=44.5, y=36.0, w=23.0, h=54.0,
        bg="#180e29", border="#a855f7",
        title="3. 3D ANATOMY & FAT QUANTIFICATION",
        title_color="#c084fc", title_size=13,
        subtitle="Unified Multi-Foundation & Geometric Engine",
        border_width=2.0, radius=1.2,
    )

    # Subcard: Model Selection Track
    draw_card(
        x=45.8, y=70.0, w=20.4, h=16.5,
        bg="#251242", border="#7e22ce",
        title="Selectable Chamber Backend (--chamber-backend)",
        title_color="#e9d5ff", title_size=10,
        subtitle="Multi-Foundation Model Options",
    )
    model_tracks = [
        ("• monai (Default)", "NGC SegResNet 1.5mm isotropic, 104 labels", "#38bdf8"),
        ("• totalsegmentator", "nnU-Net Task253, Bit-exact Harvard match", "#34d399"),
        ("• nvsegment / nvsegment_ct", "NVIDIA-MedTech VISTA-3D foundation model", "#fbbf24"),
        ("• DeepHeartSeg ONNX", "Zero-drift pericardial sac envelope (heart.seg.nrrd)", "#f472b6"),
    ]
    for idx, (m_title, m_desc, m_col) in enumerate(model_tracks):
        ax.text(46.5, 79.5 - idx * 2.3, m_title, color=m_col, fontsize=8.2, fontweight="bold", va="top", zorder=4)
        ax.text(47.5, 78.3 - idx * 2.3, m_desc, color="#cbd5e1", fontsize=7.5, va="top", zorder=4)

    # Subcard: Spatial Morphology & Invariants
    draw_card(
        x=45.8, y=51.0, w=20.4, h=17.5,
        bg="#251242", border="#7e22ce",
        title="Clinical Invariants & 3D Spatial Transforms",
        title_color="#e9d5ff", title_size=10,
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
        ax.text(46.5, 62.0 - idx * 1.9, b, color="#f3e8ff", fontsize=7.8, va="top", zorder=4)

    # Subcard: 8 Compartment Palette
    draw_card(
        x=45.8, y=38.0, w=20.4, h=11.5,
        bg="#251242", border="#7e22ce",
        title="8-Compartment Segmentation (eat_pvat.seg.nrrd)",
        title_color="#e9d5ff", title_size=9.5,
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
        cy = 43.5 - row * 2.3
        p = Rectangle((cx, cy), 1.0, 1.4, facecolor=c_color, edgecolor="#ffffff", linewidth=0.6, zorder=4)
        ax.add_patch(p)
        ax.text(cx + 1.3, cy + 0.7, c_name, color="#ffffff", fontsize=7.2, va="center", zorder=4)

    # Connect Dual-Mode to Segmentation
    draw_arrow(39.7, 75.0, 44.5, 70.0, color="#38bdf8", width=2.0)
    draw_arrow(39.7, 50.0, 44.5, 55.0, color="#10b981", width=2.0)

    # ==========================================
    # 4. COLUMN 4: DUAL DOWNSTREAM DESTINATIONS (X: 71.0 - 97.5)
    # ==========================================
    # Destination A: LOCAL HOSPITAL PACS & EHR (Top)
    draw_card(
        x=71.0, y=65.5, w=26.5, h=24.5,
        bg="#0b233a", border="#0284c7",
        title="LOCAL EHR, PACS & WORKSTATION",
        title_color="#38bdf8", title_size=12.5,
        subtitle="Hospital Clinical Production Deployment",
        border_width=2.0, radius=1.2,
    )
    # Sub-item 1: PACS
    draw_card(
        x=72.2, y=67.0, w=11.8, h=17.5,
        bg="#0f3456", border="#0369a1",
        title="Enterprise PACS Archive",
        title_color="#93c5fd", title_size=9.5,
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
        ax.text(72.8, 77.8 - idx * 1.6, b, color="#e0f2fe", fontsize=7.6, va="top", zorder=4)

    # Sub-item 2: EHR & OHIF
    draw_card(
        x=84.5, y=67.0, w=12.0, h=17.5,
        bg="#0f3456", border="#0369a1",
        title="Hospital EHR & OHIF",
        title_color="#93c5fd", title_size=9.5,
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
        ax.text(85.1, 77.8 - idx * 1.6, b, color="#e0f2fe", fontsize=7.6, va="top", zorder=4)

    # Destination B: (WORLD) FEDERATED LEARNING CONSORTIUM (Bottom)
    draw_card(
        x=71.0, y=36.0, w=26.5, h=26.5,
        bg="#291a08", border="#d97706",
        title="(WORLD) FEDERATED LEARNING",
        title_color="#fbbf24", title_size=12.5,
        subtitle="Privacy-Preserving Multi-Hospital AI Consortium",
        border_width=2.0, radius=1.2,
    )
    # FL Subcard 1: FL Framework
    draw_card(
        x=72.2, y=37.5, w=11.8, h=19.5,
        bg="#3d270c", border="#b45309",
        title="NVIDIA FLARE 2.5+",
        title_color="#fde68a", title_size=9.5,
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
        col = "#fca5a5" if "Zero" in b else "#fef3c7"
        ax.text(72.8, 50.5 - idx * 1.65, b, color=col, fontsize=7.6, va="top", zorder=4)

    # FL Subcard 2: Federated Analytics
    draw_card(
        x=84.5, y=37.5, w=12.0, h=19.5,
        bg="#3d270c", border="#b45309",
        title="Federated Analytics",
        title_color="#fde68a", title_size=9.5,
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
        ax.text(85.1, 50.5 - idx * 1.65, b, color="#fef3c7", fontsize=7.6, va="top", zorder=4)

    # Connect Segmentation to Destinations
    draw_arrow(67.5, 68.0, 71.0, 74.0, color="#38bdf8", width=2.2, label="Clinical Output", label_color="#38bdf8")
    draw_arrow(67.5, 48.0, 71.0, 48.0, color="#f59e0b", width=2.2, label="De-ID Model Gradients", label_color="#f59e0b")

    # Feedback loop from FL to Model selection (Dashed arrow)
    draw_arrow(84.0, 36.0, 56.0, 36.0, color="#f59e0b", width=1.6, dashed=True, label="PCCP Distilled Global Weights", label_color="#fbbf24", label_y_offset=-1.2)

    # ==========================================
    # 5. FOUNDATION FOR REGULATORY FILING (SaMD) (Bottom Banner: Y: 3.0 - 32.0)
    # ==========================================
    draw_card(
        x=2.5, y=3.0, w=95.0, h=28.5,
        bg="#111927", border="#6366f1",
        title="FOUNDATION FOR REGULATORY FILING: FDA Class II 510(k) Software as a Medical Device (SaMD)",
        title_color="#818cf8", title_size=13,
        subtitle="Comprehensive Quality Management System (QMS), IEC 62304 Life Cycle, ISO 14971 Risk Management & AI/ML PCCP",
        border_width=2.2, radius=1.2,
    )

    # 4 Pillar Cards
    pillars = [
        (
            "Pillar 1: FDA 510(k) Device Profile",
            "#4338ca", "#312e81", "#c7d2fe",
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
            "#0f766e", "#134e4a", "#99f6e4",
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
            "Pillar 3: ISO 14971 Risk Management",
            "#b45309", "#78350f", "#fde68a",
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
            "#be185d", "#831843", "#fbcfe8",
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
        px = 4.2 + idx * 23.3
        py = 4.8
        pw = 22.2
        ph = 21.0
        draw_card(
            x=px, y=py, w=pw, h=ph,
            bg=p_bg, border=p_border,
            title=p_title, title_color=p_text_col, title_size=10.0,
            border_width=1.4,
        )
        for b_idx, b in enumerate(p_bullets):
            ax.text(px + 0.8, py + ph - 4.5 - b_idx * 2.5, b, color="#f1f5f9", fontsize=7.6, va="top", zorder=4)

    # Save high-res figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path,
        dpi=300,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)
    print(f"Successfully generated 300 DPI Big Picture SaMD figure at: {output_path}")


if __name__ == "__main__":
    out_file = Path("docs/figures/samd_regulatory_pipeline_big_picture.png")
    create_big_picture_figure(out_file)
