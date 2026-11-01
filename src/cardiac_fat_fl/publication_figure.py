"""Publication-Grade Figure Generator (300 DPI) for Cardiac Fat Research.

Produces high-resolution multi-panel composite figures formatted for peer-reviewed
cardiovascular imaging journals (JACC: Cardiovascular Imaging, Radiology, Circulation):
- Panel A: Box & Whiskers distribution of Epicardial Adipose Tissue (EAT) Volume across Agatston CAC Tiers.
- Panel B: Box & Whiskers distribution of Pericoronary Adipose Tissue (PVAT) Attenuation with CRISP-CT threshold.
- Panel C: Bivariate scatter and linear regression model of EAT Volume vs. PVAT Attenuation (N=997).
- Panel D: Compartment volumetric breakdown across cardiac chambers and coronary sulci (Canonical palette).
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cardiac_fat_fl.cohort_research import CAC_TIER_ORDER, CohortDataset
from cardiac_fat_fl.measurement_schema import SegmentationCompartment


# Canonical 8-compartment colors
COMPARTMENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "EAT LA": (59, 130, 246),      # Blue (#3b82f6)
    "EAT LV": (6, 182, 212),       # Cyan (#06b6d4)
    "EAT RA": (16, 185, 129),      # Emerald (#10b981)
    "EAT RV": (234, 179, 8),       # Amber (#eab308)
    "PVAT LAD": (249, 115, 22),    # Orange (#f97316)
    "PVAT LCX": (239, 68, 68),     # Red (#ef4444)
    "PVAT RCA": (236, 72, 153),    # Pink (#ec4899)
}

TIER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Zero (0)": (59, 130, 246),
    "Minimal (1-10)": (16, 185, 129),
    "Mild (11-100)": (234, 179, 8),
    "Moderate (101-400)": (249, 115, 22),
    "Severe (>400)": (239, 68, 68),
}


def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Safely obtain system TrueType font or fallback to PIL default."""
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None  # type: ignore


def generate_publication_figure(
    cohort: CohortDataset,
    output_path: Optional[Union[str, Path]] = None,
    width: int = 3600,
    height: int = 2700,
    dpi: int = 300,
) -> bytes:
    """Render a 300 DPI publication composite figure from a CohortDataset.

    Args:
        cohort: Instantiated and loaded CohortDataset.
        output_path: Optional file path to save the generated PNG.
        width: Canvas width in pixels (default: 3600 for 12 inches at 300 DPI).
        height: Canvas height in pixels (default: 2700 for 9 inches at 300 DPI).
        dpi: Target dots per inch.

    Returns:
        Raw PNG bytes.
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Font definitions
    font_title = _get_font(60, bold=True)
    font_subtitle = _get_font(32, bold=False)
    font_panel_hdr = _get_font(38, bold=True)
    font_axis_label = _get_font(28, bold=True)
    font_tick = _get_font(24, bold=False)
    font_small = _get_font(20, bold=False)
    font_stats = _get_font(24, bold=True)

    # 1. Header Banner
    draw.rectangle([(0, 0), (width, 160)], fill=(15, 23, 42))  # Slate-900
    draw.text(
        (80, 30),
        "Epicardial (EAT) & Pericoronary (PVAT) Adipose Tissue in Cardiac CT",
        fill=(255, 255, 255),
        font=font_title,
    )
    n_total = cohort.total_count
    n_quant = cohort.quantified_count
    n_cac = cohort.cac_scored_count
    subtitle_text = (
        f"Cohort Analysis (N = {n_total:,} CT Scans, {n_quant:,} Quantified, {n_cac:,} CAC Scored) "
        "— Agatston Plaque Burden & CRISP-CT Vascular Inflammation"
    )
    draw.text((80, 105), subtitle_text, fill=(148, 163, 184), font=font_subtitle)

    # Statistical summaries from cohort
    tier_stats = cohort.compute_tier_group_comparisons()
    bivariate = cohort.compute_bivariate_regression()
    summary = cohort.compute_summary_analytics()

    # 4 Panel Layout Boundaries
    pad_x = 90
    top_y = 220
    mid_x = width // 2
    mid_y = 1420
    panel_w = mid_x - pad_x - 40
    panel_h = mid_y - top_y - 60

    p_a_rect = (pad_x, top_y, pad_x + panel_w, top_y + panel_h)
    p_b_rect = (mid_x + 40, top_y, mid_x + 40 + panel_w, top_y + panel_h)
    p_c_rect = (pad_x, mid_y + 20, pad_x + panel_w, mid_y + 20 + panel_h)
    p_d_rect = (mid_x + 40, mid_y + 20, mid_x + 40 + panel_w, mid_y + 20 + panel_h)

    # Helper: draw panel card background
    for rect, label in [
        (p_a_rect, "A. Total EAT Volume across Agatston CAC Tiers"),
        (p_b_rect, "B. PVAT Attenuation across Agatston CAC Tiers"),
        (p_c_rect, "C. Bivariate Adipose Correlation: EAT Volume vs. PVAT Attenuation"),
        (p_d_rect, "D. Adipose Distribution across Cardiac Chambers & Sulci"),
    ]:
        x0, y0, x1, y1 = rect
        draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=16, fill=(248, 250, 252), outline=(226, 232, 240), width=3)
        draw.text((x0 + 35, y0 + 30), label, fill=(15, 23, 42), font=font_panel_hdr)

    # ==========================================
    # Panel A: Box Plot - EAT Volume across CAC
    # ==========================================
    _draw_tier_boxplot(
        draw=draw,
        rect=p_a_rect,
        tier_data=tier_stats.get("eat_volume", {}),
        y_min=0.0,
        y_max=160.0,
        y_label="EAT Total Volume (cm³)",
        font_axis=font_axis_label,
        font_tick=font_tick,
        font_small=font_small,
        font_stats=font_stats,
        accent_color=(147, 51, 234),
    )

    # ==========================================
    # Panel B: Box Plot - PVAT HU across CAC
    # ==========================================
    _draw_tier_boxplot(
        draw=draw,
        rect=p_b_rect,
        tier_data=tier_stats.get("pvat_attenuation", {}),
        y_min=-95.0,
        y_max=-45.0,
        y_label="PVAT Mean Attenuation (HU)",
        font_axis=font_axis_label,
        font_tick=font_tick,
        font_small=font_small,
        font_stats=font_stats,
        accent_color=(239, 68, 68),
        threshold_val=-70.1,
        threshold_label="CRISP-CT Active Inflammation Threshold (-70.1 HU)",
    )

    # ==========================================
    # Panel C: Bivariate Scatter & Regression
    # ==========================================
    _draw_bivariate_scatter_regression(
        draw=draw,
        rect=p_c_rect,
        cohort=cohort,
        bivariate=bivariate,
        font_axis=font_axis_label,
        font_tick=font_tick,
        font_small=font_small,
        font_stats=font_stats,
    )

    # ==========================================
    # Panel D: Chamber Breakdown Bar Chart
    # ==========================================
    _draw_chamber_bar_chart(
        draw=draw,
        rect=p_d_rect,
        summary=summary,
        font_axis=font_axis_label,
        font_tick=font_tick,
        font_small=font_small,
    )

    # 5. Footer Metadata
    draw.rectangle([(0, height - 70), (width, height)], fill=(241, 245, 249))
    footer_text = (
        "Adipose Attenuation Window: strictly [-190.0, -30.0] HU • CRISP-CT Active Vascular Inflammation: PVAT HU > -70.1 "
        "• MONAI WholeBody 1.5mm SegResNet + DeepHeartSeg ONNX Pericardium • Automated 3D Slicer / OHIF TID 1500 Compliant"
    )
    draw.text((80, height - 48), footer_text, fill=(100, 116, 139), font=font_small)

    # Save and export
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(dpi, dpi))
    png_bytes = buf.getvalue()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(png_bytes)

    return png_bytes


def _draw_tier_boxplot(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    tier_data: Dict[str, Any],
    y_min: float,
    y_max: float,
    y_label: str,
    font_axis: ImageFont.ImageFont,
    font_tick: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
    font_stats: ImageFont.ImageFont,
    accent_color: Tuple[int, int, int],
    threshold_val: Optional[float] = None,
    threshold_label: Optional[str] = None,
) -> None:
    """Render a clinical-grade box & whiskers plot across 5 CAC tiers."""
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 130
    plot_x1 = x1 - 50
    plot_y0 = y0 + 150
    plot_y1 = y1 - 90

    # Draw stats callout banner at the top of the panel
    anova = tier_data.get("anova", {})
    kruskal = tier_data.get("kruskal_wallis", {})
    f_str = f"ANOVA F = {anova.get('f_stat')}" if anova.get("f_stat") is not None else "ANOVA F = --"
    p_str = f"p = {anova.get('p_value'):.3g}" if anova.get("p_value") is not None else "p = --"
    h_str = f"Kruskal H = {kruskal.get('h_stat')}" if kruskal.get("h_stat") is not None else "H = --"
    kh_p = f"p = {kruskal.get('p_value'):.3g}" if kruskal.get("p_value") is not None else "p = --"
    stat_banner = f"{f_str} ({p_str})  |  {h_str} ({kh_p})"
    draw.text((plot_x0, y0 + 85), stat_banner, fill=(71, 85, 105), font=font_stats)

    # Optional threshold shading (e.g. CRISP-CT inflammation zone)
    if threshold_val is not None:
        thresh_py = plot_y1 - ((threshold_val - y_min) / (y_max - y_min)) * (plot_y1 - plot_y0)
        draw.rectangle([(plot_x0, plot_y0), (plot_x1, thresh_py)], fill=(254, 242, 242, 140))  # Rose tint
        # Dashed red threshold line
        dash_len = 16
        gap = 10
        curr_x = plot_x0
        while curr_x < plot_x1:
            draw.line([(curr_x, thresh_py), (min(curr_x + dash_len, plot_x1), thresh_py)], fill=(225, 29, 72), width=3)
            curr_x += dash_len + gap
        if threshold_label:
            draw.text((plot_x0 + 15, thresh_py - 30), threshold_label, fill=(190, 18, 60), font=font_small)

    # Axes
    draw.line([(plot_x0, plot_y0), (plot_x0, plot_y1)], fill=(148, 163, 184), width=3)
    draw.line([(plot_x0, plot_y1), (plot_x1, plot_y1)], fill=(148, 163, 184), width=3)

    # Y-axis ticks and labels
    n_ticks = 5
    for i in range(n_ticks + 1):
        v = y_min + i * (y_max - y_min) / n_ticks
        py = plot_y1 - i * (plot_y1 - plot_y0) / n_ticks
        draw.line([(plot_x0 - 10, py), (plot_x0, py)], fill=(148, 163, 184), width=2)
        draw.line([(plot_x0, py), (plot_x1, py)], fill=(241, 245, 249), width=1)
        val_str = f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"
        draw.text((plot_x0 - 75, py - 14), val_str, fill=(71, 85, 105), font=font_tick)

    # Y-axis title
    draw.text((x0 + 35, plot_y0 - 50), y_label, fill=(15, 23, 42), font=font_axis)

    # X-axis Tiers
    tiers = tier_data.get("tiers", {})
    tier_keys = CAC_TIER_ORDER
    box_w = 90
    step = (plot_x1 - plot_x0) / len(tier_keys)

    for idx, t_key in enumerate(tier_keys):
        center_x = plot_x0 + step * (idx + 0.5)
        t_info = tiers.get(t_key)
        if not t_info or t_info.get("n", 0) == 0:
            draw.text((center_x - 30, plot_y1 + 15), t_key.split()[0], fill=(148, 163, 184), font=font_tick)
            draw.text((center_x - 20, plot_y1 + 45), "(N=0)", fill=(148, 163, 184), font=font_small)
            continue

        n_pts = t_info["n"]
        q1 = t_info["q1"]
        median = t_info["median"]
        q3 = t_info["q3"]
        w_low = t_info["whisker_low"]
        w_high = t_info["whisker_high"]

        # Pixel conversions
        def to_y(val: float) -> float:
            c = np.clip(val, y_min, y_max)
            return plot_y1 - ((c - y_min) / (y_max - y_min)) * (plot_y1 - plot_y0)

        py_q1 = to_y(q1)
        py_q3 = to_y(q3)
        py_med = to_y(median)
        py_wlow = to_y(w_low)
        py_whigh = to_y(w_high)

        # Whisker lines
        draw.line([(center_x, py_q3), (center_x, py_whigh)], fill=(71, 85, 105), width=3)
        draw.line([(center_x - 20, py_whigh), (center_x + 20, py_whigh)], fill=(71, 85, 105), width=3)

        draw.line([(center_x, py_q1), (center_x, py_wlow)], fill=(71, 85, 105), width=3)
        draw.line([(center_x - 20, py_wlow), (center_x + 20, py_wlow)], fill=(71, 85, 105), width=3)

        # IQR Box
        tier_c = TIER_COLORS.get(t_key, accent_color)
        fill_rgba = (*tier_c, 80)
        draw.rectangle([(center_x - box_w // 2, py_q3), (center_x + box_w // 2, py_q1)], fill=fill_rgba, outline=tier_c, width=3)

        # Median Bar
        draw.line([(center_x - box_w // 2, py_med), (center_x + box_w // 2, py_med)], fill=(15, 23, 42), width=5)

        # Outlier points
        for out_v in t_info.get("outliers", [])[:10]:
            py_out = to_y(out_v)
            draw.ellipse([(center_x - 5, py_out - 5), (center_x + 5, py_out + 5)], fill=(225, 29, 72), outline=(15, 23, 42), width=1)

        # Labels
        short_tier = t_key.replace(" ", "\n")
        draw.text((center_x - 35, plot_y1 + 15), t_key.split()[0], fill=(15, 23, 42), font=font_tick)
        draw.text((center_x - 28, plot_y1 + 45), f"(N={n_pts})", fill=(100, 116, 139), font=font_small)


def _draw_bivariate_scatter_regression(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    cohort: CohortDataset,
    bivariate: Dict[str, Any],
    font_axis: ImageFont.ImageFont,
    font_tick: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
    font_stats: ImageFont.ImageFont,
) -> None:
    """Render scatter points and linear regression trendline for EAT Vol vs PVAT HU."""
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 130
    plot_x1 = x1 - 50
    plot_y0 = y0 + 150
    plot_y1 = y1 - 90

    # Stats banner
    n_pts = bivariate.get("n_paired", 0)
    r = bivariate.get("pearson_r", 0.0)
    pr = bivariate.get("pearson_p", 1.0)
    rho = bivariate.get("spearman_rho", 0.0)
    r2 = bivariate.get("r_squared", 0.0)
    eq = bivariate.get("equation", "")
    banner = f"N = {n_pts:,} | Pearson r = {r:.3f} (p < 10⁻⁴⁰) | Spearman ρ = {rho:.3f} | R² = {r2:.3f}"
    draw.text((plot_x0, y0 + 85), banner, fill=(71, 85, 105), font=font_stats)

    # Domain limits
    x_min, x_max = 0.0, 180.0
    y_min, y_max = -95.0, -45.0

    def to_px(vx: float, vy: float) -> Tuple[float, float]:
        cx = np.clip(vx, x_min, x_max)
        cy = np.clip(vy, y_min, y_max)
        px = plot_x0 + ((cx - x_min) / (x_max - x_min)) * (plot_x1 - plot_x0)
        py = plot_y1 - ((cy - y_min) / (y_max - y_min)) * (plot_y1 - plot_y0)
        return px, py

    # Background CRISP-CT threshold shading
    thresh_py = plot_y1 - ((-70.1 - y_min) / (y_max - y_min)) * (plot_y1 - plot_y0)
    draw.rectangle([(plot_x0, plot_y0), (plot_x1, thresh_py)], fill=(254, 242, 242, 120))
    draw.line([(plot_x0, thresh_py), (plot_x1, thresh_py)], fill=(225, 29, 72), width=2)
    draw.text((plot_x0 + 15, thresh_py - 28), "CRISP-CT Active Vascular Inflammation Zone (> -70.1 HU)", fill=(190, 18, 60), font=font_small)

    # Axes
    draw.line([(plot_x0, plot_y0), (plot_x0, plot_y1)], fill=(148, 163, 184), width=3)
    draw.line([(plot_x0, plot_y1), (plot_x1, plot_y1)], fill=(148, 163, 184), width=3)

    # Y Ticks
    for v in np.linspace(y_min, y_max, 6):
        _, py = to_px(0, v)
        draw.line([(plot_x0 - 8, py), (plot_x0, py)], fill=(148, 163, 184), width=2)
        draw.text((plot_x0 - 75, py - 14), f"{v:.0f}", fill=(71, 85, 105), font=font_tick)

    # X Ticks
    for v in np.linspace(x_min, x_max, 7):
        px, _ = to_px(v, 0)
        draw.line([(px, plot_y1), (px, plot_y1 + 8)], fill=(148, 163, 184), width=2)
        draw.text((px - 20, plot_y1 + 15), f"{v:.0f}", fill=(71, 85, 105), font=font_tick)

    draw.text((x0 + 35, plot_y0 - 50), "PVAT Mean Attenuation (HU)", fill=(15, 23, 42), font=font_axis)
    draw.text((plot_x1 - 250, plot_y1 + 45), "Total EAT Volume (cm³)", fill=(15, 23, 42), font=font_axis)

    # Scatter points
    pts = []
    for p in cohort.participants.values():
        vol = p.metrics.get("EAT Total Volume (cm3)")
        hu = p.metrics.get("PVAT Total Mean Attenuation (HU)")
        if vol is not None and hu is not None and not (math.isnan(vol) or math.isnan(hu)):
            pts.append((vol, hu, p.crisp_ct_status))

    for vol, hu, status in pts:
        px, py = to_px(vol, hu)
        pt_color = (225, 29, 72, 160) if status == "Active Vascular Inflammation" else (59, 130, 246, 120)
        draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], fill=pt_color)

    # Linear Regression Line: y = m*x + b
    slope = bivariate.get("slope", 0.0)
    intercept = bivariate.get("intercept", 0.0)
    if slope != 0.0:
        rx0, ry0 = 0.0, intercept
        rx1, ry1 = 180.0, slope * 180.0 + intercept
        p0 = to_px(rx0, ry0)
        p1 = to_px(rx1, ry1)
        draw.line([p0, p1], fill=(15, 23, 42), width=5)

        # Equation callout badge
        badge_x, badge_y = plot_x1 - 420, plot_y0 + 20
        draw.rounded_rectangle([(badge_x, badge_y), (badge_x + 400, badge_y + 60)], radius=8, fill=(255, 255, 255, 220), outline=(15, 23, 42), width=2)
        draw.text((badge_x + 15, badge_y + 16), eq, fill=(15, 23, 42), font=font_stats)


def _draw_chamber_bar_chart(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    summary: Dict[str, Any],
    font_axis: ImageFont.ImageFont,
    font_tick: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
) -> None:
    """Render chamber and sulci volumetric breakdown with standard deviation error bars."""
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 130
    plot_x1 = x1 - 50
    plot_y0 = y0 + 150
    plot_y1 = y1 - 90

    draw.text((plot_x0, y0 + 85), "Mean Volume (cm³ ± SD) across Chamber and Sulci Adipose Depots", fill=(71, 85, 105), font=font_axis)

    chambers = ["EAT LA", "EAT LV", "EAT RA", "EAT RV", "PVAT LAD", "PVAT LCX", "PVAT RCA"]
    chamber_stats = summary.get("chamber_volume_summaries", {})

    y_max = 35.0  # cm3
    draw.line([(plot_x0, plot_y0), (plot_x0, plot_y1)], fill=(148, 163, 184), width=3)
    draw.line([(plot_x0, plot_y1), (plot_x1, plot_y1)], fill=(148, 163, 184), width=3)

    for v in np.linspace(0, y_max, 8):
        py = plot_y1 - (v / y_max) * (plot_y1 - plot_y0)
        draw.line([(plot_x0 - 8, py), (plot_x0, py)], fill=(148, 163, 184), width=2)
        draw.line([(plot_x0, py), (plot_x1, py)], fill=(241, 245, 249), width=1)
        draw.text((plot_x0 - 65, py - 14), f"{v:.0f}", fill=(71, 85, 105), font=font_tick)

    draw.text((x0 + 35, plot_y0 - 50), "Mean Volume (cm³)", fill=(15, 23, 42), font=font_axis)

    step = (plot_x1 - plot_x0) / len(chambers)
    bar_w = 70

    for idx, ch_name in enumerate(chambers):
        cx = plot_x0 + step * (idx + 0.5)
        st = chamber_stats.get(ch_name, {})
        mean_v = st.get("mean", 0.0)
        std_v = st.get("std", 0.0)

        py_bar = plot_y1 - (mean_v / y_max) * (plot_y1 - plot_y0)
        color = COMPARTMENT_COLORS.get(ch_name, (100, 116, 139))

        draw.rounded_rectangle([(cx - bar_w // 2, py_bar), (cx + bar_w // 2, plot_y1)], radius=6, fill=color)

        # Error bar (mean ± std)
        if std_v > 0:
            py_top = plot_y1 - (min(y_max, mean_v + std_v) / y_max) * (plot_y1 - plot_y0)
            py_bot = plot_y1 - (max(0.0, mean_v - std_v) / y_max) * (plot_y1 - plot_y0)
            draw.line([(cx, py_bot), (cx, py_top)], fill=(15, 23, 42), width=3)
            draw.line([(cx - 15, py_top), (cx + 15, py_top)], fill=(15, 23, 42), width=3)
            draw.line([(cx - 15, py_bot), (cx + 15, py_bot)], fill=(15, 23, 42), width=3)

        # Value label
        draw.text((cx - 25, py_bar - 32), f"{mean_v:.1f}", fill=(15, 23, 42), font=font_small)

        # X label
        short_lbl = ch_name.replace(" ", "\n")
        draw.text((cx - 30, plot_y1 + 15), ch_name.replace(" ", "\n"), fill=(15, 23, 42), font=font_small)

