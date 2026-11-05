"""Stanford COCA Stratified 25-Case Multi-Model Benchmark Pilot.

Executes and benchmarks all five cardiac adipose segmentation tracks across
a balanced 25-participant cohort from Stanford COCA (5 cases per CAC tier):
  1. Harvard Reference (circ-ml/CardiacFatQuant: TotalSeg v1 + Voronoi/7-Rule)
  2. MONAI WholeBody (monai_wholebody_ct_segmentation 1.5mm SegResNet + 7-Rule)
  3. TotalSegmentator Track (TotalSeg v1 + 7-Rule Morphology)
  4. NVIDIA MedTech NV-Segment-CTMR (VISTA-3D 346-class Foundation + 7-Rule)
  5. NVIDIA MedTech NV-Segment-CT (VISTA-3D 133-class CT-Dedicated + 7-Rule)

Tiers:
  - Zero (0): ['nongated_1', 'nongated_52', 'nongated_91', 'nongated_127', 'nongated_167']
  - Minimal (1-10): ['nongated_12', 'nongated_22', 'nongated_54', 'nongated_122', 'nongated_182']
  - Mild (11-100): ['nongated_13', 'nongated_66', 'nongated_102', 'nongated_142', 'nongated_175']
  - Moderate (101-400): ['nongated_2', 'nongated_25', 'nongated_57', 'nongated_98', 'nongated_128']
  - Severe (>400): ['0', '95', '190', '285', '380']
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from cardiac_fat_fl.anatomy_segmenter import UnifiedAnatomySegmenter
from cardiac_fat_fl.cohort_research import CohortDataset, categorize_calcium_score, CAC_TIER_ORDER
from cardiac_fat_fl.io_metatensor import (
    read_medical_image_as_metatensor,
    read_nrrd_as_metatensor,
    write_eat_pvat_seg_nrrd,
    write_monai_wholebody_ct_segmentation_seg_nrrd,
)
from cardiac_fat_fl.measurement_schema import (
    MEASUREMENT_CSV_COLUMNS,
    SegmentationCompartment,
    calculate_volume_cm3,
)
from cardiac_fat_fl.transforms import (
    ComputeEatPvatMasksd,
    QuantifyAdiposeMetricsd,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("coca_pilot")


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient between two binary masks."""
    b1 = mask1.astype(bool)
    b2 = mask2.astype(bool)
    intersection = np.logical_and(b1, b2).sum()
    total = b1.sum() + b2.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def run_coca_stratified_pilot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Stanford COCA Stratified Benchmark Pilot on %s (%s)", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    coca_dir = Path(r"E:\CT\Stanford_COCA").resolve()
    ds = CohortDataset(coca_dir)
    logger.info("Loaded Stanford COCA Cohort: %d participants discovered.", len(ds.participants))

    selected_cases_by_tier = {
        "Zero (0)": ["nongated_1", "nongated_52", "nongated_91", "nongated_127", "nongated_167"],
        "Minimal (1-10)": ["nongated_12", "nongated_22", "nongated_54", "nongated_122", "nongated_182"],
        "Mild (11-100)": ["nongated_13", "nongated_66", "nongated_102", "nongated_142", "nongated_175"],
        "Moderate (101-400)": ["nongated_2", "nongated_25", "nongated_57", "nongated_98", "nongated_128"],
        "Severe (>400)": ["0", "95", "190", "285", "380"],
    }

    all_case_ids = []
    case_to_tier = {}
    for tier, ids in selected_cases_by_tier.items():
        for cid in ids:
            all_case_ids.append(cid)
            case_to_tier[cid] = tier

    # Initialize NVIDIA foundation model segmenters
    logger.info("Loading Track 4: NVIDIA NV-Segment-CTMR (VISTA-3D CT+MR)...")
    seg_nv_ctmr = UnifiedAnatomySegmenter(backend="onnx", chamber_backend="nvsegment_ctmr", device=device)

    logger.info("Loading Track 5: NVIDIA NV-Segment-CT (VISTA-3D CT)...")
    seg_nv_ct = UnifiedAnatomySegmenter(backend="onnx", chamber_backend="nvsegment_ct", device=device)

    tx_quant = QuantifyAdiposeMetricsd(keys=["image"])

    all_results = []
    timing_records = {
        "nv_segment_ctmr": [],
        "nv_segment_ct": [],
    }

    output_csv = coca_dir / "coca_stratified_pilot_results.csv"
    output_json = coca_dir / "coca_stratified_pilot_summary.json"

    for idx, case_id in enumerate(all_case_ids, 1):
        p = ds.participants.get(case_id)
        if p is None:
            logger.warning("Case %s not found in dataset; skipping.", case_id)
            continue

        case_dir = p.study_dir
        tier = case_to_tier[case_id]
        cac_score = p.calcium_score

        logger.info("\n" + "=" * 70)
        logger.info("[%d/%d] Processing Case %s | CAC Tier: %s (Agatston: %s)", idx, len(all_case_ids), case_id, tier, cac_score)
        logger.info("Study Directory: %s", case_dir)
        logger.info("=" * 70)

        # 1. Ingest 3D CT scan
        if p.scan_path and (p.scan_path.is_dir() or p.scan_path.is_file()):
            meta_ct = read_medical_image_as_metatensor(p.scan_path)
        else:
            logger.warning("No valid scan path for %s; skipping.", case_id)
            continue

        ref_header = meta_ct.meta.get("nrrd_header")
        spacing = tuple(float(x) for x in meta_ct.meta.get("spacing", (1.0, 1.0, 1.0)))
        origin = tuple(float(x) for x in meta_ct.meta.get("origin", (0.0, 0.0, 0.0))) if "origin" in meta_ct.meta else None

        # 2. Ingest Pre-existing Track 1 (Harvard) & Track 2 (MONAI)
        harvard_seg_path = case_dir / "eat_pvat_harvard_cfq.seg.nrrd"
        harvard_csv_path = case_dir / "harvard_cfq_measurements.csv"
        monai_seg_path = case_dir / "eat_pvat.seg.nrrd"
        monai_csv_path = case_dir / "cardiac_fat_measurements.csv"

        if harvard_seg_path.is_file() and harvard_csv_path.is_file():
            h_mt = read_nrrd_as_metatensor(harvard_seg_path)
            h_arr = h_mt.detach().cpu().numpy()
            if h_arr.ndim == 5 and h_arr.shape[0] == 1:
                h_arr = h_arr[0]
            h_metrics = pd.read_csv(harvard_csv_path).iloc[0].to_dict()
        else:
            logger.warning("Harvard reference missing for %s; using zeros.", case_id)
            h_arr = np.zeros((8,) + meta_ct.shape[1:], dtype=np.uint8)
            h_metrics = {}

        if monai_seg_path.is_file() and monai_csv_path.is_file():
            m_mt = read_nrrd_as_metatensor(monai_seg_path)
            m_arr = m_mt.detach().cpu().numpy()
            if m_arr.ndim == 5 and m_arr.shape[0] == 1:
                m_arr = m_arr[0]
            m_metrics = pd.read_csv(monai_csv_path).iloc[0].to_dict()
        else:
            logger.warning("MONAI segmentation missing for %s; using zeros.", case_id)
            m_arr = np.zeros((8,) + meta_ct.shape[1:], dtype=np.uint8)
            m_metrics = {}

        # 3. Run Track 4: NVIDIA NV-Segment-CTMR
        logger.info("Executing Track 4: NV-Segment-CTMR...")
        t0 = time.time()
        tx_nv_ctmr = ComputeEatPvatMasksd(keys=["image"], segmenter=seg_nv_ctmr)
        out_nv_ctmr = tx_nv_ctmr({"image": meta_ct, "participant_id": case_id})
        nv_ctmr_arr = out_nv_ctmr["pred"]
        if isinstance(nv_ctmr_arr, torch.Tensor):
            nv_ctmr_arr = nv_ctmr_arr.detach().cpu().numpy()
        if nv_ctmr_arr.ndim == 5 and nv_ctmr_arr.shape[0] == 1:
            nv_ctmr_arr = nv_ctmr_arr[0]

        quant_nv_ctmr = tx_quant({"image": meta_ct, "pred": nv_ctmr_arr, "participant_id": case_id})["metrics"]
        t_nv_ctmr = time.time() - t0
        timing_records["nv_segment_ctmr"].append(t_nv_ctmr)

        # Save Track 4 Deliverables
        write_eat_pvat_seg_nrrd(
            seg_data=nv_ctmr_arr,
            output_path=case_dir / "eat_pvat_nvsegment.seg.nrrd",
            reference_header=ref_header,
            spacing=spacing,
            origin=origin,
        )
        pd.DataFrame([quant_nv_ctmr])[MEASUREMENT_CSV_COLUMNS].to_csv(
            case_dir / "nvsegment_measurements.csv", index=False
        )

        nv_ctmr_anat = out_nv_ctmr.get("anatomy_masks")
        if nv_ctmr_anat is not None:
            ch_mask = np.zeros_like(nv_ctmr_anat.pericardium, dtype=np.uint16)
            if nv_ctmr_anat.aorta is not None:
                ch_mask[nv_ctmr_anat.aorta] = 7
            if nv_ctmr_anat.myocardium is not None:
                ch_mask[nv_ctmr_anat.myocardium] = 44
            ch_mask[nv_ctmr_anat.la] = 45
            ch_mask[nv_ctmr_anat.lv] = 46
            ch_mask[nv_ctmr_anat.ra] = 47
            ch_mask[nv_ctmr_anat.rv] = 48
            write_monai_wholebody_ct_segmentation_seg_nrrd(
                seg_data=ch_mask,
                output_path=case_dir / "nv_segment_ctmr_masks.seg.nrrd",
                reference_header=ref_header,
                spacing=spacing,
                origin=origin,
            )

        # 4. Run Track 5: NVIDIA NV-Segment-CT
        logger.info("Executing Track 5: NV-Segment-CT...")
        t0 = time.time()
        tx_nv_ct = ComputeEatPvatMasksd(keys=["image"], segmenter=seg_nv_ct)
        out_nv_ct = tx_nv_ct({"image": meta_ct, "participant_id": case_id})
        nv_ct_arr = out_nv_ct["pred"]
        if isinstance(nv_ct_arr, torch.Tensor):
            nv_ct_arr = nv_ct_arr.detach().cpu().numpy()
        if nv_ct_arr.ndim == 5 and nv_ct_arr.shape[0] == 1:
            nv_ct_arr = nv_ct_arr[0]

        quant_nv_ct = tx_quant({"image": meta_ct, "pred": nv_ct_arr, "participant_id": case_id})["metrics"]
        t_nv_ct = time.time() - t0
        timing_records["nv_segment_ct"].append(t_nv_ct)

        # Save Track 5 Deliverables
        write_eat_pvat_seg_nrrd(
            seg_data=nv_ct_arr,
            output_path=case_dir / "eat_pvat_nvsegment_ct.seg.nrrd",
            reference_header=ref_header,
            spacing=spacing,
            origin=origin,
        )
        pd.DataFrame([quant_nv_ct])[MEASUREMENT_CSV_COLUMNS].to_csv(
            case_dir / "nvsegment_ct_measurements.csv", index=False
        )

        nv_ct_anat = out_nv_ct.get("anatomy_masks")
        if nv_ct_anat is not None:
            ch_mask_ct = np.zeros_like(nv_ct_anat.pericardium, dtype=np.uint16)
            if nv_ct_anat.aorta is not None:
                ch_mask_ct[nv_ct_anat.aorta] = 7
            if nv_ct_anat.myocardium is not None:
                ch_mask_ct[nv_ct_anat.myocardium] = 44
            ch_mask_ct[nv_ct_anat.la] = 45
            ch_mask_ct[nv_ct_anat.lv] = 46
            ch_mask_ct[nv_ct_anat.ra] = 47
            ch_mask_ct[nv_ct_anat.rv] = 48
            write_monai_wholebody_ct_segmentation_seg_nrrd(
                seg_data=ch_mask_ct,
                output_path=case_dir / "nv_segment_ct_masks.seg.nrrd",
                reference_header=ref_header,
                spacing=spacing,
                origin=origin,
            )

        # 5. Compute Cross-Model Concordance Metrics (Dice)
        eat_h = h_arr[:4].sum(axis=0) > 0
        eat_m = m_arr[:4].sum(axis=0) > 0
        eat_nv_ctmr = nv_ctmr_arr[:4].sum(axis=0) > 0
        eat_nv_ct = nv_ct_arr[:4].sum(axis=0) > 0

        pvat_h = h_arr[4:7].sum(axis=0) > 0
        pvat_m = m_arr[4:7].sum(axis=0) > 0
        pvat_nv_ctmr = nv_ctmr_arr[4:7].sum(axis=0) > 0
        pvat_nv_ct = nv_ct_arr[4:7].sum(axis=0) > 0

        dice_eat_h_m = compute_dice(eat_h, eat_m)
        dice_eat_h_nv_ctmr = compute_dice(eat_h, eat_nv_ctmr)
        dice_eat_h_nv_ct = compute_dice(eat_h, eat_nv_ct)
        dice_eat_m_nv_ctmr = compute_dice(eat_m, eat_nv_ctmr)
        dice_eat_nv_ctmr_vs_ct = compute_dice(eat_nv_ctmr, eat_nv_ct)

        dice_pvat_h_m = compute_dice(pvat_h, pvat_m)
        dice_pvat_h_nv_ctmr = compute_dice(pvat_h, pvat_nv_ctmr)
        dice_pvat_h_nv_ct = compute_dice(pvat_h, pvat_nv_ct)
        dice_pvat_m_nv_ctmr = compute_dice(pvat_m, nv_ctmr_arr[4:7].sum(axis=0) > 0)
        dice_pvat_nv_ctmr_vs_ct = compute_dice(pvat_nv_ctmr, pvat_nv_ct)

        logger.info(
            "Case %s [%s] Concordance: EAT Dice (H vs M): %.4f | (H vs NV-CTMR): %.4f | (CTMR vs CT): %.4f",
            case_id, tier, dice_eat_h_m, dice_eat_h_nv_ctmr, dice_eat_nv_ctmr_vs_ct,
        )
        logger.info("Latencies: NV-CTMR: %.2fs | NV-CT: %.2fs", t_nv_ctmr, t_nv_ct)

        row = {
            "participant_id": case_id,
            "cac_tier": tier,
            "calcium_score": cac_score,
            # EAT Volumes
            "eat_vol_harvard": h_metrics.get("EAT Total Volume (cm3)"),
            "eat_vol_monai": m_metrics.get("EAT Total Volume (cm3)"),
            "eat_vol_nvsegment_ctmr": quant_nv_ctmr.get("EAT Total Volume (cm3)"),
            "eat_vol_nvsegment_ct": quant_nv_ct.get("EAT Total Volume (cm3)"),
            # PVAT Volumes
            "pvat_vol_harvard": h_metrics.get("PVAT Total Volume (cm3)"),
            "pvat_vol_monai": m_metrics.get("PVAT Total Volume (cm3)"),
            "pvat_vol_nvsegment_ctmr": quant_nv_ctmr.get("PVAT Total Volume (cm3)"),
            "pvat_vol_nvsegment_ct": quant_nv_ct.get("PVAT Total Volume (cm3)"),
            # Attenuations
            "eat_hu_harvard": h_metrics.get("EAT Total Mean Attenuation (HU)"),
            "eat_hu_monai": m_metrics.get("EAT Total Mean Attenuation (HU)"),
            "eat_hu_nvsegment_ctmr": quant_nv_ctmr.get("EAT Total Mean Attenuation (HU)"),
            "eat_hu_nvsegment_ct": quant_nv_ct.get("EAT Total Mean Attenuation (HU)"),
            "pvat_hu_harvard": h_metrics.get("PVAT Total Mean Attenuation (HU)"),
            "pvat_hu_monai": m_metrics.get("PVAT Total Mean Attenuation (HU)"),
            "pvat_hu_nvsegment_ctmr": quant_nv_ctmr.get("PVAT Total Mean Attenuation (HU)"),
            "pvat_hu_nvsegment_ct": quant_nv_ct.get("PVAT Total Mean Attenuation (HU)"),
            # Dice Concordance
            "dice_eat_harvard_vs_monai": dice_eat_h_m,
            "dice_eat_harvard_vs_nvsegment_ctmr": dice_eat_h_nv_ctmr,
            "dice_eat_harvard_vs_nvsegment_ct": dice_eat_h_nv_ct,
            "dice_eat_monai_vs_nvsegment_ctmr": dice_eat_m_nv_ctmr,
            "dice_eat_nvsegment_ctmr_vs_ct": dice_eat_nv_ctmr_vs_ct,
            "dice_pvat_harvard_vs_monai": dice_pvat_h_m,
            "dice_pvat_harvard_vs_nvsegment_ctmr": dice_pvat_h_nv_ctmr,
            "dice_pvat_harvard_vs_nvsegment_ct": dice_pvat_h_nv_ct,
            "dice_pvat_monai_vs_nvsegment_ctmr": dice_pvat_m_nv_ctmr,
            "dice_pvat_nvsegment_ctmr_vs_ct": dice_pvat_nv_ctmr_vs_ct,
            # Latency
            "latency_nv_ctmr_sec": t_nv_ctmr,
            "latency_nv_ct_sec": t_nv_ct,
        }
        all_results.append(row)

        # Intermittent save after every participant
        pd.DataFrame(all_results).to_csv(output_csv, index=False)

    # Final summary statistics
    res_df = pd.DataFrame(all_results)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases_evaluated": len(res_df),
        "tiers_represented": CAC_TIER_ORDER,
        "mean_latency_nv_ctmr": float(res_df["latency_nv_ctmr_sec"].mean()),
        "mean_latency_nv_ct": float(res_df["latency_nv_ct_sec"].mean()),
        "mean_dice_eat_h_vs_m": float(res_df["dice_eat_harvard_vs_monai"].mean()),
        "mean_dice_eat_h_vs_nv_ctmr": float(res_df["dice_eat_harvard_vs_nvsegment_ctmr"].mean()),
        "mean_dice_eat_nv_ctmr_vs_ct": float(res_df["dice_eat_nvsegment_ctmr_vs_ct"].mean()),
        "mean_dice_pvat_h_vs_m": float(res_df["dice_pvat_harvard_vs_monai"].mean()),
        "mean_dice_pvat_h_vs_nv_ctmr": float(res_df["dice_pvat_harvard_vs_nvsegment_ctmr"].mean()),
    }
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Successfully finished 25-case stratified pilot benchmark!")
    logger.info("Deliverables saved to: %s and %s", output_csv, output_json)


if __name__ == "__main__":
    run_coca_stratified_pilot()

