"""Extensible 3D Segmentation Evaluation and DICE Comparison Engine.

Provides clinical and geometric validation metrics for comparing multi-compartment
cardiac adipose and anatomical segmentations against ground truth or alternative methods:
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU / Jaccard)
  - Relative Volume Error (RVE %) and Absolute Volume Difference (cm3)
  - Mean Absolute Surface Distance (MASD) and 95th percentile Hausdorff Distance (HD95)
  - Attenuation Delta (HU) between methods
  - Extensible method comparator interface for benchmarking novel models against baselines
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt

from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    SEGMENTATION_COMPARTMENTS,
    SegmentationCompartment,
    calculate_volume_cm3,
)


def compute_compartment_dice(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Compute the Dice Similarity Coefficient (DSC) between two binary masks.

    Formula: DSC = (2 * |A ∩ B|) / (|A| + |B| + smooth)
    If both masks are empty, returns 1.0 (perfect agreement on negative case).
    """
    pred_bool = (pred_mask > 0).astype(bool)
    target_bool = (target_mask > 0).astype(bool)

    pred_sum = np.count_nonzero(pred_bool)
    target_sum = np.count_nonzero(target_bool)

    if pred_sum == 0 and target_sum == 0:
        return 1.0

    intersection = np.count_nonzero(pred_bool & target_bool)
    return float((2.0 * intersection) / (pred_sum + target_sum + smooth))


def compute_compartment_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Compute Intersection over Union (IoU / Jaccard index) between two binary masks.

    Formula: IoU = |A ∩ B| / (|A ∪ B| + smooth)
    If both masks are empty, returns 1.0.
    """
    pred_bool = (pred_mask > 0).astype(bool)
    target_bool = (target_mask > 0).astype(bool)

    pred_sum = np.count_nonzero(pred_bool)
    target_sum = np.count_nonzero(target_bool)

    if pred_sum == 0 and target_sum == 0:
        return 1.0

    intersection = np.count_nonzero(pred_bool & target_bool)
    union = np.count_nonzero(pred_bool | target_bool)
    return float(intersection / (union + smooth))


def compute_volume_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute volumetric comparison metrics between candidate and reference masks.

    Returns:
        Dict containing:
          - pred_volume_cm3
          - target_volume_cm3
          - abs_volume_diff_cm3
          - relative_volume_error_pct
    """
    pred_count = int(np.count_nonzero(pred_mask > 0))
    target_count = int(np.count_nonzero(target_mask > 0))

    pred_vol = calculate_volume_cm3(pred_count, spacing)
    target_vol = calculate_volume_cm3(target_count, spacing)
    abs_diff = abs(pred_vol - target_vol)

    if target_vol > 0.0:
        rel_err = float(((pred_vol - target_vol) / target_vol) * 100.0)
    else:
        rel_err = 0.0 if pred_vol == 0.0 else float("inf")

    return {
        "pred_volume_cm3": round(pred_vol, 3),
        "target_volume_cm3": round(target_vol, 3),
        "abs_volume_diff_cm3": round(abs_diff, 3),
        "relative_volume_error_pct": round(rel_err, 2) if not math.isinf(rel_err) else 999.0,
    }


def compute_surface_distance_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Optional[float]]:
    """Compute Mean Absolute Surface Distance (MASD) and 95th Percentile Hausdorff Distance (HD95).

    Returns:
        Dict with "masd_mm" and "hd95_mm" (or None if one/both masks are empty).
    """
    pred_bool = (pred_mask > 0).astype(bool)
    target_bool = (target_mask > 0).astype(bool)

    if not np.any(pred_bool) or not np.any(target_bool):
        return {"masd_mm": None, "hd95_mm": None}

    # Extract surface voxels by morphological boundary check
    from scipy.ndimage import binary_erosion

    pred_border = pred_bool ^ binary_erosion(pred_bool)
    target_border = target_bool ^ binary_erosion(target_bool)

    if not np.any(pred_border) or not np.any(target_border):
        return {"masd_mm": None, "hd95_mm": None}

    dt_target = distance_transform_edt(~target_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    distances_pred_to_target = dt_target[pred_border]
    distances_target_to_pred = dt_pred[target_border]

    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    masd = float(np.mean(all_distances))
    hd95 = float(np.percentile(all_distances, 95))

    return {
        "masd_mm": round(masd, 3),
        "hd95_mm": round(hd95, 3),
    }


@dataclass
class CaseCompartmentMetric:
    """Metrics for a single compartment within a participant comparison."""

    compartment_name: str
    dice: float
    iou: float
    pred_volume_cm3: float
    target_volume_cm3: float
    abs_volume_diff_cm3: float
    relative_volume_error_pct: float
    masd_mm: Optional[float] = None
    hd95_mm: Optional[float] = None
    pred_mean_hu: Optional[float] = None
    target_mean_hu: Optional[float] = None
    hu_delta: Optional[float] = None


@dataclass
class ParticipantBenchmarkResult:
    """Benchmark evaluation summary for a single participant scan."""

    participant_id: str
    candidate_name: str
    reference_name: str
    mean_dice: float
    compartments: Dict[str, CaseCompartmentMetric] = field(default_factory=dict)
    pericardium_dice: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        d = asdict(self)
        d["compartments"] = {k: asdict(v) for k, v in self.compartments.items()}
        return d


@dataclass
class CohortBenchmarkReport:
    """Cohort-wide aggregated segmentation benchmark report."""

    candidate_name: str
    reference_name: str
    participant_count: int
    mean_overall_dice: float
    median_overall_dice: float
    compartment_summary: Dict[str, Dict[str, float]]
    cases: List[ParticipantBenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "candidate_name": self.candidate_name,
            "reference_name": self.reference_name,
            "participant_count": self.participant_count,
            "mean_overall_dice": round(self.mean_overall_dice, 4),
            "median_overall_dice": round(self.median_overall_dice, 4),
            "compartment_summary": self.compartment_summary,
            "cases": [c.to_dict() for c in self.cases],
        }


class BaseSegmentationMethod:
    """Abstract interface for extensible segmentation methods."""

    def __init__(self, name: str) -> None:
        self.name = name

    def segment(
        self,
        ct_volume: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Run segmentation and return 8-channel mask (uint8) of shape (8, Z, Y, X)."""
        raise NotImplementedError


class SegmentationBenchmarkEngine:
    """Extensible engine for benchmarking multi-compartment cardiac segmentation methods."""

    def __init__(
        self,
        candidate_name: str = "Candidate_Method",
        reference_name: str = "Ground_Truth",
        compute_surface_distances: bool = False,
    ) -> None:
        self.candidate_name = candidate_name
        self.reference_name = reference_name
        self.compute_surface_distances = compute_surface_distances
        self._registered_methods: Dict[str, BaseSegmentationMethod] = {}

    def register_method(self, method: BaseSegmentationMethod) -> None:
        """Register a new segmentation method for automated comparative benchmarking."""
        self._registered_methods[method.name] = method

    def evaluate_case_tensors(
        self,
        participant_id: str,
        pred_tensor: np.ndarray,
        target_tensor: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ct_volume: Optional[np.ndarray] = None,
        pred_pericardium: Optional[np.ndarray] = None,
        target_pericardium: Optional[np.ndarray] = None,
    ) -> ParticipantBenchmarkResult:
        """Evaluate an 8-channel prediction tensor against target tensor.

        Tensors must have shape (8, Z, Y, X) corresponding to:
          Ch 0: EAT_Total
          Ch 1: EAT_LA
          Ch 2: EAT_LV
          Ch 3: EAT_RA
          Ch 4: EAT_RV
          Ch 5: PVAT_LAD
          Ch 6: PVAT_LCX
          Ch 7: PVAT_RCA
        """
        # Ensure 4D tensors (8, Z, Y, X)
        while pred_tensor.ndim > 4 and pred_tensor.shape[0] == 1:
            pred_tensor = pred_tensor[0]
        while target_tensor.ndim > 4 and target_tensor.shape[0] == 1:
            target_tensor = target_tensor[0]

        # If channel dimension is last (Z, Y, X, 8), transpose to (8, Z, Y, X)
        if pred_tensor.ndim == 4 and pred_tensor.shape[-1] == 8 and pred_tensor.shape[0] != 8:
            pred_tensor = np.moveaxis(pred_tensor, -1, 0)
        if target_tensor.ndim == 4 and target_tensor.shape[-1] == 8 and target_tensor.shape[0] != 8:
            target_tensor = np.moveaxis(target_tensor, -1, 0)

        # If 3D integer label map, expand to (8, Z, Y, X)
        if pred_tensor.ndim == 3:
            expanded_p = np.zeros((8, *pred_tensor.shape), dtype=np.uint8)
            for c in range(min(8, int(np.max(pred_tensor)) + 1)):
                expanded_p[c] = (pred_tensor == c).astype(np.uint8)
            pred_tensor = expanded_p
        if target_tensor.ndim == 3:
            expanded_t = np.zeros((8, *target_tensor.shape), dtype=np.uint8)
            for c in range(min(8, int(np.max(target_tensor)) + 1)):
                expanded_t[c] = (target_tensor == c).astype(np.uint8)
            target_tensor = expanded_t

        if pred_tensor.ndim != 4 or pred_tensor.shape[0] != 8:
            raise ValueError(f"pred_tensor must have shape (8, Z, Y, X), got {pred_tensor.shape}")
        if target_tensor.ndim != 4 or target_tensor.shape[0] != 8:
            raise ValueError(f"target_tensor must have shape (8, Z, Y, X), got {target_tensor.shape}")

        comp_metrics: Dict[str, CaseCompartmentMetric] = {}
        dice_scores: List[float] = []

        # 1. Evaluate the 8 primary compartments
        for ch_idx, comp_enum in enumerate(SEGMENTATION_COMPARTMENTS):
            comp_name = comp_enum.label_name
            p_mask = pred_tensor[ch_idx]
            t_mask = target_tensor[ch_idx]

            d = compute_compartment_dice(p_mask, t_mask)
            iou = compute_compartment_iou(p_mask, t_mask)
            vol_m = compute_volume_metrics(p_mask, t_mask, spacing=spacing)

            surf_m = (
                compute_surface_distance_metrics(p_mask, t_mask, spacing=spacing)
                if self.compute_surface_distances
                else {"masd_mm": None, "hd95_mm": None}
            )

            p_hu: Optional[float] = None
            t_hu: Optional[float] = None
            hu_delta: Optional[float] = None
            if ct_volume is not None:
                p_vox = ct_volume[p_mask > 0]
                t_vox = ct_volume[t_mask > 0]
                if len(p_vox) > 0:
                    p_hu = round(float(np.mean(p_vox)), 2)
                if len(t_vox) > 0:
                    t_hu = round(float(np.mean(t_vox)), 2)
                if p_hu is not None and t_hu is not None:
                    hu_delta = round(abs(p_hu - t_hu), 2)

            metric_entry = CaseCompartmentMetric(
                compartment_name=comp_name,
                dice=round(d, 4),
                iou=round(iou, 4),
                pred_volume_cm3=vol_m["pred_volume_cm3"],
                target_volume_cm3=vol_m["target_volume_cm3"],
                abs_volume_diff_cm3=vol_m["abs_volume_diff_cm3"],
                relative_volume_error_pct=vol_m["relative_volume_error_pct"],
                masd_mm=surf_m["masd_mm"],
                hd95_mm=surf_m["hd95_mm"],
                pred_mean_hu=p_hu,
                target_mean_hu=t_hu,
                hu_delta=hu_delta,
            )
            comp_metrics[comp_name] = metric_entry
            dice_scores.append(d)

        # 2. Evaluate derived PVAT Total (Union of Ch 5, 6, 7)
        pred_pvat_tot = (pred_tensor[5] > 0) | (pred_tensor[6] > 0) | (pred_tensor[7] > 0)
        target_pvat_tot = (target_tensor[5] > 0) | (target_tensor[6] > 0) | (target_tensor[7] > 0)
        d_pvat = compute_compartment_dice(pred_pvat_tot, target_pvat_tot)
        iou_pvat = compute_compartment_iou(pred_pvat_tot, target_pvat_tot)
        vol_pvat = compute_volume_metrics(pred_pvat_tot, target_pvat_tot, spacing=spacing)

        comp_metrics["PVAT_Total"] = CaseCompartmentMetric(
            compartment_name="PVAT_Total",
            dice=round(d_pvat, 4),
            iou=round(iou_pvat, 4),
            pred_volume_cm3=vol_pvat["pred_volume_cm3"],
            target_volume_cm3=vol_pvat["target_volume_cm3"],
            abs_volume_diff_cm3=vol_pvat["abs_volume_diff_cm3"],
            relative_volume_error_pct=vol_pvat["relative_volume_error_pct"],
        )

        # Pericardium Dice if provided
        peri_dice: Optional[float] = None
        if pred_pericardium is not None and target_pericardium is not None:
            peri_dice = round(compute_compartment_dice(pred_pericardium, target_pericardium), 4)

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

        return ParticipantBenchmarkResult(
            participant_id=participant_id,
            candidate_name=self.candidate_name,
            reference_name=self.reference_name,
            mean_dice=round(mean_dice, 4),
            compartments=comp_metrics,
            pericardium_dice=peri_dice,
        )

    def evaluate_cohort(
        self,
        case_pairs: List[Tuple[str, np.ndarray, np.ndarray, Tuple[float, float, float]]],
    ) -> CohortBenchmarkReport:
        """Evaluate a collection of participant segmentations across the cohort.

        Args:
            case_pairs: List of tuples (participant_id, pred_tensor_8ch, target_tensor_8ch, spacing)
        """
        results: List[ParticipantBenchmarkResult] = []
        for pid, pred, target, sp in case_pairs:
            res = self.evaluate_case_tensors(
                participant_id=pid,
                pred_tensor=pred,
                target_tensor=target,
                spacing=sp,
            )
            results.append(res)

        if not results:
            return CohortBenchmarkReport(
                candidate_name=self.candidate_name,
                reference_name=self.reference_name,
                participant_count=0,
                mean_overall_dice=0.0,
                median_overall_dice=0.0,
                compartment_summary={},
                cases=[],
            )

        overall_dices = [r.mean_dice for r in results]
        all_comps = list(results[0].compartments.keys())
        summary: Dict[str, Dict[str, float]] = {}

        for c_name in all_comps:
            c_dices = [r.compartments[c_name].dice for r in results if c_name in r.compartments]
            c_ious = [r.compartments[c_name].iou for r in results if c_name in r.compartments]
            c_rves = [r.compartments[c_name].relative_volume_error_pct for r in results if c_name in r.compartments]
            c_diffs = [r.compartments[c_name].abs_volume_diff_cm3 for r in results if c_name in r.compartments]

            summary[c_name] = {
                "mean_dice": round(float(np.mean(c_dices)), 4),
                "std_dice": round(float(np.std(c_dices)), 4),
                "median_dice": round(float(np.median(c_dices)), 4),
                "iqr_dice": round(float(np.percentile(c_dices, 75) - np.percentile(c_dices, 25)), 4),
                "mean_iou": round(float(np.mean(c_ious)), 4),
                "mean_abs_volume_diff_cm3": round(float(np.mean(c_diffs)), 3),
                "mean_relative_volume_error_pct": round(float(np.mean(c_rves)), 2),
            }

        return CohortBenchmarkReport(
            candidate_name=self.candidate_name,
            reference_name=self.reference_name,
            participant_count=len(results),
            mean_overall_dice=round(float(np.mean(overall_dices)), 4),
            median_overall_dice=round(float(np.median(overall_dices)), 4),
            compartment_summary=summary,
            cases=results,
        )

    @staticmethod
    def export_summary_csv(report: CohortBenchmarkReport, output_csv: Path) -> Path:
        """Export per-case benchmark breakdown to CSV."""
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        for case in report.cases:
            row: Dict[str, Any] = {
                "participant_id": case.participant_id,
                "candidate": case.candidate_name,
                "reference": case.reference_name,
                "mean_dice": case.mean_dice,
                "pericardium_dice": case.pericardium_dice,
            }
            for comp_name, m in case.compartments.items():
                row[f"{comp_name}_dice"] = m.dice
                row[f"{comp_name}_iou"] = m.iou
                row[f"{comp_name}_vol_cand_cm3"] = m.pred_volume_cm3
                row[f"{comp_name}_vol_ref_cm3"] = m.target_volume_cm3
                row[f"{comp_name}_abs_diff_cm3"] = m.abs_volume_diff_cm3
                row[f"{comp_name}_rve_pct"] = m.relative_volume_error_pct
                if m.masd_mm is not None:
                    row[f"{comp_name}_masd_mm"] = m.masd_mm
                if m.hd95_mm is not None:
                    row[f"{comp_name}_hd95_mm"] = m.hd95_mm
            rows.append(row)

        df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        return output_csv

    @staticmethod
    def export_summary_json(report: CohortBenchmarkReport, output_json: Path) -> Path:
        """Export comprehensive benchmark report to JSON."""
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        return output_json
