"""Cohort Research Data Engine & Clinical Dataset Aggregator.

Designed for cardiovascular imaging research across clinical cohorts (including
large datasets like Stanford COCA, Harvard/MGH CardiacFatQuant cohorts, and institutional
DICOM/PACS studies).

Provides:
  - Smart discovery across diverse cohort directory layouts (flat, nested, gated/non-gated).
  - Multi-source clinical Agatston calcium score linking (scores_gated.csv, scores_nongated.csv, XML).
  - Risk tier stratification (Zero, Minimal 1-10, Mild 11-100, Moderate 101-400, Severe >400).
  - CRISP-CT vascular inflammation screening (PVAT Mean Attenuation > -70.1 HU).
  - Cohort correlation analytics (Pearson r, Spearman rho, p-values for EAT & PVAT vs CAC).
  - Multi-method segmentation discovery for comparative DICE evaluation.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from cardiac_fat_fl.io_metatensor import _find_dicom_series_dir, discover_participants
from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    MEASUREMENT_CSV_COLUMNS,
    REPORTED_REGIONS,
    ReportedRegion,
    SegmentationCompartment,
    calculate_volume_cm3,
)


def categorize_calcium_score(score: Optional[float]) -> str:
    """Categorize Agatston coronary artery calcium (CAC) score into standard clinical tiers."""
    if score is None or math.isnan(score):
        return "Unknown"
    if score <= 0.0:
        return "Zero (0)"
    if score <= 10.0:
        return "Minimal (1-10)"
    if score <= 100.0:
        return "Mild (11-100)"
    if score <= 400.0:
        return "Moderate (101-400)"
    return "Severe (>400)"


@dataclass
class CohortParticipant:
    """Represents a single research participant in a cardiac CT cohort."""

    participant_id: str
    study_dir: Path
    scan_path: Optional[Path] = None
    seg_path: Optional[Path] = None
    available_segmentations: Dict[str, Path] = field(default_factory=dict)
    measurements_csv: Optional[Path] = None
    report_pdf: Optional[Path] = None
    report_html: Optional[Path] = None
    report_dcm: Optional[Path] = None
    ohif_export_dir: Optional[Path] = None
    calcium_score: Optional[float] = None
    calcium_risk_tier: str = "Unknown"
    metrics: Dict[str, float] = field(default_factory=dict)
    crisp_ct_status: str = "Unknown"
    scan_type: str = "CT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert participant record to dictionary."""
        return {
            "participant_id": self.participant_id,
            "study_dir": str(self.study_dir),
            "scan_path": str(self.scan_path) if self.scan_path else None,
            "has_seg": self.seg_path is not None,
            "seg_path": str(self.seg_path) if self.seg_path else None,
            "available_segmentations": {k: str(v) for k, v in self.available_segmentations.items()},
            "has_measurements": bool(self.metrics),
            "has_report_pdf": self.report_pdf is not None,
            "has_report_html": self.report_html is not None,
            "has_ohif": self.ohif_export_dir is not None,
            "calcium_score": self.calcium_score,
            "calcium_risk_tier": self.calcium_risk_tier,
            "crisp_ct_status": self.crisp_ct_status,
            "scan_type": self.scan_type,
            "metrics": self.metrics,
        }


class CohortDataset:
    """Cohort research manager and statistical aggregator."""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Cohort directory does not exist: {self.root_dir}")

        self.calcium_scores: Dict[str, float] = {}
        self.participants: Dict[str, CohortParticipant] = {}
        self._load_cohort()

    def _discover_calcium_scores(self) -> None:
        """Discover and load CAC scores from scores CSVs or XML files in the cohort."""
        # 1. Search for scores_gated.csv, scores_nongated.csv, scores.csv
        score_candidates = list(self.root_dir.glob("**/scores*.csv")) + list(self.root_dir.glob("scores*.csv"))
        for sc in score_candidates:
            try:
                df = pd.read_csv(sc)
                cols_lower = {c.lower(): c for c in df.columns}
                id_col = cols_lower.get("patientid") or cols_lower.get("patient_id") or cols_lower.get("id") or df.columns[0]
                score_col = cols_lower.get("totalscore") or cols_lower.get("total_score") or cols_lower.get("score") or cols_lower.get("cac") or (df.columns[1] if len(df.columns) > 1 else None)

                if score_col:
                    for _, row in df.iterrows():
                        pid = str(row[id_col]).strip()
                        try:
                            score_val = float(row[score_col])
                            self.calcium_scores[pid] = score_val
                        except (ValueError, TypeError):
                            continue
            except Exception:
                pass

        # 2. Check for XML score files (e.g. Stanford COCA calcium_xml/<pid>.xml)
        xml_files = list(self.root_dir.glob("**/calcium_xml/*.xml"))
        for xf in xml_files:
            pid = xf.stem
            if pid not in self.calcium_scores:
                # Basic parse if needed
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(xf)
                    root = tree.getroot()
                    score_elem = root.find(".//TotalScore") or root.find(".//Score")
                    if score_elem is not None and score_elem.text:
                        self.calcium_scores[pid] = float(score_elem.text)
                except Exception:
                    pass

    def _load_cohort(self) -> None:
        """Scan cohort directory and assemble participant records."""
        self._discover_calcium_scores()

        # Check for root or nested measurement CSV
        root_csvs = list(self.root_dir.glob("cardiac_fat_measurements*.csv"))
        root_metrics_by_pid: Dict[str, Dict[str, float]] = {}
        for rcsv in root_csvs:
            try:
                df = pd.read_csv(rcsv)
                if "participant_id" in df.columns:
                    for _, row in df.iterrows():
                        pid = str(row["participant_id"]).strip()
                        m_dict: Dict[str, float] = {}
                        for c in df.columns:
                            if c != "participant_id":
                                try:
                                    val = float(row[c])
                                    m_dict[c] = val
                                except (ValueError, TypeError):
                                    m_dict[c] = float("nan")
                        root_metrics_by_pid[pid] = m_dict
            except Exception:
                pass

        # Find participant scan locations via discover_participants
        discovered_scans = []
        try:
            discovered_scans = discover_participants(self.root_dir)
        except Exception:
            pass

        # If empty, check subfolders like cocacoronarycalciumandchestcts-2/Gated_release_final/patient
        if not discovered_scans:
            sub_patient_dirs = list(self.root_dir.glob("**/patient"))
            for sp in sub_patient_dirs:
                if sp.is_dir():
                    try:
                        disc = discover_participants(sp)
                        discovered_scans.extend(disc)
                    except Exception:
                        pass

        scan_dict: Dict[str, Path] = {pid: path for pid, path in discovered_scans}

        # Candidate patient folders (either directory containing scans or subfolders)
        patient_dirs: Dict[str, Path] = {}
        for pid, scan_p in scan_dict.items():
            if scan_p.is_dir():
                # If scan_p is a nested DICOM series folder (e.g. patient_id/series_name/), use parent
                patient_dirs[pid] = scan_p.parent if scan_p.parent.name == pid else scan_p
            else:
                patient_dirs[pid] = scan_p.parent

        # Also discover participant folders containing eat_pvat segmentation masks
        for seg_candidate in self.root_dir.glob("**/eat_pvat.seg.*"):
            p_folder = seg_candidate.parent
            pid = p_folder.name
            if pid not in patient_dirs:
                patient_dirs[pid] = p_folder

        # Also check participant IDs from measurement CSVs
        for pid in root_metrics_by_pid:
            if pid not in patient_dirs:
                matching = [d for d in self.root_dir.glob(f"**/{pid}") if d.is_dir()]
                if matching:
                    patient_dirs[pid] = matching[0]

        # Also inspect direct children of root_dir if they look like participants
        for entry in self.root_dir.iterdir():
            if entry.is_dir() and entry.name not in ["graphs", "ohif_export", ".venv", "models", "scratch"]:
                if entry.name not in patient_dirs:
                    if not any(entry in p_folder.parents for p_folder in patient_dirs.values()):
                        patient_dirs[entry.name] = entry

        # Build records
        for pid, p_dir in patient_dirs.items():
            scan_path = scan_dict.get(pid)
            if scan_path is None:
                dcm_dir = _find_dicom_series_dir(p_dir)
                scan_path = dcm_dir if dcm_dir else None

            # Primary segmentation mask
            seg_candidates = [
                p_dir / "eat_pvat.seg.nrrd",
                p_dir / "eat_pvat.seg.nii.gz",
                p_dir / "eat_pvat.seg.nii",
            ]
            if p_dir.parent != self.root_dir:
                seg_candidates.extend([
                    p_dir.parent / "eat_pvat.seg.nrrd",
                    p_dir.parent / "eat_pvat.seg.nii.gz",
                ])
            seg_path = next((f for f in seg_candidates if f.is_file()), None)

            # Discover all other segmentation masks for comparison
            available_segs: Dict[str, Path] = {}
            if seg_path:
                available_segs["eat_pvat"] = seg_path

            for seg_file in p_dir.glob("*.seg.*"):
                seg_label = seg_file.name.split(".seg.")[0]
                available_segs[seg_label] = seg_file

            # Check for ground truth or comparator masks
            gt_candidates = [
                p_dir / "ground_truth.seg.nrrd",
                p_dir / "gt.seg.nrrd",
                p_dir / "reference.seg.nrrd",
                p_dir / "deepheartseg.seg.nrrd",
                p_dir / "heart.seg.nrrd",
            ]
            for gtc in gt_candidates:
                if gtc.is_file():
                    available_segs[gtc.stem.split(".")[0]] = gtc

            # Reports
            report_pdf = p_dir / "cardiac_fat_report.pdf"
            report_html = p_dir / "cardiac_fat_report.html"
            report_dcm = p_dir / "eat_pvat_report.dcm"
            ohif_dir = p_dir / "ohif_export"

            # Measurements
            local_csv = p_dir / "cardiac_fat_measurements.csv"
            metrics = dict(root_metrics_by_pid.get(pid, {}))
            if local_csv.is_file():
                try:
                    df_loc = pd.read_csv(local_csv)
                    if len(df_loc) > 0:
                        row = df_loc.iloc[0]
                        for c in df_loc.columns:
                            if c != "participant_id":
                                try:
                                    metrics[c] = float(row[c])
                                except (ValueError, TypeError):
                                    metrics[c] = float("nan")
                except Exception:
                    pass

            # Calcium score
            cac = self.calcium_scores.get(pid)
            tier = categorize_calcium_score(cac)

            # CRISP-CT vascular inflammation classification
            crisp_status = "Unknown"
            pvat_hu = metrics.get("PVAT Total Mean Attenuation (HU)")
            if pvat_hu is not None and not math.isnan(pvat_hu):
                crisp_status = "Active Vascular Inflammation" if pvat_hu > -70.1 else "Normal Attenuation"

            scan_type = "Gated CS" if "gated" in str(p_dir).lower() else "CT"

            self.participants[pid] = CohortParticipant(
                participant_id=pid,
                study_dir=p_dir,
                scan_path=scan_path,
                seg_path=seg_path,
                available_segmentations=available_segs,
                measurements_csv=local_csv if local_csv.is_file() else None,
                report_pdf=report_pdf if report_pdf.is_file() else None,
                report_html=report_html if report_html.is_file() else None,
                report_dcm=report_dcm if report_dcm.is_file() else None,
                ohif_export_dir=ohif_dir if ohif_dir.is_dir() else None,
                calcium_score=cac,
                calcium_risk_tier=tier,
                metrics=metrics,
                crisp_ct_status=crisp_status,
                scan_type=scan_type,
            )

    @property
    def total_count(self) -> int:
        return len(self.participants)

    @property
    def segmented_count(self) -> int:
        return sum(1 for p in self.participants.values() if p.seg_path is not None)

    @property
    def cac_scored_count(self) -> int:
        return sum(1 for p in self.participants.values() if p.calcium_score is not None)

    def compute_summary_analytics(self) -> Dict[str, Any]:
        """Compute comprehensive cohort research analytics."""
        p_list = list(self.participants.values())
        total_n = len(p_list)
        if total_n == 0:
            return {
                "total_count": 0,
                "segmented_count": 0,
                "cac_scored_count": 0,
                "eat_volume_stats": {},
                "pvat_attenuation_stats": {},
                "correlations": {},
                "risk_tier_counts": {},
            }

        # 1. Metric arrays
        eat_vols = [
            p.metrics.get("EAT Total Volume (cm3)")
            for p in p_list
            if p.metrics.get("EAT Total Volume (cm3)") is not None
            and not math.isnan(p.metrics.get("EAT Total Volume (cm3)", float("nan")))
        ]
        pvat_hus = [
            p.metrics.get("PVAT Total Mean Attenuation (HU)")
            for p in p_list
            if p.metrics.get("PVAT Total Mean Attenuation (HU)") is not None
            and not math.isnan(p.metrics.get("PVAT Total Mean Attenuation (HU)", float("nan")))
        ]
        cacs = [p.calcium_score for p in p_list if p.calcium_score is not None and not math.isnan(p.calcium_score)]

        # 2. Risk tier breakdown
        tier_counts: Dict[str, int] = {}
        for p in p_list:
            tier_counts[p.calcium_risk_tier] = tier_counts.get(p.calcium_risk_tier, 0) + 1

        # 3. Basic descriptive stats helper
        def calc_stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {}
            a = np.array(arr)
            return {
                "count": len(a),
                "mean": round(float(np.mean(a)), 2),
                "std": round(float(np.std(a)), 2),
                "median": round(float(np.median(a)), 2),
                "iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 2),
                "min": round(float(np.min(a)), 2),
                "max": round(float(np.max(a)), 2),
            }

        # 4. Correlations: EAT Volume vs Calcium Score, and PVAT HU vs Calcium Score
        corrs: Dict[str, Any] = {}
        paired_eat_cac = [
            (p.metrics["EAT Total Volume (cm3)"], p.calcium_score)
            for p in p_list
            if p.calcium_score is not None
            and "EAT Total Volume (cm3)" in p.metrics
            and not math.isnan(p.metrics["EAT Total Volume (cm3)"])
            and not math.isnan(p.calcium_score)
        ]
        if len(paired_eat_cac) >= 3:
            vols, scores = zip(*paired_eat_cac)
            r_val, p_val = stats.pearsonr(vols, scores)
            rho_val, sp_val = stats.spearmanr(vols, scores)
            corrs["eat_vs_cac"] = {
                "n_paired": len(paired_eat_cac),
                "pearson_r": round(float(r_val), 3),
                "pearson_p": float(p_val),
                "spearman_rho": round(float(rho_val), 3),
                "spearman_p": float(sp_val),
            }

        paired_pvat_cac = [
            (p.metrics["PVAT Total Mean Attenuation (HU)"], p.calcium_score)
            for p in p_list
            if p.calcium_score is not None
            and "PVAT Total Mean Attenuation (HU)" in p.metrics
            and not math.isnan(p.metrics["PVAT Total Mean Attenuation (HU)"])
            and not math.isnan(p.calcium_score)
        ]
        if len(paired_pvat_cac) >= 3:
            hus, scores = zip(*paired_pvat_cac)
            r_val, p_val = stats.pearsonr(hus, scores)
            rho_val, sp_val = stats.spearmanr(hus, scores)
            corrs["pvat_hu_vs_cac"] = {
                "n_paired": len(paired_pvat_cac),
                "pearson_r": round(float(r_val), 3),
                "pearson_p": float(p_val),
                "spearman_rho": round(float(rho_val), 3),
                "spearman_p": float(sp_val),
            }

        # 5. CRISP-CT Active Inflammation Count
        inflam_count = sum(1 for p in p_list if p.crisp_ct_status == "Active Vascular Inflammation")
        inflam_pct = round((inflam_count / len(pvat_hus)) * 100.0, 1) if pvat_hus else 0.0

        # 6. Chamber volume summaries
        chamber_summaries: Dict[str, Dict[str, float]] = {}
        for chamber in ["EAT LA", "EAT LV", "EAT RA", "EAT RV", "PVAT LAD", "PVAT LCX", "PVAT RCA"]:
            vals = [
                p.metrics.get(f"{chamber} Volume (cm3)")
                for p in p_list
                if p.metrics.get(f"{chamber} Volume (cm3)") is not None
                and not math.isnan(p.metrics.get(f"{chamber} Volume (cm3)", float("nan")))
            ]
            chamber_summaries[chamber] = calc_stats(vals)

        return {
            "total_count": total_n,
            "segmented_count": self.segmented_count,
            "segmented_pct": round((self.segmented_count / total_n) * 100.0, 1) if total_n else 0.0,
            "cac_scored_count": self.cac_scored_count,
            "cac_scored_pct": round((self.cac_scored_count / total_n) * 100.0, 1) if total_n else 0.0,
            "active_inflammation_count": inflam_count,
            "active_inflammation_pct": inflam_pct,
            "eat_volume_stats": calc_stats(eat_vols),
            "pvat_attenuation_stats": calc_stats(pvat_hus),
            "calcium_score_stats": calc_stats(cacs),
            "chamber_volume_summaries": chamber_summaries,
            "risk_tier_counts": tier_counts,
            "correlations": corrs,
        }

    def get_scatter_data(self) -> List[Dict[str, Any]]:
        """Extract paired datapoints for interactive research scatter plots."""
        pts = []
        for p in self.participants.values():
            pts.append({
                "participant_id": p.participant_id,
                "eat_volume_cm3": p.metrics.get("EAT Total Volume (cm3)"),
                "pvat_volume_cm3": p.metrics.get("PVAT Total Volume (cm3)"),
                "pvat_mean_hu": p.metrics.get("PVAT Total Mean Attenuation (HU)"),
                "calcium_score": p.calcium_score,
                "calcium_risk_tier": p.calcium_risk_tier,
                "crisp_ct_status": p.crisp_ct_status,
                "has_seg": p.seg_path is not None,
            })
        return pts
