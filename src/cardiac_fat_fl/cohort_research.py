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
import fnmatch
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from cardiac_fat_fl.discovery import _find_dicom_series_dir, discover_participants
from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    MEASUREMENT_CSV_COLUMNS,
    REPORTED_REGIONS,
    ReportedRegion,
    SegmentationCompartment,
    calculate_volume_cm3,
)

if TYPE_CHECKING:
    import numpy as np


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


CAC_TIER_ORDER: List[str] = [
    "Zero (0)",
    "Minimal (1-10)",
    "Mild (11-100)",
    "Moderate (101-400)",
    "Severe (>400)",
]


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
    has_nv_reason: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert participant record to dictionary with JSON-safe values."""
        clean_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_metrics[k] = None
            else:
                clean_metrics[k] = v

        cac_val = self.calcium_score
        if isinstance(cac_val, float) and (math.isnan(cac_val) or math.isinf(cac_val)):
            cac_val = None

        has_pdf = (self.report_pdf is not None and self.report_pdf.is_file()) or (self.study_dir / "cardiac_fat_report.pdf").is_file() or (self.study_dir / "report" / "cardiac_fat_report.pdf").is_file()
        has_html = (self.report_html is not None and self.report_html.is_file()) or (self.study_dir / "cardiac_fat_report.html").is_file() or (self.study_dir / "report" / "cardiac_fat_report.html").is_file()
        has_dcm = (self.report_dcm is not None and self.report_dcm.is_file()) or (self.study_dir / "eat_pvat_report.dcm").is_file() or (self.study_dir / "report" / "eat_pvat_report.dcm").is_file()
        has_vlm = self.has_nv_reason or (self.study_dir / "nv_reason_narrative.txt").is_file() or (self.study_dir / "nv_reason_clinical_narrative.txt").is_file() or (self.study_dir / "report" / "nv_reason_narrative.txt").is_file()

        return {
            "participant_id": self.participant_id,
            "study_dir": str(self.study_dir),
            "scan_path": str(self.scan_path) if self.scan_path else None,
            "has_seg": self.seg_path is not None,
            "seg_path": str(self.seg_path) if self.seg_path else None,
            "available_segmentations": {k: str(v) for k, v in self.available_segmentations.items()},
            "has_measurements": bool(self.metrics),
            "has_report_pdf": has_pdf,
            "has_report_html": has_html,
            "has_report_dcm": has_dcm,
            "has_nv_reason": has_vlm,
            "has_ohif": self.ohif_export_dir is not None,
            "calcium_score": cac_val,
            "calcium_risk_tier": self.calcium_risk_tier,
            "crisp_ct_status": self.crisp_ct_status,
            "scan_type": self.scan_type,
            "metrics": clean_metrics,
        }


def _read_csv_dicts(csv_path: Path) -> List[Dict[str, str]]:
    """Read a CSV file via stdlib ``csv.DictReader`` with stripped column headers."""
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        clean_fields = [c.strip() if c is not None else "" for c in reader.fieldnames]
        for raw_row in reader:
            row: Dict[str, str] = {}
            for raw_k, clean_k in zip(reader.fieldnames, clean_fields):
                if clean_k and clean_k not in row:
                    val = raw_row.get(raw_k)
                    row[clean_k] = val if val is not None else ""
            rows.append(row)
    return rows


def _safe_float(raw: Any) -> float:
    """Parse float value from CSV cell, returning NaN for blank or non-numeric entries."""
    if raw is None:
        return float("nan")
    s = str(raw).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


class CohortDataset:
    """Cohort research manager and statistical aggregator."""

    _CACHE_VERSION = 2
    _CACHE_THRESHOLD = 20

    def __init__(self, root_dir: Union[str, Path], *, use_cache: bool = True) -> None:
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Cohort directory does not exist: {self.root_dir}")

        self.calcium_scores: Dict[str, float] = {}
        self.participants: Dict[str, CohortParticipant] = {}
        if use_cache and self._try_load_cache():
            return
        self._load_cohort()
        if use_cache and len(self.participants) >= self._CACHE_THRESHOLD:
            self._save_cache()

    @staticmethod
    def _cache_dir() -> Path:
        return Path(tempfile.gettempdir()) / "cardiac_fat_fl_cohort_cache"

    def _cache_file_path(self) -> Path:
        key_hash = hashlib.sha256(str(self.root_dir).encode("utf-8")).hexdigest()[:20]
        return self._cache_dir() / f"cohort_{key_hash}.json"

    def _compute_fingerprint(self) -> str:
        """Compute a fast directory + CSV mtime fingerprint for safe cache invalidation."""
        parts: List[str] = [f"v{self._CACHE_VERSION}", str(self.root_dir)]
        candidate_roots = [
            self.root_dir,
            self.root_dir / "cocacoronarycalciumandchestcts-2",
        ]
        probe_dirs: List[Path] = []
        for base in candidate_roots:
            if not base.is_dir():
                continue
            probe_dirs.extend([
                base,
                base / "Gated_release_final",
                base / "Gated_release_final" / "calcium_xml",
                base / "Gated_release_final" / "patient",
                base / "deidentified_nongated",
            ])

        seen_dirs: set[Path] = set()
        for d in probe_dirs:
            if d in seen_dirs or not d.is_dir():
                continue
            seen_dirs.add(d)
            try:
                st = d.stat()
                child_names: List[str] = []
                with os.scandir(d) as it:
                    for entry in it:
                        child_names.append(entry.name)
                        lname = entry.name.lower()
                        if d in candidate_roots and lname.endswith(".csv") and entry.is_file():
                            est = entry.stat()
                            parts.append(f"csv:{d.name}/{entry.name}:{est.st_mtime_ns}:{est.st_size}")
                parts.append(f"dir:{d}:{st.st_mtime_ns}:{len(child_names)}")
            except OSError:
                continue

        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def _try_load_cache(self) -> bool:
        """Load cached participant index from user temp directory if valid."""
        cache_file = self._cache_file_path()
        if not cache_file.is_file():
            return False
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if payload.get("version") != self._CACHE_VERSION:
                return False
            if payload.get("root_dir") != str(self.root_dir):
                return False
            records = payload.get("participants")
            if not isinstance(records, list) or len(records) < self._CACHE_THRESHOLD:
                return False
            if payload.get("fingerprint") != self._compute_fingerprint():
                return False

            self.calcium_scores = {
                str(k): float(v)
                for k, v in (payload.get("calcium_scores") or {}).items()
                if v is not None
            }
            participants: Dict[str, CohortParticipant] = {}
            for rec in records:
                pid = str(rec["participant_id"])
                participants[pid] = CohortParticipant(
                    participant_id=pid,
                    study_dir=Path(rec["study_dir"]),
                    scan_path=Path(rec["scan_path"]) if rec.get("scan_path") else None,
                    seg_path=Path(rec["seg_path"]) if rec.get("seg_path") else None,
                    available_segmentations={
                        k: Path(v) for k, v in (rec.get("available_segmentations") or {}).items()
                    },
                    measurements_csv=Path(rec["measurements_csv"]) if rec.get("measurements_csv") else None,
                    report_pdf=Path(rec["report_pdf"]) if rec.get("report_pdf") else None,
                    report_html=Path(rec["report_html"]) if rec.get("report_html") else None,
                    report_dcm=Path(rec["report_dcm"]) if rec.get("report_dcm") else None,
                    ohif_export_dir=Path(rec["ohif_export_dir"]) if rec.get("ohif_export_dir") else None,
                    calcium_score=float(rec["calcium_score"]) if rec.get("calcium_score") is not None else None,
                    calcium_risk_tier=str(rec.get("calcium_risk_tier", "Unknown")),
                    metrics={
                        k: (float("nan") if v is None else float(v))
                        for k, v in (rec.get("metrics") or {}).items()
                    },
                    crisp_ct_status=str(rec.get("crisp_ct_status", "Unknown")),
                    scan_type=str(rec.get("scan_type", "CT")),
                    has_nv_reason=bool(rec.get("has_nv_reason", False)),
                )
            self.participants = participants
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        """Persist participant index to user temp directory (never inside cohort directory)."""
        try:
            cache_dir = self._cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_file_path()
            records = []
            for p in self.participants.values():
                clean_metrics = {
                    k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v)
                    for k, v in p.metrics.items()
                }
                cac_val = p.calcium_score
                if isinstance(cac_val, float) and (math.isnan(cac_val) or math.isinf(cac_val)):
                    cac_val = None
                records.append({
                    "participant_id": p.participant_id,
                    "study_dir": str(p.study_dir),
                    "scan_path": str(p.scan_path) if p.scan_path else None,
                    "seg_path": str(p.seg_path) if p.seg_path else None,
                    "available_segmentations": {k: str(v) for k, v in p.available_segmentations.items()},
                    "measurements_csv": str(p.measurements_csv) if p.measurements_csv else None,
                    "report_pdf": str(p.report_pdf) if p.report_pdf else None,
                    "report_html": str(p.report_html) if p.report_html else None,
                    "report_dcm": str(p.report_dcm) if p.report_dcm else None,
                    "ohif_export_dir": str(p.ohif_export_dir) if p.ohif_export_dir else None,
                    "calcium_score": cac_val,
                    "calcium_risk_tier": p.calcium_risk_tier,
                    "metrics": clean_metrics,
                    "crisp_ct_status": p.crisp_ct_status,
                    "scan_type": p.scan_type,
                    "has_nv_reason": p.has_nv_reason,
                })
            payload = {
                "version": self._CACHE_VERSION,
                "root_dir": str(self.root_dir),
                "fingerprint": self._compute_fingerprint(),
                "calcium_scores": {
                    k: v
                    for k, v in self.calcium_scores.items()
                    if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                },
                "participants": records,
            }
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _bounded_glob(self, pattern: str, max_depth: int = 4) -> List[Path]:
        """Fast bounded ``os.scandir`` search up to *max_depth* skipping leaf DICOM stacks."""
        results: List[Path] = []
        seen: set[Path] = set()
        pat_lower = pattern.lower()
        has_slash = "/" in pat_lower
        skip_dirs = {".git", ".venv", "__pycache__", "ohif_export"}
        want_dcm = ".dcm" in pat_lower

        def _walk(cur_dir: Path, rel_prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                with os.scandir(cur_dir) as it:
                    entries = list(it)
            except OSError:
                return

            has_dcm_file = False
            subdirs: List[Tuple[str, Path]] = []
            for entry in entries:
                ename = entry.name
                ename_l = ename.lower()
                rel_p = f"{rel_prefix}/{ename_l}" if rel_prefix else ename_l
                full_p = Path(entry.path)
                is_dir = False
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue

                if is_dir:
                    if ename_l in skip_dirs or ename_l.startswith("."):
                        continue
                    if not has_slash and fnmatch.fnmatch(ename_l, pat_lower):
                        if full_p not in seen:
                            seen.add(full_p)
                            results.append(full_p)
                    subdirs.append((rel_p, full_p))
                else:
                    if ename_l.endswith(".dcm"):
                        has_dcm_file = True
                    matched = (
                        fnmatch.fnmatch(rel_p, pat_lower) or fnmatch.fnmatch(rel_p, f"*/{pat_lower}")
                        if has_slash
                        else fnmatch.fnmatch(ename_l, pat_lower)
                    )
                    if matched and full_p not in seen:
                        seen.add(full_p)
                        results.append(full_p)

            if has_dcm_file and not want_dcm:
                return
            if depth < max_depth:
                for sub_rel, sub_path in subdirs:
                    _walk(sub_path, sub_rel, depth + 1)

        _walk(self.root_dir, "", 0)
        return results

    def _discover_calcium_scores(self) -> None:
        """Discover and load CAC scores from scores CSVs or XML files in the cohort."""
        # 1. Search for scores_gated.csv, scores_nongated.csv, scores.csv
        score_candidates = self._bounded_glob("scores*.csv", max_depth=3)
        for sc in score_candidates:
            try:
                rows = _read_csv_dicts(sc)
                if not rows:
                    continue
                cols = list(rows[0].keys())
                cols_lower = {c.strip().lower(): c for c in cols}
                id_col = (
                    cols_lower.get("patientid")
                    or cols_lower.get("patient_id")
                    or cols_lower.get("filename")
                    or cols_lower.get("id")
                    or cols[0]
                )
                score_col = (
                    cols_lower.get("totalscore")
                    or cols_lower.get("total_score")
                    or cols_lower.get("total")
                    or cols_lower.get("score")
                    or cols_lower.get("cac")
                    or (cols[1] if len(cols) > 1 else None)
                )

                is_nongated = "nongated" in sc.name.lower()
                if score_col:
                    for row in rows:
                        pid = str(row.get(id_col, "")).strip()
                        score_val = _safe_float(row.get(score_col))
                        if math.isnan(score_val):
                            continue
                        if is_nongated or pid.endswith(("A", "a")):
                            clean_id = pid.rstrip("Aa").strip()
                            if clean_id:
                                self.calcium_scores[f"nongated_{clean_id}"] = score_val
                                self.calcium_scores[f"{clean_id}A"] = score_val
                        else:
                            self.calcium_scores[pid] = score_val
            except Exception:
                pass

        # 2. Check for XML score files (e.g. Stanford COCA calcium_xml/<pid>.xml)
        xml_files = self._bounded_glob("calcium_xml/*.xml", max_depth=3)
        if xml_files:
            import xml.etree.ElementTree as ET  # nosec B405

            for xf in xml_files:
                pid = xf.stem
                if pid not in self.calcium_scores:
                    try:
                        tree = ET.parse(xf)  # nosec B314
                        root = tree.getroot()
                        score_elem = root.find(".//TotalScore") or root.find(".//Score")
                        if score_elem is not None and score_elem.text:
                            self.calcium_scores[pid] = float(score_elem.text)
                    except Exception:
                        pass

    def _load_cohort(self) -> None:
        """Scan cohort directory and assemble participant records."""
        self._discover_calcium_scores()

        # Check for root measurement CSVs
        root_metrics_by_pid: Dict[str, Dict[str, float]] = {}
        try:
            with os.scandir(self.root_dir) as it:
                root_csvs = sorted(
                    Path(e.path)
                    for e in it
                    if e.is_file() and fnmatch.fnmatch(e.name.lower(), "cardiac_fat_measurements*.csv")
                )
        except OSError:
            root_csvs = []

        for rcsv in root_csvs:
            try:
                for row in _read_csv_dicts(rcsv):
                    if "participant_id" in row:
                        pid = str(row["participant_id"]).strip()
                        m_dict: Dict[str, float] = {}
                        for c, raw_v in row.items():
                            if c != "participant_id":
                                m_dict[c] = _safe_float(raw_v)
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
            sub_patient_dirs = self._bounded_glob("patient", max_depth=3)
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
        seen_dirs: set[Path] = set()
        for pid, scan_p in scan_dict.items():
            if scan_p.is_dir():
                p_dir = scan_p.parent if scan_p.parent.name == pid else scan_p
            else:
                p_dir = scan_p.parent
            patient_dirs[pid] = p_dir
            seen_dirs.add(p_dir)

        def _overlaps_existing(candidate: Path) -> bool:
            if candidate in seen_dirs:
                return True
            if any(par in seen_dirs for par in candidate.parents):
                return True
            return False

        # Also discover participant folders containing eat_pvat segmentation masks
        for seg_candidate in self._bounded_glob("eat_pvat.seg.*", max_depth=4):
            p_folder = seg_candidate.parent
            pid = p_folder.name
            if (
                pid not in patient_dirs
                and f"nongated_{pid}" not in patient_dirs
                and not _overlaps_existing(p_folder)
                and not any(p_folder in ep.parents for ep in seen_dirs)
            ):
                patient_dirs[pid] = p_folder
                seen_dirs.add(p_folder)

        # Also check participant IDs from measurement CSVs without O(N^2) tree walks
        missing_csv_pids = [
            pid
            for pid in root_metrics_by_pid
            if (
                pid not in patient_dirs
                and f"nongated_{pid}" not in patient_dirs
                and not any(pid == ep.replace("nongated_", "") for ep in patient_dirs)
            )
        ]
        if missing_csv_pids:
            missing_set = set(missing_csv_pids)
            dir_index: Dict[str, Path] = {}
            for d in self._bounded_glob("*", max_depth=3):
                if d.name in missing_set and d.is_dir() and d.name not in dir_index:
                    dir_index[d.name] = d
            for pid in missing_csv_pids:
                matched_d = dir_index.get(pid)
                if matched_d is not None and matched_d not in seen_dirs:
                    patient_dirs[pid] = matched_d
                    seen_dirs.add(matched_d)

        # Also inspect direct children of root_dir if they look like participants
        ignored_root_dirs = {
            "graphs",
            "ohif_export",
            ".venv",
            "models",
            "scratch",
            "fhir_storage",
            "report",
            "reports",
        }
        ancestor_dirs: set[Path] = set()
        for p_folder in seen_dirs:
            ancestor_dirs.update(p_folder.parents)

        for entry in self.root_dir.iterdir():
            if (
                entry.is_dir()
                and not entry.name.startswith(".")
                and entry.name not in ignored_root_dirs
                and not entry.name.startswith("parity_verification")
            ):
                if entry.name not in patient_dirs and entry not in seen_dirs and entry not in ancestor_dirs:
                    if (
                        _find_dicom_series_dir(entry) is not None
                        or any(entry.glob("*.nrrd"))
                        or any(entry.glob("*.nii*"))
                    ):
                        patient_dirs[entry.name] = entry
                        seen_dirs.add(entry)

        vlm_filenames = (
            "nv_reason_narrative.txt",
            "nv_reason_clinical_narrative.txt",
            "nv_reason_ct_cardiac_specialized_report.txt",
            "nv_reason_ct_generated_report.txt",
        )
        gt_filenames = (
            "ground_truth.seg.nrrd",
            "gt.seg.nrrd",
            "reference.seg.nrrd",
            "deepheartseg.seg.nrrd",
            "heart.seg.nrrd",
        )

        # Build records using single os.scandir pass per participant directory
        for pid, p_dir in patient_dirs.items():
            scan_path = scan_dict.get(pid)
            if scan_path is None:
                dcm_dir = _find_dicom_series_dir(p_dir)
                scan_path = dcm_dir if dcm_dir else None

            dir_files: Dict[str, Path] = {}
            dir_subdirs: set[str] = set()
            try:
                with os.scandir(p_dir) as it:
                    for e in it:
                        ep = Path(e.path)
                        if e.is_dir():
                            dir_subdirs.add(e.name)
                        elif e.is_file():
                            dir_files[e.name] = ep
            except OSError:
                pass

            # Primary segmentation mask
            seg_path: Optional[Path] = None
            for cand_name in ("eat_pvat.seg.nrrd", "eat_pvat.seg.nii.gz", "eat_pvat.seg.nii"):
                if cand_name in dir_files:
                    seg_path = dir_files[cand_name]
                    break
            if seg_path is None and p_dir.parent != self.root_dir:
                for cand_name in ("eat_pvat.seg.nrrd", "eat_pvat.seg.nii.gz"):
                    par_cand = p_dir.parent / cand_name
                    if par_cand.is_file():
                        seg_path = par_cand
                        break

            # Discover all other segmentation masks for comparison
            available_segs: Dict[str, Path] = {}
            if seg_path:
                available_segs["eat_pvat"] = seg_path
            for fname, fpath in dir_files.items():
                if ".seg." in fname:
                    seg_label = fname.split(".seg.")[0]
                    available_segs[seg_label] = fpath
            for gt_name in gt_filenames:
                if gt_name in dir_files:
                    available_segs[gt_name.split(".")[0]] = dir_files[gt_name]

            # Reports & VLM narrative status
            report_pdf = dir_files.get("cardiac_fat_report.pdf")
            report_html = dir_files.get("cardiac_fat_report.html")
            report_dcm = dir_files.get("eat_pvat_report.dcm")
            has_vlm = any(vf in dir_files for vf in vlm_filenames)
            if "report" in dir_subdirs:
                report_sub = p_dir / "report"
                if report_pdf is None and (report_sub / "cardiac_fat_report.pdf").is_file():
                    report_pdf = report_sub / "cardiac_fat_report.pdf"
                if report_html is None and (report_sub / "cardiac_fat_report.html").is_file():
                    report_html = report_sub / "cardiac_fat_report.html"
                if report_dcm is None and (report_sub / "eat_pvat_report.dcm").is_file():
                    report_dcm = report_sub / "eat_pvat_report.dcm"
                if not has_vlm and (report_sub / "nv_reason_narrative.txt").is_file():
                    has_vlm = True

            ohif_dir = (p_dir / "ohif_export") if "ohif_export" in dir_subdirs else None

            # Measurements
            local_csv = dir_files.get("cardiac_fat_measurements.csv")
            metrics = dict(root_metrics_by_pid.get(pid, {}))
            if local_csv is not None:
                try:
                    local_rows = _read_csv_dicts(local_csv)
                    if local_rows:
                        row = local_rows[0]
                        for c, raw_v in row.items():
                            if c != "participant_id":
                                metrics[c] = _safe_float(raw_v)
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

            if "nongated" in str(p_dir).lower() or pid.startswith("nongated"):
                scan_type = "Non-Gated CT"
            elif "gated" in str(p_dir).lower():
                scan_type = "Gated CS"
            else:
                scan_type = "CT"

            self.participants[pid] = CohortParticipant(
                participant_id=pid,
                study_dir=p_dir,
                scan_path=scan_path,
                seg_path=seg_path,
                available_segmentations=available_segs,
                measurements_csv=local_csv,
                report_pdf=report_pdf,
                report_html=report_html,
                report_dcm=report_dcm,
                ohif_export_dir=ohif_dir,
                calcium_score=cac,
                calcium_risk_tier=tier,
                metrics=metrics,
                crisp_ct_status=crisp_status,
                scan_type=scan_type,
                has_nv_reason=has_vlm,
            )

        # Ensure participants are sorted in natural human numerical order (0, 1, 2, ..., 10, 11, ..., 100)
        self.participants = dict(
            sorted(
                self.participants.items(),
                key=lambda item: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(item[0]))],
            )
        )

    @property
    def total_count(self) -> int:
        return len(self.participants)

    @property
    def quantified_count(self) -> int:
        return sum(1 for p in self.participants.values() if bool(p.metrics))

    @property
    def segmented_count(self) -> int:
        return sum(1 for p in self.participants.values() if p.seg_path is not None)

    @property
    def cac_scored_count(self) -> int:
        return sum(1 for p in self.participants.values() if p.calcium_score is not None)

    def compute_summary_analytics(self) -> Dict[str, Any]:
        """Compute comprehensive cohort research analytics."""
        import numpy as np
        from scipy import stats

        p_list = list(self.participants.values())
        total_n = len(p_list)
        if total_n == 0:
            return {
                "total_count": 0,
                "quantified_count": 0,
                "quantified_pct": 0.0,
                "segmented_count": 0,
                "segmented_pct": 0.0,
                "mask_saved_count": 0,
                "mask_saved_pct": 0.0,
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
        eat_hus = [
            p.metrics.get("EAT Total Mean Attenuation (HU)")
            for p in p_list
            if p.metrics.get("EAT Total Mean Attenuation (HU)") is not None
            and not math.isnan(p.metrics.get("EAT Total Mean Attenuation (HU)", float("nan")))
        ]
        pvat_vols = [
            p.metrics.get("PVAT Total Volume (cm3)")
            for p in p_list
            if p.metrics.get("PVAT Total Volume (cm3)") is not None
            and not math.isnan(p.metrics.get("PVAT Total Volume (cm3)", float("nan")))
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
            a = np.asarray(arr, dtype=np.float64)
            mean_val = float(np.mean(a))
            max_dev = float(np.max(np.abs(a - mean_val)))
            eps = float(np.finfo(a.dtype).eps) * 10.0
            # Guard constant or near-constant inputs so SciPy skew avoids NaN / cancellation warnings
            has_spread = len(a) > 2 and (
                (mean_val == 0.0 and max_dev > 0.0)
                or (mean_val != 0.0 and (max_dev / abs(mean_val)) >= eps)
            )
            sk = round(float(stats.skew(a)), 2) if has_spread else 0.0
            return {
                "count": len(a),
                "mean": round(float(np.mean(a)), 2),
                "std": round(float(np.std(a)), 2),
                "median": round(float(np.median(a)), 2),
                "iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 2),
                "min": round(float(np.min(a)), 2),
                "max": round(float(np.max(a)), 2),
                "skewness": sk,
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

        # 7. Advanced group comparisons and bivariate regression
        tier_comparisons = self.compute_tier_group_comparisons()
        bivariate_reg = self.compute_bivariate_regression()

        return {
            "total_count": total_n,
            "quantified_count": self.quantified_count,
            "quantified_pct": round((self.quantified_count / total_n) * 100.0, 1) if total_n else 0.0,
            "segmented_count": self.segmented_count,
            "segmented_pct": round((self.segmented_count / total_n) * 100.0, 1) if total_n else 0.0,
            "mask_saved_count": self.segmented_count,
            "mask_saved_pct": round((self.segmented_count / total_n) * 100.0, 1) if total_n else 0.0,
            "cac_scored_count": self.cac_scored_count,
            "cac_scored_pct": round((self.cac_scored_count / total_n) * 100.0, 1) if total_n else 0.0,
            "active_inflammation_count": inflam_count,
            "active_inflammation_pct": inflam_pct,
            "eat_volume_stats": calc_stats(eat_vols),
            "eat_attenuation_stats": calc_stats(eat_hus),
            "pvat_volume_stats": calc_stats(pvat_vols),
            "pvat_attenuation_stats": calc_stats(pvat_hus),
            "calcium_score_stats": calc_stats(cacs),
            "chamber_volume_summaries": chamber_summaries,
            "risk_tier_counts": tier_counts,
            "correlations": corrs,
            "tier_group_comparisons": tier_comparisons,
            "bivariate_regression": bivariate_reg,
        }

    def compute_bivariate_regression(self) -> Dict[str, Any]:
        """Compute bivariate linear regression and correlation for EAT Volume vs PVAT HU."""
        from scipy import stats

        paired: List[Tuple[float, float]] = []
        for p in self.participants.values():
            vol = p.metrics.get("EAT Total Volume (cm3)")
            hu = p.metrics.get("PVAT Total Mean Attenuation (HU)")
            if vol is not None and hu is not None and not (math.isnan(vol) or math.isnan(hu)):
                paired.append((float(vol), float(hu)))

        if len(paired) < 3:
            return {
                "n_paired": len(paired),
                "pearson_r": 0.0,
                "pearson_p": 1.0,
                "spearman_rho": 0.0,
                "spearman_p": 1.0,
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "std_err": 0.0,
                "equation": "N/A",
                "crisp_ct_threshold": -70.1,
            }

        vols, hus = zip(*paired)
        r_val, p_r = stats.pearsonr(vols, hus)
        rho_val, p_rho = stats.spearmanr(vols, hus)
        slope, intercept, r_value, p_value, std_err = stats.linregress(vols, hus)

        return {
            "n_paired": len(paired),
            "x_label": "Total EAT Volume (cm3)",
            "y_label": "PVAT Total Mean Attenuation (HU)",
            "pearson_r": round(float(r_val), 3),
            "pearson_p": float(p_r),
            "spearman_rho": round(float(rho_val), 3),
            "spearman_p": float(p_rho),
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 2),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": float(p_value),
            "std_err": round(float(std_err), 4),
            "equation": f"PVAT HU = {slope:.3f} × EAT Vol + {intercept:.2f}",
            "crisp_ct_threshold": -70.1,
        }

    def compute_tier_group_comparisons(self) -> Dict[str, Any]:
        """Compute comprehensive distribution metrics and hypothesis testing across 5 CAC tiers."""
        import numpy as np
        from scipy import stats

        results: Dict[str, Any] = {"CAC_TIERS": CAC_TIER_ORDER}

        targets = [
            ("eat_volume", "EAT Total Volume (cm3)", "cm3"),
            ("pvat_attenuation", "PVAT Total Mean Attenuation (HU)", "HU"),
        ]

        for metric_id, metric_key, unit in targets:
            tier_arrays: Dict[str, List[float]] = {t: [] for t in CAC_TIER_ORDER}
            for p in self.participants.values():
                if p.calcium_risk_tier in tier_arrays:
                    v = p.metrics.get(metric_key)
                    if v is not None and not math.isnan(v):
                        tier_arrays[p.calcium_risk_tier].append(float(v))

            tiers_out: Dict[str, Dict[str, Any]] = {}
            for t_name, arr in tier_arrays.items():
                if not arr:
                    tiers_out[t_name] = {
                        "n": 0, "mean": None, "std": None, "median": None,
                        "q1": None, "q3": None, "iqr": None, "min": None,
                        "max": None, "whisker_low": None, "whisker_high": None,
                        "outliers": [], "raw_sample": [],
                    }
                    continue

                a = np.array(arr)
                q1 = float(np.percentile(a, 25))
                med = float(np.median(a))
                q3 = float(np.percentile(a, 75))
                iqr = q3 - q1
                w_low_limit = q1 - 1.5 * iqr
                w_high_limit = q3 + 1.5 * iqr
                non_outliers = a[(a >= w_low_limit) & (a <= w_high_limit)]
                w_low = float(np.min(non_outliers)) if len(non_outliers) > 0 else float(np.min(a))
                w_high = float(np.max(non_outliers)) if len(non_outliers) > 0 else float(np.max(a))
                outliers = [float(x) for x in a[(a < w_low_limit) | (a > w_high_limit)]]

                tiers_out[t_name] = {
                    "n": len(a),
                    "mean": round(float(np.mean(a)), 2),
                    "std": round(float(np.std(a)), 2),
                    "median": round(med, 2),
                    "q1": round(q1, 2),
                    "q3": round(q3, 2),
                    "iqr": round(iqr, 2),
                    "min": round(float(np.min(a)), 2),
                    "max": round(float(np.max(a)), 2),
                    "whisker_low": round(w_low, 2),
                    "whisker_high": round(w_high, 2),
                    "outliers": [round(x, 2) for x in outliers],
                    "raw_sample": [round(x, 2) for x in a[:50]],  # Jittered points sample for UI
                }

            # Omnibus testing
            valid_arrays = [arr for arr in tier_arrays.values() if len(arr) >= 2]
            anova_res: Dict[str, Any] = {"f_stat": None, "p_value": None, "significant": False}
            kruskal_res: Dict[str, Any] = {"h_stat": None, "p_value": None, "significant": False}

            if len(valid_arrays) >= 2:
                try:
                    f_val, f_p = stats.f_oneway(*valid_arrays)
                    anova_res = {
                        "f_stat": round(float(f_val), 3),
                        "p_value": float(f_p),
                        "significant": bool(f_p < 0.05),
                    }
                except Exception:
                    pass

                try:
                    h_val, h_p = stats.kruskal(*valid_arrays)
                    kruskal_res = {
                        "h_stat": round(float(h_val), 3),
                        "p_value": float(h_p),
                        "significant": bool(h_p < 0.05),
                    }
                except Exception:
                    pass

            # Post-hoc pairwise comparisons with Bonferroni correction
            pairwise: List[Dict[str, Any]] = []
            active_tiers = [t for t in CAC_TIER_ORDER if len(tier_arrays[t]) >= 2]
            num_comparisons = max(1, len(active_tiers) * (len(active_tiers) - 1) // 2)

            for i in range(len(active_tiers)):
                for j in range(i + 1, len(active_tiers)):
                    t1, t2 = active_tiers[i], active_tiers[j]
                    try:
                        u_stat, p_raw = stats.mannwhitneyu(tier_arrays[t1], tier_arrays[t2], alternative="two-sided")
                        p_adj = min(1.0, float(p_raw) * num_comparisons)
                        stars = "ns"
                        if p_adj < 0.0001:
                            stars = "****"
                        elif p_adj < 0.001:
                            stars = "***"
                        elif p_adj < 0.01:
                            stars = "**"
                        elif p_adj < 0.05:
                            stars = "*"
                        pairwise.append({
                            "group1": t1,
                            "group2": t2,
                            "u_stat": round(float(u_stat), 1),
                            "p_raw": float(p_raw),
                            "p_adj": float(p_adj),
                            "stars": stars,
                        })
                    except Exception:
                        pass

            results[metric_id] = {
                "metric_label": metric_key,
                "unit": unit,
                "tiers": tiers_out,
                "anova": anova_res,
                "kruskal_wallis": kruskal_res,
                "pairwise": pairwise,
            }

        return results

    def get_scatter_data(self) -> List[Dict[str, Any]]:
        """Extract paired datapoints for interactive research scatter plots with strictly JSON-compliant values."""
        pts = []

        def _clean(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)
                return None if (math.isnan(f) or math.isinf(f)) else f
            except (ValueError, TypeError):
                return None

        for p in self.participants.values():
            pts.append({
                "participant_id": p.participant_id,
                "eat_volume_cm3": _clean(p.metrics.get("EAT Total Volume (cm3)")),
                "pvat_volume_cm3": _clean(p.metrics.get("PVAT Total Volume (cm3)")),
                "pvat_mean_hu": _clean(p.metrics.get("PVAT Total Mean Attenuation (HU)")),
                "calcium_score": _clean(p.calcium_score),
                "calcium_risk_tier": p.calcium_risk_tier,
                "crisp_ct_status": p.crisp_ct_status,
                "has_seg": p.seg_path is not None,
            })
        return pts
