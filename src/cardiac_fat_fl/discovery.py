"""Lightweight, dependency-free CT participant & DICOM series discovery helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union


def _has_at_least_n_dicoms(dir_path: Path, min_count: int = 5) -> bool:
    """Return True if *dir_path* directly contains at least *min_count* ``.dcm`` files."""
    count = 0
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.name.lower().endswith(".dcm") and entry.is_file():
                    count += 1
                    if count >= min_count:
                        return True
    except OSError:
        return False
    return False


def _find_dicom_series_dir(dir_path: Path) -> Optional[Path]:
    """Check if dir_path or an immediate subdirectory contains a DICOM series."""
    if _has_at_least_n_dicoms(dir_path, 5):
        return dir_path

    # Check immediate subdirectories (e.g. participant/series_name/*.dcm)
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.is_dir():
                    sub = Path(entry.path)
                    if _has_at_least_n_dicoms(sub, 5):
                        return sub
    except OSError:
        return None

    return None


def discover_participants(
    source_dir: Union[str, Path],
    scan_filename: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, Path]]:
    """Discover participant CT scans in a source directory supporting DICOM, NRRD, and NIfTI.

    Supports multiple directory layouts:
      1. DICOM series folders: `source_dir/<participant_id>/<series_name>/*.dcm` or `source_dir/<participant_id>/*.dcm`
      2. 3D Volume files: `source_dir/<participant_id>/scan.nrrd` (or `*.nii.gz`, `*.mha`, etc.)
      3. Flat volumes: `source_dir/<participant_id>.nrrd` or `source_dir/<participant_id>.nii.gz`

    Files matching `*.seg.*`, `*.mask.*`, or `eat_pvat*` are strictly ignored.

    Args:
        source_dir: Root directory containing participant scans or subfolders.
        scan_filename: Optional preferred filename (e.g. "scan.nrrd").
        ignore_patterns: Substrings/patterns to ignore when identifying CT scans.

    Returns:
        List of tuples: `(participant_id, path_to_scan_or_dicom_dir)` sorted by participant_id.
    """
    root = Path(source_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")

    if ignore_patterns is None:
        ignore_patterns = [".seg.", "_seg.", ".mask.", "eat_pvat", "calcium_xml", "graphs", "ohif_export"]

    ignore_lower = tuple(pat.lower() for pat in ignore_patterns)

    # Check if root itself is a single participant study directory (contains DICOM series directly, or a single series subfolder)
    subdirs = [
        d
        for d in root.iterdir()
        if d.is_dir() and not any(pat in d.name.lower() for pat in ignore_lower)
    ]
    dcm_root = _find_dicom_series_dir(root)
    if dcm_root is not None and (dcm_root == root or (dcm_root.parent == root and len(subdirs) == 1)):
        return [(root.name, dcm_root)]

    participants: List[Tuple[str, Path]] = []

    for entry in sorted(root.iterdir()):
        # Ignore segmentation masks and non-target directories
        entry_lower = entry.name.lower()
        if any(pat in entry_lower for pat in ignore_lower):
            continue

        if entry.is_dir():
            participant_id = entry.name

            # 1. Preferred explicit filename
            if scan_filename is not None:
                target = entry / scan_filename
                if target.is_file():
                    participants.append((participant_id, target))
                    continue

            # 2. Check for DICOM series directory (direct or nested)
            dicom_dir = _find_dicom_series_dir(entry)
            if dicom_dir is not None:
                participants.append((participant_id, dicom_dir))
                continue

            # 3. Look for 3D Volume files in folder (.nrrd, .nii, .nii.gz, .mha)
            vol_candidates = [
                f
                for f in entry.iterdir()
                if f.is_file()
                and "." in f.name
                and f.suffix.lower() in [".nrrd", ".nii", ".gz", ".mha", ".mhd"]
                and not any(pat in f.name.lower() for pat in ignore_lower)
            ]
            if len(vol_candidates) == 1:
                participants.append((participant_id, vol_candidates[0]))
            elif len(vol_candidates) > 1:
                ranked = sorted(
                    vol_candidates,
                    key=lambda p: (
                        "scan" not in p.name.lower(),
                        "ct" not in p.name.lower(),
                        len(p.name),
                    ),
                )
                participants.append((participant_id, ranked[0]))

        elif entry.is_file():
            # Flat file layout: source_dir/<participant_id>.nrrd or .nii.gz
            ext = entry.suffix.lower()
            if ext in [".nrrd", ".mha"] or entry_lower.endswith(".nii.gz"):
                pid = entry.name.split(".")[0]
                participants.append((pid, entry))

    # Check for nested Stanford COCA layout (Gated_release_final/patient and deidentified_nongated)
    if not participants:
        coca_roots = [root, root / "cocacoronarycalciumandchestcts-2"]
        for c_root in coca_roots:
            gated_dir = c_root / "Gated_release_final" / "patient"
            nongated_dir = c_root / "deidentified_nongated"
            if gated_dir.is_dir():
                for g_pid, g_path in discover_participants(gated_dir, scan_filename, ignore_patterns):
                    participants.append((g_pid, g_path))
            if nongated_dir.is_dir():
                for ng_pid, ng_path in discover_participants(nongated_dir, scan_filename, ignore_patterns):
                    prefix_pid = f"nongated_{ng_pid}" if gated_dir.is_dir() else ng_pid
                    participants.append((prefix_pid, ng_path))
            if participants:
                participants.sort(
                    key=lambda item: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(item[0]))]
                )
                return participants

    # Fallback: check if root itself is a single participant study directory
    if not participants:
        dcm_dir = _find_dicom_series_dir(root)
        if dcm_dir is not None:
            participants.append((root.name, dcm_dir))
        else:
            vol_candidates = [
                f
                for f in root.iterdir()
                if f.is_file()
                and "." in f.name
                and f.suffix.lower() in [".nrrd", ".nii", ".gz", ".mha", ".mhd"]
                and not any(pat in f.name.lower() for pat in ignore_lower)
            ]
            if vol_candidates:
                ranked = sorted(
                    vol_candidates,
                    key=lambda p: (
                        "scan" not in p.name.lower(),
                        "ct" not in p.name.lower(),
                        len(p.name),
                    ),
                )
                participants.append((root.name, ranked[0]))

    participants.sort(key=lambda item: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(item[0]))])
    return participants
