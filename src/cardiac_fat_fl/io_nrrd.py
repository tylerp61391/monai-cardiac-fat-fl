"""Robust Medical Image I/O and Participant Directory Discovery using MONAI and SimpleITK.

Supports loading 3D Cardiac CT from:
  - DICOM Series (single folder of .dcm files or nested series folders)
  - NRRD (.nrrd)
  - NIfTI (.nii, .nii.gz)
  - MetaImage (.mha, .mhd)

Preserves exact origin, spacing, space direction matrices, and clinical Hounsfield Units (HU)
within MONAI MetaTensor objects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nrrd
import numpy as np
import SimpleITK as sitk
import torch
from monai.data import MetaTensor
from monai.data.image_reader import ITKReader

from cardiac_fat_fl.measurement_schema import SegmentationCompartment


def _find_dicom_series_dir(dir_path: Path) -> Optional[Path]:
    """Check if dir_path or an immediate subdirectory contains a DICOM series."""
    # Check current directory
    dcm_files = list(dir_path.glob("*.dcm"))
    if len(dcm_files) >= 5:
        return dir_path

    # Check immediate subdirectories (e.g. participant/series_name/*.dcm)
    for sub in dir_path.iterdir():
        if sub.is_dir():
            sub_dcm = list(sub.glob("*.dcm"))
            if len(sub_dcm) >= 5:
                return sub

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
        ignore_patterns = [".seg.", "_seg.", ".mask.", "eat_pvat", "calcium_xml", "graphs"]

    participants: List[Tuple[str, Path]] = []

    for entry in sorted(root.iterdir()):
        # Ignore segmentation masks and non-target directories
        if any(pat.lower() in entry.name.lower() for pat in ignore_patterns):
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
                f for f in entry.glob("*.*")
                if f.suffix.lower() in [".nrrd", ".nii", ".gz", ".mha", ".mhd"]
                and not any(pat.lower() in f.name.lower() for pat in ignore_patterns)
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
            if ext in [".nrrd", ".mha"] or entry.name.lower().endswith(".nii.gz"):
                pid = entry.name.split(".")[0]
                participants.append((pid, entry))

    return participants


def _compute_affine_from_sitk(image: sitk.Image) -> np.ndarray:
    """Compute 4x4 spatial affine matrix from SimpleITK image metadata."""
    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    origin = np.array(image.GetOrigin(), dtype=np.float64)
    direction = np.array(image.GetDirection(), dtype=np.float64)

    dim = image.GetDimension()
    affine = np.eye(4, dtype=np.float64)

    if dim == 3 and len(direction) == 9:
        dir_mat = direction.reshape(3, 3)
        affine[:3, :3] = dir_mat * spacing
        affine[:3, 3] = origin
    elif dim == 3:
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        affine[:3, 3] = origin

    return affine


def read_medical_image_as_metatensor(path_or_dir: Union[str, Path]) -> MetaTensor:
    """Read a medical CT scan from a DICOM directory, NRRD, NIfTI, or MetaImage into MONAI MetaTensor.

    Automatically handles DICOM series loading, applies rescale slope and intercept for true HU,
    and constructs a MONAI MetaTensor with full spatial affine matrix.

    Args:
        path_or_dir: File path or DICOM series directory.

    Returns:
        MONAI MetaTensor of shape (1, D, H, W) with complete spatial metadata.
    """
    p = Path(path_or_dir).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Medical image path not found: {p}")

    # Case 1: Directory containing DICOM series
    if p.is_dir():
        dcm_dir = _find_dicom_series_dir(p)
        if dcm_dir is None:
            raise ValueError(f"Directory does not contain a recognized DICOM series: {p}")

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dcm_dir))
        if not series_ids:
            # Fallback: get all .dcm files directly
            dcm_files = [str(f) for f in sorted(dcm_dir.glob("*.dcm"))]
            if not dcm_files:
                raise ValueError(f"No DICOM slices found in {dcm_dir}")
            reader.SetFileNames(dcm_files)
        else:
            file_names = reader.GetGDCMSeriesFileNames(str(dcm_dir), series_ids[0])
            reader.SetFileNames(file_names)

        sitk_image = reader.Execute()
        data_np = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # Shape (D, H, W)
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        affine = _compute_affine_from_sitk(sitk_image)

        meta: Dict[str, Any] = {
            "spatial_shape": list(data_np.shape),
            "spacing": [float(s) for s in spacing],
            "origin": [float(o) for o in origin],
            "direction": list(direction),
            "filename_or_obj": str(p),
            "format": "DICOM",
        }

        tensor_data = torch.from_numpy(data_np).unsqueeze(0)  # Shape (1, D, H, W)
        return MetaTensor(tensor_data, affine=torch.from_numpy(affine), meta=meta)

    # Case 2: File (.nrrd, .nii, .nii.gz, .mha)
    if p.name.lower().endswith(".nrrd"):
        data_np, header, spacing = read_nrrd_ct(p)
        affine = np.eye(4, dtype=np.float64)
        if "space directions" in header and header["space directions"] is not None:
            dirs = np.array(header["space directions"], dtype=np.float64)
            if dirs.shape == (3, 3):
                affine[:3, :3] = dirs.T
        else:
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2]

        if "space origin" in header and header["space origin"] is not None:
            affine[:3, 3] = np.array(header["space origin"], dtype=np.float64)

        meta = {
            "spatial_shape": list(data_np.shape),
            "spacing": list(spacing),
            "nrrd_header": header,
            "filename_or_obj": str(p),
            "format": "NRRD",
        }
        tensor_data = torch.from_numpy(data_np).unsqueeze(0)
        return MetaTensor(tensor_data, affine=torch.from_numpy(affine), meta=meta)

    # Fallback to SimpleITK / MONAI ITKReader for NIfTI / MetaImage / other 3D formats
    sitk_image = sitk.ReadImage(str(p))
    data_np = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()
    affine = _compute_affine_from_sitk(sitk_image)

    meta = {
        "spatial_shape": list(data_np.shape),
        "spacing": [float(s) for s in spacing],
        "origin": [float(o) for o in origin],
        "direction": list(direction),
        "filename_or_obj": str(p),
        "format": p.suffix.upper(),
    }
    tensor_data = torch.from_numpy(data_np).unsqueeze(0)
    return MetaTensor(tensor_data, affine=torch.from_numpy(affine), meta=meta)


# Backward-compatible aliases
read_nrrd_as_metatensor = read_medical_image_as_metatensor


def read_nrrd_ct(
    file_path: Union[str, Path]
) -> Tuple[np.ndarray, Dict[str, Any], Tuple[float, float, float]]:
    """Read a 3D CT scan from an NRRD file, preserving complete spatial metadata.

    Args:
        file_path: Path to the input .nrrd file.

    Returns:
        data: 3D float32 numpy array of CT voxel intensities in Hounsfield Units.
        header: Raw NRRD header dictionary.
        spacing: Tuple of (sx, sy, sz) voxel spacing in millimeters.
    """
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"NRRD file not found: {file_path}")

    data, header = nrrd.read(str(file_path))
    data = data.astype(np.float32)

    # Extract voxel spacing
    spacing: Tuple[float, float, float]
    if "space directions" in header and header["space directions"] is not None:
        dirs = header["space directions"]
        sp_list = []
        for d in dirs:
            if d is not None:
                norm = float(np.linalg.norm(np.array(d, dtype=np.float64)))
                sp_list.append(norm if norm > 0 else 1.0)
            else:
                sp_list.append(1.0)
        spacing = (sp_list[0], sp_list[1], sp_list[2])
    elif "spacings" in header and header["spacings"] is not None:
        raw_sp = header["spacings"]
        spacing = (float(raw_sp[0]), float(raw_sp[1]), float(raw_sp[2]))
    else:
        spacing = (1.0, 1.0, 1.0)

    return data, header, spacing


# Color palette for 8 EAT/PVAT compartments (normalized RGB for 3D Slicer)
SLICER_COMPARTMENT_COLORS = [
    "0.576 0.200 0.918",  # EAT_Total (Purple)
    "0.231 0.510 0.965",  # EAT_LA (Blue)
    "0.024 0.714 0.831",  # EAT_LV (Cyan)
    "0.063 0.725 0.506",  # EAT_RA (Green)
    "0.918 0.702 0.031",  # EAT_RV (Yellow)
    "0.976 0.451 0.086",  # PVAT_LAD (Orange)
    "0.937 0.267 0.267",  # PVAT_LCX (Red)
    "0.925 0.282 0.596",  # PVAT_RCA (Pink)
]

# Cardiac anatomical labels matching monai_wholebody_ct_segmentation & MGH convention
MONAI_WHOLEBODY_CT_CARDIAC_LABELS = [
    {"id": "aorta", "name": "aorta", "label": 7, "color": "0.850 0.150 0.150"},
    {"id": "heart_atrium_left", "name": "heart_atrium_left", "label": 45, "color": "0.231 0.510 0.965"},
    {"id": "heart_ventricle_left", "name": "heart_ventricle_left", "label": 46, "color": "0.024 0.714 0.831"},
    {"id": "heart_atrium_right", "name": "heart_atrium_right", "label": 47, "color": "0.063 0.725 0.506"},
    {"id": "heart_ventricle_right", "name": "heart_ventricle_right", "label": 48, "color": "0.918 0.702 0.031"},
    {"id": "coronary_artery_lad", "name": "coronary_artery_lad", "label": 5, "color": "0.976 0.451 0.086"},
    {"id": "coronary_artery_lcx", "name": "coronary_artery_lcx", "label": 6, "color": "0.937 0.267 0.267"},
    {"id": "coronary_artery_rca", "name": "coronary_artery_rca", "label": 8, "color": "0.925 0.282 0.596"},
]
TOTALSEGMENTATOR_CARDIAC_LABELS = MONAI_WHOLEBODY_CT_CARDIAC_LABELS


def write_eat_pvat_seg_nrrd(
    seg_data: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    reference_header: Optional[Dict[str, Any]] = None,
    spacing: Optional[Tuple[float, float, float]] = None,
    origin: Optional[Tuple[float, float, float]] = None,
    space_directions: Optional[Any] = None,
    is_totalseg: bool = False,
) -> Path:
    """Write an EAT/PVAT or anatomical segmentation to `.seg.nrrd` compatible with 3D Slicer.

    Strictly preserves the reference header spatial coordinate system and embeds
    3D Slicer segment metadata (names, layers, label values, colors).
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(seg_data, torch.Tensor):
        seg_np = seg_data.detach().cpu().numpy()
    else:
        seg_np = np.asarray(seg_data)

    # Check if input is a multi-label integer volume (e.g. TotalSegmentator masks)
    has_multilabel = is_totalseg or (np.issubdtype(seg_np.dtype, np.integer) and np.max(seg_np) > 1)
    if has_multilabel:
        seg_np = seg_np.astype(np.uint16)
    else:
        seg_np = (seg_np > 0.5).astype(np.uint8)

    out_header: Dict[str, Any] = {}

    if reference_header is not None:
        for key in [
            "space",
            "space dimension",
            "space origin",
            "space directions",
            "measurement frame",
            "endian",
            "encoding",
        ]:
            if key in reference_header:
                out_header[key] = reference_header[key]

    if origin is not None:
        out_header["space origin"] = list(origin)
    if "space" not in out_header:
        out_header["space"] = "left-posterior-superior"

    # 3D Slicer segmentation representation tags
    out_header["Segmentation_MasterRepresentation"] = "Binary labelmap"
    out_header["Segmentation_ContainedRepresentationNames"] = "Binary labelmap|"

    # Handle 8-channel vs 3D representation
    if seg_np.ndim == 4:
        if seg_np.shape[0] == 8:
            out_header["kinds"] = ["list", "domain", "domain", "domain"]
            if "space directions" in out_header and out_header["space directions"] is not None:
                orig_dirs = list(out_header["space directions"])
                if len(orig_dirs) == 3:
                    out_header["space directions"] = np.array(
                        [[np.nan, np.nan, np.nan], orig_dirs[0], orig_dirs[1], orig_dirs[2]],
                        dtype=float,
                    )
            elif space_directions is not None:
                out_header["space directions"] = np.array(
                    [[np.nan, np.nan, np.nan], space_directions[0], space_directions[1], space_directions[2]],
                    dtype=float,
                )
            elif spacing is not None:
                out_header["space directions"] = np.array(
                    [[np.nan, np.nan, np.nan], [spacing[0], 0.0, 0.0], [0.0, spacing[1], 0.0], [0.0, 0.0, spacing[2]]],
                    dtype=float,
                )

            # Add 3D Slicer metadata for the 8 compartments across layers
            compartment_names = SegmentationCompartment.get_names()
            for idx, name in enumerate(compartment_names):
                out_header[f"Segment{idx}_ID"] = f"Segment_{idx + 1}"
                out_header[f"Segment{idx}_Name"] = name
                out_header[f"Segment{idx}_Layer"] = str(idx)
                out_header[f"Segment{idx}_LabelValue"] = "1"
                out_header[f"Segment{idx}_Color"] = SLICER_COMPARTMENT_COLORS[idx]
                out_header[f"Segment{idx}_ColorAutoGenerated"] = "0"
                pts = np.where(seg_np[idx] > 0)
                if len(pts[0]) > 0:
                    out_header[f"Segment{idx}_Extent"] = f"{int(pts[0].min())} {int(pts[0].max())} {int(pts[1].min())} {int(pts[1].max())} {int(pts[2].min())} {int(pts[2].max())}"

        elif seg_np.shape[-1] == 8:
            out_header["kinds"] = ["domain", "domain", "domain", "list"]
            if "space directions" in out_header and out_header["space directions"] is not None:
                orig_dirs = list(out_header["space directions"])
                if len(orig_dirs) == 3:
                    out_header["space directions"] = np.array(
                        [orig_dirs[0], orig_dirs[1], orig_dirs[2], [np.nan, np.nan, np.nan]],
                        dtype=float,
                    )
            elif spacing is not None:
                out_header["space directions"] = np.array(
                    [[spacing[0], 0.0, 0.0], [0.0, spacing[1], 0.0], [0.0, 0.0, spacing[2]], [np.nan, np.nan, np.nan]],
                    dtype=float,
                )

            compartment_names = SegmentationCompartment.get_names()
            for idx, name in enumerate(compartment_names):
                out_header[f"Segment{idx}_ID"] = f"Segment_{idx + 1}"
                out_header[f"Segment{idx}_Name"] = name
                out_header[f"Segment{idx}_Layer"] = str(idx)
                out_header[f"Segment{idx}_LabelValue"] = "1"
                out_header[f"Segment{idx}_Color"] = SLICER_COMPARTMENT_COLORS[idx]
                out_header[f"Segment{idx}_ColorAutoGenerated"] = "0"

    elif seg_np.ndim == 3:
        out_header["kinds"] = ["domain", "domain", "domain"]
        if spacing is not None and "space directions" not in out_header:
            out_header["space directions"] = [
                [spacing[0], 0.0, 0.0],
                [0.0, spacing[1], 0.0],
                [0.0, 0.0, spacing[2]],
            ]

        if has_multilabel:
            # Multi-label monai_wholebody_ct_segmentation chamber and vessel segmentation
            seg_idx = 0
            for item in TOTALSEGMENTATOR_CARDIAC_LABELS:
                lbl = item["label"]
                if np.any(seg_np == lbl):
                    out_header[f"Segment{seg_idx}_ID"] = item["id"]
                    out_header[f"Segment{seg_idx}_Name"] = item["name"]
                    out_header[f"Segment{seg_idx}_Layer"] = "0"
                    out_header[f"Segment{seg_idx}_LabelValue"] = str(lbl)
                    out_header[f"Segment{seg_idx}_Color"] = item["color"]
                    out_header[f"Segment{seg_idx}_ColorAutoGenerated"] = "0"
                    pts = np.where(seg_np == lbl)
                    if len(pts[0]) > 0:
                        out_header[f"Segment{seg_idx}_Extent"] = f"{int(pts[0].min())} {int(pts[0].max())} {int(pts[1].min())} {int(pts[1].max())} {int(pts[2].min())} {int(pts[2].max())}"
                    seg_idx += 1
        else:
            # Single-compartment binary mask (e.g. pericardium / heart.seg.nrrd)
            out_header["Segment0_ID"] = "heart"
            out_header["Segment0_Name"] = "pericardium"
            out_header["Segment0_Layer"] = "0"
            out_header["Segment0_LabelValue"] = "1"
            out_header["Segment0_Color"] = "0.85 0.15 0.15"
            out_header["Segment0_ColorAutoGenerated"] = "0"
            pts = np.where(seg_np > 0)
            if len(pts[0]) > 0:
                out_header["Segment0_Extent"] = f"{int(pts[0].min())} {int(pts[0].max())} {int(pts[1].min())} {int(pts[1].max())} {int(pts[2].min())} {int(pts[2].max())}"

    out_header["encoding"] = "gzip"
    nrrd.write(str(output_path), seg_np, out_header)

    return output_path


def write_monai_wholebody_ct_segmentation_seg_nrrd(
    seg_data: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    reference_header: Optional[Dict[str, Any]] = None,
    spacing: Optional[Tuple[float, float, float]] = None,
    origin: Optional[Tuple[float, float, float]] = None,
    space_directions: Optional[Any] = None,
) -> Path:
    """Convenience alias to write multi-label TotalSegmentator masks to `.seg.nrrd`."""
    return write_eat_pvat_seg_nrrd(
        seg_data=seg_data,
        output_path=output_path,
        reference_header=reference_header,
        spacing=spacing,
        origin=origin,
        space_directions=space_directions,
        is_totalseg=True,
    )


write_totalsegmentator_seg_nrrd = write_monai_wholebody_ct_segmentation_seg_nrrd
