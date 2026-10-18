"""NRRD and ITK Header-Preserving I/O and Participant Directory Discovery.

Provides robust reading and writing for 3D cardiac CT scans and 8-channel EAT/PVAT
segmentations in NRRD format while strictly preserving origin, spacing, space
directions, and coordinate system headers.
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

from cardiac_fat_fl.measurement_schema import SegmentationCompartment


def discover_participants(
    source_dir: Union[str, Path],
    scan_filename: str = "scan.nrrd",
    ignore_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, Path]]:
    """Discover participant CT scans in a source directory.

    Expected layout: `source_dir/<participant_id>/scan.nrrd`.
    Files matching `*.seg.nrrd` or existing segmentation masks are strictly ignored.

    Args:
        source_dir: Root directory containing participant subfolders.
        scan_filename: Preferred filename for the CT scan (default: "scan.nrrd").
        ignore_patterns: Suffixes/patterns to ignore when identifying CT scans.

    Returns:
        List of tuples: `(participant_id, path_to_scan_nrrd)` sorted by participant_id.
    """
    root = Path(source_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")

    if ignore_patterns is None:
        ignore_patterns = [".seg.nrrd", "_seg.nrrd", ".mask.nrrd", "eat_pvat.seg.nrrd"]

    participants: List[Tuple[str, Path]] = []

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            participant_id = entry.name
            target_scan = entry / scan_filename

            if target_scan.is_file():
                participants.append((participant_id, target_scan))
            else:
                # Fallback: look for any .nrrd in directory that is not a segmentation mask
                candidate_nrrds = [
                    f for f in entry.glob("*.nrrd")
                    if not any(f.name.lower().endswith(pat.lower()) for pat in ignore_patterns)
                ]
                if len(candidate_nrrds) == 1:
                    participants.append((participant_id, candidate_nrrds[0]))
                elif len(candidate_nrrds) > 1:
                    # Prefer file containing 'scan' or 'ct' in name
                    ranked = sorted(
                        candidate_nrrds,
                        key=lambda p: (
                            "scan" not in p.name.lower(),
                            "ct" not in p.name.lower(),
                            len(p.name),
                        ),
                    )
                    participants.append((participant_id, ranked[0]))

    return participants


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
        # Fallback to 1.0mm isotropic if missing
        spacing = (1.0, 1.0, 1.0)

    return data, header, spacing


def read_nrrd_as_metatensor(file_path: Union[str, Path]) -> MetaTensor:
    """Read an NRRD CT file directly into a MONAI MetaTensor with affine matrix.

    Args:
        file_path: Path to input NRRD file.

    Returns:
        MetaTensor with shape (1, D, H, W) or (1, X, Y, Z), float32, and spatial affine.
    """
    data, header, spacing = read_nrrd_ct(file_path)

    # Construct 4x4 spatial affine
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

    # Convert to 1-channel torch tensor
    tensor_data = torch.from_numpy(data).unsqueeze(0)  # Shape (1, X, Y, Z)
    meta = {
        "affine": torch.from_numpy(affine),
        "spatial_shape": list(data.shape),
        "spacing": list(spacing),
        "nrrd_header": header,
        "filename_or_obj": str(file_path),
    }

    return MetaTensor(tensor_data, affine=torch.from_numpy(affine), meta=meta)


def write_eat_pvat_seg_nrrd(
    seg_data: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    reference_header: Optional[Dict[str, Any]] = None,
    spacing: Optional[Tuple[float, float, float]] = None,
    origin: Optional[Tuple[float, float, float]] = None,
    space_directions: Optional[Any] = None,
) -> Path:
    """Write an 8-channel EAT/PVAT segmentation to `eat_pvat.seg.nrrd`.

    Strictly preserves the reference header spatial coordinate system and tags
    the 8 compartments matching the CardiacFatQuant specification.

    Args:
        seg_data: Binary/multichannel segmentation tensor or array.
                  Expected shape: (8, X, Y, Z) or (X, Y, Z, 8) or 3D multi-label.
        output_path: Target path for `eat_pvat.seg.nrrd`.
        reference_header: Optional dictionary from the input CT scan to copy spatial geometry.
        spacing: Optional (sx, sy, sz) voxel spacing.
        origin: Optional (ox, oy, oz) coordinate origin.
        space_directions: Optional 3x3 direction matrix.

    Returns:
        Path to the saved NRRD segmentation file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(seg_data, torch.Tensor):
        seg_np = seg_data.detach().cpu().numpy()
    else:
        seg_np = np.asarray(seg_data)

    # Cast to uint8 binary masks
    seg_np = (seg_np > 0.5).astype(np.uint8)

    # Build NRRD header
    out_header: Dict[str, Any] = {}

    if reference_header is not None:
        # Clone spatial keys from reference
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

    # Handle 8-channel vs 3D representation
    if seg_np.ndim == 4:
        if seg_np.shape[0] == 8:
            # Shape is (8, X, Y, Z) -> first axis is list kind: kinds = ['list', 'domain', 'domain', 'domain']
            out_header["kinds"] = ["list", "domain", "domain", "domain"]
            if "space directions" in out_header and out_header["space directions"] is not None:
                orig_dirs = list(out_header["space directions"])
                if len(orig_dirs) == 3:
                    out_header["space directions"] = np.array(
                        [[np.nan, np.nan, np.nan], orig_dirs[0], orig_dirs[1], orig_dirs[2]],
                        dtype=float,
                    )
        elif seg_np.shape[-1] == 8:
            out_header["kinds"] = ["domain", "domain", "domain", "list"]
            if "space directions" in out_header and out_header["space directions"] is not None:
                orig_dirs = list(out_header["space directions"])
                if len(orig_dirs) == 3:
                    out_header["space directions"] = np.array(
                        [orig_dirs[0], orig_dirs[1], orig_dirs[2], [np.nan, np.nan, np.nan]],
                        dtype=float,
                    )

    elif seg_np.ndim == 3:
        out_header["kinds"] = ["domain", "domain", "domain"]
        if spacing is not None and "space directions" not in out_header:
            out_header["space directions"] = [
                [spacing[0], 0.0, 0.0],
                [0.0, spacing[1], 0.0],
                [0.0, 0.0, spacing[2]],
            ]

    # Add segmentation segment names in metadata
    compartment_names = SegmentationCompartment.get_names()
    for idx, name in enumerate(compartment_names):
        out_header[f"Segment{idx}_Name"] = name
        out_header[f"Segment{idx}_LabelValue"] = str(idx + 1)

    out_header["encoding"] = "gzip"
    nrrd.write(str(output_path), seg_np, out_header)

    return output_path

