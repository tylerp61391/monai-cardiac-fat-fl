"""Pytest Fixtures for Cardiac Fat Quantification Testing.

Generates mathematically and clinically consistent synthetic 3D cardiac CT .nrrd phantoms
with known attenuation windows (-190 to -30 HU), chambers, pericardium, and coronary vessels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, Tuple

import nrrd
import numpy as np
import pytest
import torch

from cardiac_fat_fl.anatomy_segmenter import AnatomyMasks
from cardiac_fat_fl.io_metatensor import write_eat_pvat_seg_nrrd


def generate_synthetic_cardiac_ct(
    shape: Tuple[int, int, int] = (64, 64, 64),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    origin: Tuple[float, float, float] = (-48.0, -48.0, -64.0),
) -> Tuple[np.ndarray, Dict[str, Any], AnatomyMasks, np.ndarray]:
    """Generate a 3D synthetic cardiac CT scan with ground-truth anatomical and adipose compartments.

    Hounsfield Unit Distribution:
      - Background / Lungs: -800 HU
      - Soft Tissue / Myocardium: +40 HU
      - Blood pool in chambers: +150 HU
      - Coronary vessels (contrast-enhanced): +350 HU
      - Epicardial & Pericoronary Fat: -110 HU (strictly within [-190, -30] HU)
      - Calcification / Spine: +800 HU

    Returns:
      ct_volume: 3D float32 array in HU.
      header: NRRD header dictionary.
      anatomy: Ground-truth AnatomyMasks object.
      fat_mask: Ground-truth boolean mask of adipose voxels.
    """
    nx, ny, nz = shape
    ct = np.full(shape, -800.0, dtype=np.float32)  # Background lung / air

    grid_x, grid_y, grid_z = np.ogrid[:nx, :ny, :nz]
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0

    # 1. Pericardial Sac (Heart Envelope)
    rx, ry, rz = nx * 0.38, ny * 0.38, nz * 0.38
    pericardium = (
        ((grid_x - cx) / rx) ** 2
        + ((grid_y - cy) / ry) ** 2
        + ((grid_z - cz) / rz) ** 2
    ) <= 1.0

    # Set base heart tissue to soft tissue (+40 HU)
    ct[pericardium] = 40.0

    # 2. Cardiac Chambers (LA, LV, RA, RV) - Enhanced with blood pool (+150 HU)
    # LA: Posterior-Superior-Left
    la = (
        (((grid_x - (cx + 8)) / 9.0) ** 2
        + ((grid_y - (cy + 8)) / 9.0) ** 2
        + ((grid_z - (cz + 8)) / 9.0) ** 2) <= 1.0
    ) & pericardium

    # LV: Posterior-Inferior-Left
    lv = (
        (((grid_x - (cx + 9)) / 11.0) ** 2
        + ((grid_y - (cy - 7)) / 11.0) ** 2
        + ((grid_z - (cz - 7)) / 11.0) ** 2) <= 1.0
    ) & pericardium

    # RA: Anterior-Superior-Right
    ra = (
        (((grid_x - (cx - 8)) / 9.0) ** 2
        + ((grid_y - (cy + 8)) / 9.0) ** 2
        + ((grid_z - (cz + 8)) / 9.0) ** 2) <= 1.0
    ) & pericardium

    # RV: Anterior-Inferior-Right
    rv = (
        (((grid_x - (cx - 8)) / 10.0) ** 2
        + ((grid_y - (cy - 8)) / 10.0) ** 2
        + ((grid_z - (cz - 6)) / 10.0) ** 2) <= 1.0
    ) & pericardium

    ct[la | lv | ra | rv] = 150.0

    # 3. Coronary Arteries (Contrast enhanced: +350 HU)
    # LAD: Anterior interventricular groove
    lad = np.zeros(shape, dtype=bool)
    lad_x = int(cx)
    lad_y = int(cy - ry * 0.75)
    z_min, z_max = int(cz - 12), int(cz + 12)
    lad[lad_x - 1 : lad_x + 2, lad_y - 1 : lad_y + 2, z_min:z_max] = True
    lad = lad & pericardium
    ct[lad] = 350.0

    # LCX: Left AV groove
    lcx = np.zeros(shape, dtype=bool)
    lcx_x = int(cx + rx * 0.75)
    lcx_y = int(cy)
    lcx[lcx_x - 1 : lcx_x + 2, lcx_y - 1 : lcx_y + 2, z_min:z_max] = True
    lcx = lcx & pericardium
    ct[lcx] = 350.0

    # RCA: Right AV groove
    rca = np.zeros(shape, dtype=bool)
    rca_x = int(cx - rx * 0.75)
    rca_y = int(cy)
    rca[rca_x - 1 : rca_x + 2, rca_y - 1 : rca_y + 2, z_min:z_max] = True
    rca = rca & pericardium
    ct[rca] = 350.0

    # 4. Known Adipose Fat Deposits (-110 HU, strictly in [-190, -30])
    # Place epicardial fat in subepicardial space (near outer pericardial boundary)
    outer_shell = (
        ((grid_x - cx) / rx) ** 2
        + ((grid_y - cy) / ry) ** 2
        + ((grid_z - cz) / rz) ** 2
    ) >= 0.75
    fat_zone = pericardium & outer_shell & ~(la | lv | ra | rv | lad | lcx | rca)
    ct[fat_zone] = -110.0  # Known fat HU

    # Pericoronary fat layer immediately surrounding LAD vessel
    pvat_lad_zone = (
        (np.abs(grid_x - lad_x) <= 3)
        & (np.abs(grid_y - lad_y) <= 3)
        & (grid_z >= z_min)
        & (grid_z <= z_max)
    ) & pericardium & ~lad
    ct[pvat_lad_zone] = -95.0

    # Pericoronary fat layer immediately surrounding LCX vessel
    pvat_lcx_zone = (
        (np.abs(grid_x - lcx_x) <= 3)
        & (np.abs(grid_y - lcx_y) <= 3)
        & (grid_z >= z_min)
        & (grid_z <= z_max)
    ) & pericardium & ~lcx
    ct[pvat_lcx_zone] = -105.0

    # Pericoronary fat layer immediately surrounding RCA vessel
    pvat_rca_zone = (
        (np.abs(grid_x - rca_x) <= 3)
        & (np.abs(grid_y - rca_y) <= 3)
        & (grid_z >= z_min)
        & (grid_z <= z_max)
    ) & pericardium & ~rca
    ct[pvat_rca_zone] = -100.0

    # Spine/Bone (+800 HU) outside pericardium
    ct[int(cx) - 2 : int(cx) + 3, int(ny - 6) : int(ny - 2), :] = 800.0

    # Build NRRD header
    header = {
        "type": "float",
        "dimension": 3,
        "space": "left-posterior-superior",
        "sizes": list(shape),
        "space directions": [
            [spacing[0], 0.0, 0.0],
            [0.0, spacing[1], 0.0],
            [0.0, 0.0, spacing[2]],
        ],
        "kinds": ["domain", "domain", "domain"],
        "endian": "little",
        "encoding": "gzip",
        "space origin": list(origin),
    }

    anatomy = AnatomyMasks(
        pericardium=pericardium,
        la=la,
        lv=lv,
        ra=ra,
        rv=rv,
        lad=lad,
        lcx=lcx,
        rca=rca,
    )

    fat_mask = (ct >= -190.0) & (ct <= -30.0)

    return ct, header, anatomy, fat_mask


@pytest.fixture
def synthetic_ct_fixture() -> Tuple[np.ndarray, Dict[str, Any], AnatomyMasks, np.ndarray]:
    """Pytest fixture providing in-memory synthetic 3D cardiac CT data."""
    return generate_synthetic_cardiac_ct()


@pytest.fixture
def synthetic_nrrd_scan(tmp_path: Path) -> Path:
    """Pytest fixture saving a single synthetic 3D cardiac CT scan.nrrd to disk."""
    ct, header, _, _ = generate_synthetic_cardiac_ct()
    participant_dir = tmp_path / "participant_001"
    participant_dir.mkdir(parents=True, exist_ok=True)
    scan_path = participant_dir / "scan.nrrd"
    nrrd.write(str(scan_path), ct, header)
    return scan_path


@pytest.fixture
def synthetic_cohort_dir(tmp_path: Path) -> Path:
    """Pytest fixture creating a synthetic cohort directory with 2 participants."""
    cohort_dir = tmp_path / "synthetic_cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    for i in (1, 2):
        pid = f"PATIENT_{i:03d}"
        pdir = cohort_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)
        scan_path = pdir / "scan.nrrd"
        ct, header, _, _ = generate_synthetic_cardiac_ct(
            spacing=(1.5, 1.5, 2.0),
            origin=(-48.0, -48.0, -64.0),
        )
        nrrd.write(str(scan_path), ct, header)

    return cohort_dir

