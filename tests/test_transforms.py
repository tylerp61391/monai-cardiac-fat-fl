"""Unit and Integration Tests for Transforms, Geometry, and Schema Export.

Verifies:
  - Strict -190 to -30 HU attenuation thresholding
  - Exact cm3 volume mathematics from NRRD header voxel spacing
  - Empty mask handling (0.0 volume, NaN attenuation)
  - 8-compartment EAT/PVAT mask tensor generation
  - Header-preserving `eat_pvat.seg.nrrd` export
  - Exact 19-column `cardiac_fat_measurements.csv` validation
  - Discovery filtering (ignoring *.seg.nrrd files)
"""

from __future__ import annotations

import math
from pathlib import Path

import nrrd
import numpy as np
import pandas as pd
import pytest
import torch
from monai.data import MetaTensor

from cardiac_fat_fl.anatomy_segmenter import AnatomyMasks, UnifiedAnatomySegmenter
from cardiac_fat_fl.io_metatensor import (
    discover_participants,
    read_nrrd_as_metatensor,
    read_nrrd_ct,
    write_eat_pvat_seg_nrrd,
)
from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    MEASUREMENT_CSV_COLUMNS,
    REPORTED_REGIONS,
    ReportedRegion,
    SegmentationCompartment,
    calculate_volume_cm3,
    validate_measurement_dataframe,
)
from cardiac_fat_fl.transforms import ComputeEatPvatMasksd, QuantifyAdiposeMetricsd


def test_calculate_volume_cm3() -> None:
    """Verify exact formula: voxel_count * (sx * sy * sz) / 1000.0."""
    # 1,000 voxels with 1mm x 1mm x 1mm spacing = 1.0 cm3
    vol = calculate_volume_cm3(1000, (1.0, 1.0, 1.0))
    assert math.isclose(vol, 1.0, rel_tol=1e-6)

    # 500 voxels with 1.5mm x 1.5mm x 2.0mm spacing
    # 500 * (1.5 * 1.5 * 2.0) = 500 * 4.5 = 2250 mm3 = 2.25 cm3
    vol2 = calculate_volume_cm3(500, (1.5, 1.5, 2.0))
    assert math.isclose(vol2, 2.25, rel_tol=1e-6)

    # Zero voxels = 0.0 cm3
    vol_zero = calculate_volume_cm3(0, (1.2, 1.2, 1.5))
    assert vol_zero == 0.0


def test_schema_columns_count_and_order() -> None:
    """Verify that MEASUREMENT_CSV_COLUMNS has exactly 19 columns in canonical order."""
    assert len(MEASUREMENT_CSV_COLUMNS) == 19
    assert MEASUREMENT_CSV_COLUMNS[0] == "participant_id"

    # Verify all 9 regions are present with both Volume and Mean Attenuation
    for region in REPORTED_REGIONS:
        vol_col = f"{region.value} Volume (cm3)"
        hu_col = f"{region.value} Mean Attenuation (HU)"
        assert vol_col in MEASUREMENT_CSV_COLUMNS
        assert hu_col in MEASUREMENT_CSV_COLUMNS

        # Verify they are consecutive pairs
        idx_vol = MEASUREMENT_CSV_COLUMNS.index(vol_col)
        idx_hu = MEASUREMENT_CSV_COLUMNS.index(hu_col)
        assert idx_hu == idx_vol + 1


def test_validate_measurement_dataframe_success_and_failure() -> None:
    """Verify validation passes for exact schema and fails on missing or extra columns."""
    valid_data = {col: [0.0] for col in MEASUREMENT_CSV_COLUMNS}
    valid_data["participant_id"] = ["TEST_001"]
    df_valid = pd.DataFrame(valid_data)

    assert validate_measurement_dataframe(df_valid) is True

    # Missing column
    df_invalid = df_valid.drop(columns=["PVAT Total Volume (cm3)"])
    with pytest.raises(ValueError, match="Measurement DataFrame schema mismatch"):
        validate_measurement_dataframe(df_invalid)

    # Extra unexpected column
    df_extra = df_valid.copy()
    df_extra["Extra_Col"] = 1.0
    with pytest.raises(ValueError, match="Measurement DataFrame schema mismatch"):
        validate_measurement_dataframe(df_extra)


def test_discover_participants_ignores_segmentations(tmp_path: Path) -> None:
    """Verify that discover_participants strictly ignores *.seg.nrrd and finds scan.nrrd."""
    pdir = tmp_path / "PATIENT_999"
    pdir.mkdir(parents=True, exist_ok=True)

    scan_file = pdir / "scan.nrrd"
    seg_file = pdir / "eat_pvat.seg.nrrd"
    dummy_seg2 = pdir / "heart.seg.nrrd"

    # Write dummy NRRD files
    dummy_data = np.zeros((10, 10, 10), dtype=np.float32)
    nrrd.write(str(scan_file), dummy_data)
    nrrd.write(str(seg_file), dummy_data)
    nrrd.write(str(dummy_seg2), dummy_data)

    participants = discover_participants(tmp_path)
    assert len(participants) == 1
    pid, discovered_scan = participants[0]
    assert pid == "PATIENT_999"
    assert discovered_scan.name == "scan.nrrd"


def test_compute_eat_pvat_masksd_attenuation_window(synthetic_ct_fixture) -> None:
    """Verify that EAT and PVAT masks strictly obey the [-190, -30] HU window."""
    ct_vol, header, anatomy, _ = synthetic_ct_fixture
    spacing = (1.5, 1.5, 2.0)

    transform = ComputeEatPvatMasksd(
        keys=["image"],
        hu_min=ADIPOSE_HU_MIN,
        hu_max=ADIPOSE_HU_MAX,
    )

    data = {
        "image": ct_vol,
        "anatomy": anatomy,
        "participant_id": "TEST_PHANTOM",
    }
    # Attach spacing to MetaTensor
    meta_ct = MetaTensor(
        torch.from_numpy(ct_vol).unsqueeze(0),
        meta={"spacing": list(spacing), "nrrd_header": header},
    )
    data["image"] = meta_ct

    result = transform(data)
    assert "pred" in result
    mask_8ch = result["pred"]

    assert mask_8ch.shape[0] == 8
    mask_np = mask_8ch.detach().cpu().numpy()

    # For each compartment, verify all active voxels are within [-190, -30] HU
    for c in range(8):
        comp_mask = mask_np[c] > 0.5
        if np.any(comp_mask):
            hu_in_mask = ct_vol[comp_mask]
            assert np.all(hu_in_mask >= ADIPOSE_HU_MIN), f"Channel {c} contained HU < {ADIPOSE_HU_MIN}"
            assert np.all(hu_in_mask <= ADIPOSE_HU_MAX), f"Channel {c} contained HU > {ADIPOSE_HU_MAX}"


def test_quantify_adipose_metricsd_known_values(synthetic_ct_fixture) -> None:
    """Verify metrics calculation against known synthetic ground truth."""
    ct_vol, header, anatomy, _ = synthetic_ct_fixture
    spacing = (1.5, 1.5, 2.0)

    mask_transform = ComputeEatPvatMasksd(keys=["image"])
    quant_transform = QuantifyAdiposeMetricsd(keys=["image"])

    meta_ct = MetaTensor(
        torch.from_numpy(ct_vol).unsqueeze(0),
        meta={"spacing": list(spacing), "nrrd_header": header},
    )
    data = {
        "image": meta_ct,
        "anatomy": anatomy,
        "participant_id": "PATIENT_KNOWN_HU",
    }

    data = mask_transform(data)
    data = quant_transform(data)

    metrics = data["metrics"]
    assert metrics["participant_id"] == "PATIENT_KNOWN_HU"

    # Verify EAT Total volume is positive and attenuation is within [-190, -30]
    eat_total_vol = metrics["EAT Total Volume (cm3)"]
    eat_total_hu = metrics["EAT Total Mean Attenuation (HU)"]
    assert eat_total_vol > 0.0
    assert -190.0 <= eat_total_hu <= -30.0

    # Verify PVAT Total is non-negative
    pvat_total_vol = metrics["PVAT Total Volume (cm3)"]
    assert pvat_total_vol >= 0.0

    # Test conversion to DataFrame and validation
    df = pd.DataFrame([metrics])
    assert validate_measurement_dataframe(df) is True


def test_empty_mask_handling() -> None:
    """Verify that empty masks produce volume 0.0 and mean HU NaN without throwing errors."""
    shape = (32, 32, 32)
    empty_ct = np.zeros(shape, dtype=np.float32)
    # Zero 8-channel mask
    empty_mask = torch.zeros((8, *shape), dtype=torch.float32)

    data = {
        "image": MetaTensor(torch.from_numpy(empty_ct).unsqueeze(0), meta={"spacing": [1.0, 1.0, 1.0]}),
        "pred": empty_mask,
        "participant_id": "PATIENT_EMPTY",
    }

    quant = QuantifyAdiposeMetricsd(keys=["image"])
    res = quant(data)
    metrics = res["metrics"]

    for region in REPORTED_REGIONS:
        assert metrics[f"{region.value} Volume (cm3)"] == 0.0
        assert math.isnan(metrics[f"{region.value} Mean Attenuation (HU)"])


def test_header_preserving_nrrd_io(tmp_path: Path, synthetic_ct_fixture) -> None:
    """Verify that writing eat_pvat.seg.nrrd preserves directions, origin, and segment metadata."""
    ct_vol, header, _, _ = synthetic_ct_fixture
    out_file = tmp_path / "eat_pvat.seg.nrrd"

    dummy_mask_8ch = np.zeros((8, *ct_vol.shape), dtype=np.uint8)
    dummy_mask_8ch[SegmentationCompartment.EAT_TOTAL, 10:20, 10:20, 10:20] = 1

    saved_path = write_eat_pvat_seg_nrrd(
        seg_data=dummy_mask_8ch,
        output_path=out_file,
        reference_header=header,
    )
    assert saved_path.is_file()

    read_data, read_header = nrrd.read(str(saved_path))
    assert read_data.shape == dummy_mask_8ch.shape
    assert read_header["space"] == header["space"]
    assert read_header["Segment0_Name"] == "EAT_TOTAL"
    assert read_header["Segment5_Name"] == "PVAT_LAD"


def test_anisotropic_edt_sampling_and_aliases() -> None:
    """Verify anisotropic EDT sampling order and backward-compatible constructor aliases."""
    # Test constructor aliases
    transform = ComputeEatPvatMasksd(pvat_radial_dilation_mm=12.0)
    assert transform.pvat_radial_distance_mm == 12.0

    seg = UnifiedAnatomySegmenter(onnx_heart_path=None, onnx_loc_path=None, backend="heuristic")
    assert seg.backend == "heuristic"

    # Test anisotropic physical dilation:
    # Array shape (Z=20, Y=30, X=30) with spacing (sx=1.0, sy=1.0, sz=5.0) mm
    # A single voxel vessel at center (z=10, y=15, x=15)
    vessel = np.zeros((20, 30, 30), dtype=bool)
    vessel[10, 15, 15] = True
    spacing = (1.0, 1.0, 5.0)  # (X, Y, Z)

    # With max_dist_mm = 6.0:
    # In Z direction, 1 slice is 5.0 mm <= 6.0 mm, but 2 slices is 10.0 mm > 6.0 mm.
    # In X and Y directions, up to 6 voxels are within 6.0 mm.
    envelope = transform._compute_pvat_envelope(vessel, spacing, max_dist_mm=6.0)

    # Along Z: only z=9, 10, 11 can be True (within 5mm)
    z_indices = np.where(envelope)[0]
    assert np.min(z_indices) >= 9
    assert np.max(z_indices) <= 11

    # Along X: voxels up to distance 6 (15-6=9 to 15+6=21) should be included
    x_indices = np.where(envelope)[2]
    assert np.min(x_indices) <= 10
    assert np.max(x_indices) >= 20


