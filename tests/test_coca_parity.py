"""Unit and Integration Tests for End-to-End Architectural Parity (monai-cardiac-fat-fl vs. CardiacFatQuant)."""

from __future__ import annotations

import plistlib
from pathlib import Path

import numpy as np
import pytest

from cardiac_fat_fl.coca_parity import (
    align_eat_pvat_channels,
    compute_cardiacfatquant_eat_pvat_masks,
    rasterize_coca_calcium_xml,
    run_architectural_parity_audit,
)
from cardiac_fat_fl.dashboard import create_dashboard_app
from cardiac_fat_fl.measurement_schema import SegmentationCompartment


def test_align_eat_pvat_channels_from_cardiacfatquant_header() -> None:
    """Verify that CardiacFatQuant channel order (5:LCX, 6:RCA, 7:LAD) is reordered to canonical (5:LAD, 6:LCX, 7:RCA)."""
    raw = np.zeros((8, 4, 4, 4), dtype=np.uint8)
    for i in range(8):
        raw[i] = i + 10

    # CardiacFatQuant header with Segment{i}_Name
    cfq_header = {
        "Segment0_Name": "EAT",
        "Segment1_Name": "EAT LA",
        "Segment2_Name": "EAT LV",
        "Segment3_Name": "EAT RA",
        "Segment4_Name": "EAT RV",
        "Segment5_Name": "PVAT LCX",
        "Segment6_Name": "PVAT RCA",
        "Segment7_Name": "PVAT LAD",
    }
    aligned = align_eat_pvat_channels(raw, cfq_header)
    assert np.all(aligned[SegmentationCompartment.EAT_TOTAL] == 10)
    assert np.all(aligned[SegmentationCompartment.PVAT_LAD] == 17)  # Came from channel 7
    assert np.all(aligned[SegmentationCompartment.PVAT_LCX] == 15)  # Came from channel 5
    assert np.all(aligned[SegmentationCompartment.PVAT_RCA] == 16)  # Came from channel 6

    # CardiacFatQuant header with only Segment{i}_Layer (no Segment{i}_Name)
    cfq_layer_only_header = {f"Segment{i}_Layer": str(i) for i in range(8)}
    aligned2 = align_eat_pvat_channels(raw, cfq_layer_only_header)
    assert np.all(aligned2[SegmentationCompartment.PVAT_LAD] == 17)
    assert np.all(aligned2[SegmentationCompartment.PVAT_LCX] == 15)
    assert np.all(aligned2[SegmentationCompartment.PVAT_RCA] == 16)


def test_compute_cardiacfatquant_eat_pvat_masks_7_rules() -> None:
    """Verify the 7 morphological rules of CardiacFatQuant/generate_eat_pvat_masks.py."""
    shape = (40, 40, 40)
    spacing = (1.0, 1.0, 1.0)
    ct_arr = np.full(shape, -100.0, dtype=np.float32)  # Uniform adipose HU (-100 HU)

    pericardium = np.zeros(shape, dtype=bool)
    pericardium[5:35, 5:35, 5:35] = True

    la = np.zeros(shape, dtype=bool)
    la[15:25, 10:18, 10:18] = True

    lv = np.zeros(shape, dtype=bool)
    lv[15:25, 20:28, 12:18] = True

    myo = np.zeros(shape, dtype=bool)
    myo[15:25, 19:29, 11:19] = True

    ra = np.zeros(shape, dtype=bool)
    ra[15:25, 10:18, 22:30] = True

    rv = np.zeros(shape, dtype=bool)
    rv[15:25, 20:28, 22:30] = True

    aorta = np.zeros(shape, dtype=bool)
    aorta[25:32, 16:22, 16:22] = True

    masks_8ch = compute_cardiacfatquant_eat_pvat_masks(
        ct_arr=ct_arr,
        spacing=spacing,
        pericardium=pericardium,
        la=la,
        lv=lv,
        ra=ra,
        rv=rv,
        myocardium=myo,
        aorta=aorta,
    )
    assert masks_8ch.shape == (8, 40, 40, 40)
    # EAT_Total must be non-empty and contain all sub-compartments
    eat_total = masks_8ch[0] > 0.5
    assert np.any(eat_total)
    for c_idx in range(1, 8):
        sub_c = masks_8ch[c_idx] > 0.5
        assert np.all(sub_c <= eat_total)


def test_rasterize_coca_calcium_xml(tmp_path: Path) -> None:
    """Verify Stanford AIMI COCA calcium_xml polygon rasterization and Agatston integration."""
    xml_file = tmp_path / "001.xml"
    sample_plist = {
        "Images": [
            {
                "ImageIndex": 5,
                "ROIs": [
                    {
                        "Name": "Left Anterior Descending",
                        "Area": 0.12,
                        "Max": 450.0,
                        "NumberOfPixels": 9,
                        "Point_px": [
                            "(10.0, 10.0)",
                            "(14.0, 10.0)",
                            "(14.0, 14.0)",
                            "(10.0, 14.0)",
                        ],
                    }
                ],
            }
        ]
    }
    with open(xml_file, "wb") as f:
        plistlib.dump(sample_plist, f)

    ct_arr = np.zeros((10, 32, 32), dtype=np.float32)
    ct_arr[5, 10:15, 10:15] = 450.0  # > 400 HU -> Agatston weight 4
    res = rasterize_coca_calcium_xml(xml_file, ct_arr, spacing=(0.5, 0.5, 3.0))

    assert res["has_xml"] is True
    assert res["roi_count"] == 1
    assert res["total_voxels"] > 0
    assert res["total_agatston"] > 0.0
    assert res["vessel_agatston"]["Left Anterior Descending"] > 0.0


def test_run_architectural_parity_audit_and_dashboard_endpoint() -> None:
    """Verify 4-checkpoint parity audit runner and Flask /api/parity/run endpoint."""
    cohort_dir = Path("example_source_dir")
    if not cohort_dir.is_dir():
        pytest.skip("example_source_dir not present")

    report = run_architectural_parity_audit(
        cohort_dir=cohort_dir,
        mode="controlled_stage3",
        limit=2,
        compute_surface_distances=False,
    )
    assert report["participant_count"] >= 1
    assert math.isclose(report["mean_overall_dice"], 1.0, abs_tol=1e-5)
    assert report["checkpoint_a"]["status"] == "PASS"
    assert report["checkpoint_b"]["status"] == "PASS"
    assert report["checkpoint_c"]["status"] == "PASS"
    assert report["checkpoint_d"]["status"] == "PASS"
    assert report["checkpoint_d"]["eat_volume_mae_cm3"] == 0.0

    # Test Dashboard endpoint
    app = create_dashboard_app(cohort_dir=cohort_dir)
    app.config["TESTING"] = True
    client = app.test_client()
    res = client.post(
        "/api/parity/run",
        json={"mode": "controlled_stage3", "limit": 1, "compute_surface_distances": False},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["mean_overall_dice"] == 1.0
    assert "checkpoint_d" in data


import math

