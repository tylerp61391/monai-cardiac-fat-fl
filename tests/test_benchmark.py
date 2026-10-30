"""Unit and Integration Tests for Segmentation Benchmarking & DICE Engine."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from cardiac_fat_fl.segmentation_evaluator import (
    CohortBenchmarkReport,
    ParticipantBenchmarkResult,
    SegmentationBenchmarkEngine,
    compute_compartment_dice,
    compute_compartment_iou,
    compute_surface_distance_metrics,
    compute_volume_metrics,
)


def test_compute_compartment_dice_and_iou():
    """Verify Dice and IoU formulas on controlled boolean arrays."""
    # Perfect match
    a = np.ones((10, 10, 10), dtype=np.uint8)
    b = np.ones((10, 10, 10), dtype=np.uint8)
    assert pytest.approx(compute_compartment_dice(a, b), 1e-4) == 1.0
    assert pytest.approx(compute_compartment_iou(a, b), 1e-4) == 1.0

    # Completely disjoint
    c = np.zeros((10, 10, 10), dtype=np.uint8)
    c[:5, :, :] = 1
    d = np.zeros((10, 10, 10), dtype=np.uint8)
    d[5:, :, :] = 1
    assert pytest.approx(compute_compartment_dice(c, d), 1e-4) == 0.0
    assert pytest.approx(compute_compartment_iou(c, d), 1e-4) == 0.0

    # Both empty masks -> 1.0 agreement on true negative
    empty1 = np.zeros((10, 10, 10), dtype=np.uint8)
    empty2 = np.zeros((10, 10, 10), dtype=np.uint8)
    assert compute_compartment_dice(empty1, empty2) == 1.0
    assert compute_compartment_iou(empty1, empty2) == 1.0

    # 50% overlap: intersection=500, pred=1000, target=1000 -> dice = (2*500)/(1000+1000) = 0.5
    half_overlap = np.zeros((10, 10, 10), dtype=np.uint8)
    half_overlap[:5, :, :] = 1
    assert pytest.approx(compute_compartment_dice(a, half_overlap), 1e-4) == (2.0 * 500) / (1000 + 500)


def test_compute_volume_metrics():
    """Verify volumetric comparison calculations."""
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    pred[:5, :, :] = 1  # 500 voxels
    target = np.ones((10, 10, 10), dtype=np.uint8)  # 1000 voxels

    # spacing 2.0 x 2.0 x 2.0 mm -> 8 mm3 per voxel
    # pred: 500 * 8 / 1000 = 4.0 cm3
    # target: 1000 * 8 / 1000 = 8.0 cm3
    m = compute_volume_metrics(pred, target, spacing=(2.0, 2.0, 2.0))
    assert pytest.approx(m["pred_volume_cm3"], 1e-3) == 4.0
    assert pytest.approx(m["target_volume_cm3"], 1e-3) == 8.0
    assert pytest.approx(m["abs_volume_diff_cm3"], 1e-3) == 4.0
    assert pytest.approx(m["relative_volume_error_pct"], 1e-2) == -50.0


def test_compute_surface_distance_metrics():
    """Verify surface distance metrics (MASD and HD95)."""
    # Create two spheres or concentric cubes
    a = np.zeros((20, 20, 20), dtype=np.uint8)
    a[5:15, 5:15, 5:15] = 1

    b = np.zeros((20, 20, 20), dtype=np.uint8)
    b[4:16, 4:16, 4:16] = 1

    surf = compute_surface_distance_metrics(a, b, spacing=(1.0, 1.0, 1.0))
    assert surf["masd_mm"] is not None
    assert surf["hd95_mm"] is not None
    assert surf["masd_mm"] > 0.0
    assert surf["hd95_mm"] >= surf["masd_mm"]

    # Empty mask fallback returns None
    empty = np.zeros((20, 20, 20), dtype=np.uint8)
    empty_surf = compute_surface_distance_metrics(a, empty, spacing=(1.0, 1.0, 1.0))
    assert empty_surf["masd_mm"] is None
    assert empty_surf["hd95_mm"] is None


def test_segmentation_benchmark_engine_case_and_cohort(tmp_path: Path):
    """Verify SegmentationBenchmarkEngine evaluating 8-channel tensors and cohort reporting."""
    engine = SegmentationBenchmarkEngine(
        candidate_name="Method_A",
        reference_name="Method_B",
        compute_surface_distances=False,
    )

    # Create synthetic 8-channel tensors (8, 16, 32, 32)
    shape = (8, 16, 32, 32)
    t1 = np.zeros(shape, dtype=np.uint8)
    t2 = np.zeros(shape, dtype=np.uint8)

    # Compartment 0 (EAT_Total)
    t1[0, 4:12, 8:24, 8:24] = 1
    t2[0, 4:12, 8:24, 8:24] = 1  # perfect match for Ch 0

    # Compartment 1 (EAT_LA)
    t1[1, 2:6, 10:20, 10:20] = 1
    t2[1, 2:6, 10:20, 12:22] = 1  # partial match for Ch 1

    res = engine.evaluate_case_tensors(
        participant_id="CASE_001",
        pred_tensor=t1,
        target_tensor=t2,
        spacing=(1.0, 1.0, 1.0),
    )

    assert res.participant_id == "CASE_001"
    assert pytest.approx(res.compartments["EAT_TOTAL"].dice, 1e-4) == 1.0
    assert 0.0 < res.compartments["EAT_LA"].dice < 1.0
    assert "PVAT_Total" in res.compartments
    assert res.mean_dice > 0.0

    # Cohort evaluation
    case_pairs = [
        ("CASE_001", t1, t2, (1.0, 1.0, 1.0)),
        ("CASE_002", t1, t1, (1.0, 1.0, 1.0)),  # perfect match
    ]
    report = engine.evaluate_cohort(case_pairs)
    assert report.participant_count == 2
    assert report.mean_overall_dice > 0.0
    assert "EAT_TOTAL" in report.compartment_summary
    assert report.compartment_summary["EAT_TOTAL"]["mean_dice"] == 1.0

    # Export CSV & JSON
    csv_file = tmp_path / "benchmark.csv"
    json_file = tmp_path / "benchmark.json"

    engine.export_summary_csv(report, csv_file)
    engine.export_summary_json(report, json_file)

    assert csv_file.is_file()
    assert json_file.is_file()

    with open(json_file, "r") as f:
        data = json.load(f)
    assert data["participant_count"] == 2
    assert "cases" in data

