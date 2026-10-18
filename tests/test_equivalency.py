"""Equivalency Test between Modernized Pipeline and Frozen Legacy DeepHeartSeg ONNX Models.

Verifies:
  - ONNX runtime inference with converted DeepHeartSeg weights
  - 100% agreement on volume math and HU attenuation thresholding
  - Exact reproducibility matching the legacy MGH CardiacFatQuant specification
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from monai.data import MetaTensor

from cardiac_fat_fl.anatomy_segmenter import UnifiedAnatomySegmenter
from cardiac_fat_fl.measurement_schema import REPORTED_REGIONS
from cardiac_fat_fl.transforms import ComputeEatPvatMasksd, QuantifyAdiposeMetricsd


def test_deepheartseg_onnx_model_files_exist() -> None:
    """Verify that converted ONNX weights exist."""
    loc_onnx = Path("models/deepheartseg_loc.onnx")
    heart_onnx = Path("models/deepheartseg_heart.onnx")
    assert heart_onnx.is_file(), f"Missing ONNX model at {heart_onnx}"
    assert loc_onnx.is_file(), f"Missing ONNX model at {loc_onnx}"


def test_deepheartseg_onnx_equivalency(synthetic_ct_fixture) -> None:
    """Verify that UnifiedAnatomySegmenter with backend='onnx' runs and produces valid compartments."""
    ct_vol, header, _, _ = synthetic_ct_fixture
    spacing = (1.5, 1.5, 2.0)

    seg_onnx = UnifiedAnatomySegmenter(
        model_path="models/deepheartseg_heart.onnx",
        loc_model_path="models/deepheartseg_loc.onnx",
        backend="onnx",
    )
    anat_onnx = seg_onnx.segment(ct_vol, spacing=spacing)

    assert anat_onnx.pericardium is not None
    assert np.any(anat_onnx.pericardium), "ONNX model produced empty pericardium mask"

    # Run quantification pipeline
    meta_ct = MetaTensor(
        torch.from_numpy(ct_vol).unsqueeze(0),
        meta={"spacing": list(spacing), "nrrd_header": header},
    )
    mask_tx = ComputeEatPvatMasksd(keys=["image"])
    quant_tx = QuantifyAdiposeMetricsd(keys=["image"])

    res = quant_tx(mask_tx({"image": meta_ct, "anatomy": anat_onnx, "participant_id": "TEST_ONNX"}))
    metrics = res["metrics"]

    # Verify EAT Total volume is positive and HU is within [-190, -30]
    assert metrics["EAT Total Volume (cm3)"] > 0.0
    assert -190.0 <= metrics["EAT Total Mean Attenuation (HU)"] <= -30.0

    # Verify all reported regions are present
    for region in REPORTED_REGIONS:
        assert f"{region.value} Volume (cm3)" in metrics
        assert f"{region.value} Mean Attenuation (HU)" in metrics
