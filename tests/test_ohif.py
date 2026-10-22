"""Unit tests for OHIF Viewer DICOM and DICOM-SEG Export."""

from __future__ import annotations

from pathlib import Path
import pydicom
import pytest

from cardiac_fat_fl.measurement_schema import SegmentationCompartment
from cardiac_fat_fl.ohif_exporter import (
    CT_IMAGE_STORAGE_UID,
    SEGMENTATION_STORAGE_UID,
    export_sample_to_ohif,
)


def test_export_sample_to_ohif_synthetic(synthetic_nrrd_scan: Path) -> None:
    """Test exporting a participant sample directory into OHIF DICOM and DICOM-SEG."""
    sample_dir = synthetic_nrrd_scan.parent
    
    # Create a synthetic 8-channel eat_pvat.seg.nrrd in sample_dir
    import nrrd
    import numpy as np
    seg_dummy = np.zeros((8, 64, 64, 64), dtype=np.uint8)
    seg_dummy[:, 20:44, 20:44, 20:44] = 1
    nrrd.write(str(sample_dir / "eat_pvat.seg.nrrd"), seg_dummy)

    out_dir = sample_dir / "ohif_export"

    result_dir = export_sample_to_ohif(
        sample_dir=sample_dir,
        output_dir=out_dir,
        summary_metrics={"EAT Total Volume (cm3)": 45.2, "EAT Total Mean Attenuation (HU)": -65.4},
    )

    assert result_dir.exists()
    assert (result_dir / "index.html").exists()

    ct_dir = result_dir / "ct"
    assert ct_dir.exists()
    ct_files = sorted(list(ct_dir.glob("*.dcm")))
    assert len(ct_files) > 0

    # Verify first CT slice
    ref_ct = pydicom.dcmread(str(ct_files[0]))
    assert ref_ct.Modality == "CT"
    assert ref_ct.SOPClassUID == CT_IMAGE_STORAGE_UID
    assert hasattr(ref_ct, "PixelSpacing")

    # Verify DICOM-SEG
    seg_file = result_dir / "eat_pvat_seg.dcm"
    assert seg_file.exists()
    seg_ds = pydicom.dcmread(str(seg_file))
    assert seg_ds.Modality == "SEG"
    assert seg_ds.SOPClassUID == SEGMENTATION_STORAGE_UID
    assert len(seg_ds.SegmentSequence) == 8

    # Verify 8 compartment labels
    expected_names = SegmentationCompartment.get_names()
    actual_labels = [s.SegmentLabel for s in seg_ds.SegmentSequence]
    assert actual_labels == expected_names

    # Verify OHIF v3.13 studies.json manifest
    studies_json = result_dir / "studies.json"
    assert studies_json.exists()
    import json
    manifest = json.loads(studies_json.read_text(encoding="utf-8"))
    assert manifest["version"] == "3.13.0"
    assert "https://ohif.org/release-notes/3p13" in manifest["documentation"]
    assert len(manifest["studies"]) > 0
    assert len(manifest["studies"][0]["series"]) == 2
    assert manifest["studies"][0]["series"][0]["Modality"] == "CT"
    assert manifest["studies"][0]["series"][1]["Modality"] == "SEG"
