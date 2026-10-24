"""Unit tests for DICOM Structured Reporting (DICOM-SR TID 1500)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import Comprehensive3DSRStorage, ExplicitVRLittleEndian, generate_uid

from cardiac_fat_fl.dicom_sr import create_cardiac_fat_sr, export_cardiac_fat_sr
from cardiac_fat_fl.measurement_schema import REPORTED_REGIONS


def _create_mock_ct_dataset() -> Dataset:
    """Create a minimal mock CT dataset for metadata testing."""
    ds = Dataset()
    ds.PatientID = "TEST_PATIENT_123"
    ds.PatientName = "Smith^John"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.StudyDate = "20261024"
    ds.StudyTime = "120000"
    return ds


def test_create_cardiac_fat_sr_basic():
    meas = {
        "EAT Total Volume (cm3)": 85.5,
        "EAT Total Mean Attenuation (HU)": -68.2,
        "EAT LA Volume (cm3)": 15.2,
        "EAT LA Mean Attenuation (HU)": -70.1,
        "PVAT Total Volume (cm3)": 42.1,
        "PVAT Total Mean Attenuation (HU)": -67.8,
    }

    mock_ct = _create_mock_ct_dataset()
    sr = create_cardiac_fat_sr(meas, source_ct_dcm=mock_ct)

    # Basic DICOM attributes
    assert sr.Modality == "SR"
    assert sr.SOPClassUID == Comprehensive3DSRStorage
    assert sr.PatientID == "TEST_PATIENT_123"
    assert sr.PatientName == "Smith^John"
    assert sr.StudyInstanceUID == mock_ct.StudyInstanceUID

    # Root Container check
    assert sr.ValueType == "CONTAINER"
    assert sr.ContinuityOfContent == "SEPARATE"
    assert sr.ConceptNameCodeSequence[0].CodeValue == "126000"

    # Template check
    assert sr.ContentTemplateSequence[0].TemplateIdentifier == "1500"

    # Find Measurement Groups
    meas_container = [c for c in sr.ContentSequence if c.ValueType == "CONTAINER" and c.ConceptNameCodeSequence[0].CodeValue == "126010"]
    assert len(meas_container) == 1

    groups = meas_container[0].ContentSequence
    # We provided EAT Total, EAT LA, PVAT Total
    assert len(groups) == 3

    # Check EAT Total group
    eat_group = groups[0]
    num_items = [item for item in eat_group.ContentSequence if item.ValueType == "NUM"]
    assert len(num_items) == 2  # Volume and Mean HU
    vol_num = num_items[0]
    assert float(vol_num.MeasuredValueSequence[0].NumericValue) == pytest.approx(85.5, rel=1e-3)
    assert vol_num.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == "cm3"

    hu_num = num_items[1]
    assert float(hu_num.MeasuredValueSequence[0].NumericValue) == pytest.approx(-68.2, rel=1e-2)
    assert hu_num.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == "[hnsf'U]"


def test_create_cardiac_fat_sr_with_nan():
    # Test handling of NaN mean attenuation (empty compartment)
    meas = {
        "EAT Total Volume (cm3)": 0.0,
        "EAT Total Mean Attenuation (HU)": float("nan"),
    }
    sr = create_cardiac_fat_sr(meas)
    meas_container = [c for c in sr.ContentSequence if c.ValueType == "CONTAINER" and c.ConceptNameCodeSequence[0].CodeValue == "126010"][0]
    eat_group = meas_container.ContentSequence[0]
    num_items = [item for item in eat_group.ContentSequence if item.ValueType == "NUM"]
    # Should only contain volume (0.0 cm3), skipping NaN HU
    assert len(num_items) == 1
    assert float(num_items[0].MeasuredValueSequence[0].NumericValue) == 0.0


def test_export_cardiac_fat_sr_roundtrip(tmp_path: Path):
    meas_csv = tmp_path / "cardiac_fat_measurements.csv"
    row = {
        "participant_id": "0506",
        "EAT Total Volume (cm3)": 91.2,
        "EAT Total Mean Attenuation (HU)": -66.5,
        "PVAT Total Volume (cm3)": 35.4,
        "PVAT Total Mean Attenuation (HU)": -68.9,
    }
    df = pd.DataFrame([row])
    df.to_csv(meas_csv, index=False)

    sr_out = tmp_path / "eat_pvat_sr.dcm"
    saved = export_cardiac_fat_sr(
        measurements=meas_csv,
        output_path=sr_out,
        participant_id="0506",
    )

    assert saved.exists()
    assert saved == sr_out

    # Read back
    ds = pydicom.dcmread(str(saved))
    assert ds.Modality == "SR"
    assert ds.SOPClassUID == Comprehensive3DSRStorage

