"""DICOM Structured Reporting (DICOM-SR / TID 1500) Generator for Cardiac Fat Quantification.

Constructs DICOM Comprehensive 3D SR objects (SOP Class 1.2.840.10008.5.1.4.1.1.88.34)
containing the 19 canonical quantitative volumetric (cm3) and attenuation (HU) metrics,
referencing the patient CT series and eat_pvat_seg.dcm segmentation instances.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import Comprehensive3DSRStorage, ExplicitVRLittleEndian, generate_uid

from cardiac_fat_fl.measurement_schema import REPORTED_REGIONS, ReportedRegion


def _create_coded_concept(code_value: str, scheme: str, meaning: str) -> Dataset:
    d = Dataset()
    d.CodeValue = str(code_value)
    d.CodingSchemeDesignator = str(scheme)
    d.CodeMeaning = str(meaning)
    return d


# Mapping of regions to RadLex / SNOMED anatomical concepts
REGION_ANATOMY_CODES = {
    ReportedRegion.EAT_TOTAL: ("T-D3000", "SRT", "Epicardial adipose tissue"),
    ReportedRegion.EAT_LA: ("T-32300", "SRT", "Left atrium epicardial adipose tissue"),
    ReportedRegion.EAT_LV: ("T-32600", "SRT", "Left ventricle epicardial adipose tissue"),
    ReportedRegion.EAT_RA: ("T-32100", "SRT", "Right atrium epicardial adipose tissue"),
    ReportedRegion.EAT_RV: ("T-32400", "SRT", "Right ventricle epicardial adipose tissue"),
    ReportedRegion.PVAT_TOTAL: ("128100", "DCM", "Total pericoronary adipose tissue"),
    ReportedRegion.PVAT_LAD: ("T-43110", "SRT", "Left anterior descending artery pericoronary adipose tissue"),
    ReportedRegion.PVAT_LCX: ("T-43120", "SRT", "Left circumflex artery pericoronary adipose tissue"),
    ReportedRegion.PVAT_RCA: ("T-43200", "SRT", "Right coronary artery pericoronary adipose tissue"),
}


def create_cardiac_fat_sr(
    measurements: Dict[str, Any],
    source_ct_dcm: Optional[Union[str, Path, Dataset]] = None,
    seg_dcm: Optional[Union[str, Path, Dataset]] = None,
    series_number: int = 1002,
    instance_number: int = 1,
) -> Dataset:
    """Construct a standard DICOM TID 1500 Comprehensive 3D Structured Report (SR).

    Args:
        measurements: Dict containing metrics (e.g. 'EAT Total Volume (cm3)', 'EAT Total Mean Attenuation (HU)').
        source_ct_dcm: Reference CT DICOM dataset or path to inherit Patient/Study metadata.
        seg_dcm: Reference DICOM-SEG dataset or path to create ReferencedSOPSequence links.
        series_number: Series number for the SR series (default: 1002).
        instance_number: Instance number for the SR document (default: 1).

    Returns:
        pydicom.dataset.Dataset containing standard TID 1500 SR attributes.
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    # Read reference dataset if provided
    ref_ds: Optional[Dataset] = None
    if isinstance(source_ct_dcm, Dataset):
        ref_ds = source_ct_dcm
    elif source_ct_dcm and Path(source_ct_dcm).exists():
        try:
            ref_ds = pydicom.dcmread(str(source_ct_dcm), stop_before_pixels=True)
        except Exception:
            ref_ds = None

    # Read segmentation dataset if provided
    seg_ds: Optional[Dataset] = None
    if isinstance(seg_dcm, Dataset):
        seg_ds = seg_dcm
    elif seg_dcm and Path(seg_dcm).exists():
        try:
            seg_ds = pydicom.dcmread(str(seg_dcm), stop_before_pixels=True)
        except Exception:
            seg_ds = None

    # File Meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = Comprehensive3DSRStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Patient & Study attributes
    ds.PatientID = getattr(ref_ds, "PatientID", "ANONYMOUS")
    ds.PatientName = getattr(ref_ds, "PatientName", "ANONYMOUS")
    ds.PatientBirthDate = getattr(ref_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(ref_ds, "PatientSex", "")

    ds.StudyInstanceUID = getattr(ref_ds, "StudyInstanceUID", generate_uid())
    ds.StudyDate = getattr(ref_ds, "StudyDate", date_str)
    ds.StudyTime = getattr(ref_ds, "StudyTime", time_str)
    ds.StudyID = getattr(ref_ds, "StudyID", "1")
    ds.AccessionNumber = getattr(ref_ds, "AccessionNumber", "")
    ds.StudyDescription = getattr(ref_ds, "StudyDescription", "Cardiac CT Fat Quantification")

    # Series & Instance attributes
    ds.Modality = "SR"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.SeriesDate = date_str
    ds.SeriesTime = time_str
    ds.SeriesDescription = "Cardiac EAT/PVAT Quantitative Report (TID 1500)"

    ds.SOPClassUID = Comprehensive3DSRStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceCreationDate = date_str
    ds.InstanceCreationTime = time_str

    # Root Document / Concept (TID 1500 Imaging Measurement Report)
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = Sequence([
        _create_coded_concept("126000", "DCM", "Imaging Measurement Report")
    ])

    # Template Identification
    tmpl = Dataset()
    tmpl.MappingResource = "DCMR"
    tmpl.TemplateIdentifier = "1500"
    ds.ContentTemplateSequence = Sequence([tmpl])

    content_items = []

    # 1. Language of Content Item
    lang_item = Dataset()
    lang_item.RelationshipType = "HAS CONCEPT MOD"
    lang_item.ValueType = "CODE"
    lang_item.ConceptNameCodeSequence = Sequence([
        _create_coded_concept("121049", "DCM", "Language of Content Item and Descendants")
    ])
    lang_item.ConceptCodeSequence = Sequence([
        _create_coded_concept("eng", "RFC5646", "English")
    ])
    content_items.append(lang_item)

    # 2. Observer Context - Device / Software
    obs_item = Dataset()
    obs_item.RelationshipType = "HAS OBS CONTEXT"
    obs_item.ValueType = "CODE"
    obs_item.ConceptNameCodeSequence = Sequence([
        _create_coded_concept("121005", "DCM", "Observer Type")
    ])
    obs_item.ConceptCodeSequence = Sequence([
        _create_coded_concept("121007", "DCM", "Device")
    ])
    content_items.append(obs_item)

    algo_name_item = Dataset()
    algo_name_item.RelationshipType = "HAS OBS CONTEXT"
    algo_name_item.ValueType = "TEXT"
    algo_name_item.ConceptNameCodeSequence = Sequence([
        _create_coded_concept("121012", "DCM", "Device Observer Name")
    ])
    algo_name_item.TextValue = "monai-cardiac-fat-fl"
    content_items.append(algo_name_item)

    # 3. Measurement Groups Container
    meas_container = Dataset()
    meas_container.RelationshipType = "CONTAINS"
    meas_container.ValueType = "CONTAINER"
    meas_container.ContinuityOfContent = "SEPARATE"
    meas_container.ConceptNameCodeSequence = Sequence([
        _create_coded_concept("126010", "DCM", "Imaging Measurements")
    ])

    group_items = []

    for region in REPORTED_REGIONS:
        reg_name = region.value
        vol_key = f"{reg_name} Volume (cm3)"
        hu_key = f"{reg_name} Mean Attenuation (HU)"

        vol_val = measurements.get(vol_key)
        hu_val = measurements.get(hu_key)

        if vol_val is None:
            continue

        c_val, c_scheme, c_meaning = REGION_ANATOMY_CODES.get(region, ("99_EAT", "LOCAL", reg_name))

        group = Dataset()
        group.RelationshipType = "CONTAINS"
        group.ValueType = "CONTAINER"
        group.ContinuityOfContent = "SEPARATE"
        group.ConceptNameCodeSequence = Sequence([
            _create_coded_concept("125007", "DCM", "Measurement Group")
        ])

        inner_items = []

        # Tracking Identifier
        track_id = Dataset()
        track_id.RelationshipType = "HAS OBS CONTEXT"
        track_id.ValueType = "TEXT"
        track_id.ConceptNameCodeSequence = Sequence([
            _create_coded_concept("112039", "DCM", "Tracking Identifier")
        ])
        track_id.TextValue = reg_name
        inner_items.append(track_id)

        # Finding Site
        site_item = Dataset()
        site_item.RelationshipType = "HAS CONCEPT MOD"
        site_item.ValueType = "CODE"
        site_item.ConceptNameCodeSequence = Sequence([
            _create_coded_concept("363698007", "SCT", "Finding Site")
        ])
        site_item.ConceptCodeSequence = Sequence([
            _create_coded_concept(c_val, c_scheme, c_meaning)
        ])
        inner_items.append(site_item)

        # Volume measurement
        vol_item = Dataset()
        vol_item.RelationshipType = "CONTAINS"
        vol_item.ValueType = "NUM"
        vol_item.ConceptNameCodeSequence = Sequence([
            _create_coded_concept("G-D705", "SRT", "Volume")
        ])
        vol_meas = Dataset()
        vol_meas.NumericValue = f"{float(vol_val):.4f}"
        vol_meas.MeasurementUnitsCodeSequence = Sequence([
            _create_coded_concept("cm3", "UCUM", "cubic centimeter")
        ])
        vol_item.MeasuredValueSequence = Sequence([vol_meas])
        inner_items.append(vol_item)

        # Mean Attenuation (HU)
        if hu_val is not None and not (isinstance(hu_val, float) and np.isnan(hu_val)):
            try:
                hu_float = float(hu_val)
                if not np.isnan(hu_float):
                    hu_item = Dataset()
                    hu_item.RelationshipType = "CONTAINS"
                    hu_item.ValueType = "NUM"
                    hu_item.ConceptNameCodeSequence = Sequence([
                        _create_coded_concept("111031", "DCM", "Mean Hounsfield Unit")
                    ])
                    hu_meas = Dataset()
                    hu_meas.NumericValue = f"{hu_float:.2f}"
                    hu_meas.MeasurementUnitsCodeSequence = Sequence([
                        _create_coded_concept("[hnsf'U]", "UCUM", "Hounsfield unit")
                    ])
                    hu_item.MeasuredValueSequence = Sequence([hu_meas])
                    inner_items.append(hu_item)
            except (ValueError, TypeError):
                pass

        # Reference to Segmentation SOP Instance if present
        if seg_ds is not None and hasattr(seg_ds, "SOPInstanceUID"):
            ref_seg_item = Dataset()
            ref_seg_item.RelationshipType = "CONTAINS"
            ref_seg_item.ValueType = "IMAGE"
            ref_seg_item.ConceptNameCodeSequence = Sequence([
                _create_coded_concept("121191", "DCM", "Referenced Segment")
            ])
            ref_sop = Dataset()
            ref_sop.ReferencedSOPClassUID = getattr(seg_ds, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.66.4")
            ref_sop.ReferencedSOPInstanceUID = seg_ds.SOPInstanceUID
            ref_seg_item.ReferencedSOPSequence = Sequence([ref_sop])
            inner_items.append(ref_seg_item)

        group.ContentSequence = Sequence(inner_items)
        group_items.append(group)

    meas_container.ContentSequence = Sequence(group_items)
    content_items.append(meas_container)

    ds.ContentSequence = Sequence(content_items)
    return ds


def export_cardiac_fat_sr(
    measurements: Union[Dict[str, Any], Path, str, pd.DataFrame],
    output_path: Union[str, Path],
    source_ct_dcm: Optional[Union[str, Path, Dataset]] = None,
    seg_dcm: Optional[Union[str, Path, Dataset]] = None,
    participant_id: Optional[str] = None,
) -> Path:
    """Generate and serialize a standard DICOM-SR TID 1500 file to disk.

    Args:
        measurements: Dict of metrics, path to cardiac_fat_measurements.csv, or DataFrame.
        output_path: Destination file path (e.g. `path/to/eat_pvat_sr.dcm`).
        source_ct_dcm: Optional source CT DICOM slice or dataset for patient metadata.
        seg_dcm: Optional DICOM-SEG dataset or path for cross-referencing.
        participant_id: Required if measurements is a CSV file or multi-row DataFrame.

    Returns:
        Path to saved DICOM-SR file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    meas_dict: Dict[str, Any] = {}

    if isinstance(measurements, (str, Path)):
        p = Path(measurements)
        if p.is_file() and p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            if participant_id is not None:
                match = df[df["participant_id"].astype(str) == str(participant_id)]
                if not match.empty:
                    meas_dict = match.iloc[0].to_dict()
                else:
                    meas_dict = df.iloc[0].to_dict()
            else:
                meas_dict = df.iloc[0].to_dict()
    elif isinstance(measurements, pd.DataFrame):
        if participant_id is not None and "participant_id" in measurements.columns:
            match = measurements[measurements["participant_id"].astype(str) == str(participant_id)]
            meas_dict = match.iloc[0].to_dict() if not match.empty else measurements.iloc[0].to_dict()
        else:
            meas_dict = measurements.iloc[0].to_dict()
    elif isinstance(measurements, dict):
        meas_dict = measurements

    ds = create_cardiac_fat_sr(
        measurements=meas_dict,
        source_ct_dcm=source_ct_dcm,
        seg_dcm=seg_dcm,
    )

    pydicom.dcmwrite(str(out_file), ds, write_like_original=False)
    return out_file

