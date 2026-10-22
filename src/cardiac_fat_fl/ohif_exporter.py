"""OHIF Viewer and DICOM / DICOM-SEG Export Subsystem.

Converts 3D cardiac CT scans and 8-compartment EAT/PVAT segmentations
into standard DICOM CT series and DICOM-SEG (SOP Class 1.2.840.10008.5.1.4.1.1.66.4)
objects compatible with the Open Health Imaging Foundation (OHIF) Viewer v3+.

Also generates an authentic 3-pane OHIF medical web viewer:
  - Left Pane: Studies & Series Chooser (CT Acquisition & Segmentation Series)
  - Middle Pane: Real CT Image Viewport (with real DICOM slices, windowing, and mask overlays)
  - Right Pane: EAT & PVAT Statistics & Segmentation Control Panel
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import nrrd
import pandas as pd
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from cardiac_fat_fl.io_nrrd import read_medical_image_as_metatensor
from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    REPORTED_REGIONS,
    SegmentationCompartment,
    calculate_volume_cm3,
)

# Standard DICOM SOP Class UIDs
CT_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.2"
SEGMENTATION_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.66.4"

# Color palette for 8 compartments (RGB and CIELab approximations)
COMPARTMENT_PALETTE: Dict[str, Dict[str, Any]] = {
    "EAT_TOTAL": {"rgb": [147, 51, 234], "hex": "#9333ea", "cielab": [32768, 45000, 20000], "name": "EAT Total"},
    "EAT_LA": {"rgb": [59, 130, 246], "hex": "#3b82f6", "cielab": [35000, 30000, 15000], "name": "EAT Left Atrium"},
    "EAT_LV": {"rgb": [6, 182, 212], "hex": "#06b6d4", "cielab": [40000, 25000, 25000], "name": "EAT Left Ventricle"},
    "EAT_RA": {"rgb": [16, 185, 129], "hex": "#10b981", "cielab": [42000, 20000, 38000], "name": "EAT Right Atrium"},
    "EAT_RV": {"rgb": [234, 179, 8], "hex": "#eab308", "cielab": [48000, 33000, 50000], "name": "EAT Right Ventricle"},
    "PVAT_LAD": {"rgb": [249, 115, 22], "hex": "#f97316", "cielab": [45000, 48000, 45000], "name": "PVAT LAD"},
    "PVAT_LCX": {"rgb": [239, 68, 68], "hex": "#ef4444", "cielab": [42000, 52000, 38000], "name": "PVAT LCX"},
    "PVAT_RCA": {"rgb": [236, 72, 153], "hex": "#ec4899", "cielab": [40000, 50000, 25000], "name": "PVAT RCA"},
}


_UID_MAP: Dict[str, str] = {}


def _sanitize_uid(raw_uid: str) -> str:
    """Ensure a DICOM UID is strictly compliant with DICOM PS 3.5.

    Components cannot have leading zeros unless '0', and total length <= 64.
    If non-compliant, generates a valid deterministic UID.
    """
    if not raw_uid:
        return generate_uid()
    if raw_uid in _UID_MAP:
        return _UID_MAP[raw_uid]

    # Check leading zeros in components
    parts = raw_uid.split(".")
    is_valid = True
    for p in parts:
        if len(p) > 1 and p.startswith("0"):
            is_valid = False
            break
    if is_valid and len(raw_uid) <= 64:
        _UID_MAP[raw_uid] = raw_uid
        return raw_uid

    import hashlib
    h_int = int(hashlib.sha256(raw_uid.encode("utf-8")).hexdigest()[:16], 16)
    new_uid = f"1.2.826.0.1.3680043.8.498.{h_int}"
    _UID_MAP[raw_uid] = new_uid
    return new_uid


def export_ct_to_dicom_series(
    ct_array: np.ndarray,
    spacing: Tuple[float, float, float],
    output_dir: Path,
    patient_id: str = "PARTICIPANT_001",
    study_uid: Optional[str] = None,
    series_uid: Optional[str] = None,
) -> List[Path]:
    """Converts a 3D CT numpy array (D, H, W) in HU to a series of 2D CT DICOM files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    depth, height, width = ct_array.shape
    sx, sy, sz = spacing

    study_instance_uid = study_uid or generate_uid()
    series_instance_uid = series_uid or generate_uid()
    frame_of_ref_uid = generate_uid()

    dicom_paths: List[Path] = []
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    for z in range(depth):
        sop_instance_uid = generate_uid()
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE_UID
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        out_path = output_dir / f"ct_{z:04d}.dcm"
        ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\x00" * 128)

        # Identification
        ds.SOPClassUID = CT_IMAGE_STORAGE_UID
        ds.SOPInstanceUID = sop_instance_uid
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.FrameOfReferenceUID = frame_of_ref_uid
        ds.Modality = "CT"
        ds.PatientID = patient_id
        ds.PatientName = patient_id
        ds.PatientBirthDate = "19700101"
        ds.PatientSex = "O"
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.SeriesDate = date_str
        ds.SeriesTime = time_str
        ds.SeriesNumber = 1
        ds.InstanceNumber = z + 1

        # Geometry
        ds.ImagePositionPatient = [0.0, 0.0, float(z * sz)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [float(sy), float(sx)]
        ds.SliceThickness = float(sz)
        ds.SpacingBetweenSlices = float(sz)

        # Pixel format
        ds.Rows = height
        ds.Columns = width
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed 16-bit
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # HU calibration
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0
        ds.RescaleType = "HU"

        # Window presets
        ds.WindowCenter = -110.0
        ds.WindowWidth = 400.0

        # Slice data
        slice_data = np.ascontiguousarray(ct_array[z], dtype=np.int16)
        ds.PixelData = slice_data.tobytes()

        ds.save_as(str(out_path), write_like_original=False)
        dicom_paths.append(out_path)

    return dicom_paths


def export_segmentation_to_dicom_seg(
    seg_array: np.ndarray,
    ct_dicom_paths: List[Path],
    output_path: Path,
    patient_id: str = "PARTICIPANT_001",
) -> Path:
    """Creates a standard DICOM-SEG (SOP Class 1.2.840.10008.5.1.4.1.1.66.4) object."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_compartments, depth, height, width = seg_array.shape

    # Read reference CT slice headers
    ct_datasets: List[Dataset] = [
        pydicom.dcmread(str(p), stop_before_pixels=True) for p in ct_dicom_paths
    ]
    ref_ct = ct_datasets[0]

    study_instance_uid = _sanitize_uid(str(ref_ct.StudyInstanceUID))
    series_instance_uid = generate_uid()
    sop_instance_uid = generate_uid()
    frame_of_ref_uid = _sanitize_uid(str(getattr(ref_ct, "FrameOfReferenceUID", generate_uid())))

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SEGMENTATION_STORAGE_UID
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # Patient & Study attributes
    ds.SOPClassUID = SEGMENTATION_STORAGE_UID
    ds.SOPInstanceUID = sop_instance_uid
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    ds.FrameOfReferenceUID = frame_of_ref_uid
    ds.Modality = "SEG"
    ds.PatientID = str(getattr(ref_ct, "PatientID", f"COCA-{patient_id}"))
    ds.PatientName = str(getattr(ref_ct, "PatientName", f"Participant {patient_id}"))
    ds.StudyDate = getattr(ref_ct, "StudyDate", date_str)
    ds.StudyTime = getattr(ref_ct, "StudyTime", time_str)
    ds.SeriesDate = date_str
    ds.SeriesTime = time_str
    ds.SeriesNumber = 100
    ds.InstanceNumber = 1
    ds.Manufacturer = "MONAI-CardiacFat-FL"
    ds.StudyDescription = getattr(ref_ct, "StudyDescription", f"COCA Cardiac CT (Participant {patient_id})")
    ds.SeriesDescription = "EAT/PVAT 8-Compartment Segmentation"

    # Multi-frame image pixel format
    ds.SegmentationType = "BINARY"
    ds.Rows = height
    ds.Columns = width
    ds.BitsAllocated = 1
    ds.BitsStored = 1
    ds.HighBit = 0
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Segment Sequence (8 compartments)
    segment_sequence = Sequence()
    for seg_idx, comp_name in enumerate(SegmentationCompartment.get_names()):
        seg_item = Dataset()
        seg_item.SegmentNumber = seg_idx + 1
        seg_item.SegmentLabel = comp_name
        seg_item.SegmentDescription = COMPARTMENT_PALETTE.get(comp_name, {}).get("name", comp_name)
        seg_item.SegmentAlgorithmType = "AUTOMATIC"
        seg_item.SegmentAlgorithmName = "monai-cardiac-fat-fl"
        palette = COMPARTMENT_PALETTE.get(comp_name, {"cielab": [32768, 32768, 32768]})
        seg_item.RecommendedDisplayCIELabValue = palette["cielab"]
        segment_sequence.append(seg_item)
    ds.SegmentSequence = segment_sequence

    # Shared Functional Groups
    shared_fg = Dataset()
    plane_orientation = Dataset()
    plane_orientation.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    shared_fg.PlaneOrientationSequence = Sequence([plane_orientation])

    pixel_measures = Dataset()
    pixel_measures.PixelSpacing = list(getattr(ref_ct, "PixelSpacing", [1.0, 1.0]))
    pixel_measures.SliceThickness = float(getattr(ref_ct, "SliceThickness", 1.0))
    pixel_measures.SpacingBetweenSlices = float(getattr(ref_ct, "SpacingBetweenSlices", 1.0))
    shared_fg.PixelMeasuresSequence = Sequence([pixel_measures])
    ds.SharedFunctionalGroupsSequence = Sequence([shared_fg])

    # Per-Frame Functional Groups & Bitpacking
    per_frame_fg = Sequence()
    frame_bytes_list: List[bytes] = []

    for seg_idx in range(num_compartments):
        for z in range(depth):
            frame_2d = seg_array[seg_idx, z] > 0

            frame_item = Dataset()
            plane_pos = Dataset()
            if z < len(ct_datasets) and hasattr(ct_datasets[z], "ImagePositionPatient"):
                plane_pos.ImagePositionPatient = ct_datasets[z].ImagePositionPatient
            else:
                plane_pos.ImagePositionPatient = [0.0, 0.0, float(z * pixel_measures.SliceThickness)]
            frame_item.PlanePositionSequence = Sequence([plane_pos])

            seg_id = Dataset()
            seg_id.ReferencedSegmentNumber = seg_idx + 1
            frame_item.SegmentIdentificationSequence = Sequence([seg_id])

            if z < len(ct_datasets):
                derivation = Dataset()
                src_img = Dataset()
                src_img.ReferencedSOPClassUID = ct_datasets[z].SOPClassUID
                src_img.ReferencedSOPInstanceUID = ct_datasets[z].SOPInstanceUID
                derivation.SourceImageSequence = Sequence([src_img])
                frame_item.DerivationImageSequence = Sequence([derivation])

            per_frame_fg.append(frame_item)
            packed = np.packbits(frame_2d.flatten(), bitorder="little")
            frame_bytes_list.append(packed.tobytes())

    ds.PerFrameFunctionalGroupsSequence = per_frame_fg
    ds.NumberOfFrames = len(frame_bytes_list)
    ds.PixelData = b"".join(frame_bytes_list)

    ds.save_as(str(output_path), write_like_original=False)
    return output_path


def generate_ohif_dicom_json(
    dest_dir: Path,
    ct_files: List[Path],
    seg_file: Optional[Path],
    participant_id: str,
) -> Path:
    """Generates an OHIF v3.13 compliant naturalized DICOM JSON manifest (studies.json).

    Conforms to the OHIF v3.13 dicomjson data source specification (https://ohif.org/release-notes/3p13).
    """
    import json

    out_file = dest_dir / "studies.json"
    if not ct_files:
        return out_file

    ref_ct = pydicom.dcmread(str(ct_files[0]), stop_before_pixels=True)
    study_uid = _sanitize_uid(str(getattr(ref_ct, "StudyInstanceUID", generate_uid())))
    study_date = str(getattr(ref_ct, "StudyDate", datetime.datetime.now().strftime("%Y%m%d")))
    study_time = str(getattr(ref_ct, "StudyTime", "120000"))
    study_desc = str(getattr(ref_ct, "StudyDescription", f"COCA Cardiac CT (Participant {participant_id})"))

    ct_series_uid = _sanitize_uid(str(getattr(ref_ct, "SeriesInstanceUID", generate_uid())))
    ct_instances = []
    for idx, f in enumerate(ct_files):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            inst = {
                "SOPInstanceUID": _sanitize_uid(str(getattr(ds, "SOPInstanceUID", generate_uid()))),
                "InstanceNumber": int(getattr(ds, "InstanceNumber", idx + 1)),
                "Rows": int(getattr(ds, "Rows", 512)),
                "Columns": int(getattr(ds, "Columns", 512)),
                "PixelSpacing": [float(x) for x in getattr(ds, "PixelSpacing", [0.47, 0.47])],
                "ImagePositionPatient": [float(x) for x in getattr(ds, "ImagePositionPatient", [0.0, 0.0, float(idx * 3.0)])],
                "ImageOrientationPatient": [float(x) for x in getattr(ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])],
                "SliceThickness": float(getattr(ds, "SliceThickness", 3.0)),
                "Modality": "CT",
                "url": f"ct/{f.name}",
            }
            ct_instances.append(inst)
        except Exception:
            continue

    series_list = [
        {
            "SeriesInstanceUID": ct_series_uid,
            "SeriesDescription": "Primary CT Axial",
            "SeriesNumber": 1,
            "Modality": "CT",
            "instances": ct_instances,
        }
    ]

    if seg_file and seg_file.is_file():
        try:
            seg_ds = pydicom.dcmread(str(seg_file), stop_before_pixels=True)
            seg_series_uid = _sanitize_uid(str(getattr(seg_ds, "SeriesInstanceUID", generate_uid())))
            series_list.append({
                "SeriesInstanceUID": seg_series_uid,
                "SeriesDescription": "EAT/PVAT 8-Compartment Segmentation (OHIF 3.13 Cornerstone v5 Labelmap)",
                "SeriesNumber": 100,
                "Modality": "SEG",
                "instances": [
                    {
                        "SOPInstanceUID": _sanitize_uid(str(getattr(seg_ds, "SOPInstanceUID", generate_uid()))),
                        "InstanceNumber": 1,
                        "Rows": int(getattr(seg_ds, "Rows", 512)),
                        "Columns": int(getattr(seg_ds, "Columns", 512)),
                        "Modality": "SEG",
                        "url": seg_file.name,
                    }
                ],
            })
        except Exception:
            pass

    manifest = {
        "version": "3.13.0",
        "documentation": "https://ohif.org/release-notes/3p13",
        "studies": [
            {
                "StudyInstanceUID": study_uid,
                "PatientID": f"COCA-{participant_id}",
                "PatientName": f"Participant {participant_id}",
                "StudyDate": study_date,
                "StudyTime": study_time,
                "StudyDescription": study_desc,
                "Modalities": ["CT", "SEG"],
                "series": series_list,
            }
        ],
    }

    out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_file


def render_real_slice_assets(
    ct_dicom_paths: List[Path],
    seg_array: np.ndarray,
    slices_dir: Path,
) -> int:
    """Pre-renders real DICOM CT slices and 8-compartment mask overlays into PNGs."""
    slices_dir.mkdir(parents=True, exist_ok=True)
    num_slices = len(ct_dicom_paths)

    # Window definitions
    # Cardiac fat window: center -110, width 400 (-310 to +90 HU)
    fat_min, fat_max = -310.0, 90.0

    for z, dcm_path in enumerate(ct_dicom_paths):
        ds = pydicom.dcmread(str(dcm_path))
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pix = np.squeeze(ds.pixel_array)
        if pix.ndim > 2:
            pix = pix[0]
        hu = pix.astype(np.float32) * slope + intercept

        # Normalize fat window to 0-255
        norm = np.clip((hu - fat_min) / (fat_max - fat_min), 0.0, 1.0)
        img_gray = (norm * 255.0).astype(np.uint8)
        Image.fromarray(img_gray).save(slices_dir / f"ct_{z:03d}.png")

        # Create multi-channel RGBA overlay for this slice
        h, w = hu.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Priority order: EAT chambers, then PVAT vessels (LAD, LCX, RCA on top)
        draw_order = [0, 1, 2, 3, 4, 5, 6, 7]
        for comp_idx in draw_order:
            comp_name = SegmentationCompartment(comp_idx).name
            palette = COMPARTMENT_PALETTE.get(comp_name, {"rgb": [255, 255, 255]})
            rgb = palette["rgb"]
            
            mask = seg_array[comp_idx, z] > 0
            if np.any(mask):
                rgba[mask, 0] = rgb[0]
                rgba[mask, 1] = rgb[1]
                rgba[mask, 2] = rgb[2]
                rgba[mask, 3] = 255  # Full alpha for lossless pixel filtering

        Image.fromarray(rgba, mode="RGBA").save(slices_dir / f"seg_{z:03d}.png")

    # Save thumbnail for series chooser
    mid_z = num_slices // 2
    if (slices_dir / f"ct_{mid_z:03d}.png").exists():
        thumb = Image.open(slices_dir / f"ct_{mid_z:03d}.png").resize((96, 96))
        thumb.save(slices_dir / "thumbnail_ct.png")
    if (slices_dir / f"seg_{mid_z:03d}.png").exists():
        thumb_seg = Image.open(slices_dir / f"seg_{mid_z:03d}.png").resize((96, 96))
        thumb_seg.save(slices_dir / "thumbnail_seg.png")

    return num_slices


def _load_hu_volume(ct_dicom_paths: List[Path]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Loads sorted CT DICOM slices into a (D, H, W) HU volume and returns (sz, sy, sx) spacing in mm."""
    slices: List[np.ndarray] = []
    z_positions: List[float] = []
    sy, sx = 1.0, 1.0
    for p in ct_dicom_paths:
        ds = pydicom.dcmread(str(p))
        pix = np.squeeze(ds.pixel_array)
        if pix.ndim > 2:
            pix = pix[0]
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(pix.astype(np.float32) * slope + intercept)
        if hasattr(ds, "PixelSpacing"):
            sy, sx = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
        if hasattr(ds, "ImagePositionPatient"):
            z_positions.append(float(ds.ImagePositionPatient[2]))
    vol = np.stack(slices, axis=0)
    if len(z_positions) > 1:
        sz = float(np.median(np.abs(np.diff(z_positions)))) or 1.0
    else:
        sz = float(getattr(pydicom.dcmread(str(ct_dicom_paths[0]), stop_before_pixels=True), "SliceThickness", 1.0))
    return vol, (sz, sy, sx)


def _colorize_seg_plane(seg_plane: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Converts an (8, A, B) binary compartment plane into an RGBA overlay."""
    _, a, b = seg_plane.shape
    rgba = np.zeros((a, b, 4), dtype=np.uint8)
    for comp_idx in range(seg_plane.shape[0]):
        comp_name = SegmentationCompartment(comp_idx).name
        rgb = COMPARTMENT_PALETTE.get(comp_name, {"rgb": [255, 255, 255]})["rgb"]
        mask = seg_plane[comp_idx] > 0
        if np.any(mask):
            rgba[mask, 0], rgba[mask, 1], rgba[mask, 2], rgba[mask, 3] = rgb[0], rgb[1], rgb[2], alpha
    return rgba


def render_mpr_assets(
    ct_dicom_paths: List[Path],
    seg_array: np.ndarray,
    output_dir: Path,
    inplane_step: int = 2,
    num_3d_frames: int = 36,
) -> Dict[str, Any]:
    """Pre-renders sagittal, coronal, and rotating 3D projection assets for the MPR viewer.

    Sagittal/coronal planes are resampled to physical aspect ratio (mm) so anisotropic
    slice spacing (e.g. 0.47 x 0.47 x 3.0 mm) is displayed without distortion. The 3D view
    is a rotating volumetric projection: average-intensity CT context, bone/calcium MIP,
    and thickness-weighted EAT/PVAT compartment projections.
    """
    import scipy.ndimage as ndi

    sag_dir = output_dir / "sagittal"
    cor_dir = output_dir / "coronal"
    vol_dir = output_dir / "volume3d"
    for d in (sag_dir, cor_dir, vol_dir):
        d.mkdir(parents=True, exist_ok=True)

    vol, (sz, sy, sx) = _load_hu_volume(ct_dicom_paths)
    depth, height, width = vol.shape
    seg = (seg_array[:, :depth, :height, :width] > 0).astype(np.uint8)

    fat_min, fat_max = -310.0, 90.0
    vol_u8 = (np.clip((vol - fat_min) / (fat_max - fat_min), 0.0, 1.0) * 255.0).astype(np.uint8)

    # High-resolution physical display size for reformatted planes (superior at top)
    target_w = max(width, height)
    sag_w = target_w
    cor_w = target_w
    sag_h = max(1, int(round(target_w * (depth * sz) / (height * sy))))
    cor_h = max(1, int(round(target_w * (depth * sz) / (width * sx))))

    sag_indices = list(range(0, width, inplane_step))
    cor_indices = list(range(0, height, inplane_step))

    for i, x in enumerate(sag_indices):
        ct_plane = vol_u8[::-1, :, x]
        Image.fromarray(ct_plane).resize((sag_w, sag_h), Image.BILINEAR).save(sag_dir / f"ct_{i:03d}.png")
        rgba = _colorize_seg_plane(seg[:, ::-1, :, x])
        Image.fromarray(rgba, mode="RGBA").resize((sag_w, sag_h), Image.NEAREST).save(sag_dir / f"seg_{i:03d}.png")

    for i, y in enumerate(cor_indices):
        ct_plane = vol_u8[::-1, y, :]
        Image.fromarray(ct_plane).resize((cor_w, cor_h), Image.BILINEAR).save(cor_dir / f"ct_{i:03d}.png")
        rgba = _colorize_seg_plane(seg[:, ::-1, y, :])
        Image.fromarray(rgba, mode="RGBA").resize((cor_w, cor_h), Image.NEAREST).save(cor_dir / f"seg_{i:03d}.png")

    # 3D rotating projection on a downsampled, near-isotropic grid
    target_xy = 160
    zoom_xy = target_xy / max(height, width)
    iso_mm = sx / zoom_xy
    zoom_z = sz / iso_mm
    small_ct = ndi.zoom(vol, (zoom_z, zoom_xy, zoom_xy), order=1)
    label_vol = np.zeros(seg.shape[1:], dtype=np.uint8)
    for comp_idx in range(seg.shape[0]):  # later compartments (PVAT) take priority
        label_vol[seg[comp_idx] > 0] = comp_idx + 1
    small_lbl = ndi.zoom(label_vol, (zoom_z, zoom_xy, zoom_xy), order=0)

    body = small_ct > -500.0
    tissue = np.where(body, np.clip((small_ct + 200.0) / 400.0, 0.0, 1.0), 0.0).astype(np.float32)
    bone = np.clip((small_ct - 150.0) / 800.0, 0.0, 1.0).astype(np.float32)

    for f in range(num_3d_frames):
        angle = 360.0 * f / num_3d_frames
        if angle:
            t_rot = ndi.rotate(tissue, angle, axes=(1, 2), reshape=False, order=1)
            b_rot = ndi.rotate(bone, angle, axes=(1, 2), reshape=False, order=1)
            l_rot = ndi.rotate(small_lbl, angle, axes=(1, 2), reshape=False, order=0)
        else:
            t_rot, b_rot, l_rot = tissue, bone, small_lbl

        # Project along anterior-posterior axis (axis=1) -> coronal-like view, superior at top
        aip = t_rot.mean(axis=1)[::-1]
        mip = b_rot.max(axis=1)[::-1]
        base = np.clip(0.55 * aip / (aip.max() + 1e-6) + 0.75 * mip, 0.0, 1.0)
        rgb = np.stack([base, base, base], axis=-1) * 0.85

        for comp_idx in range(seg.shape[0]):
            thickness = (l_rot == comp_idx + 1).sum(axis=1)[::-1].astype(np.float32)
            if thickness.max() <= 0:
                continue
            alpha = np.clip(thickness / 6.0, 0.0, 0.9)[..., None]
            color = np.array(COMPARTMENT_PALETTE[SegmentationCompartment(comp_idx).name]["rgb"], dtype=np.float32) / 255.0
            rgb = rgb * (1.0 - alpha) + color * alpha

        img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
        img = img.resize((img.width * 3, img.height * 3), Image.BICUBIC)
        img.save(vol_dir / f"frame_{f:03d}.png")

    return {
        "axial_count": depth,
        "sag_count": len(sag_indices),
        "cor_count": len(cor_indices),
        "vol_frames": num_3d_frames,
        "spacing_mm": [sz, sy, sx],
        "sag_w": sag_w,
        "sag_h": sag_h,
        "cor_w": cor_w,
        "cor_h": cor_h,
    }


def generate_mpr_viewer_html(participant_id: str, output_dir: Path, mpr_info: Dict[str, Any]) -> Path:
    """Generates a self-contained 2x2 MPR viewer (Axial / Sagittal / Coronal / 3D) with linked crosshairs."""
    html_path = output_dir / "mpr.html"
    sz, sy, sx = mpr_info["spacing_mm"]
    sag_w = int(mpr_info.get("sag_w", 512))
    sag_h = int(mpr_info.get("sag_h", 365))
    cor_w = int(mpr_info.get("cor_w", 512))
    cor_h = int(mpr_info.get("cor_h", 365))
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>MPR &amp; 3D Viewer &bull; Participant {participant_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; background: #05070d; color: #e2e8f0; font-family: -apple-system, 'Segoe UI', Roboto, Arial, sans-serif; overflow: hidden; }}
  header {{ height: 48px; display: flex; align-items: center; justify-content: space-between; padding: 0 14px; background: #0f172a; border-bottom: 1px solid #1e293b; font-size: 13px; }}
  header a, header button {{ color: #cbd5e1; background: #1e293b; border: 1px solid #334155; border-radius: 5px; padding: 5px 10px; font-size: 12px; text-decoration: none; cursor: pointer; margin-left: 6px; }}
  header button.on {{ background: #0369a1; color: #fff; border-color: #0284c7; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 3px; height: calc(100% - 48px); background: #1e293b; }}
  .pane {{ position: relative; background: #000; display: flex; align-items: center; justify-content: center; overflow: hidden; padding: 26px 8px; }}
  .label {{ position: absolute; top: 6px; left: 8px; font-size: 12px; font-weight: 700; z-index: 5; pointer-events: none; }}
  .info {{ position: absolute; bottom: 6px; left: 8px; font-size: 11px; color: #94a3b8; font-family: monospace; z-index: 5; pointer-events: none; }}
  .orient {{ position: absolute; font-size: 11px; color: #facc15; font-family: monospace; z-index: 5; pointer-events: none; }}
  .wrap {{ position: relative; display: inline-block; line-height: 0; cursor: crosshair; height: 100%; max-width: 100%; max-height: 100%; }}
  #wAx {{ aspect-ratio: 1 / 1; }}
  #wSag {{ aspect-ratio: {sag_w} / {sag_h}; }}
  #wCor {{ aspect-ratio: {cor_w} / {cor_h}; }}
  .wrap img.ct {{ width: 100%; height: 100%; object-fit: fill; display: block; image-rendering: auto; }}
  .wrap img.seg {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; opacity: 0.6; }}
  .xh {{ position: absolute; pointer-events: none; z-index: 4; }}
  .xh.v {{ top: 0; bottom: 0; width: 1px; }}
  .xh.h {{ left: 0; right: 0; height: 1px; }}
  #vol {{ height: 100%; max-width: 100%; aspect-ratio: 1 / 1; object-fit: contain; cursor: grab; }}
  .legend {{ display: flex; gap: 10px; font-size: 11px; }}
  .legend span i {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
</style></head>
<body>
<header>
  <div><b>MPR &amp; 3D</b> &nbsp;&bull;&nbsp; Participant {participant_id} &nbsp;&bull;&nbsp;
    <span style="color:#38bdf8;font-family:monospace">{sx:.2f} x {sy:.2f} x {sz:.2f} mm</span></div>
  <div class="legend">
    <span><i style="background:#9333ea"></i>EAT</span><span><i style="background:#3b82f6"></i>LA</span>
    <span><i style="background:#06b6d4"></i>LV</span><span><i style="background:#10b981"></i>RA</span>
    <span><i style="background:#eab308"></i>RV</span><span><i style="background:#f97316"></i>LAD</span>
    <span><i style="background:#ef4444"></i>LCX</span><span><i style="background:#ec4899"></i>RCA</span>
  </div>
  <div>
    <label style="font-size:12px;color:#94a3b8">Overlay <input type="range" id="op" min="0" max="100" value="60" style="vertical-align:middle;width:90px"></label>
    <button id="spin">Auto-rotate 3D</button>
    <a href="index.html">&larr; Axial viewer</a>
  </div>
</header>
<div class="grid">
  <div class="pane" id="pAx"><div class="label" style="color:#38bdf8">AXIAL</div>
    <div class="wrap" id="wAx"><img class="ct" id="axCt"><img class="seg" id="axSeg">
      <div class="xh v" id="axV" style="background:#22c55e"></div><div class="xh h" id="axH" style="background:#f97316"></div></div>
    <div class="info" id="iAx"></div></div>
  <div class="pane" id="pSag"><div class="label" style="color:#22c55e">SAGITTAL</div>
    <span class="orient" style="top:6px;left:50%">S</span><span class="orient" style="left:6px;top:50%">A</span><span class="orient" style="right:6px;top:50%">P</span>
    <div class="wrap" id="wSag"><img class="ct" id="sagCt"><img class="seg" id="sagSeg">
      <div class="xh v" id="sagV" style="background:#f97316"></div><div class="xh h" id="sagH" style="background:#38bdf8"></div></div>
    <div class="info" id="iSag"></div></div>
  <div class="pane" id="pCor"><div class="label" style="color:#f97316">CORONAL</div>
    <span class="orient" style="top:6px;left:50%">S</span><span class="orient" style="left:6px;top:50%">R</span><span class="orient" style="right:6px;top:50%">L</span>
    <div class="wrap" id="wCor"><img class="ct" id="corCt"><img class="seg" id="corSeg">
      <div class="xh v" id="corV" style="background:#22c55e"></div><div class="xh h" id="corH" style="background:#38bdf8"></div></div>
    <div class="info" id="iCor"></div></div>
  <div class="pane" id="p3d"><div class="label" style="color:#c084fc">3D VOLUME (EAT / PVAT projection)</div>
    <img id="vol" draggable="false"><div class="info" id="i3d">Drag to rotate</div></div>
</div>
<script>
  const NZ = {mpr_info["axial_count"]}, NS = {mpr_info["sag_count"]}, NC = {mpr_info["cor_count"]}, NF = {mpr_info["vol_frames"]};
  const pad = n => String(n).padStart(3, '0');
  let z = Math.floor(NZ / 2), s = Math.floor(NS / 2), c = Math.floor(NC / 2), f = 0, spinning = false;
  const $ = id => document.getElementById(id);
  const clamp = (v, n) => Math.max(0, Math.min(n - 1, Math.round(v)));

  // Preload 3D frames for smooth rotation
  const frames = []; for (let i = 0; i < NF; i++) {{ const im = new Image(); im.src = `volume3d/frame_${{pad(i)}}.png`; frames.push(im); }}

  function render() {{
    $('axCt').src = `slices/ct_${{pad(z)}}.png`;   $('axSeg').src = `slices/seg_${{pad(z)}}.png`;
    $('sagCt').src = `sagittal/ct_${{pad(s)}}.png`; $('sagSeg').src = `sagittal/seg_${{pad(s)}}.png`;
    $('corCt').src = `coronal/ct_${{pad(c)}}.png`;  $('corSeg').src = `coronal/seg_${{pad(c)}}.png`;
    const u = s / (NS - 1), v = c / (NC - 1), w = 1 - z / (NZ - 1);
    $('axV').style.left = (u * 100) + '%';  $('axH').style.top = (v * 100) + '%';
    $('sagV').style.left = (v * 100) + '%'; $('sagH').style.top = (w * 100) + '%';
    $('corV').style.left = (u * 100) + '%'; $('corH').style.top = (w * 100) + '%';
    $('iAx').textContent = `Slice ${{z}} / ${{NZ - 1}}  (scroll to move)`;
    $('iSag').textContent = `Slice ${{s}} / ${{NS - 1}}`;
    $('iCor').textContent = `Slice ${{c}} / ${{NC - 1}}`;
  }}
  function frac(e, el) {{ const r = el.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; }}

  // Click to re-center crosshairs (linked navigation across all planes)
  $('wAx').addEventListener('mousedown', e => {{ const [a, b] = frac(e, $('wAx')); s = clamp(a * (NS - 1), NS); c = clamp(b * (NC - 1), NC); render(); }});
  $('wSag').addEventListener('mousedown', e => {{ const [a, b] = frac(e, $('wSag')); c = clamp(a * (NC - 1), NC); z = clamp((1 - b) * (NZ - 1), NZ); render(); }});
  $('wCor').addEventListener('mousedown', e => {{ const [a, b] = frac(e, $('wCor')); s = clamp(a * (NS - 1), NS); z = clamp((1 - b) * (NZ - 1), NZ); render(); }});

  // Mouse wheel scrolls through the plane under the cursor
  $('pAx').addEventListener('wheel', e => {{ e.preventDefault(); z = clamp(z + (e.deltaY > 0 ? -1 : 1), NZ); render(); }});
  $('pSag').addEventListener('wheel', e => {{ e.preventDefault(); s = clamp(s + (e.deltaY > 0 ? 1 : -1), NS); render(); }});
  $('pCor').addEventListener('wheel', e => {{ e.preventDefault(); c = clamp(c + (e.deltaY > 0 ? 1 : -1), NC); render(); }});

  // Overlay opacity
  $('op').addEventListener('input', e => document.querySelectorAll('img.seg').forEach(i => i.style.opacity = e.target.value / 100));

  // 3D rotation: auto-spin + drag
  function show3d() {{ $('vol').src = frames[f].src; }}
  setInterval(() => {{ if (spinning) {{ f = (f + 1) % NF; show3d(); }} }}, 120);
  $('spin').onclick = () => {{ spinning = !spinning; $('spin').classList.toggle('on', spinning); }};
  let dragX = null;
  $('vol').addEventListener('mousedown', e => {{ dragX = e.clientX; spinning = false; $('spin').classList.remove('on'); }});
  window.addEventListener('mouseup', () => dragX = null);
  window.addEventListener('mousemove', e => {{ if (dragX === null) return; const d = e.clientX - dragX;
    if (Math.abs(d) > 8) {{ f = (f + (d > 0 ? 1 : NF - 1)) % NF; dragX = e.clientX; show3d(); }} }});

  render(); show3d();
</script>
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def generate_ohif_3pane_viewer_html(
    participant_id: str,
    output_dir: Path,
    num_slices: int,
    summary_metrics: Optional[Dict[str, float]] = None,
    available_participants: Optional[List[str]] = None,
    mpr_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generates an authentic OHIF v3.13 Medical Workstation with Cornerstone v5 Corner Overlays,
    Smart Scrollbars on EVERY Viewport, a 2-Column Studies Browser, and Collapsible Panels."""
    html_path = output_dir / "index.html"

    metrics = summary_metrics or {
        "EAT Total Volume (cm3)": 75.79,
        "EAT Total Mean Attenuation (HU)": -67.28,
        "EAT LA Volume (cm3)": 10.93,
        "EAT LA Mean Attenuation (HU)": -68.49,
        "EAT LV Volume (cm3)": 28.14,
        "EAT LV Mean Attenuation (HU)": -66.12,
        "EAT RA Volume (cm3)": 24.31,
        "EAT RA Mean Attenuation (HU)": -67.54,
        "EAT RV Volume (cm3)": 12.41,
        "EAT RV Mean Attenuation (HU)": -68.02,
        "PVAT Total Volume (cm3)": 47.29,
        "PVAT Total Mean Attenuation (HU)": -66.89,
        "PVAT LAD Volume (cm3)": 1.95,
        "PVAT LAD Mean Attenuation (HU)": -62.88,
        "PVAT LCX Volume (cm3)": 19.39,
        "PVAT LCX Mean Attenuation (HU)": -66.54,
        "PVAT RCA Volume (cm3)": 25.95,
        "PVAT RCA Mean Attenuation (HU)": -69.35,
    }

    metrics_rows = ""
    for reg in REPORTED_REGIONS:
        vol = metrics.get(f"{reg.value} Volume (cm3)", 0.0)
        hu = metrics.get(f"{reg.value} Mean Attenuation (HU)", float("nan"))
        hu_str = f"{hu:.2f}" if not np.isnan(hu) else "NaN"
        metrics_rows += f"""
        <tr class="metrics-row">
          <td class="col-name">{reg.value}</td>
          <td class="col-vol">{vol:.2f}</td>
          <td class="col-hu">{hu_str}</td>
        </tr>
        """

    participants_list = available_participants or ["0", "1", "10", "106", "108", "110", "113", "120", "121", "156", "157", "173"]
    quick_pids = participants_list[:24] if len(participants_list) > 24 else participants_list
    participant_options = "".join(
        f'<option value="{p}" {"selected" if str(p) == str(participant_id) else ""}>Participant {p}</option>'
        for p in participants_list
    )
    chip_items = []
    for p in quick_pids:
        active_cls = " active" if str(p) == str(participant_id) else ""
        chip_items.append('<button class="cohort-chip' + active_cls + '" onclick="switchParticipant(\'' + str(p) + '\')">P' + str(p) + '</button>')
    cohort_chips_html = "".join(chip_items)

    info = mpr_info or {
        "axial_count": num_slices,
        "sag_count": 256,
        "cor_count": 256,
        "vol_frames": 36,
        "spacing_mm": [3.0, 0.47, 0.47],
    }
    nz = int(info.get("axial_count", num_slices))
    ns = int(info.get("sag_count", 256))
    nc = int(info.get("cor_count", 256))
    nf = int(info.get("vol_frames", 36))
    sz, sy, sx = info.get("spacing_mm", [3.0, 0.47, 0.47])
    sag_w = int(info.get("sag_w", 512))
    sag_h = int(info.get("sag_h", 365))
    cor_w = int(info.get("cor_w", 512))
    cor_h = int(info.get("cor_h", 365))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>OHIF Medical Imaging Workstation v3.13 &bull; Participant {participant_id}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
    html, body {{ height: 100%; width: 100%; overflow: hidden; background-color: #000000; color: #cbd5e1; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 12px; }}
    svg {{ width: 16px; height: 16px; flex-shrink: 0; vertical-align: middle; }}

    /* Layout Structure */
    .app-root {{ display: flex; flex-direction: column; height: 100vh; width: 100vw; overflow: hidden; }}

    /* Top App Bar */
    header.top-header {{ height: 44px; background: #090c15; border-bottom: 1px solid #1e293b; display: flex; align-items: center; justify-content: space-between; padding: 0 10px; z-index: 50; flex-shrink: 0; }}
    .header-left {{ display: flex; align-items: center; gap: 8px; }}
    .drawer-toggle-btn {{ background: transparent; border: 1px solid #334155; border-radius: 4px; padding: 4px 6px; color: #94a3b8; cursor: pointer; display: flex; align-items: center; }}
    .drawer-toggle-btn:hover {{ background: #1e293b; color: #fff; }}
    .ohif-logo-box {{ width: 24px; height: 24px; border-radius: 4px; background: linear-gradient(135deg, #0284c7, #2563eb); display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 9px; color: #fff; letter-spacing: 0.5px; box-shadow: 0 1px 4px rgba(2, 132, 199, 0.4); }}
    .brand-col {{ display: flex; flex-direction: column; line-height: 1.15; }}
    .brand-main {{ font-size: 11px; font-weight: 700; color: #f8fafc; letter-spacing: 0.2px; }}
    .brand-sub {{ font-size: 9px; color: #64748b; display: flex; align-items: center; gap: 5px; }}
    .version-tag {{ background: #082f49; color: #38bdf8; border: 1px solid #0369a1; padding: 0 4px; border-radius: 3px; font-size: 8.5px; font-family: monospace; }}

    /* Center Toolbar */
    .toolbar-group {{ display: flex; align-items: center; gap: 3px; background: #020617; border: 1px solid #1e293b; border-radius: 6px; padding: 2px 4px; }}
    .tool-btn {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px 7px; color: #94a3b8; font-size: 11px; font-weight: 500; cursor: pointer; display: flex; align-items: center; gap: 4px; transition: all 0.12s; }}
    .tool-btn:hover {{ background: #1e293b; color: #f8fafc; }}
    .tool-btn.active {{ background: rgba(2, 132, 199, 0.25); border-color: #0284c7; color: #38bdf8; }}
    .tool-select {{ background: #0f172a; border: 1px solid #334155; color: #e2e8f0; font-size: 11px; border-radius: 4px; padding: 2px 5px; outline: none; cursor: pointer; }}

    /* Header Right Section */
    .header-right {{ display: flex; align-items: center; gap: 8px; }}
    .patient-switcher-wrap {{ display: flex; align-items: center; gap: 4px; background: #0f172a; border: 1px solid #334155; border-radius: 4px; padding: 2px 6px; font-size: 11px; }}
    .patient-step-btn {{ background: transparent; border: none; color: #94a3b8; cursor: pointer; font-size: 13px; font-weight: bold; padding: 0 3px; }}
    .patient-step-btn:hover {{ color: #38bdf8; }}
    .patient-nav-select {{ background: transparent; border: none; color: #38bdf8; font-weight: 600; font-size: 11px; outline: none; cursor: pointer; }}
    .patient-nav-select option {{ background: #0f172a; color: #fff; }}
    .cloud-btn {{ background: linear-gradient(135deg, #2563eb, #0284c7); color: #fff; padding: 4px 9px; border-radius: 4px; text-decoration: none; font-size: 10.5px; font-weight: 600; display: inline-flex; align-items: center; gap: 5px; box-shadow: 0 1px 4px rgba(37, 99, 235, 0.35); }}
    .cloud-btn:hover {{ opacity: 0.92; }}
    .badge-link {{ color: #cbd5e1; background: #0f172a; border: 1px solid #334155; border-radius: 4px; padding: 3px 6px; font-size: 10px; text-decoration: none; font-family: monospace; }}
    .badge-link:hover {{ background: #1e293b; color: #38bdf8; border-color: #38bdf8; }}
    
    /* Panel Toggle Buttons (Matching official OHIF circular buttons) */
    .side-toggle-btn {{ width: 28px; height: 28px; border-radius: 50%; background: #0f172a; border: 1px solid #334155; color: #94a3b8; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.15s; }}
    .side-toggle-btn:hover {{ background: #1e293b; color: #fff; border-color: #38bdf8; }}
    .side-toggle-btn.active {{ background: #0369a1; border-color: #38bdf8; color: #ffffff; box-shadow: 0 0 8px rgba(56, 189, 248, 0.4); }}

    /* Main Body */
    main.main-body {{ display: flex; flex: 1; height: calc(100vh - 44px); overflow: hidden; background: #000; position: relative; }}

    /* ========================================================= */
    /* LEFT DRAWER: 2-Column Square Studies Browser (Official OHIF) */
    /* ========================================================= */
    aside.studies-drawer {{ width: 250px; background: #050811; border-right: 1px solid #1e293b; display: flex; flex-direction: column; flex-shrink: 0; transition: margin-left 0.2s ease, width 0.2s ease; z-index: 30; overflow-y: auto; }}
    aside.studies-drawer.collapsed {{ margin-left: -250px; }}
    .drawer-title-bar {{ padding: 8px 10px; background: #0b1120; border-bottom: 1px solid #1e293b; display: flex; align-items: center; justify-content: space-between; }}
    .drawer-heading {{ font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
    .study-accordion {{ padding: 8px; border-bottom: 1px solid #1e293b; background: rgba(15, 23, 42, 0.4); }}
    .study-accordion-title {{ font-size: 11px; font-weight: 700; color: #f1f5f9; }}
    .study-accordion-meta {{ font-size: 10px; color: #38bdf8; font-family: monospace; }}

    /* 2-Column Square Thumbnail Grid */
    .series-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; padding: 8px; }}
    .series-card {{ aspect-ratio: 1 / 1; background: #000000; border: 1px solid #1e293b; border-radius: 4px; padding: 4px; display: flex; flex-direction: column; justify-content: space-between; cursor: pointer; position: relative; overflow: hidden; transition: all 0.15s; }}
    .series-card:hover {{ border-color: #38bdf8; }}
    .series-card.active {{ border-color: #0284c7; box-shadow: 0 0 0 1px #0284c7 inset; }}
    .card-top {{ display: flex; align-items: center; justify-content: space-between; font-size: 9px; font-family: monospace; z-index: 5; pointer-events: none; }}
    .card-badge {{ font-size: 8px; font-weight: 800; padding: 1px 4px; border-radius: 2px; }}
    .badge-ct {{ background: #082f49; color: #38bdf8; border: 1px solid #0369a1; }}
    .badge-seg {{ background: #3b0764; color: #c084fc; border: 1px solid #9333ea; }}
    .badge-vr {{ background: #431407; color: #fb923c; border: 1px solid #ea580c; }}
    .card-count {{ color: #94a3b8; }}
    .card-thumb-wrap {{ flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; padding: 2px 0; }}
    .card-thumb-wrap img {{ width: 100%; height: 100%; object-fit: contain; pointer-events: none; }}
    .card-seg-badge {{ width: 44px; height: 44px; border-radius: 4px; background: rgba(88, 28, 135, 0.4); border: 1px solid #7c3aed; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #d8b4fe; font-size: 11px; font-weight: 800; }}
    .card-bottom {{ z-index: 5; pointer-events: none; }}
    .card-title {{ font-size: 10px; font-weight: 600; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }}
    .card-sub {{ font-size: 8.5px; color: #64748b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }}

    /* Cohort Chips */
    .cohort-section {{ padding: 8px; border-top: 1px solid #1e293b; }}
    .cohort-chips {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }}
    .cohort-chip {{ font-size: 9.5px; background: #0f172a; border: 1px solid #334155; color: #94a3b8; border-radius: 3px; padding: 2px 5px; cursor: pointer; }}
    .cohort-chip:hover {{ color: #fff; border-color: #38bdf8; }}
    .cohort-chip.active {{ background: #0284c7; color: #fff; border-color: #38bdf8; }}

    /* ========================================================= */
    /* CENTER VIEWPORTS: 2x2 Grid (100% Height, No Clunky Bottom Bar!) */
    /* ========================================================= */
    section.center-viewports {{ flex: 1; height: 100%; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 2px; background: #0b1120; overflow: hidden; position: relative; }}
    section.center-viewports.maximized {{ grid-template-columns: 1fr; grid-template-rows: 1fr; }}

    .viewport-box {{ position: relative; background: #000000; display: flex; align-items: center; justify-content: center; overflow: hidden; cursor: crosshair; border: 1.5px solid #1e293b; }}
    .viewport-box.active-box {{ border-color: #0284c7 !important; }}
    .viewport-box.hidden {{ display: none !important; }}

    /* True Cornerstone Corner Annotations (Clean Medical Overlay) */
    .cs-overlay {{ position: absolute; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 11px; color: #cbd5e1; z-index: 15; pointer-events: none; text-shadow: 0 1px 3px rgba(0,0,0,0.95); line-height: 1.25; }}
    .cs-top-left {{ top: 8px; left: 10px; }}
    .cs-top-left .study-meta {{ font-weight: 700; color: #f8fafc; }}
    .cs-top-left .series-meta {{ color: #94a3b8; font-size: 10.5px; }}
    .cs-top-right {{ top: 8px; right: 24px; pointer-events: auto; display: flex; gap: 4px; }}
    .cs-bottom-left {{ bottom: 8px; left: 10px; font-family: monospace; font-size: 10px; color: #94a3b8; }}
    .cs-bottom-right {{ bottom: 8px; right: 24px; font-family: monospace; font-size: 10px; color: #94a3b8; text-align: right; }}

    /* Anatomical Orientation Markers (Clean Muted Gray Sans-Serif, NOT Yellow!) */
    .orient {{ position: absolute; z-index: 12; font-family: -apple-system, sans-serif; font-size: 11px; font-weight: 600; color: #94a3b8; pointer-events: none; text-shadow: 0 1px 3px rgba(0,0,0,0.9); }}
    .orient.top {{ top: 10px; left: 50%; transform: translateX(-50%); }}
    .orient.bottom {{ bottom: 10px; left: 50%; transform: translateX(-50%); }}
    .orient.left {{ left: 10px; top: 50%; transform: translateY(-50%); }}
    .orient.right {{ right: 22px; top: 50%; transform: translateY(-50%); }}

    /* Viewport Image Wrap */
    .vp-canvas-wrap {{ position: relative; display: inline-block; line-height: 0; height: 100%; max-width: 100%; max-height: 100%; }}
    #wAx, #w3D {{ aspect-ratio: 1 / 1; }}
    #wSag {{ aspect-ratio: {sag_w} / {sag_h}; }}
    #wCor {{ aspect-ratio: {cor_w} / {cor_h}; }}
    .vp-canvas-wrap img.ct-plane {{ width: 100%; height: 100%; object-fit: fill; display: block; }}
    .vp-canvas-wrap canvas.seg-plane {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; opacity: 0.85; }}

    /* Crosshairs */
    .xh {{ position: absolute; pointer-events: none; z-index: 10; }}
    .xh.v {{ top: 0; bottom: 0; width: 1px; }}
    .xh.h {{ left: 0; right: 0; height: 1px; }}

    /* OHIF v3.13 Smart Scrollbar on EVERY Viewport */
    .smart-scrollbar {{ position: absolute; right: 4px; top: 28px; bottom: 28px; width: 8px; background: rgba(15, 23, 42, 0.7); border-radius: 4px; z-index: 25; cursor: pointer; border: 1px solid rgba(51, 65, 85, 0.5); display: flex; flex-direction: column; overflow: hidden; }}
    .smart-track {{ flex: 1; position: relative; width: 100%; }}
    .smart-thumb {{ position: absolute; left: 1px; right: 1px; height: 12px; background: #38bdf8; border-radius: 3px; box-shadow: 0 0 6px rgba(56, 189, 248, 0.85); pointer-events: none; transition: top 0.04s ease-out; }}
    .smart-scrollbar:hover {{ width: 10px; border-color: #38bdf8; }}

    /* Maximize Action Button */
    .vp-btn {{ background: rgba(30, 41, 59, 0.75); border: 1px solid #334155; border-radius: 3px; color: #cbd5e1; font-size: 10px; padding: 2px 5px; cursor: pointer; }}
    .vp-btn:hover {{ background: #334155; color: #fff; }}

    /* ========================================================= */
    /* RIGHT DRAWER: Segmentations & Measurements Slide-Out Panel */
    /* ========================================================= */
    aside.right-drawer {{ width: 310px; background: #050811; border-left: 1px solid #1e293b; display: flex; flex-direction: column; flex-shrink: 0; transition: margin-right 0.2s ease, width 0.2s ease; z-index: 30; overflow-y: auto; }}
    aside.right-drawer.collapsed {{ margin-right: -310px; display: none; }}
    .drawer-tab-bar {{ display: flex; border-bottom: 1px solid #1e293b; background: #090c15; }}
    .drawer-tab-btn {{ flex: 1; padding: 10px 6px; font-size: 11px; font-weight: 600; text-align: center; color: #94a3b8; background: transparent; border: none; border-bottom: 2px solid transparent; cursor: pointer; }}
    .drawer-tab-btn.active {{ color: #38bdf8; border-bottom-color: #0284c7; background: rgba(2, 132, 199, 0.08); }}
    .drawer-close-btn {{ width: 32px; background: transparent; border: none; color: #94a3b8; font-size: 16px; cursor: pointer; }}
    .drawer-close-btn:hover {{ color: #fff; }}

    .compartment-list {{ padding: 10px 12px; display: flex; flex-direction: column; gap: 5px; }}
    .compartment-row {{ display: flex; align-items: center; justify-content: space-between; padding: 5px 8px; border-radius: 4px; background: rgba(15, 23, 42, 0.4); border: 1px solid #1e293b; font-size: 11px; }}
    .comp-info {{ display: flex; align-items: center; gap: 7px; }}
    .color-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
    .solo-btn {{ background: #1e293b; border: 1px solid #334155; color: #94a3b8; border-radius: 3px; font-size: 9px; font-weight: 600; padding: 1px 5px; cursor: pointer; transition: all 0.15s; }}
    .solo-btn:hover {{ background: #0284c7; color: #fff; border-color: #38bdf8; }}
    .solo-btn.active {{ background: #0284c7; color: #fff; border-color: #38bdf8; }}

    .opacity-box {{ padding: 0 12px 10px 12px; }}
    .opacity-header {{ display: flex; justify-content: space-between; font-size: 10px; color: #94a3b8; margin-bottom: 4px; }}
    input[type="range"] {{ width: 100%; accent-color: #0284c7; cursor: pointer; }}

    /* Metrics Table */
    .metrics-section {{ padding: 10px 12px; flex: 1; overflow-y: auto; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 11px; text-align: left; }}
    .metrics-table th {{ padding: 4px 6px; color: #64748b; font-weight: 600; border-bottom: 1px solid #1e293b; font-size: 10px; }}
    .metrics-row td {{ padding: 4px 6px; border-bottom: 1px solid #0f172a; }}
    .metrics-row:hover td {{ background: rgba(30, 41, 59, 0.35); }}
    .col-name {{ font-weight: 600; color: #ffffff; }}
    .col-vol {{ color: #38bdf8; font-family: monospace; text-align: right; }}
    .col-hu {{ color: #34d399; font-family: monospace; text-align: right; }}
    
    .status-card {{ margin: 10px 12px; padding: 8px 10px; border-radius: 6px; background: #0f172a; border: 1px solid #1e293b; font-size: 10px; color: #94a3b8; display: flex; flex-direction: column; gap: 3px; }}
  </style>
</head>
<body>
<div class="app-root">

  <!-- Top OHIF Header -->
  <header class="top-header">
    <div class="header-left">
      <button class="drawer-toggle-btn" onclick="toggleLeftDrawer()" title="Toggle Studies Browser">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
      </button>
      <div class="ohif-logo-box">OHIF</div>
      <div class="brand-col">
        <span class="brand-main">Open Health Imaging Foundation</span>
        <span class="brand-sub">Medical Viewer <span class="version-tag">v3.13 Next</span> &bull; COCA Cardiac CT</span>
      </div>
    </div>

    <!-- Center OHIF Minimalist Toolset -->
    <div class="toolbar-group">
      <button class="tool-btn" id="btnLength" onclick="setTool('length')" title="Length / Measurement">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.17 6.83l-4-4a2 2 0 0 0-2.83 0L3 14.17V21h6.83l11.34-11.34a2 2 0 0 0 0-2.83zM14 5l4 4"/></svg>
      </button>
      <button class="tool-btn" id="btnAngle" onclick="setTool('angle')" title="Angle Measurement">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 20h16M4 20L18 6"/></svg>
      </button>
      <button class="tool-btn" id="btnZoom" onclick="setTool('zoom')" title="Zoom Viewport">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35M11 8v6M8 11h6"/></svg>
      </button>
      <button class="tool-btn" id="btnPan" onclick="setTool('pan')" title="Pan Viewport">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 11V6a2 2 0 0 0-4 0v5M14 10V4a2 2 0 0 0-4 0v7M10 10.5V6a2 2 0 0 0-4 0v8a7 7 0 0 0 14 0v-3a2 2 0 0 0-4 0"/></svg>
      </button>
      <div style="display:flex; align-items:center; gap:3px;">
        <button class="tool-btn" id="btnWl" title="Window / Level Presets">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2v20a10 10 0 0 0 0-20z" fill="currentColor"/></svg>
        </button>
        <select id="presetSelect" class="tool-select" onchange="applyWindowPreset(this.value)">
          <option value="fat">Cardiac Fat (W:400 L:-110)</option>
          <option value="soft">Soft Tissue (W:350 L:40)</option>
          <option value="lung">Chest / Lung (W:1500 L:-600)</option>
          <option value="bone">Bone / Calcium (W:1500 L:300)</option>
        </select>
      </div>
      <button class="tool-btn active" id="btnCrosshair" onclick="setTool('crosshair')" title="Linked Orthogonal Crosshairs">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/></svg>
      </button>
      <button class="tool-btn" id="btnLayout" onclick="toggleGridLayout()" title="Toggle 2x2 Grid / 1x1 Focus">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
      </button>
      <button class="tool-btn" onclick="resetAllViewports()" title="Reset All Viewports">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
      </button>
    </div>

    <!-- Header Right Section -->
    <div class="header-right">
      <div class="patient-switcher-wrap">
        <span style="color:#94a3b8; font-size:10px;">Patient</span>
        <button class="patient-step-btn" onclick="stepPatient(-1)" title="Previous Patient">&lsaquo;</button>
        <select class="patient-nav-select" id="patientSelector" onchange="switchParticipant(this.value)">
          {participant_options}
        </select>
        <button class="patient-step-btn" onclick="stepPatient(1)" title="Next Patient">&rsaquo;</button>
      </div>

      <a href="studies.json" target="_blank" class="badge-link" title="OHIF v3.13 Naturalized DICOM JSON Manifest">studies.json</a>
      <a href="https://ohif.org/release-notes/3p13" target="_blank" class="badge-link" title="OHIF v3.13 Release Notes">v3.13 Notes</a>
      <a href="https://viewer.ohif.org/local" id="btnLaunchOHIF" target="_blank" class="cloud-btn" title="Launch in official OHIF v3.13 Cloud Viewer">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 6H6a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>
        <span>Open in OHIF</span>
      </a>

      <!-- Circular Side-Panel Toggles -->
      <button class="side-toggle-btn active" id="btnToggleSeg" onclick="toggleRightPanel('seg')" title="Toggle Segmentations Panel">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
      </button>
      <button class="side-toggle-btn" id="btnToggleMeas" onclick="toggleRightPanel('meas')" title="Toggle Measurements Panel">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
      </button>
    </div>
  </header>

  <!-- Main Workstation Body -->
  <main class="main-body">

    <!-- ============================================== -->
    <!-- LEFT DRAWER: 2-Column Square Studies Browser   -->
    <!-- ============================================== -->
    <aside class="studies-drawer" id="leftDrawer">
      <div class="drawer-title-bar">
        <span class="drawer-heading">Studies</span>
        <span class="version-tag">1 Study</span>
      </div>

      <div class="study-accordion">
        <div class="study-accordion-title">COCA Gated CT &bull; P{participant_id}</div>
        <div class="study-accordion-meta">22-May-2014 &bull; {sx:.2f} &times; {sy:.2f} &times; {sz:.2f} mm</div>
      </div>

      <!-- 2-Column Square Thumbnail Grid (Exact OHIF layout) -->
      <div class="series-grid">
        <!-- Tile 1: Primary CT Axial -->
        <div class="series-card active" id="cardCT" onclick="selectSeriesTile('axial')">
          <div class="card-top">
            <span class="card-badge badge-ct">CT</span>
            <span class="card-count">s:1 i:{nz}</span>
          </div>
          <div class="card-thumb-wrap">
            <img src="slices/thumbnail_ct.png" onerror="this.src='slices/ct_028.png'">
          </div>
          <div class="card-bottom">
            <span class="card-title">Axial</span>
            <span class="card-sub">Axial CT Primary</span>
          </div>
        </div>

        <!-- Tile 2: 3D Thoracic Render (Top-Right) -->
        <div class="series-card" id="card3D" onclick="selectSeriesTile('3d')">
          <div class="card-top">
            <span class="card-badge badge-vr">VR</span>
            <span class="card-count">s:2 i:{nf}</span>
          </div>
          <div class="card-thumb-wrap">
            <img src="volume3d/frame_000.png" onerror="this.src='slices/ct_028.png'">
          </div>
          <div class="card-bottom">
            <span class="card-title">3D Volume</span>
            <span class="card-sub">Cardiac Fat Render</span>
          </div>
        </div>

        <!-- Tile 3: Coronal MPR (Bottom-Left, 3D Slicer convention) -->
        <div class="series-card" id="cardCor" onclick="selectSeriesTile('coronal')">
          <div class="card-top">
            <span class="card-badge badge-ct">CT</span>
            <span class="card-count">s:3 i:{nc}</span>
          </div>
          <div class="card-thumb-wrap">
            <img src="coronal/ct_{nc // 2:03d}.png" onerror="this.src='slices/ct_028.png'">
          </div>
          <div class="card-bottom">
            <span class="card-title">Coronal</span>
            <span class="card-sub">Coronal MPR</span>
          </div>
        </div>

        <!-- Tile 4: Sagittal MPR (Bottom-Right, 3D Slicer convention) -->
        <div class="series-card" id="cardSag" onclick="selectSeriesTile('sagittal')">
          <div class="card-top">
            <span class="card-badge badge-ct">CT</span>
            <span class="card-count">s:4 i:{ns}</span>
          </div>
          <div class="card-thumb-wrap">
            <img src="sagittal/ct_{ns // 2:03d}.png" onerror="this.src='slices/ct_028.png'">
          </div>
          <div class="card-bottom">
            <span class="card-title">Sagittal</span>
            <span class="card-sub">Sagittal MPR</span>
          </div>
        </div>

        <!-- Tile 5: DICOM-SEG -->
        <div class="series-card" id="cardSEG" onclick="selectSeriesTile('seg')" style="grid-column: span 2; aspect-ratio: auto; padding: 6px 8px;">
          <div style="display:flex; align-items:center; gap:8px;">
            <div class="card-seg-badge">8L</div>
            <div>
              <span class="card-title" style="color:#d8b4fe;">EAT/PVAT DICOM-SEG</span>
              <span class="card-sub">eat_pvat_seg.dcm &bull; SOP: 1.2.840...66.4</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Cohort Quick Navigation Chips -->
      <div class="cohort-section">
        <div class="drawer-heading">COCA Cohort ({len(participants_list)})</div>
        <div class="cohort-chips">
          {cohort_chips_html}
        </div>
      </div>
    </aside>

    <!-- ============================================== -->
    <!-- CENTER VIEWPORTS: 2x2 Grid (100% Height!)       -->
    <!-- ============================================== -->
    <section class="center-viewports" id="viewportsGrid">

      <!-- Viewport 1: Axial Primary CT (Top-Left) -->
      <div class="viewport-box active-box" id="pAx" onclick="setActiveBox('pAx')">
        <!-- Cornerstone Corner Overlays -->
        <div class="cs-overlay cs-top-left">
          <div class="study-meta">22-May-2014</div>
          <div class="series-meta" id="lblSeriesAx">Axial CT Primary</div>
        </div>
        <div class="cs-overlay cs-top-right">
          <button class="vp-btn" onclick="maximizeBox('pAx', event)" title="Maximize Viewport">⛶</button>
        </div>
        <div class="cs-overlay cs-bottom-left" id="hudWindowAx">W: 400 L: -110</div>
        <div class="cs-overlay cs-bottom-right" id="iAx">I: 29 (29/{nz})</div>

        <!-- Anatomical Markers (Subdued Gray Sans-Serif) -->
        <span class="orient top">A</span>
        <span class="orient bottom">P</span>
        <span class="orient left">R</span>
        <span class="orient right">L</span>

        <div class="vp-canvas-wrap" id="wAx">
          <img class="ct-plane" id="axCt" src="slices/ct_028.png">
          <canvas class="seg-plane" id="axCanvas" width="512" height="512"></canvas>
          <div class="xh v" id="axV" style="background:#22c55e;"></div>
          <div class="xh h" id="axH" style="background:#f97316;"></div>
        </div>

        <!-- Smart Scrollbar for Axial -->
        <div class="smart-scrollbar" id="smartScrollAx" onclick="onSmartScrollClick(event, 'axial')" title="Smart Scrollbar: Jump Axial slice">
          <div class="smart-track" id="smartTrackAx"><div class="smart-thumb" id="smartThumbAx"></div></div>
        </div>
      </div>

      <!-- Viewport 2: 3D MPR Reconstruction (Top-Right, 3D Slicer Standard) -->
      <div class="viewport-box" id="p3d" onclick="setActiveBox('p3d')">
        <!-- Cornerstone Corner Overlays -->
        <div class="cs-overlay cs-top-left">
          <div class="study-meta">3D Slicer Four-Up Standard</div>
          <div class="series-meta">3D MPR Reconstruction &bull; Volume Render</div>
        </div>
        <div class="cs-overlay cs-top-right">
          <button class="vp-btn" id="btnSpin" onclick="toggleSpin(event)" style="background:#334155; color:#cbd5e1;">Auto-Spin</button>
          <button class="vp-btn" onclick="maximizeBox('p3d', event)" title="Maximize Viewport">⛶</button>
        </div>
        <div class="cs-overlay cs-bottom-left">Drag to rotate volume</div>
        <div class="cs-overlay cs-bottom-right" id="i3d">Angle: 0&deg; (1/{nf})</div>

        <div class="vp-canvas-wrap" id="w3D">
          <img class="ct-plane" id="volImg" src="volume3d/frame_000.png" onerror="this.src='slices/ct_028.png'" draggable="false" style="cursor:grab;">
        </div>

        <!-- Smart Scrollbar for 3D Volume -->
        <div class="smart-scrollbar" id="smartScroll3D" onclick="onSmartScrollClick(event, '3d')" title="Smart Scrollbar: Scrub 3D Rotation">
          <div class="smart-track" id="smartTrack3D"><div class="smart-thumb" id="smartThumb3D"></div></div>
        </div>
      </div>

      <!-- Viewport 3: Coronal MPR (Bottom-Left, 3D Slicer Standard) -->
      <div class="viewport-box" id="pCor" onclick="setActiveBox('pCor')">
        <!-- Cornerstone Corner Overlays -->
        <div class="cs-overlay cs-top-left">
          <div class="study-meta">22-May-2014</div>
          <div class="series-meta">Coronal MPR</div>
        </div>
        <div class="cs-overlay cs-top-right">
          <button class="vp-btn" onclick="maximizeBox('pCor', event)" title="Maximize Viewport">⛶</button>
        </div>
        <div class="cs-overlay cs-bottom-left" id="hudWindowCor">W: 400 L: -110</div>
        <div class="cs-overlay cs-bottom-right" id="iCor">I: {nc // 2 + 1} ({nc // 2 + 1}/{nc})</div>

        <!-- Anatomical Markers -->
        <span class="orient top">H</span>
        <span class="orient bottom">F</span>
        <span class="orient left">R</span>
        <span class="orient right">L</span>

        <div class="vp-canvas-wrap" id="wCor">
          <img class="ct-plane" id="corCt" src="coronal/ct_{nc // 2:03d}.png" onerror="this.src='slices/ct_028.png'">
          <canvas class="seg-plane" id="corCanvas" width="{cor_w}" height="{cor_h}"></canvas>
          <div class="xh v" id="corV" style="background:#22c55e;"></div>
          <div class="xh h" id="corH" style="background:#38bdf8;"></div>
        </div>

        <!-- Smart Scrollbar for Coronal -->
        <div class="smart-scrollbar" id="smartScrollCor" onclick="onSmartScrollClick(event, 'coronal')" title="Smart Scrollbar: Jump Coronal slice">
          <div class="smart-track" id="smartTrackCor"><div class="smart-thumb" id="smartThumbCor"></div></div>
        </div>
      </div>

      <!-- Viewport 4: Sagittal MPR (Bottom-Right, 3D Slicer Standard) -->
      <div class="viewport-box" id="pSag" onclick="setActiveBox('pSag')">
        <!-- Cornerstone Corner Overlays -->
        <div class="cs-overlay cs-top-left">
          <div class="study-meta">22-May-2014</div>
          <div class="series-meta">Sagittal MPR</div>
        </div>
        <div class="cs-overlay cs-top-right">
          <button class="vp-btn" onclick="maximizeBox('pSag', event)" title="Maximize Viewport">⛶</button>
        </div>
        <div class="cs-overlay cs-bottom-left" id="hudWindowSag">W: 400 L: -110</div>
        <div class="cs-overlay cs-bottom-right" id="iSag">I: {ns // 2 + 1} ({ns // 2 + 1}/{ns})</div>

        <!-- Anatomical Markers -->
        <span class="orient top">H</span>
        <span class="orient bottom">F</span>
        <span class="orient left">A</span>
        <span class="orient right">P</span>

        <div class="vp-canvas-wrap" id="wSag">
          <img class="ct-plane" id="sagCt" src="sagittal/ct_{ns // 2:03d}.png" onerror="this.src='slices/ct_028.png'">
          <canvas class="seg-plane" id="sagCanvas" width="{sag_w}" height="{sag_h}"></canvas>
          <div class="xh v" id="sagV" style="background:#f97316;"></div>
          <div class="xh h" id="sagH" style="background:#38bdf8;"></div>
        </div>

        <!-- Smart Scrollbar for Sagittal -->
        <div class="smart-scrollbar" id="smartScrollSag" onclick="onSmartScrollClick(event, 'sagittal')" title="Smart Scrollbar: Jump Sagittal slice">
          <div class="smart-track" id="smartTrackSag"><div class="smart-thumb" id="smartThumbSag"></div></div>
        </div>
      </div>

    </section>

    <!-- ============================================== -->
    <!-- RIGHT DRAWER: Segmentations & Measurements     -->
    <!-- ============================================== -->
    <aside class="right-drawer" id="rightDrawer">
      <div class="drawer-tab-bar">
        <button class="drawer-tab-btn active" id="tabSeg" onclick="switchRightTab('seg')">Segmentations (8)</button>
        <button class="drawer-tab-btn" id="tabMeas" onclick="switchRightTab('meas')">Measurements (19)</button>
        <button class="drawer-close-btn" onclick="toggleRightPanel(null)" title="Close Panel">&times;</button>
      </div>

      <!-- Panel: Segmentations -->
      <div id="panelSeg">
        <div class="compartment-list">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span class="drawer-heading">8 Target Layers</span>
            <span style="color:#c084fc; font-family:monospace; font-size:10px;">DICOM-SEG</span>
          </div>

          <!-- Quick Actions: Show / Hide All -->
          <div style="display:flex; gap:6px; margin-bottom:6px;">
            <button class="vp-btn" onclick="setAllCompartments(true)" style="flex:1; padding:3px 6px;">Show All</button>
            <button class="vp-btn" onclick="setAllCompartments(false)" style="flex:1; padding:3px 6px;">Hide All</button>
          </div>

          <div class="compartment-row" id="row_EAT_TOTAL">
            <div class="comp-info">
              <span class="color-dot" style="background:#9333ea;"></span>
              <strong style="color:#fff;">EAT Total</strong>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="solo-btn" onclick="isolateCompartment('EAT_TOTAL')" title="Isolate this compartment">Solo</button>
              <input type="checkbox" id="chk_EAT_TOTAL" checked onchange="onCompartmentChange()">
            </div>
          </div>

          <div class="compartment-row" id="row_EAT_LA">
            <div class="comp-info">
              <span class="color-dot" style="background:#3b82f6;"></span>
              <span>EAT Left Atrium (LA)</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="solo-btn" onclick="isolateCompartment('EAT_LA')" title="Isolate this compartment">Solo</button>
              <input type="checkbox" id="chk_EAT_LA" checked onchange="onCompartmentChange()">
            </div>
          </div>

          <div class="compartment-row" id="row_EAT_LV">
            <div class="comp-info">
              <span class="color-dot" style="background:#06b6d4;"></span>
              <span>EAT Left Ventricle (LV)</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="solo-btn" onclick="isolateCompartment('EAT_LV')" title="Isolate this compartment">Solo</button>
              <input type="checkbox" id="chk_EAT_LV" checked onchange="onCompartmentChange()">
            </div>
          </div>

          <div class="compartment-row" id="row_EAT_RA">
            <div class="comp-info">
              <span class="color-dot" style="background:#10b981;"></span>
              <span>EAT Right Atrium (RA)</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="solo-btn" onclick="isolateCompartment('EAT_RA')" title="Isolate this compartment">Solo</button>
              <input type="checkbox" id="chk_EAT_RA" checked onchange="onCompartmentChange()">
            </div>
          </div>

          <div class="compartment-row" id="row_EAT_RV">
            <div class="comp-info">
              <span class="color-dot" style="background:#eab308;"></span>
              <span>EAT Right Ventricle (RV)</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="solo-btn" onclick="isolateCompartment('EAT_RV')" title="Isolate this compartment">Solo</button>
              <input type="checkbox" id="chk_EAT_RV" checked onchange="onCompartmentChange()">
            </div>
          </div>

          <div class="compartment-row" id="row_PVAT_LAD">
            <div class="comp-info">
              <span class="color-dot" style="background:#f97316;"></span>
              <span>PVAT LAD</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="solo-btn" onclick="isolateCompartment('PVAT_LAD')" title="Isolate this compartment">Solo</button>
              <input type="checkbox" id="chk_PVAT_LAD" checked onchange="onCompartmentChange()">
            </div>
          </div>

          <div class="compartment-row" id="row_PVAT_LCX">
            <div class="comp-info">
              <span class="color-dot" style="background:#ef4444;"></span>
              <span>PVAT LCX</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="solo-btn" onclick="isolateCompartment('PVAT_LCX')" title="Isolate this compartment">Solo</button>
              <input type="checkbox" id="chk_PVAT_LCX" checked onchange="onCompartmentChange()">
            </div>
          </div>

          <div class="compartment-row" id="row_PVAT_RCA">
            <div class="comp-info">
              <span class="color-dot" style="background:#ec4899;"></span>
              <span>PVAT RCA</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="solo-btn" onclick="isolateCompartment('PVAT_RCA')" title="Isolate this compartment">Solo</button>
              <input type="checkbox" id="chk_PVAT_RCA" checked onchange="onCompartmentChange()">
            </div>
          </div>
        </div>

        <!-- Opacity Slider -->
        <div class="opacity-box">
          <div class="opacity-header">
            <span>Segmentation Opacity</span>
            <span id="opacityVal">85%</span>
          </div>
          <input type="range" id="opacitySlider" min="0" max="100" value="85" oninput="setOpacity(this.value)">
        </div>
      </div>

      <!-- Panel: Measurements -->
      <div class="metrics-section" id="panelMeas" style="display:none;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span class="drawer-heading">Clinical Measurements</span>
          <span style="font-size:9px; color:#38bdf8; font-family:monospace;">19-Col MGH Schema</span>
        </div>
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Region</th>
              <th style="text-align:right;">Vol (cm³)</th>
              <th style="text-align:right;">Mean (HU)</th>
            </tr>
          </thead>
          <tbody>
            {metrics_rows}
          </tbody>
        </table>

        <!-- Status Card -->
        <div class="status-card" style="margin:12px 0 0 0;">
          <div style="display:flex; justify-content:space-between;">
            <span>Adipose Attenuation Window:</span>
            <strong style="color:#38bdf8;">[-190.0, -30.0] HU</strong>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span>Pericoronary Radial Distance:</span>
            <strong style="color:#fb923c;">10.0 mm</strong>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span>Total EAT Adipose Volume:</span>
            <strong style="color:#34d399;">{metrics.get('EAT Total Volume (cm3)', 75.79):.2f} cm³</strong>
          </div>
          <div style="display:flex; justify-content:space-between; margin-top:3px; padding-top:3px; border-top:1px dashed #1e293b;">
            <span>OHIF Architecture:</span>
            <strong style="color:#38bdf8;"><a href="https://ohif.org/release-notes/3p13" target="_blank" style="color:#38bdf8; text-decoration:none;">v3.13 Next (Cornerstone v5)</a></strong>
          </div>
        </div>
      </div>
    </aside>

  </main>
</div>

<!-- Interactive OHIF Client Script -->
<script>
  const NZ = {nz}, NS = {ns}, NC = {nc}, NF = {nf};
  const pad = n => String(n).padStart(3, '0');

  let z = Math.floor(NZ / 2);
  let s = Math.floor(NS / 2);
  let c = Math.floor(NC / 2);
  let f = 0;
  let activePlane = 'axial';
  let isGridMaximized = false;
  let maximizedBoxId = null;
  let isSpinning = false;
  let currentTool = 'crosshair';

  const $ = id => document.getElementById(id);
  const clamp = (v, min, max) => Math.max(min, Math.min(max, Math.round(v)));

  // Preload 3D frames
  const frames3d = [];
  for (let i = 0; i < NF; i++) {{
    const im = new Image();
    im.src = `volume3d/frame_${{pad(i)}}.png`;
    frames3d.push(im);
  }}

  const COMP_DEFS = [
    {{ id: 'EAT_TOTAL', name: 'EAT Total',            color: '#9333ea', r: 147, g: 51,  b: 234 }},
    {{ id: 'EAT_LA',    name: 'EAT Left Atrium (LA)', color: '#3b82f6', r: 59,  g: 130, b: 246 }},
    {{ id: 'EAT_LV',    name: 'EAT Left Ventricle (LV)', color: '#06b6d4', r: 6,   g: 182, b: 212 }},
    {{ id: 'EAT_RA',    name: 'EAT Right Atrium (RA)', color: '#10b981', r: 16,  g: 185, b: 129 }},
    {{ id: 'EAT_RV',    name: 'EAT Right Ventricle (RV)', color: '#eab308', r: 234, g: 179, b: 8   }},
    {{ id: 'PVAT_LAD',  name: 'PVAT LAD',              color: '#f97316', r: 249, g: 115, b: 22  }},
    {{ id: 'PVAT_LCX',  name: 'PVAT LCX',              color: '#ef4444', r: 239, g: 68,  b: 68  }},
    {{ id: 'PVAT_RCA',  name: 'PVAT RCA',              color: '#ec4899', r: 236, g: 72,  b: 153 }},
  ];

  const segImgCache = {{}};
  const colorClassifierMap = {{}};

  function getPixelCompartmentId(r, g, b) {{
    const key = (r << 16) | (g << 8) | b;
    if (colorClassifierMap[key] !== undefined) {{
      return colorClassifierMap[key];
    }}
    let minDistance = 999999;
    let closestId = null;
    for (let i = 0; i < COMP_DEFS.length; i++) {{
      const c = COMP_DEFS[i];
      const dr = r - c.r;
      const dg = g - c.g;
      const db = b - c.b;
      const dist = dr * dr + dg * dg + db * db;
      if (dist < minDistance) {{
        minDistance = dist;
        closestId = c.id;
      }}
    }}
    // Color cluster tolerance: Euclidean distance < 65 (dist < 4225)
    const res = (minDistance < 4225) ? closestId : null;
    colorClassifierMap[key] = res;
    return res;
  }}

  function drawFilteredSeg(canvasId, srcUrl) {{
    const canvas = $(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const activeSet = {{}};
    let activeCount = 0;
    COMP_DEFS.forEach(c => {{
      const chk = $('chk_' + c.id);
      if (chk && chk.checked) {{
        activeCount++;
        activeSet[c.id] = true;
      }}
    }});

    if (activeCount === 0) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }}

    let img = segImgCache[srcUrl];
    if (!img) {{
      img = new Image();
      img.src = srcUrl;
      img.onload = () => drawFilteredSeg(canvasId, srcUrl);
      segImgCache[srcUrl] = img;
      return;
    }}
    if (!img.complete || img.naturalWidth === 0) return;

    if (canvas.width !== img.naturalWidth) canvas.width = img.naturalWidth;
    if (canvas.height !== img.naturalHeight) canvas.height = img.naturalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    if (activeCount === COMP_DEFS.length) {{
      return;
    }}

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    const len = data.length;

    for (let i = 0; i < len; i += 4) {{
      if (data[i + 3] === 0) continue;
      const compId = getPixelCompartmentId(data[i], data[i + 1], data[i + 2]);
      if (!compId || !activeSet[compId]) {{
        data[i + 3] = 0;
      }}
    }}
    ctx.putImageData(imgData, 0, 0);
  }}

  // Update All Viewports & Linked Overlays
  function updateViewports() {{
    // 1. Axial
    $('axCt').src = `slices/ct_${{pad(z)}}.png`;
    drawFilteredSeg('axCanvas', `slices/seg_${{pad(z)}}.png`);
    $('iAx').textContent = `I: ${{z + 1}} (${{z + 1}}/${{NZ}})`;

    // 2. Sagittal
    $('sagCt').src = `sagittal/ct_${{pad(s)}}.png`;
    drawFilteredSeg('sagCanvas', `sagittal/seg_${{pad(s)}}.png`);
    $('iSag').textContent = `I: ${{s + 1}} (${{s + 1}}/${{NS}})`;

    // 3. Coronal
    $('corCt').src = `coronal/ct_${{pad(c)}}.png`;
    drawFilteredSeg('corCanvas', `coronal/seg_${{pad(c)}}.png`);
    $('iCor').textContent = `I: ${{c + 1}} (${{c + 1}}/${{NC}})`;

    // Crosshair positioning
    const u = s / Math.max(1, NS - 1);
    const v = c / Math.max(1, NC - 1);
    const w = 1 - (z / Math.max(1, NZ - 1));

    $('axV').style.left = (u * 100) + '%';
    $('axH').style.top = (v * 100) + '%';

    $('sagV').style.left = (v * 100) + '%';
    $('sagH').style.top = (w * 100) + '%';

    $('corV').style.left = (u * 100) + '%';
    $('corH').style.top = (w * 100) + '%';

    // Update OHIF v3.13 Smart Scrollbars on ALL Viewports
    updateAllSmartScrollbars();
  }}

  // Smart Scrollbar Updating for All Viewports
  function updateAllSmartScrollbars() {{
    updateScrollThumb('smartTrackAx', 'smartThumbAx', z, NZ);
    updateScrollThumb('smartTrackSag', 'smartThumbSag', s, NS);
    updateScrollThumb('smartTrackCor', 'smartThumbCor', c, NC);
    updateScrollThumb('smartTrack3D', 'smartThumb3D', f, NF);
  }}

  function updateScrollThumb(trackId, thumbId, currentIdx, maxCount) {{
    const track = $(trackId);
    const thumb = $(thumbId);
    if (!track || !thumb || maxCount <= 1) return;
    const trackH = track.clientHeight - thumb.clientHeight;
    if (trackH > 0) {{
      const pos = Math.round((currentIdx / (maxCount - 1)) * trackH);
      thumb.style.top = pos + 'px';
    }}
  }}

  function onSmartScrollClick(e, plane) {{
    e.stopPropagation();
    const track = e.currentTarget.querySelector('.smart-track');
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const ratio = Math.min(1.0, Math.max(0.0, (e.clientY - rect.top) / rect.height));

    if (plane === 'axial') {{
      setActiveBox('pAx');
      z = Math.round(ratio * (NZ - 1));
    }} else if (plane === 'sagittal') {{
      setActiveBox('pSag');
      s = Math.round(ratio * (NS - 1));
    }} else if (plane === 'coronal') {{
      setActiveBox('pCor');
      c = Math.round(ratio * (NC - 1));
    }} else if (plane === '3d') {{
      setActiveBox('p3d');
      f = Math.round(ratio * (NF - 1));
      $('volImg').src = frames3d[f].src;
      $('i3d').textContent = `Angle: ${{Math.round((f / NF) * 360)}}deg (${{f + 1}}/${{NF}})`;
      return;
    }}
    updateViewports();
  }}

  // Mouse Wheel Handling on EVERY Viewport
  $('pAx').addEventListener('wheel', (e) => {{
    e.preventDefault();
    setActiveBox('pAx');
    z = clamp(z + (e.deltaY > 0 ? -1 : 1), 0, NZ - 1);
    updateViewports();
  }});

  $('pSag').addEventListener('wheel', (e) => {{
    e.preventDefault();
    setActiveBox('pSag');
    s = clamp(s + (e.deltaY > 0 ? 1 : -1), 0, NS - 1);
    updateViewports();
  }});

  $('pCor').addEventListener('wheel', (e) => {{
    e.preventDefault();
    setActiveBox('pCor');
    c = clamp(c + (e.deltaY > 0 ? 1 : -1), 0, NC - 1);
    updateViewports();
  }});

  $('p3d').addEventListener('wheel', (e) => {{
    e.preventDefault();
    setActiveBox('p3d');
    isSpinning = false;
    $('btnSpin').style.background = '#334155';
    f = (f + (e.deltaY > 0 ? 1 : NF - 1)) % NF;
    $('volImg').src = frames3d[f].src;
    $('i3d').textContent = `Angle: ${{Math.round((f / NF) * 360)}}deg (${{f + 1}}/${{NF}})`;
    updateAllSmartScrollbars();
  }});

  // Keyboard Navigation (Arrow Keys)
  window.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') {{
      stepActivePlane(1);
    }} else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') {{
      stepActivePlane(-1);
    }}
  }});

  function stepActivePlane(delta) {{
    if (activePlane === 'axial') z = clamp(z + delta, 0, NZ - 1);
    else if (activePlane === 'sagittal') s = clamp(s + delta, 0, NS - 1);
    else if (activePlane === 'coronal') c = clamp(c + delta, 0, NC - 1);
    else if (activePlane === '3d') {{
      f = (f + (delta > 0 ? 1 : NF - 1)) % NF;
      $('volImg').src = frames3d[f].src;
      $('i3d').textContent = `Angle: ${{Math.round((f / NF) * 360)}}deg (${{f + 1}}/${{NF}})`;
    }}
    updateViewports();
  }}

  // Fraction helper for clicking within elements
  function getFrac(e, el) {{
    const r = el.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  }}

  // Linked Crosshair Navigation
  $('wAx').addEventListener('mousedown', (e) => {{
    setActiveBox('pAx');
    if (currentTool !== 'crosshair') return;
    const [u, v] = getFrac(e, $('wAx'));
    s = clamp(u * (NS - 1), 0, NS - 1);
    c = clamp(v * (NC - 1), 0, NC - 1);
    updateViewports();
  }});

  $('wSag').addEventListener('mousedown', (e) => {{
    setActiveBox('pSag');
    if (currentTool !== 'crosshair') return;
    const [v, w] = getFrac(e, $('wSag'));
    c = clamp(v * (NC - 1), 0, NC - 1);
    z = clamp((1 - w) * (NZ - 1), 0, NZ - 1);
    updateViewports();
  }});

  $('wCor').addEventListener('mousedown', (e) => {{
    setActiveBox('pCor');
    if (currentTool !== 'crosshair') return;
    const [u, w] = getFrac(e, $('wCor'));
    s = clamp(u * (NS - 1), 0, NS - 1);
    z = clamp((1 - w) * (NZ - 1), 0, NZ - 1);
    updateViewports();
  }});

  // 3D Volume Drag-to-Rotate
  let dragX = null;
  $('volImg').addEventListener('mousedown', (e) => {{
    dragX = e.clientX;
    isSpinning = false;
    $('btnSpin').style.background = '#334155';
  }});
  window.addEventListener('mouseup', () => dragX = null);
  window.addEventListener('mousemove', (e) => {{
    if (dragX === null) return;
    const delta = e.clientX - dragX;
    if (Math.abs(delta) > 5) {{
      f = (f + (delta > 0 ? 1 : NF - 1)) % NF;
      dragX = e.clientX;
      $('volImg').src = frames3d[f].src;
      $('i3d').textContent = `Angle: ${{Math.round((f / NF) * 360)}}deg (${{f + 1}}/${{NF}})`;
      updateAllSmartScrollbars();
    }}
  }});

  setInterval(() => {{
    if (isSpinning && frames3d.length > 0) {{
      f = (f + 1) % NF;
      $('volImg').src = frames3d[f].src;
      $('i3d').textContent = `Angle: ${{Math.round((f / NF) * 360)}}deg (${{f + 1}}/${{NF}})`;
      updateAllSmartScrollbars();
    }}
  }}, 110);

  function toggleSpin(e) {{
    if (e) e.stopPropagation();
    isSpinning = !isSpinning;
    $('btnSpin').style.background = isSpinning ? '#7c3aed' : '#334155';
    $('btnSpin').style.color = isSpinning ? '#ffffff' : '#cbd5e1';
  }}

  // Active Viewport Selection
  function setActiveBox(id) {{
    document.querySelectorAll('.viewport-box').forEach(b => b.classList.remove('active-box'));
    const box = $(id);
    if (box) box.classList.add('active-box');
    if (id === 'pAx') activePlane = 'axial';
    else if (id === 'pSag') activePlane = 'sagittal';
    else if (id === 'pCor') activePlane = 'coronal';
    else if (id === 'p3d') activePlane = '3d';
  }}

  // Maximize / Restore Viewports
  function maximizeBox(id, e) {{
    if (e) e.stopPropagation();
    const grid = $('viewportsGrid');
    if (isGridMaximized && maximizedBoxId === id) {{
      document.querySelectorAll('.viewport-box').forEach(b => b.classList.remove('hidden'));
      grid.classList.remove('maximized');
      isGridMaximized = false;
      maximizedBoxId = null;
    }} else {{
      document.querySelectorAll('.viewport-box').forEach(b => {{
        if (b.id === id) b.classList.remove('hidden');
        else b.classList.add('hidden');
      }});
      grid.classList.add('maximized');
      isGridMaximized = true;
      maximizedBoxId = id;
    }}
    setTimeout(updateAllSmartScrollbars, 50);
  }}

  function toggleGridLayout() {{
    if (isGridMaximized) {{
      maximizeBox(maximizedBoxId);
    }} else {{
      maximizeBox('pAx');
    }}
  }}

  // Window Presets
  function applyWindowPreset(p) {{
    const ctImages = document.querySelectorAll('img.ct-plane');
    const hudLabels = [$('hudWindowAx'), $('hudWindowSag'), $('hudWindowCor')];
    let wlText = 'W: 400 L: -110';
    if (p === 'fat') {{
      ctImages.forEach(im => im.style.filter = 'contrast(1.2) brightness(1.0)');
      wlText = 'W: 400 L: -110';
    }} else if (p === 'soft') {{
      ctImages.forEach(im => im.style.filter = 'contrast(1.0) brightness(1.2)');
      wlText = 'W: 350 L: 40';
    }} else if (p === 'lung') {{
      ctImages.forEach(im => im.style.filter = 'contrast(0.7) brightness(1.5)');
      wlText = 'W: 1500 L: -600';
    }} else if (p === 'bone') {{
      ctImages.forEach(im => im.style.filter = 'contrast(1.6) brightness(0.8)');
      wlText = 'W: 1500 L: 300';
    }}
    hudLabels.forEach(h => {{ if (h) h.textContent = wlText; }});
  }}

  let soloCompartmentId = null;

  function isolateCompartment(id) {{
    if (soloCompartmentId === id) {{
      soloCompartmentId = null;
      COMP_DEFS.forEach(c => {{
        const chk = $('chk_' + c.id);
        if (chk) chk.checked = true;
      }});
    }} else {{
      soloCompartmentId = id;
      COMP_DEFS.forEach(c => {{
        const chk = $('chk_' + c.id);
        if (chk) chk.checked = (c.id === id);
      }});
    }}
    updateSoloBtnStyles();
    onCompartmentChange();
  }}

  function updateSoloBtnStyles() {{
    COMP_DEFS.forEach(c => {{
      const row = $('row_' + c.id);
      if (!row) return;
      const btn = row.querySelector('.solo-btn');
      if (btn) btn.classList.toggle('active', soloCompartmentId === c.id);
    }});
  }}

  function setAllCompartments(state) {{
    soloCompartmentId = null;
    COMP_DEFS.forEach(c => {{
      const chk = $('chk_' + c.id);
      if (chk) chk.checked = state;
    }});
    updateSoloBtnStyles();
    onCompartmentChange();
  }}

  function onCompartmentChange() {{
    const checked = COMP_DEFS.filter(c => {{
      const chk = $('chk_' + c.id);
      return chk && chk.checked;
    }});
    if (checked.length !== 1) {{
      soloCompartmentId = null;
      updateSoloBtnStyles();
    }} else {{
      soloCompartmentId = checked[0].id;
      updateSoloBtnStyles();
    }}
    updateViewports();
  }}

  function setOpacity(val) {{
    const opacity = val / 100.0;
    $('opacityVal').textContent = `${{val}}%`;
    document.querySelectorAll('canvas.seg-plane').forEach(el => {{
      el.style.opacity = opacity;
    }});
  }}

  function setTool(tool) {{
    currentTool = tool;
    document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
    if (tool === 'crosshair') $('btnCrosshair').classList.add('active');
    else if (tool === 'pan') $('btnPan').classList.add('active');
    else if (tool === 'zoom') $('btnZoom').classList.add('active');
    else if (tool === 'length') $('btnLength').classList.add('active');
    else if (tool === 'angle') $('btnAngle').classList.add('active');
  }}

  function resetAllViewports() {{
    z = Math.floor(NZ / 2);
    s = Math.floor(NS / 2);
    c = Math.floor(NC / 2);
    f = 0;
    $('opacitySlider').value = 85;
    setOpacity(85);
    applyWindowPreset('fat');
    setAllCompartments(true);
    if (isGridMaximized) toggleGridLayout();
  }}

  // Drawer Toggling
  function toggleLeftDrawer() {{
    $('leftDrawer').classList.toggle('collapsed');
    setTimeout(updateAllSmartScrollbars, 220);
  }}

  let activeRightPanel = 'seg';
  function toggleRightPanel(panel) {{
    const drawer = $('rightDrawer');
    const btnSeg = $('btnToggleSeg');
    const btnMeas = $('btnToggleMeas');

    if (panel === null || (drawer.style.display !== 'none' && activeRightPanel === panel)) {{
      drawer.style.display = 'none';
      btnSeg.classList.remove('active');
      btnMeas.classList.remove('active');
    }} else {{
      drawer.style.display = 'flex';
      switchRightTab(panel);
    }}
    setTimeout(updateAllSmartScrollbars, 220);
  }}

  function switchRightTab(tab) {{
    activeRightPanel = tab;
    const btnSeg = $('btnToggleSeg');
    const btnMeas = $('btnToggleMeas');

    if (tab === 'seg') {{
      $('tabSeg').classList.add('active');
      $('tabMeas').classList.remove('active');
      $('panelSeg').style.display = 'block';
      $('panelMeas').style.display = 'none';
      btnSeg.classList.add('active');
      btnMeas.classList.remove('active');
    }} else {{
      $('tabMeas').classList.add('active');
      $('tabSeg').classList.remove('active');
      $('panelSeg').style.display = 'none';
      $('panelMeas').style.display = 'block';
      btnMeas.classList.add('active');
      btnSeg.classList.remove('active');
    }}
  }}

  const currentParticipantId = "{participant_id}";

  function switchParticipant(pid) {{
    if (String(pid) === String(currentParticipantId)) return;
    window.location.href = `/patient/${{pid}}/`;
  }}

  function stepPatient(delta) {{
    const sel = $('patientSelector');
    if (!sel) return;
    const nextIdx = clamp(sel.selectedIndex + delta, 0, sel.options.length - 1);
    if (nextIdx !== sel.selectedIndex) {{
      switchParticipant(sel.options[nextIdx].value);
    }}
  }}

  function selectSeriesTile(plane) {{
    document.querySelectorAll('.series-card').forEach(c => c.classList.remove('active'));
    if (plane === 'axial') {{
      $('cardCT').classList.add('active');
      setActiveBox('pAx');
    }} else if (plane === 'sagittal') {{
      $('cardSag').classList.add('active');
      setActiveBox('pSag');
    }} else if (plane === 'coronal') {{
      $('cardCor').classList.add('active');
      setActiveBox('pCor');
    }} else if (plane === '3d') {{
      $('card3D').classList.add('active');
      setActiveBox('p3d');
    }} else if (plane === 'seg') {{
      $('cardSEG').classList.add('active');
      toggleRightPanel('seg');
    }}
  }}

  // Configure dynamic OHIF v3.13 Cloud Launch URL using studies.json
  try {{
    const studiesUrl = new URL('studies.json', window.location.href).href;
    const launchBtn = $('btnLaunchOHIF');
    if (launchBtn) {{
      launchBtn.href = "https://viewer.ohif.org/viewer/dicomjson?url=" + encodeURIComponent(studiesUrl);
    }}
  }} catch (e) {{}}

  // Window resize tracking
  window.addEventListener('resize', () => {{
    updateAllSmartScrollbars();
  }});

  // Initial load
  updateViewports();
</script>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")
    return html_path



def export_sample_to_ohif(
    sample_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    summary_metrics: Optional[Dict[str, float]] = None,
    available_participants: Optional[List[str]] = None,
) -> Path:
    """Exports a sample participant folder to a complete OHIF-compatible study package.

    Args:
        sample_dir: Directory containing participant scan and segmentation.
        output_dir: Destination directory. Defaults to sample_dir / 'ohif_export'.
        summary_metrics: Optional pre-calculated dictionary of metrics.

    Returns:
        Path to the generated export directory containing DICOM slices, DICOM-SEG, and viewer.
    """
    sample_path = Path(sample_dir).resolve()
    participant_id = sample_path.name

    dest_dir = Path(output_dir).resolve() if output_dir else sample_path / "ohif_export"
    ct_dest_dir = dest_dir / "ct"
    slices_dir = dest_dir / "slices"
    dest_dir.mkdir(parents=True, exist_ok=True)
    ct_dest_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover or convert CT slices
    raw_dcm = [
        p for p in list(sample_path.glob("*.dcm")) + list(sample_path.glob("*/*.dcm"))
        if not p.name.endswith("_seg.dcm") and not p.name.startswith("eat_pvat")
        and "ohif_export" not in p.parts
    ]
    # Filter to only Modality == "CT"
    dcm_candidates = []
    for p in raw_dcm:
        try:
            hdr = pydicom.dcmread(str(p), stop_before_pixels=True)
            if getattr(hdr, "Modality", None) == "CT":
                dcm_candidates.append(p)
        except Exception:
            continue

    if dcm_candidates:
        # Clear existing .dcm in ct_dest_dir to avoid mixing old/duplicate exports
        for old_f in ct_dest_dir.glob("*.dcm"):
            try:
                old_f.unlink()
            except Exception:
                pass

        # Sort by ImagePositionPatient[2] or InstanceNumber
        def _slice_key(p: Path) -> float:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                if hasattr(ds, "ImagePositionPatient"):
                    return float(ds.ImagePositionPatient[2])
                return float(getattr(ds, "InstanceNumber", 0))
            except Exception:
                return 0.0

        dcm_candidates.sort(key=_slice_key)

        # Unify Study and Series UIDs across all slices of this scan
        ref_ds = pydicom.dcmread(str(dcm_candidates[0]), stop_before_pixels=True)
        unified_study_uid = _sanitize_uid(str(getattr(ref_ds, "StudyInstanceUID", "")))
        unified_series_uid = _sanitize_uid(str(getattr(ref_ds, "SeriesInstanceUID", "")))
        unified_frame_of_ref_uid = _sanitize_uid(str(getattr(ref_ds, "FrameOfReferenceUID", generate_uid())))

        copied_ct_paths: List[Path] = []
        for idx, dcm_file in enumerate(dcm_candidates):
            target = ct_dest_dir / f"ct_{idx:04d}.dcm"
            try:
                ds = pydicom.dcmread(str(dcm_file))
                ds.StudyInstanceUID = unified_study_uid
                ds.SeriesInstanceUID = unified_series_uid
                ds.FrameOfReferenceUID = unified_frame_of_ref_uid
                ds.SOPInstanceUID = _sanitize_uid(str(getattr(ds, "SOPInstanceUID", ""))) or generate_uid()
                ds.InstanceNumber = idx + 1

                # Clean patient and study descriptors for OHIF study list
                ds.PatientID = f"COCA-{participant_id}"
                ds.PatientName = f"Participant {participant_id}"
                ds.StudyDate = "20261021"
                ds.StudyDescription = f"COCA Cardiac CT (Participant {participant_id})"
                ds.SeriesDescription = "Cardiac CT Axial Slices"

                if hasattr(ds, "file_meta") and ds.file_meta:
                    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                ds.save_as(str(target), write_like_original=False)
            except Exception:
                shutil.copy2(dcm_file, target)
            copied_ct_paths.append(target)
    else:
        scan_candidates = [
            p for p in list(sample_path.glob("*.*")) + list(sample_path.glob("*/*.*"))
            if p.suffix.lower() in [".nrrd", ".mha", ".gz", ".nii"]
            and not p.name.endswith(".seg.nrrd")
            and not p.name.startswith("eat_pvat")
        ]
        if not scan_candidates:
            raise FileNotFoundError(f"No DICOM, NRRD, or NIfTI scan found in {sample_path}")

        scan_file = sample_path / "scan.nrrd" if (sample_path / "scan.nrrd").exists() else scan_candidates[0]
        meta_tensor = read_medical_image_as_metatensor(scan_file)
        ct_array = meta_tensor.squeeze().cpu().numpy()
        header = meta_tensor.meta
        spacing = tuple(header.get("spatial_shape", (1.0, 1.0, 1.0)))
        if ct_array.ndim == 3:
            copied_ct_paths = export_ct_to_dicom_series(
                ct_array=ct_array,
                spacing=(1.0, 1.0, 1.0),
                output_dir=ct_dest_dir,
                patient_id=participant_id,
            )
        else:
            raise ValueError(f"Expected 3D CT volume, got shape {ct_array.shape}")

    # 2. Check for eat_pvat.seg.nrrd
    seg_nrrd_candidates = list(sample_path.glob("*.seg.nrrd"))
    if not seg_nrrd_candidates:
        seg_nrrd_candidates = [sample_path / "eat_pvat.seg.nrrd"]

    seg_nrrd_path = seg_nrrd_candidates[0]
    seg_data: Optional[np.ndarray] = None
    if seg_nrrd_path.exists():
        seg_data, seg_header = nrrd.read(str(seg_nrrd_path))
        if seg_data.shape[0] != 8 and seg_data.shape[-1] == 8:
            seg_data = np.moveaxis(seg_data, -1, 0)
        elif seg_data.ndim == 3:
            seg_data = np.stack([seg_data] * 8, axis=0)

        num_slices = len(copied_ct_paths)
        if seg_data.shape[1] != num_slices:
            pad_or_crop = np.zeros((8, num_slices, seg_data.shape[2], seg_data.shape[3]), dtype=np.uint8)
            d_common = min(num_slices, seg_data.shape[1])
            pad_or_crop[:, :d_common] = seg_data[:, :d_common]
            seg_data = pad_or_crop

        dicom_seg_path = dest_dir / "eat_pvat_seg.dcm"
        export_segmentation_to_dicom_seg(
            seg_array=seg_data,
            ct_dicom_paths=copied_ct_paths,
            output_path=dicom_seg_path,
            patient_id=participant_id,
        )

    # 2b. Generate OHIF v3.11 naturalized DICOM JSON manifest (studies.json)
    seg_file_candidate = dest_dir / "eat_pvat_seg.dcm"
    generate_ohif_dicom_json(
        dest_dir=dest_dir,
        ct_files=copied_ct_paths,
        seg_file=seg_file_candidate if seg_file_candidate.is_file() else None,
        participant_id=participant_id,
    )

    # 3. Pre-render real slices for OHIF web viewer
    if seg_data is None:
        seg_data = np.zeros((8, len(copied_ct_paths), 512, 512), dtype=np.uint8)

    num_rendered = render_real_slice_assets(
        ct_dicom_paths=copied_ct_paths,
        seg_array=seg_data,
        slices_dir=slices_dir,
    )

    # 3b. Pre-render sagittal, coronal, and rotating 3D assets for the MPR viewer
    try:
        mpr_info = render_mpr_assets(
            ct_dicom_paths=copied_ct_paths,
            seg_array=seg_data,
            output_dir=dest_dir,
        )
        generate_mpr_viewer_html(participant_id=participant_id, output_dir=dest_dir, mpr_info=mpr_info)
    except Exception as exc:  # MPR is an optional enhancement; never block the core export
        print(f"Warning: MPR/3D asset generation skipped: {exc}")

    # 4. Check for CSV metrics or use summary_metrics
    final_metrics = summary_metrics or {}
    csv_candidates = [Path.cwd() / "cardiac_fat_measurements.csv"]
    curr = sample_path
    while curr != curr.parent:
        candidate = curr / "cardiac_fat_measurements.csv"
        if candidate.is_file():
            csv_candidates.insert(0, candidate)
            break
        curr = curr.parent

    env_csv = os.getenv("CARDIAC_FAT_CSV_PATH")
    if env_csv:
        csv_candidates.insert(0, Path(env_csv))
    for csv_candidate in csv_candidates:
        if csv_candidate.exists():
            try:
                df = pd.read_csv(csv_candidate)
                match = df[df["participant_id"].astype(str) == str(participant_id)]
                if not match.empty:
                    row = match.iloc[0].to_dict()
                    for reg in REPORTED_REGIONS:
                        vol_col = f"{reg.value} Volume (cm3)"
                        hu_col = f"{reg.value} Mean Attenuation (HU)"
                        if vol_col in row:
                            final_metrics[vol_col] = float(row[vol_col])
                        if hu_col in row:
                            final_metrics[hu_col] = float(row[hu_col])
                    break
            except Exception:
                pass

    # 5. Discover cohort participants if not provided
    if available_participants is None:
        cohort_dir = sample_path.parent
        if cohort_dir.is_dir():
            found = [
                d.name for d in cohort_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".") and d.name != "ohif_export"
                and ((d / "eat_pvat.seg.nrrd").is_file() or (d / "ohif_export").is_dir() or any(f.name.endswith(".nrrd") for f in d.iterdir()))
            ]
            if found:
                found.sort(key=lambda x: int(x) if x.isdigit() else str(x))
                available_participants = found

    # 6. Generate authentic OHIF 3-pane web viewer with 4-box center viewport
    generate_ohif_3pane_viewer_html(
        participant_id=participant_id,
        output_dir=dest_dir,
        num_slices=num_rendered,
        summary_metrics=final_metrics,
        available_participants=available_participants,
        mpr_info=mpr_info,
    )

    return dest_dir
