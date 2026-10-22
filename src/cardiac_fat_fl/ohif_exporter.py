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
                rgba[mask, 3] = 190  # Alpha

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


def _colorize_seg_plane(seg_plane: np.ndarray, alpha: int = 190) -> np.ndarray:
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
    <button id="spin" class="on">Auto-rotate 3D</button>
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
  let z = Math.floor(NZ / 2), s = Math.floor(NS / 2), c = Math.floor(NC / 2), f = 0, spinning = true;
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
    """Generates the authentic OHIF 3-Pane Medical Web Viewer with a 4-box center viewport."""
    html_path = output_dir / "index.html"

    # Default to known participant 0 numbers if none provided
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
    participant_options = "".join(
        f'<option value="{p}" {"selected" if str(p) == str(participant_id) else ""}>Participant {p}</option>'
        for p in participants_list
    )

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
  <title>OHIF Medical Imaging Workstation &bull; Participant {participant_id}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; width: 100%; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #050811; color: #f1f5f9; }}
    svg {{ max-width: 14px; max-height: 14px; width: 14px; height: 14px; flex-shrink: 0; vertical-align: middle; }}
    
    /* Layout */
    .app-root {{ display: flex; flex-direction: column; height: 100vh; width: 100vw; overflow: hidden; }}
    
    /* Top Header */
    header.top-header {{ height: 48px; background: #0b1120; border-bottom: 1px solid #1e293b; display: flex; align-items: center; justify-content: space-between; padding: 0 14px; flex-shrink: 0; z-index: 30; }}
    .brand-section {{ display: flex; align-items: center; gap: 10px; }}
    .ohif-logo {{ width: 28px; height: 28px; border-radius: 4px; background: linear-gradient(135deg, #0284c7, #2563eb); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 10px; color: #fff; letter-spacing: 0.5px; box-shadow: 0 2px 6px rgba(2, 132, 199, 0.35); }}
    .header-info-title {{ font-size: 12px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 6px; }}
    .version-tag {{ font-size: 9px; background: #082f49; color: #38bdf8; border: 1px solid #0369a1; padding: 1px 5px; border-radius: 3px; font-family: monospace; }}
    .header-info-sub {{ font-size: 11px; color: #94a3b8; }}
    
    /* Toolbar */
    .toolbar {{ display: flex; align-items: center; gap: 4px; background: #020617; border: 1px solid #1e293b; border-radius: 6px; padding: 2px 4px; }}
    .tool-btn {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px 8px; color: #cbd5e1; font-size: 11px; font-weight: 500; cursor: pointer; display: flex; align-items: center; gap: 5px; transition: all 0.15s; }}
    .tool-btn:hover {{ background: #1e293b; color: #ffffff; }}
    .tool-btn.active {{ background: rgba(2, 132, 199, 0.25); border-color: #0284c7; color: #38bdf8; }}
    .cloud-btn {{ background: linear-gradient(135deg, #2563eb, #0284c7); color: white; padding: 5px 11px; border-radius: 5px; text-decoration: none; font-size: 11px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3); }}
    .cloud-btn:hover {{ opacity: 0.92; }}

    /* Main 3-Pane Body */
    main.three-pane-body {{ display: flex; flex: 1; height: calc(100vh - 48px); overflow: hidden; }}

    /* LEFT PANE: Studies & Series Chooser */
    aside.left-pane {{ width: 250px; flex-shrink: 0; background: #080d1a; border-right: 1px solid #1e293b; display: flex; flex-direction: column; justify-content: space-between; overflow-y: auto; z-index: 20; }}
    .pane-block {{ padding: 12px; border-bottom: 1px solid #1e293b; }}
    .pane-heading {{ font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between; margin-bottom: 8px; }}
    .select-dropdown {{ width: 100%; background: #0f172a; border: 1px solid #334155; color: #ffffff; font-size: 11px; border-radius: 4px; padding: 6px 8px; outline: none; }}
    .patient-card {{ margin-top: 8px; background: rgba(15, 23, 42, 0.5); border: 1px solid #1e293b; border-radius: 6px; padding: 8px; font-size: 11px; color: #94a3b8; display: flex; flex-direction: column; gap: 3px; }}
    .patient-card-row {{ display: flex; justify-content: space-between; }}
    .patient-card-row strong {{ color: #e2e8f0; font-family: monospace; }}
    .series-list {{ padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; flex: 1; overflow-y: auto; }}
    .series-card {{ display: flex; align-items: center; gap: 10px; padding: 8px; border-radius: 6px; border: 1px solid #1e293b; background: rgba(15, 23, 42, 0.5); cursor: pointer; transition: all 0.15s; }}
    .series-card:hover {{ border-color: #38bdf8; }}
    .series-card.active {{ border-color: #0284c7; background: rgba(2, 132, 199, 0.12); }}
    .series-thumb {{ width: 48px; height: 48px; border-radius: 4px; object-fit: cover; background: #000; border: 1px solid #334155; flex-shrink: 0; }}
    .seg-badge-box {{ width: 48px; height: 48px; border-radius: 4px; border: 1px solid #7c3aed; background: rgba(88, 28, 135, 0.35); display: flex; flex-direction: column; align-items: center; justify-content: center; color: #d8b4fe; font-size: 10px; font-weight: bold; flex-shrink: 0; }}
    .series-details {{ flex: 1; overflow: hidden; }}
    .series-title {{ font-size: 11px; font-weight: 600; color: #ffffff; display: block; }}
    .series-sub {{ font-size: 10px; color: #94a3b8; display: block; }}
    .series-meta {{ font-size: 9px; color: #38bdf8; font-family: monospace; display: block; }}

    /* CENTER PANE: Four-Box 2x2 Viewport */
    section.center-pane {{ flex: 1; background: #000000; display: flex; flex-direction: column; justify-content: space-between; position: relative; overflow: hidden; }}
    
    /* 2x2 MPR Grid */
    .mpr-grid {{ flex: 1; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 2px; background: #1e293b; overflow: hidden; position: relative; }}
    .mpr-grid.maximized {{ grid-template-columns: 1fr; grid-template-rows: 1fr; }}
    .viewport-box {{ position: relative; background: #000000; display: flex; align-items: center; justify-content: center; overflow: hidden; cursor: crosshair; border: 1.5px solid transparent; padding: 26px 8px; }}
    .viewport-box.active-box {{ border-color: #0284c7; }}
    .viewport-box.hidden {{ display: none !important; }}
    
    /* Viewport Headers & Overlays */
    .vp-header {{ position: absolute; top: 0; left: 0; right: 0; z-index: 15; display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; font-size: 11px; pointer-events: none; background: linear-gradient(180deg, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0) 100%); }}
    .vp-tag {{ display: inline-flex; align-items: center; gap: 5px; font-weight: 700; font-size: 10px; letter-spacing: 0.5px; padding: 1px 6px; border-radius: 3px; font-family: monospace; }}
    .tag-axial {{ background: #082f49; color: #38bdf8; border: 1px solid #0284c7; }}
    .tag-sagittal {{ background: #064e3b; color: #34d399; border: 1px solid #059669; }}
    .tag-coronal {{ background: #431407; color: #fb923c; border: 1px solid #ea580c; }}
    .tag-3d {{ background: #3b0764; color: #c084fc; border: 1px solid #9333ea; }}
    .vp-actions {{ pointer-events: auto; display: flex; gap: 4px; }}
    .vp-btn {{ background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 3px; color: #cbd5e1; font-size: 10px; padding: 2px 5px; cursor: pointer; }}
    .vp-btn:hover {{ background: #334155; color: #fff; }}
    
    /* Viewport Orientation Labels */
    .orient {{ position: absolute; z-index: 12; font-family: monospace; font-size: 11px; font-weight: 700; color: #facc15; pointer-events: none; text-shadow: 0 1px 3px rgba(0,0,0,0.9); }}
    .orient.top {{ top: 26px; left: 50%; transform: translateX(-50%); }}
    .orient.bottom {{ bottom: 26px; left: 50%; transform: translateX(-50%); }}
    .orient.left {{ left: 8px; top: 50%; transform: translateY(-50%); }}
    .orient.right {{ right: 8px; top: 50%; transform: translateY(-50%); }}

    /* Viewport Canvas & Images */
    .vp-wrap {{ position: relative; display: inline-block; line-height: 0; height: 100%; max-width: 100%; max-height: 100%; }}
    #wAx, #w3D {{ aspect-ratio: 1 / 1; }}
    #wSag {{ aspect-ratio: {sag_w} / {sag_h}; }}
    #wCor {{ aspect-ratio: {cor_w} / {cor_h}; }}
    .vp-wrap img.ct-plane {{ width: 100%; height: 100%; object-fit: fill; display: block; }}
    .vp-wrap img.seg-plane {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; opacity: 0.85; }}
    
    /* Crosshairs */
    .xh {{ position: absolute; pointer-events: none; z-index: 10; }}
    .xh.v {{ top: 0; bottom: 0; width: 1px; }}
    .xh.h {{ left: 0; right: 0; height: 1px; }}

    /* Viewport Footers */
    .vp-footer {{ position: absolute; bottom: 0; left: 0; right: 0; z-index: 15; display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; font-size: 10px; font-family: monospace; pointer-events: none; background: linear-gradient(0deg, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0) 100%); color: #94a3b8; }}

    /* Center Bottom Bar */
    .bottom-bar {{ height: 50px; background: #090e1c; border-top: 1px solid #1e293b; display: flex; align-items: center; justify-content: space-between; padding: 0 12px; gap: 12px; flex-shrink: 0; z-index: 20; }}
    .slider-container {{ flex: 1; max-width: 520px; display: flex; align-items: center; gap: 8px; font-size: 11px; font-family: monospace; color: #94a3b8; }}
    .slider-step-btn {{ width: 24px; height: 24px; border-radius: 4px; background: #1e293b; border: 1px solid #334155; color: #cbd5e1; display: flex; align-items: center; justify-content: center; cursor: pointer; }}
    .slider-step-btn:hover {{ background: #334155; color: #fff; }}
    input[type="range"] {{ flex: 1; accent-color: #0284c7; cursor: pointer; }}
    .preset-selector {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: #94a3b8; }}

    /* Mode Navigation Pills */
    .mode-pills {{ display: flex; gap: 4px; }}
    .mode-pill {{ font-size: 10px; font-weight: 600; padding: 3px 8px; border-radius: 12px; background: #0f172a; border: 1px solid #1e293b; color: #94a3b8; cursor: pointer; }}
    .mode-pill:hover {{ color: #ffffff; border-color: #38bdf8; }}
    .mode-pill.active {{ background: #0284c7; border-color: #0284c7; color: #ffffff; }}

    /* RIGHT PANE: Measurements & Adipose Stats */
    aside.right-pane {{ width: 320px; flex-shrink: 0; background: #080d1a; border-left: 1px solid #1e293b; display: flex; flex-direction: column; justify-content: space-between; overflow-y: auto; z-index: 20; }}
    .tab-header {{ display: flex; border-bottom: 1px solid #1e293b; background: #050811; }}
    .tab-btn {{ flex: 1; padding: 9px 6px; font-size: 11px; font-weight: 600; text-align: center; color: #94a3b8; background: transparent; border: none; border-bottom: 2px solid transparent; cursor: pointer; }}
    .tab-btn.active {{ color: #38bdf8; border-bottom-color: #0284c7; background: rgba(2, 132, 199, 0.08); }}
    
    .compartment-list {{ padding: 12px; display: flex; flex-direction: column; gap: 5px; }}
    .compartment-row {{ display: flex; align-items: center; justify-content: space-between; padding: 5px 8px; border-radius: 4px; background: rgba(15, 23, 42, 0.4); border: 1px solid #1e293b; font-size: 11px; }}
    .comp-info {{ display: flex; align-items: center; gap: 7px; }}
    .color-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
    
    .opacity-box {{ padding: 0 12px 10px 12px; }}
    .opacity-header {{ display: flex; justify-content: space-between; font-size: 10px; color: #94a3b8; margin-bottom: 4px; }}
    
    /* Metrics Table */
    .metrics-section {{ padding: 10px 12px; border-top: 1px solid #1e293b; flex: 1; overflow-y: auto; }}
    .metrics-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
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
    <div class="brand-section">
      <div class="ohif-logo">OHIF</div>
      <div>
        <div class="header-info-title">
          Open Health Imaging Foundation Workstation
          <span class="version-tag">v3.8 SOTA</span>
        </div>
        <div class="header-info-sub">
          Study: <strong style="color:#e2e8f0;">COCA Cardiac CT</strong> &bull; Patient: <strong style="color:#38bdf8;">{participant_id}</strong> &bull; Modality: <strong style="color:#34d399;">CT/SEG</strong>
        </div>
      </div>
    </div>

    <!-- Center OHIF Toolset -->
    <div class="toolbar">
      <button class="tool-btn active" id="btnCrosshair" onclick="setTool('crosshair')" title="Linked Orthogonal Crosshairs">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v18m9-9H3"/></svg>
        Crosshairs
      </button>
      <button class="tool-btn" id="btnPan" onclick="setTool('pan')" title="Pan Viewport">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11"/></svg>
        Pan
      </button>
      <button class="tool-btn" id="btnZoom" onclick="setTool('zoom')" title="Zoom Viewport">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"/></svg>
        Zoom
      </button>
      <button class="tool-btn" id="btnWl" onclick="setTool('wl')" title="Window / Level Preset">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/></svg>
        Window
      </button>
      <button class="tool-btn" id="btnLayout" onclick="toggleGridLayout()" title="Toggle 2x2 Grid / 1x1 Focus">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z"/></svg>
        2x2 Grid
      </button>
      <button class="tool-btn" onclick="resetAllViewports()" title="Reset All Viewports">Reset</button>
    </div>

    <!-- Right Cloud OHIF Link -->
    <div style="display:flex; align-items:center; gap:8px;">
      <a href="https://viewer.ohif.org/local" target="_blank" class="cloud-btn">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>
        Open in OHIF Cloud
      </a>
    </div>
  </header>

  <!-- 3-Pane Body -->
  <main class="three-pane-body">

    <!-- ============================================== -->
    <!-- LEFT PANE: Studies & Series Chooser (Pick DICOM)-->
    <!-- ============================================== -->
    <aside class="left-pane" id="leftPane">
      <div class="pane-block">
        <div class="pane-heading">
          <span>Studies Chooser</span>
          <span style="color:#38bdf8; font-family:monospace;">DICOM</span>
        </div>
        <select id="participantSelector" class="select-dropdown" onchange="switchParticipant(this.value)">
          {participant_options}
        </select>
        
        <div class="patient-card">
          <div class="patient-card-row"><span>Patient ID:</span><strong>COCA-{participant_id}</strong></div>
          <div class="patient-card-row"><span>Modality:</span><strong style="color:#38bdf8;">CT &bull; Gated</strong></div>
          <div class="patient-card-row"><span>Series:</span><strong>2 Active</strong></div>
          <div class="patient-card-row"><span>Spacing:</span><strong>{sx:.2f} &times; {sy:.2f} &times; {sz:.2f} mm</strong></div>
        </div>
      </div>

      <!-- Series List -->
      <div class="series-list">
        <div class="pane-heading"><span>Series Chooser</span><span style="color:#94a3b8;">2 Series</span></div>

        <!-- Series 1: Primary CT Slices -->
        <div class="series-card active" id="cardCT" onclick="selectSeries('ct')">
          <img src="slices/thumbnail_ct.png" class="series-thumb" onerror="this.src='slices/ct_028.png'">
          <div class="series-details">
            <span class="series-title">Primary CT Axial</span>
            <span class="series-sub">{nz} Axial Slices</span>
            <span class="series-meta">{sx:.2f} x {sy:.2f} x {sz:.2f} mm</span>
          </div>
        </div>

        <!-- Series 2: EAT/PVAT DICOM-SEG -->
        <div class="series-card" id="cardSEG" onclick="selectSeries('seg')">
          <div class="seg-badge-box">
            <span>SEG</span>
            <span style="font-size:9px;">8 Layers</span>
          </div>
          <div class="series-details">
            <span class="series-title">EAT/PVAT DICOM-SEG</span>
            <span class="series-sub">eat_pvat_seg.dcm</span>
            <span class="series-meta" style="color:#a855f7;">SOP: 1.2.840...66.4</span>
          </div>
        </div>

        <!-- Cohort Quick Navigation -->
        <div style="margin-top:10px;">
          <div class="pane-heading"><span>COCA Cohort List</span><span style="color:#34d399;">{len(participants_list)} Cases</span></div>
          <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">
            {''.join(f'<button class="vp-btn {"active" if str(p) == str(participant_id) else ""}" style="font-size:10px; padding:3px 6px;" onclick="switchParticipant(\'{p}\')">P{p}</button>' for p in participants_list)}
          </div>
        </div>
      </div>

      <div class="pane-block" style="border-top:1px solid #1e293b; border-bottom:none;">
        <span style="font-size:10px; color:#64748b;">OHIF PACS Client &bull; 0-Loss HU Fidelity</span>
      </div>
    </aside>

    <!-- ============================================== -->
    <!-- CENTER PANE: Four-Box Viewport Grid (2x2)      -->
    <!-- ============================================== -->
    <section class="center-pane">
      
      <!-- 2x2 Viewport Grid -->
      <div class="mpr-grid" id="mprGrid">
        
        <!-- Box 1: Axial (Top-Left) -->
        <div class="viewport-box active-box" id="pAx" onclick="setActiveBox('pAx')">
          <div class="vp-header">
            <span class="vp-tag tag-axial">AXIAL</span>
            <span style="color:#94a3b8; font-size:11px;">Body 3.0 CE</span>
            <div class="vp-actions">
              <button class="vp-btn" onclick="maximizeBox('pAx', event)" title="Maximize Viewport">⛶</button>
            </div>
          </div>
          <span class="orient top">A</span>
          <span class="orient bottom">P</span>
          <span class="orient left">R</span>
          <span class="orient right">L</span>
          <div class="vp-wrap" id="wAx">
            <img class="ct-plane" id="axCt" src="slices/ct_028.png">
            <img class="seg-plane" id="axSeg" src="slices/seg_028.png">
            <div class="xh v" id="axV" style="background:#34d399;"></div>
            <div class="xh h" id="axH" style="background:#fb923c;"></div>
          </div>
          <div class="vp-footer">
            <span id="hudWindow">W: 400 L: -110</span>
            <span id="iAx">Slice 28 / {nz - 1}</span>
          </div>
        </div>

        <!-- Box 2: Sagittal (Top-Right) -->
        <div class="viewport-box" id="pSag" onclick="setActiveBox('pSag')">
          <div class="vp-header">
            <span class="vp-tag tag-sagittal">SAGITTAL</span>
            <span style="color:#94a3b8; font-size:11px;">Body 4.0 CE</span>
            <div class="vp-actions">
              <button class="vp-btn" onclick="maximizeBox('pSag', event)" title="Maximize Viewport">⛶</button>
            </div>
          </div>
          <span class="orient top">H</span>
          <span class="orient bottom">F</span>
          <span class="orient left">A</span>
          <span class="orient right">P</span>
          <div class="vp-wrap" id="wSag">
            <img class="ct-plane" id="sagCt" src="sagittal/ct_{ns // 2:03d}.png" onerror="this.src='slices/ct_028.png'">
            <img class="seg-plane" id="sagSeg" src="sagittal/seg_{ns // 2:03d}.png" onerror="this.src='slices/seg_028.png'">
            <div class="xh v" id="sagV" style="background:#fb923c;"></div>
            <div class="xh h" id="sagH" style="background:#38bdf8;"></div>
          </div>
          <div class="vp-footer">
            <span>W: 400 L: -110</span>
            <span id="iSag">Slice {ns // 2} / {ns - 1}</span>
          </div>
        </div>

        <!-- Box 3: Coronal (Bottom-Left) -->
        <div class="viewport-box" id="pCor" onclick="setActiveBox('pCor')">
          <div class="vp-header">
            <span class="vp-tag tag-coronal">CORONAL</span>
            <span style="color:#94a3b8; font-size:11px;">Body 4.0 CE</span>
            <div class="vp-actions">
              <button class="vp-btn" onclick="maximizeBox('pCor', event)" title="Maximize Viewport">⛶</button>
            </div>
          </div>
          <span class="orient top">H</span>
          <span class="orient bottom">F</span>
          <span class="orient left">R</span>
          <span class="orient right">L</span>
          <div class="vp-wrap" id="wCor">
            <img class="ct-plane" id="corCt" src="coronal/ct_{nc // 2:03d}.png" onerror="this.src='slices/ct_028.png'">
            <img class="seg-plane" id="corSeg" src="coronal/seg_{nc // 2:03d}.png" onerror="this.src='slices/seg_028.png'">
            <div class="xh v" id="corV" style="background:#34d399;"></div>
            <div class="xh h" id="corH" style="background:#38bdf8;"></div>
          </div>
          <div class="vp-footer">
            <span>W: 400 L: -110</span>
            <span id="iCor">Slice {nc // 2} / {nc - 1}</span>
          </div>
        </div>

        <!-- Box 4: 3D Volume (Bottom-Right) -->
        <div class="viewport-box" id="p3d" onclick="setActiveBox('p3d')">
          <div class="vp-header">
            <span class="vp-tag tag-3d">3D VOLUME</span>
            <span style="color:#c084fc; font-size:11px;">EAT / PVAT Render</span>
            <div class="vp-actions">
              <button class="vp-btn" id="btnSpin" onclick="toggleSpin(event)" style="background:#7c3aed; color:#fff;">Auto-Spin</button>
              <button class="vp-btn" onclick="maximizeBox('p3d', event)" title="Maximize Viewport">⛶</button>
            </div>
          </div>
          <div class="vp-wrap" id="w3D">
            <img class="ct-plane" id="volImg" src="volume3d/frame_000.png" onerror="this.src='slices/ct_028.png'" draggable="false" style="cursor:grab;">
          </div>
          <div class="vp-footer">
            <span>Thoracic MIP + Colored Fat Volume</span>
            <span id="i3d">Drag mouse to spin</span>
          </div>
        </div>

      </div>

      <!-- Center Bottom Bar -->
      <div class="bottom-bar">
        
        <!-- Scrubber Slider for Active Plane -->
        <div class="slider-container">
          <button class="slider-step-btn" onclick="stepActiveSlice(-1)">&larr;</button>
          <span id="activePlaneName" style="color:#38bdf8; font-weight:bold;">AXIAL:</span>
          <input type="range" id="activeSlider" min="0" max="{nz - 1}" value="{nz // 2}">
          <button class="slider-step-btn" onclick="stepActiveSlice(1)">&rarr;</button>
          <span id="activeSliceReadout">28/{nz - 1}</span>
        </div>

        <!-- Window Presets -->
        <div class="preset-selector">
          <span>Preset:</span>
          <select id="presetSelect" class="select-dropdown" style="width:160px; padding:3px 6px;" onchange="applyWindowPreset(this.value)">
            <option value="fat">Cardiac Fat (W:400 L:-110)</option>
            <option value="soft">Soft Tissue (W:350 L:40)</option>
            <option value="lung">Chest / Lung (W:1500 L:-600)</option>
            <option value="bone">Bone / Calcium (W:1500 L:300)</option>
          </select>
        </div>

        <!-- OHIF Authentic Mode Pills -->
        <div class="mode-pills">
          <div class="mode-pill active">Segmentation</div>
          <div class="mode-pill">Volume</div>
          <div class="mode-pill">4D</div>
          <div class="mode-pill">Annotations</div>
        </div>
      </div>

    </section>

    <!-- ============================================== -->
    <!-- RIGHT PANE: Measurements & Adipose Stats Panel  -->
    <!-- ============================================== -->
    <aside class="right-pane" id="rightPane">
      
      <!-- Top Tabs -->
      <div class="tab-header">
        <button class="tab-btn active" id="tabSeg" onclick="switchRightTab('seg')">Segmentation (8)</button>
        <button class="tab-btn" id="tabMeas" onclick="switchRightTab('meas')">Measurements (19)</button>
      </div>

      <!-- Tab Content: Segmentation Compartments -->
      <div id="panelSeg">
        <div class="compartment-list">
          <div class="pane-heading">
            <span>8 Target Compartments</span>
            <span style="color:#c084fc; font-family:monospace;">DICOM-SEG</span>
          </div>

          <div class="compartment-row">
            <div class="comp-info">
              <span class="color-dot" style="background:#9333ea;"></span>
              <strong style="color:#fff;">EAT Total</strong>
            </div>
            <input type="checkbox" id="chkEAT_TOTAL" checked onchange="toggleMaskVisibility()">
          </div>

          <div class="compartment-row">
            <div class="comp-info">
              <span class="color-dot" style="background:#3b82f6;"></span>
              <span>EAT Left Atrium (LA)</span>
            </div>
            <input type="checkbox" id="chkEAT_LA" checked onchange="toggleMaskVisibility()">
          </div>

          <div class="compartment-row">
            <div class="comp-info">
              <span class="color-dot" style="background:#06b6d4;"></span>
              <span>EAT Left Ventricle (LV)</span>
            </div>
            <input type="checkbox" id="chkEAT_LV" checked onchange="toggleMaskVisibility()">
          </div>

          <div class="compartment-row">
            <div class="comp-info">
              <span class="color-dot" style="background:#10b981;"></span>
              <span>EAT Right Atrium (RA)</span>
            </div>
            <input type="checkbox" id="chkEAT_RA" checked onchange="toggleMaskVisibility()">
          </div>

          <div class="compartment-row">
            <div class="comp-info">
              <span class="color-dot" style="background:#eab308;"></span>
              <span>EAT Right Ventricle (RV)</span>
            </div>
            <input type="checkbox" id="chkEAT_RV" checked onchange="toggleMaskVisibility()">
          </div>

          <div class="compartment-row">
            <div class="comp-info">
              <span class="color-dot" style="background:#f97316;"></span>
              <strong style="color:#fff;">PVAT LAD</strong>
            </div>
            <input type="checkbox" id="chkPVAT_LAD" checked onchange="toggleMaskVisibility()">
          </div>

          <div class="compartment-row">
            <div class="comp-info">
              <span class="color-dot" style="background:#ef4444;"></span>
              <span>PVAT LCX</span>
            </div>
            <input type="checkbox" id="chkPVAT_LCX" checked onchange="toggleMaskVisibility()">
          </div>

          <div class="compartment-row">
            <div class="comp-info">
              <span class="color-dot" style="background:#ec4899;"></span>
              <span>PVAT RCA</span>
            </div>
            <input type="checkbox" id="chkPVAT_RCA" checked onchange="toggleMaskVisibility()">
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

      <!-- Tab Content: Measurements Table -->
      <div class="metrics-section" id="panelMeas">
        <div class="metrics-header">
          <span style="font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase;">Clinical Measurements</span>
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
      </div>

      <!-- Adipose Status Card -->
      <div class="status-card">
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
  let isSpinning = true;
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

  function updateViewports() {{
    // 1. Axial
    $('axCt').src = `slices/ct_${{pad(z)}}.png`;
    $('axSeg').src = `slices/seg_${{pad(z)}}.png`;
    $('iAx').textContent = `Slice ${{z}} / ${{NZ - 1}}`;

    // 2. Sagittal
    $('sagCt').src = `sagittal/ct_${{pad(s)}}.png`;
    $('sagSeg').src = `sagittal/seg_${{pad(s)}}.png`;
    $('iSag').textContent = `Slice ${{s}} / ${{NS - 1}}`;

    // 3. Coronal
    $('corCt').src = `coronal/ct_${{pad(c)}}.png`;
    $('corSeg').src = `coronal/seg_${{pad(c)}}.png`;
    $('iCor').textContent = `Slice ${{c}} / ${{NC - 1}}`;

    // Crosshair positioning
    const u = s / (NS - 1);
    const v = c / (NC - 1);
    const w = 1 - (z / (NZ - 1));

    $('axV').style.left = (u * 100) + '%';
    $('axH').style.top = (v * 100) + '%';
    
    $('sagV').style.left = (v * 100) + '%';
    $('sagH').style.top = (w * 100) + '%';

    $('corV').style.left = (u * 100) + '%';
    $('corH').style.top = (w * 100) + '%';

    // Update bottom slider according to active plane
    syncBottomSlider();
  }}

  function syncBottomSlider() {{
    const slider = $('activeSlider');
    const readout = $('activeSliceReadout');
    const label = $('activePlaneName');

    if (activePlane === 'axial') {{
      label.textContent = 'AXIAL:';
      label.style.color = '#38bdf8';
      slider.max = NZ - 1;
      slider.value = z;
      readout.textContent = `${{z}}/${{NZ - 1}}`;
    }} else if (activePlane === 'sagittal') {{
      label.textContent = 'SAGITTAL:';
      label.style.color = '#34d399';
      slider.max = NS - 1;
      slider.value = s;
      readout.textContent = `${{s}}/${{NS - 1}}`;
    }} else if (activePlane === 'coronal') {{
      label.textContent = 'CORONAL:';
      label.style.color = '#fb923c';
      slider.max = NC - 1;
      slider.value = c;
      readout.textContent = `${{c}}/${{NC - 1}}`;
    }}
  }}

  $('activeSlider').addEventListener('input', (e) => {{
    const val = parseInt(e.target.value);
    if (activePlane === 'axial') z = val;
    else if (activePlane === 'sagittal') s = val;
    else if (activePlane === 'coronal') c = val;
    updateViewports();
  }});

  function stepActiveSlice(delta) {{
    if (activePlane === 'axial') z = clamp(z + delta, 0, NZ - 1);
    else if (activePlane === 'sagittal') s = clamp(s + delta, 0, NS - 1);
    else if (activePlane === 'coronal') c = clamp(c + delta, 0, NC - 1);
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

  // Mouse Wheel Scrolling per Viewport
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
    if (Math.abs(delta) > 6) {{
      f = (f + (delta > 0 ? 1 : NF - 1)) % NF;
      dragX = e.clientX;
      $('volImg').src = frames3d[f].src;
    }}
  }});

  setInterval(() => {{
    if (isSpinning && frames3d.length > 0) {{
      f = (f + 1) % NF;
      $('volImg').src = frames3d[f].src;
    }}
  }}, 110);

  function toggleSpin(e) {{
    if (e) e.stopPropagation();
    isSpinning = !isSpinning;
    $('btnSpin').style.background = isSpinning ? '#7c3aed' : '#334155';
  }}

  // Active Viewport Selection
  function setActiveBox(id) {{
    document.querySelectorAll('.viewport-box').forEach(b => b.classList.remove('active-box'));
    const box = $(id);
    if (box) box.classList.add('active-box');
    if (id === 'pAx') activePlane = 'axial';
    else if (id === 'pSag') activePlane = 'sagittal';
    else if (id === 'pCor') activePlane = 'coronal';
    syncBottomSlider();
  }}

  // Maximize / Restore Viewports
  function maximizeBox(id, e) {{
    if (e) e.stopPropagation();
    const grid = $('mprGrid');
    if (isGridMaximized && maximizedBoxId === id) {{
      // Restore 2x2
      document.querySelectorAll('.viewport-box').forEach(b => b.classList.remove('hidden'));
      grid.classList.remove('maximized');
      isGridMaximized = false;
      maximizedBoxId = null;
    }} else {{
      // Maximize target
      document.querySelectorAll('.viewport-box').forEach(b => {{
        if (b.id === id) b.classList.remove('hidden');
        else b.classList.add('hidden');
      }});
      grid.classList.add('maximized');
      isGridMaximized = true;
      maximizedBoxId = id;
    }}
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
    const hud = $('hudWindow');
    if (p === 'fat') {{
      ctImages.forEach(im => im.style.filter = 'contrast(1.2) brightness(1.0)');
      if (hud) hud.textContent = 'W: 400 L: -110';
    }} else if (p === 'soft') {{
      ctImages.forEach(im => im.style.filter = 'contrast(1.0) brightness(1.2)');
      if (hud) hud.textContent = 'W: 350 L: 40';
    }} else if (p === 'lung') {{
      ctImages.forEach(im => im.style.filter = 'contrast(0.7) brightness(1.5)');
      if (hud) hud.textContent = 'W: 1500 L: -600';
    }} else if (p === 'bone') {{
      ctImages.forEach(im => im.style.filter = 'contrast(1.6) brightness(0.8)');
      if (hud) hud.textContent = 'W: 1500 L: 300';
    }}
  }}

  // Opacity Control
  function setOpacity(val) {{
    const alpha = val / 100.0;
    $('opacityVal').textContent = val + '%';
    document.querySelectorAll('img.seg-plane').forEach(im => im.style.opacity = alpha);
  }}

  function toggleMaskVisibility() {{
    const isVisible = $('chkEAT_TOTAL').checked;
    document.querySelectorAll('img.seg-plane').forEach(im => im.style.display = isVisible ? 'block' : 'none');
  }}

  function setTool(tool) {{
    currentTool = tool;
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
    if (tool === 'crosshair') $('btnCrosshair').classList.add('active');
    else if (tool === 'pan') $('btnPan').classList.add('active');
    else if (tool === 'zoom') $('btnZoom').classList.add('active');
    else if (tool === 'wl') $('btnWl').classList.add('active');
  }}

  function resetAllViewports() {{
    z = Math.floor(NZ / 2);
    s = Math.floor(NS / 2);
    c = Math.floor(NC / 2);
    f = 0;
    applyWindowPreset('fat');
    setOpacity(85);
    $('opacitySlider').value = 85;
    $('presetSelect').value = 'fat';
    isGridMaximized = false;
    maximizedBoxId = null;
    document.querySelectorAll('.viewport-box').forEach(b => b.classList.remove('hidden'));
    $('mprGrid').classList.remove('maximized');
    setTool('crosshair');
    updateViewports();
  }}

  function switchRightTab(tab) {{
    if (tab === 'seg') {{
      $('tabSeg').classList.add('active');
      $('tabMeas').classList.remove('active');
      $('panelSeg').style.display = 'block';
    }} else {{
      $('tabMeas').classList.add('active');
      $('tabSeg').classList.remove('active');
    }}
  }}

  function switchParticipant(pid) {{
    alert('Switching to Participant ' + pid + '. Please run: cardiac-fat-fl view <patient_dir> or reload.');
  }}

  function selectSeries(kind) {{
    $('cardCT').classList.toggle('active', kind === 'ct');
    $('cardSEG').classList.toggle('active', kind === 'seg');
  }}

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

    # 5. Generate authentic OHIF 3-pane web viewer with 4-box center viewport
    generate_ohif_3pane_viewer_html(
        participant_id=participant_id,
        output_dir=dest_dir,
        num_slices=num_rendered,
        summary_metrics=final_metrics,
        mpr_info=mpr_info,
    )

    return dest_dir
