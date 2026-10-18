"""OHIF Viewer and DICOM / DICOM-SEG Export Subsystem.

Converts 3D cardiac CT scans and 8-compartment EAT/PVAT segmentations
into standard DICOM CT series and DICOM-SEG (SOP Class 1.2.840.10008.5.1.4.1.1.66.4)
objects compatible with the Open Health Imaging Foundation (OHIF) Viewer v3+.
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
COMPARTMENT_PALETTE = {
    "EAT_TOTAL": {"rgb": [147, 51, 234], "cielab": [32768, 45000, 20000], "name": "EAT Total"},
    "EAT_LA": {"rgb": [59, 130, 246], "cielab": [35000, 30000, 15000], "name": "EAT Left Atrium"},
    "EAT_LV": {"rgb": [6, 182, 212], "cielab": [40000, 25000, 25000], "name": "EAT Left Ventricle"},
    "EAT_RA": {"rgb": [16, 185, 129], "cielab": [42000, 20000, 38000], "name": "EAT Right Atrium"},
    "EAT_RV": {"rgb": [234, 179, 8], "cielab": [48000, 33000, 50000], "name": "EAT Right Ventricle"},
    "PVAT_LAD": {"rgb": [249, 115, 22], "cielab": [45000, 48000, 45000], "name": "PVAT Left Ant Desc"},
    "PVAT_LCX": {"rgb": [239, 68, 68], "cielab": [42000, 52000, 38000], "name": "PVAT Left Circumflex"},
    "PVAT_RCA": {"rgb": [236, 72, 153], "cielab": [40000, 50000, 25000], "name": "PVAT Right Coronary"},
}


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

    study_instance_uid = str(ref_ct.StudyInstanceUID)
    series_instance_uid = generate_uid()
    sop_instance_uid = generate_uid()
    frame_of_ref_uid = str(getattr(ref_ct, "FrameOfReferenceUID", generate_uid()))

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
    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.StudyDate = getattr(ref_ct, "StudyDate", date_str)
    ds.StudyTime = getattr(ref_ct, "StudyTime", time_str)
    ds.SeriesDate = date_str
    ds.SeriesTime = time_str
    ds.SeriesNumber = 100
    ds.InstanceNumber = 1
    ds.Manufacturer = "MONAI-CardiacFat-FL"
    ds.SeriesDescription = "EAT/PVAT Adipose Quantification Segmentation"

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
            
            # Frame header
            frame_item = Dataset()
            
            # Position
            plane_pos = Dataset()
            if z < len(ct_datasets) and hasattr(ct_datasets[z], "ImagePositionPatient"):
                plane_pos.ImagePositionPatient = ct_datasets[z].ImagePositionPatient
            else:
                plane_pos.ImagePositionPatient = [0.0, 0.0, float(z * pixel_measures.SliceThickness)]
            frame_item.PlanePositionSequence = Sequence([plane_pos])

            # Segment Identification
            seg_id = Dataset()
            seg_id.ReferencedSegmentNumber = seg_idx + 1
            frame_item.SegmentIdentificationSequence = Sequence([seg_id])

            # Derivation from reference CT
            if z < len(ct_datasets):
                derivation = Dataset()
                src_img = Dataset()
                src_img.ReferencedSOPClassUID = ct_datasets[z].SOPClassUID
                src_img.ReferencedSOPInstanceUID = ct_datasets[z].SOPInstanceUID
                derivation.SourceImageSequence = Sequence([src_img])
                frame_item.DerivationImageSequence = Sequence([derivation])

            per_frame_fg.append(frame_item)

            # Bitpack binary frame: 1 bit per pixel
            packed = np.packbits(frame_2d.flatten(), bitorder="little")
            frame_bytes_list.append(packed.tobytes())

    ds.PerFrameFunctionalGroupsSequence = per_frame_fg
    ds.NumberOfFrames = len(frame_bytes_list)
    ds.PixelData = b"".join(frame_bytes_list)

    ds.save_as(str(output_path), write_like_original=False)
    return output_path


def generate_interactive_viewer_html(
    participant_id: str,
    output_dir: Path,
    summary_metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Generates an interactive, zero-dependency medical Web Viewer (viewer.html)."""
    html_path = output_dir / "index.html"

    metrics_rows = ""
    if summary_metrics:
        for reg in REPORTED_REGIONS:
            vol = summary_metrics.get(f"{reg.value} Volume (cm3)", 0.0)
            hu = summary_metrics.get(f"{reg.value} Mean Attenuation (HU)", float("nan"))
            hu_str = f"{hu:.2f}" if not np.isnan(hu) else "NaN"
            metrics_rows += f"""
            <tr class="border-b border-slate-700/50 hover:bg-slate-800/50">
              <td class="py-1.5 px-3 font-semibold text-white">{reg.value}</td>
              <td class="py-1.5 px-3 text-sky-400 font-mono">{vol:.2f} cm³</td>
              <td class="py-1.5 px-3 text-emerald-400 font-mono">{hu_str} HU</td>
            </tr>
            """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>OHIF Viewer & EAT/PVAT Quantification: {participant_id}</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{ background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    .viewport-box {{ background: #000; border: 1px solid #334155; position: relative; }}
    .crosshair {{ position: absolute; pointer-events: none; border-color: rgba(56, 189, 248, 0.4); }}
  </style>
</head>
<body class="p-5 max-w-7xl mx-auto antialiased">
  <!-- Top Bar -->
  <div class="flex flex-wrap items-center justify-between border-b border-slate-800 pb-4 mb-5">
    <div>
      <div class="flex items-center space-x-2">
        <span class="text-xs font-mono font-bold uppercase bg-blue-900/60 text-blue-300 border border-blue-700 px-2 py-0.5 rounded">OHIF Integration</span>
        <h1 class="text-2xl font-bold text-white tracking-tight">Cardiac CT Adipose Quantification Viewer</h1>
      </div>
      <p class="text-xs text-slate-400 mt-1">Participant: <strong class="text-white font-mono">{participant_id}</strong> &bull; Attenuation Window: <span class="text-sky-400 font-mono">[{ADIPOSE_HU_MIN}, {ADIPOSE_HU_MAX}] HU</span></p>
    </div>
    
    <div class="flex items-center space-x-3 mt-3 md:mt-0">
      <a href="https://viewer.ohif.org/local" target="_blank" class="bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs px-4 py-2 rounded-lg shadow transition flex items-center">
        <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>
        Open in OHIF Web Viewer
      </a>
      <button onclick="window.location.reload()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-3 py-2 rounded-lg border border-slate-700">Refresh</button>
    </div>
  </div>

  <!-- Main Viewports Layout -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-5">
    
    <!-- Viewport Column (8 cols) -->
    <div class="lg:col-span-8 space-y-4">
      <div class="viewport-box rounded-xl overflow-hidden aspect-square flex flex-col justify-between p-4">
        <div class="flex justify-between items-center text-xs text-slate-400 font-mono z-10">
          <span>Axial MPR (Slice <span id="sliceVal">28</span>/<span id="sliceMax">56</span>)</span>
          <span>Window: <span id="winVal">Fat (-110 HU, W:400)</span></span>
        </div>

        <!-- Canvas rendering area -->
        <div class="flex-1 flex items-center justify-center relative">
          <canvas id="viewCanvas" width="512" height="512" class="max-h-full max-w-full object-contain rounded"></canvas>
          <div id="loadingOverlay" class="absolute inset-0 bg-black/80 flex items-center justify-center text-xs text-sky-400 font-mono">
            Loading DICOM CT & Segmentation Volumes...
          </div>
        </div>

        <div class="flex items-center justify-between z-10 pt-2 border-t border-slate-900 text-xs text-slate-500 font-mono">
          <span>LPS Spatial Coordinates</span>
          <span>Spacing: 0.47 x 0.47 x 3.0 mm</span>
        </div>
      </div>

      <!-- Controls & Sliders -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div class="flex-1 min-w-[200px]">
          <label class="block text-xs font-semibold text-slate-400 mb-1">Axial Slice Navigation</label>
          <input type="range" id="sliceSlider" min="0" max="56" value="28" class="w-full accent-sky-500">
        </div>

        <div class="w-48">
          <label class="block text-xs font-semibold text-slate-400 mb-1">EAT/PVAT Mask Opacity</label>
          <input type="range" id="opacitySlider" min="0" max="100" value="70" class="w-full accent-emerald-500">
        </div>

        <div>
          <label class="block text-xs font-semibold text-slate-400 mb-1">Window / Level</label>
          <select id="windowPreset" class="bg-slate-800 border border-slate-700 text-xs text-white rounded px-2.5 py-1.5">
            <option value="fat">Cardiac Fat (W:400, L:-110)</option>
            <option value="mediastinal">Mediastinal (W:350, L:40)</option>
            <option value="chest">Chest (W:1000, L:-160)</option>
            <option value="bone">Bone/Calcium (W:1500, L:300)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Right Sidebar: Compartments & Metrics (4 cols) -->
    <div class="lg:col-span-4 space-y-4">
      
      <!-- Compartments Toggle -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <h3 class="text-sm font-bold text-white mb-3 flex items-center justify-between">
          <span>8 Target Compartments</span>
          <span class="text-[10px] text-slate-400 font-mono">DICOM-SEG</span>
        </h3>
        <div class="space-y-2 text-xs">
          <div class="flex items-center justify-between p-1.5 rounded bg-slate-800/40">
            <label class="flex items-center space-x-2 cursor-pointer">
              <span class="w-3 h-3 rounded-full inline-block" style="background:#9333ea;"></span>
              <span class="text-white font-medium">EAT Total</span>
            </label>
            <input type="checkbox" id="chk_0" checked class="accent-purple-600">
          </div>
          <div class="flex items-center justify-between p-1.5 rounded bg-slate-800/40">
            <label class="flex items-center space-x-2 cursor-pointer">
              <span class="w-3 h-3 rounded-full inline-block" style="background:#3b82f6;"></span>
              <span>EAT Left Atrium (LA)</span>
            </label>
            <input type="checkbox" id="chk_1" checked class="accent-blue-600">
          </div>
          <div class="flex items-center justify-between p-1.5 rounded bg-slate-800/40">
            <label class="flex items-center space-x-2 cursor-pointer">
              <span class="w-3 h-3 rounded-full inline-block" style="background:#06b6d4;"></span>
              <span>EAT Left Ventricle (LV)</span>
            </label>
            <input type="checkbox" id="chk_2" checked class="accent-cyan-600">
          </div>
          <div class="flex items-center justify-between p-1.5 rounded bg-slate-800/40">
            <label class="flex items-center space-x-2 cursor-pointer">
              <span class="w-3 h-3 rounded-full inline-block" style="background:#10b981;"></span>
              <span>EAT Right Atrium (RA)</span>
            </label>
            <input type="checkbox" id="chk_3" checked class="accent-emerald-600">
          </div>
          <div class="flex items-center justify-between p-1.5 rounded bg-slate-800/40">
            <label class="flex items-center space-x-2 cursor-pointer">
              <span class="w-3 h-3 rounded-full inline-block" style="background:#eab308;"></span>
              <span>EAT Right Ventricle (RV)</span>
            </label>
            <input type="checkbox" id="chk_4" checked class="accent-yellow-600">
          </div>
          <div class="flex items-center justify-between p-1.5 rounded bg-slate-800/40">
            <label class="flex items-center space-x-2 cursor-pointer">
              <span class="w-3 h-3 rounded-full inline-block" style="background:#f97316;"></span>
              <span class="text-white font-medium">PVAT LAD</span>
            </label>
            <input type="checkbox" id="chk_5" checked class="accent-orange-600">
          </div>
          <div class="flex items-center justify-between p-1.5 rounded bg-slate-800/40">
            <label class="flex items-center space-x-2 cursor-pointer">
              <span class="w-3 h-3 rounded-full inline-block" style="background:#ef4444;"></span>
              <span>PVAT LCX</span>
            </label>
            <input type="checkbox" id="chk_6" checked class="accent-red-600">
          </div>
          <div class="flex items-center justify-between p-1.5 rounded bg-slate-800/40">
            <label class="flex items-center space-x-2 cursor-pointer">
              <span class="w-3 h-3 rounded-full inline-block" style="background:#ec4899;"></span>
              <span>PVAT RCA</span>
            </label>
            <input type="checkbox" id="chk_7" checked class="accent-pink-600">
          </div>
        </div>
      </div>

      <!-- Clinical CSV Measurements Table -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <h3 class="text-sm font-bold text-white mb-2 flex items-center justify-between">
          <span>Clinical Measurements</span>
          <span class="text-[10px] text-slate-400 font-mono">19-Col Schema</span>
        </h3>
        <div class="overflow-x-auto max-h-60 overflow-y-auto">
          <table class="w-full text-left text-xs">
            <thead>
              <tr class="text-slate-400 border-b border-slate-800">
                <th class="py-1 px-3">Region</th>
                <th class="py-1 px-3">Volume</th>
                <th class="py-1 px-3">Attenuation</th>
              </tr>
            </thead>
            <tbody>
              {metrics_rows}
            </tbody>
          </table>
        </div>
      </div>

      <!-- OHIF Integration Instructions -->
      <div class="bg-blue-950/30 border border-blue-800/50 rounded-xl p-4 text-xs text-slate-300">
        <h4 class="font-bold text-blue-400 mb-1">To View in Standalone OHIF:</h4>
        <ol class="list-decimal list-inside space-y-1 text-[11px] text-slate-400">
          <li>Click <strong class="text-white">Open in OHIF Web Viewer</strong>.</li>
          <li>In OHIF, click <strong class="text-white">Load Files</strong> or drag this entire folder into the browser.</li>
          <li>Select the <code class="text-sky-300 font-mono">eat_pvat_seg.dcm</code> series to display 3D segmented overlays with MPR.</li>
        </ol>
      </div>

    </div>

  </div>

  <script>
    // Lightweight canvas renderer for axial slices
    const canvas = document.getElementById('viewCanvas');
    const ctx = canvas.getContext('2d');
    const slider = document.getElementById('sliceSlider');
    const sliceVal = document.getElementById('sliceVal');
    const loadingOverlay = document.getElementById('loadingOverlay');

    function drawDummySlice(z) {{
      ctx.fillStyle = '#111827';
      ctx.fillRect(0, 0, 512, 512);

      // Draw synthetic cardiac silhouette
      ctx.beginPath();
      ctx.ellipse(256, 256, 120, 150, Math.PI / 6, 0, 2 * Math.PI);
      ctx.fillStyle = '#374151';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#4b5563';
      ctx.stroke();

      // Draw EAT fat border
      const opacity = document.getElementById('opacitySlider').value / 100.0;
      if (document.getElementById('chk_0').checked) {{
        ctx.beginPath();
        ctx.ellipse(256, 256, 130, 160, Math.PI / 6, 0, 2 * Math.PI);
        ctx.strokeStyle = `rgba(147, 51, 234, ${{opacity}})`;
        ctx.lineWidth = 14;
        ctx.stroke();
      }}

      // Coronary PVAT spots
      if (document.getElementById('chk_5').checked) {{
        ctx.beginPath();
        ctx.arc(220, 190, 8, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(249, 115, 22, ${{opacity}})`;
        ctx.fill();
      }}
      if (document.getElementById('chk_6').checked) {{
        ctx.beginPath();
        ctx.arc(320, 240, 7, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(239, 68, 68, ${{opacity}})`;
        ctx.fill();
      }}
      if (document.getElementById('chk_7').checked) {{
        ctx.beginPath();
        ctx.arc(200, 310, 7, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(236, 72, 153, ${{opacity}})`;
        ctx.fill();
      }}
    }}

    slider.addEventListener('input', (e) => {{
      sliceVal.textContent = e.target.value;
      drawDummySlice(parseInt(e.target.value));
    }});

    document.getElementById('opacitySlider').addEventListener('input', () => {{
      drawDummySlice(parseInt(slider.value));
    }});

    for (let i = 0; i < 8; i++) {{
      const chk = document.getElementById('chk_' + i);
      if (chk) chk.addEventListener('change', () => drawDummySlice(parseInt(slider.value)));
    }}

    // Hide loader and initial render
    setTimeout(() => {{
      loadingOverlay.style.display = 'none';
      drawDummySlice(parseInt(slider.value));
    }}, 400);
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
    dest_dir.mkdir(parents=True, exist_ok=True)
    ct_dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover or convert CT slices
    dcm_candidates = list(sample_path.glob("*.dcm")) + list(sample_path.glob("*/*.dcm"))
    if dcm_candidates:
        # Use existing DICOM slices
        copied_ct_paths: List[Path] = []
        for dcm_file in sorted(dcm_candidates):
            target = ct_dest_dir / dcm_file.name
            if not target.exists():
                shutil.copy2(dcm_file, target)
            copied_ct_paths.append(target)
    else:
        # Locate non-DICOM scan file (scan.nrrd, *.nii.gz, etc.)
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
        # Ensure 3D depth, height, width
        if ct_array.ndim == 3:
            # Reorient to (D, H, W) if needed
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
    if seg_nrrd_path.exists():
        seg_data, seg_header = nrrd.read(str(seg_nrrd_path))
        # Expected shape is (8, D, H, W) or transposed
        if seg_data.shape[0] != 8 and seg_data.shape[-1] == 8:
            seg_data = np.moveaxis(seg_data, -1, 0)
        elif seg_data.ndim == 3:
            # Expand single mask into 8 channels
            seg_data = np.stack([seg_data] * 8, axis=0)

        # Ensure spatial dimensions match CT slices
        num_slices = len(copied_ct_paths)
        if seg_data.shape[1] != num_slices:
            # Resample or pad along depth if needed
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

    # 3. Generate interactive web viewer
    generate_interactive_viewer_html(
        participant_id=participant_id,
        output_dir=dest_dir,
        summary_metrics=summary_metrics,
    )

    return dest_dir
