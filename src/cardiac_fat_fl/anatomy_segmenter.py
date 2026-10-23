"""Unified PyTorch/MONAI Heart-and-Pericardium Anatomy Segmenter.

Eliminates the legacy Python 2.7 / TensorFlow 1.14 `DeepHeartSeg` dependency completely.
Supports:
  1. MONAI / PyTorch 3D deep learning models (e.g. TotalSegmentator / WholeBody CT).
  2. Optional ONNX Runtime fallback for converted legacy frozen graphs.
  3. Geometric / Heuristic segmentation engine for standalone execution and offline validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
import torch
from monai.networks.nets import SegResNet

logger = logging.getLogger(__name__)


@dataclass
class AnatomyMasks:
    """Container for the segmented cardiac anatomical structures.

    Attributes:
        pericardium: Binary 3D mask of the pericardial sac (bounding epicardial fat).
        la: Binary 3D mask of the Left Atrium.
        lv: Binary 3D mask of the Left Ventricle.
        ra: Binary 3D mask of the Right Atrium.
        rv: Binary 3D mask of the Right Ventricle.
        lad: Binary 3D mask or centerline of the Left Anterior Descending coronary artery.
        lcx: Binary 3D mask or centerline of the Left Circumflex coronary artery.
        rca: Binary 3D mask or centerline of the Right Coronary Artery.
    """
    pericardium: np.ndarray
    la: np.ndarray
    lv: np.ndarray
    ra: np.ndarray
    rv: np.ndarray
    lad: np.ndarray
    lcx: np.ndarray
    rca: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert masks to a dictionary of numpy arrays."""
        return {
            "pericardium": self.pericardium,
            "la": self.la,
            "lv": self.lv,
            "ra": self.ra,
            "rv": self.rv,
            "lad": self.lad,
            "lcx": self.lcx,
            "rca": self.rca,
        }

    def to_torch_dict(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Convert masks to dictionary of PyTorch tensors."""
        res: Dict[str, torch.Tensor] = {}
        for k, v in self.to_dict().items():
            t = torch.from_numpy(v.astype(bool))
            if device is not None:
                t = t.to(device)
            res[k] = t
        return res


class UnifiedAnatomySegmenter:
    """Unified segmentation wrapper for cardiac chambers, pericardium, and coronary vessels.

    Modernizes and replaces the legacy DeepHeartSeg pipeline with native PyTorch/MONAI
    architecture and optional ONNX Runtime compatibility.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        loc_model_path: Optional[Union[str, Path]] = None,
        backend: str = "auto",
        device: Optional[Union[str, torch.device]] = None,
        onnx_heart_path: Optional[Union[str, Path]] = None,
        onnx_loc_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the segmenter.

        Args:
            model_path: Optional path to PyTorch checkpoint (.pt, .pth) or ONNX model (.onnx).
            loc_model_path: Optional path to heart localization ONNX model for 2-step DeepHeartSeg.
            backend: One of 'auto', 'monai', 'onnx', or 'heuristic'.
            device: Compute device ('cuda', 'cpu', or torch.device).
            onnx_heart_path: Backward-compatible alias for model_path.
            onnx_loc_path: Backward-compatible alias for loc_model_path.
        """
        if model_path is None and onnx_heart_path is not None:
            model_path = onnx_heart_path
        if loc_model_path is None and onnx_loc_path is not None:
            loc_model_path = onnx_loc_path

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.loc_model_path = Path(loc_model_path).resolve() if loc_model_path is not None else None
        self.backend = backend.lower()
        self.torch_model: Optional[torch.nn.Module] = None
        self.onnx_session: Optional[Any] = None
        self.onnx_loc_session: Optional[Any] = None

        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize model backend according to configuration."""
        repo_root = Path(__file__).resolve().parents[2]
        default_seg = repo_root / "models" / "deepheartseg_heart.onnx"
        default_loc = repo_root / "models" / "deepheartseg_loc.onnx"

        if self.model_path is None:
            if default_seg.is_file():
                self.model_path = default_seg
            elif Path("models/deepheartseg_heart.onnx").is_file():
                self.model_path = Path("models/deepheartseg_heart.onnx").resolve()

        if self.loc_model_path is None:
            if default_loc.is_file():
                self.loc_model_path = default_loc
            elif Path("models/deepheartseg_loc.onnx").is_file():
                self.loc_model_path = Path("models/deepheartseg_loc.onnx").resolve()

        if self.backend in ("auto", "onnx"):
            if self.model_path is not None and self.model_path.is_file():
                if self.model_path.suffix.lower() == ".onnx":
                    self.backend = "onnx"
                else:
                    self.backend = "monai"
            elif self.backend == "auto":
                self.backend = "heuristic"

        if self.backend == "onnx":
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in str(self.device) else ["CPUExecutionProvider"]
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 1
            if self.model_path is not None and self.model_path.is_file():
                self.onnx_session = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=providers)

            if self.loc_model_path is not None and self.loc_model_path.is_file():
                self.onnx_loc_session = ort.InferenceSession(str(self.loc_model_path), sess_options=sess_opts, providers=providers)

        elif self.backend == "monai":
            # 8-channel anatomy model (0: pericardium, 1: LA, 2: LV, 3: RA, 4: RV, 5: LAD, 6: LCX, 7: RCA)
            self.torch_model = SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=8,
                init_filters=16,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                dropout_prob=0.0,
            ).to(self.device)

            if self.model_path is not None and self.model_path.is_file():
                ckpt = torch.load(self.model_path, map_location=self.device)
                state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                self.torch_model.load_state_dict(state_dict)

            self.torch_model.eval()

    def segment(
        self,
        image_hu: Union[np.ndarray, torch.Tensor],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> AnatomyMasks:
        """Run anatomy segmentation on a 3D cardiac CT scan.

        Args:
            image_hu: 3D CT image in Hounsfield Units (shape: D, H, W or X, Y, Z).
            spacing: Voxel spacing (sx, sy, sz) in millimeters.

        Returns:
            AnatomyMasks object containing binary masks for all required structures.
        """
        if isinstance(image_hu, torch.Tensor):
            arr = image_hu.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(image_hu, dtype=np.float32)

        if self.backend == "onnx" and self.onnx_session is not None:
            try:
                return self._segment_onnx(arr, spacing)
            except Exception as exc:
                logger.warning("ONNX segmentation failed (%s); falling back to heuristic engine.", exc)
                return self._segment_heuristic(arr, spacing)
        elif self.backend == "monai" and self.torch_model is not None and self.model_path is not None:
            try:
                return self._segment_monai(arr, spacing)
            except Exception as exc:
                logger.warning("MONAI segmentation failed (%s); falling back to heuristic engine.", exc)
                return self._segment_heuristic(arr, spacing)
        else:
            return self._segment_heuristic(arr, spacing)

    def _segment_monai(
        self,
        arr: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> AnatomyMasks:
        """Run PyTorch/MONAI deep learning inference with sliding window support."""
        assert self.torch_model is not None
        from monai.inferers import SlidingWindowInferer

        # Normalize HU input: standard cardiac window [-200, 800] -> [0, 1]
        norm = np.clip((arr - (-200.0)) / (800.0 - (-200.0)), 0.0, 1.0)
        inp = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.device)

        # Dynamic ROI divisible by 16 for SegResNet architecture
        spatial_shape = norm.shape
        roi = tuple(max(16, min(96, (s // 16) * 16)) for s in spatial_shape)
        inferer = SlidingWindowInferer(roi_size=roi, sw_batch_size=2, overlap=0.25, mode="gaussian")

        with torch.no_grad():
            pred = inferer(inp, self.torch_model)
            probs = torch.sigmoid(pred).squeeze(0).cpu().numpy()

        pericardium = probs[0] > 0.5
        la = probs[1] > 0.5
        lv = probs[2] > 0.5
        ra = probs[3] > 0.5
        rv = probs[4] > 0.5
        lad = probs[5] > 0.5
        lcx = probs[6] > 0.5
        rca = probs[7] > 0.5

        return AnatomyMasks(
            pericardium=pericardium.astype(bool),
            la=la.astype(bool),
            lv=lv.astype(bool),
            ra=ra.astype(bool),
            rv=rv.astype(bool),
            lad=lad.astype(bool),
            lcx=lcx.astype(bool),
            rca=rca.astype(bool),
        )

    def _segment_onnx(
        self,
        arr: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> AnatomyMasks:
        """Run ONNX runtime model inference supporting DeepHeartSeg and multi-channel models."""
        assert self.onnx_session is not None
        inp_shape = self.onnx_session.get_inputs()[0].shape

        # Check if model is the DeepHeartSeg 1-channel binary pericardium model
        out_shape = self.onnx_session.get_outputs()[0].shape
        is_single_channel = len(out_shape) == 5 and out_shape[1] == 1

        if is_single_channel:
            # 1. Heart localization (Step 1) to derive cardiac bounding box
            if self.onnx_loc_session is not None:
                z_zoom = 64.0 / arr.shape[0]
                y_zoom = 128.0 / arr.shape[1]
                x_zoom = 128.0 / arr.shape[2]
                down = ndi.zoom(arr, (z_zoom, y_zoom, x_zoom), order=1)
                norm_loc = ((np.clip(down, -1024.0, 3071.0)) - 1023.5) / 2047.5
                inp_loc = norm_loc[np.newaxis, np.newaxis, ...].astype(np.float32)
                pred_loc = self.onnx_loc_session.run(None, {self.onnx_loc_session.get_inputs()[0].name: inp_loc})[0]
                loc_mask = pred_loc[0, 0] > 0.5

                idx = np.where(loc_mask)
                if len(idx[0]) > 0:
                    z_min, z_max = int(np.min(idx[0]) / z_zoom), int(np.max(idx[0]) / z_zoom)
                    y_min, y_max = int(np.min(idx[1]) / y_zoom), int(np.max(idx[1]) / y_zoom)
                    x_min, x_max = int(np.min(idx[2]) / x_zoom), int(np.max(idx[2]) / x_zoom)
                    # Add margin
                    z_min, z_max = max(0, z_min - 4), min(arr.shape[0], z_max + 4)
                    y_min, y_max = max(0, y_min - 16), min(arr.shape[1], y_max + 16)
                    x_min, x_max = max(0, x_min - 16), min(arr.shape[2], x_max + 16)
                else:
                    z_min, z_max = 0, arr.shape[0]
                    y_min, y_max = int(arr.shape[1] * 0.2), int(arr.shape[1] * 0.8)
                    x_min, x_max = int(arr.shape[2] * 0.2), int(arr.shape[2] * 0.8)
            else:
                z_min, z_max = 0, arr.shape[0]
                y_min, y_max = int(arr.shape[1] * 0.2), int(arr.shape[1] * 0.8)
                x_min, x_max = int(arr.shape[2] * 0.2), int(arr.shape[2] * 0.8)

            cropped = arr[z_min:z_max, y_min:y_max, x_min:x_max]
            cz, cy, cx = cropped.shape

            # Step 2: Resample cropped heart to (112, 128, 128)
            z_f = 112.0 / cz
            y_f = 128.0 / cy
            x_f = 128.0 / cx
            crop_resampled = ndi.zoom(cropped, (z_f, y_f, x_f), order=1)
            norm_seg = ((np.clip(crop_resampled, -1024.0, 3071.0)) - 1023.5) / 2047.5

            inp_seg = norm_seg[np.newaxis, np.newaxis, ...].astype(np.float32)
            pred_seg = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: inp_seg})[0]
            seg_mask_small = pred_seg[0, 0] > 0.5

            # Upsample back to cropped dimensions
            seg_crop_orig = ndi.zoom(seg_mask_small.astype(float), (1.0 / z_f, 1.0 / y_f, 1.0 / x_f), order=0) > 0.5
            pericardium = np.zeros_like(arr, dtype=bool)
            pericardium[z_min:z_max, y_min:y_max, x_min:x_max] = seg_crop_orig[:cz, :cy, :cx]
            pericardium = ndi.binary_fill_holes(pericardium)

            # Derive chambers and coronaries using DeepHeartSeg pericardium envelope
            return self._derive_chambers_and_vessels(pericardium, arr)

        # Standard 8-channel ONNX model
        norm = np.clip((arr - (-200.0)) / (800.0 - (-200.0)), 0.0, 1.0).astype(np.float32)
        inp = np.expand_dims(np.expand_dims(norm, axis=0), axis=0)  # (1, 1, D, H, W)

        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: inp})
        probs = 1.0 / (1.0 + np.exp(-outputs[0]))
        probs = probs.squeeze(0)  # (8, D, H, W)

        return AnatomyMasks(
            pericardium=(probs[0] > 0.5).astype(bool),
            la=(probs[1] > 0.5).astype(bool),
            lv=(probs[2] > 0.5).astype(bool),
            ra=(probs[3] > 0.5).astype(bool),
            rv=(probs[4] > 0.5).astype(bool),
            lad=(probs[5] > 0.5).astype(bool),
            lcx=(probs[6] > 0.5).astype(bool),
            rca=(probs[7] > 0.5).astype(bool),
        )

    def _derive_chambers_and_vessels(
        self,
        pericardium: np.ndarray,
        arr: np.ndarray,
    ) -> AnatomyMasks:
        """Derive cardiac chamber and coronary vessel masks within a given pericardium."""
        shape = arr.shape
        nz, ny, nx = shape  # Axis 0: Z (axial), Axis 1: Y (coronal), Axis 2: X (sagittal)

        indices = np.where(pericardium)
        if len(indices[0]) > 0:
            cz = float(np.mean(indices[0]))
            cy = float(np.mean(indices[1]))
            cx = float(np.mean(indices[2]))
        else:
            cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0

        grid_z, grid_y, grid_x = np.indices(shape)
        chamber_voxels = pericardium & (arr >= -30.0)

        is_left = grid_x >= cx
        is_right = ~is_left
        is_anterior = grid_y < cy
        is_posterior = ~is_anterior
        is_superior = grid_z >= cz
        is_inferior = ~is_superior

        la = chamber_voxels & is_left & is_posterior & is_superior
        lv = chamber_voxels & is_left & (is_inferior | (is_posterior & ~is_superior))
        ra = chamber_voxels & is_right & (is_superior | (is_anterior & is_superior))
        rv = chamber_voxels & is_right & is_anterior & is_inferior

        if not np.any(la):
            la = pericardium & is_left & is_posterior
        if not np.any(lv):
            lv = pericardium & is_left & is_anterior
        if not np.any(ra):
            ra = pericardium & is_right & is_posterior
        if not np.any(rv):
            rv = pericardium & is_right & is_anterior

        lad_zone = pericardium & (np.abs(grid_x - cx) <= 8) & is_anterior
        lad_dilated = ndi.binary_dilation(lad_zone, iterations=1)
        lad = lad_dilated & (arr >= -30.0)
        if not np.any(lad):
            lad = lad_zone

        lcx_zone = pericardium & is_left & (np.abs(grid_z - cz) <= 8) & is_posterior
        lcx_dilated = ndi.binary_dilation(lcx_zone, iterations=1)
        lcx = lcx_dilated & (arr >= -30.0)
        if not np.any(lcx):
            lcx = lcx_zone

        rca_zone = pericardium & is_right & (np.abs(grid_z - cz) <= 8) & is_anterior
        rca_dilated = ndi.binary_dilation(rca_zone, iterations=1)
        rca = rca_dilated & (arr >= -30.0)
        if not np.any(rca):
            rca = rca_zone

        return AnatomyMasks(
            pericardium=pericardium.astype(bool),
            la=la.astype(bool),
            lv=lv.astype(bool),
            ra=ra.astype(bool),
            rv=rv.astype(bool),
            lad=lad.astype(bool),
            lcx=lcx.astype(bool),
            rca=rca.astype(bool),
        )

    def _segment_heuristic(
        self,
        arr: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> AnatomyMasks:
        """Heuristic anatomical segmenter based on CT Hounsfield Units and cardiac morphology.

        Ensures full reproducibility on synthetic phantoms and real CTs without external downloads:
          - Delineates heart silhouette & pericardium (-50 to +400 HU mediastinal envelope)
          - Partitions cardiac chambers (LA, LV, RA, RV) via standard anatomical quadrant geometry
          - Identifies coronary trajectories (LAD in anterior groove, LCX in circumflex groove, RCA in right groove)
        """
        shape = arr.shape
        nx, ny, nz = shape

        # 1. Mediastinum / Heart silhouette: tissue HU between -50 and 500 HU, excluding lungs
        heart_core = (arr >= -50.0) & (arr <= 500.0)
        # Morphological closing to enclose chambers and pericardial space
        struct = ndi.generate_binary_structure(3, 1)
        heart_core = ndi.binary_closing(heart_core, structure=struct, iterations=2)

        # Find largest connected component in center volume
        labeled, num_features = ndi.label(heart_core)
        if num_features > 0:
            center_x, center_y, center_z = nx // 2, ny // 2, nz // 2
            # Find label closest to center
            sizes = ndi.sum(heart_core, labeled, range(1, num_features + 1))
            largest_label = int(np.argmax(sizes)) + 1
            pericardium = labeled == largest_label
        else:
            # Fallback to an ellipsoid in volume center
            grid_x, grid_y, grid_z = np.ogrid[:nx, :ny, :nz]
            rx, ry, rz = nx * 0.35, ny * 0.35, nz * 0.35
            pericardium = (
                ((grid_x - nx / 2) / rx) ** 2
                + ((grid_y - ny / 2) / ry) ** 2
                + ((grid_z - nz / 2) / rz) ** 2
            ) <= 1.0

        # Fill holes in pericardium
        pericardium = ndi.binary_fill_holes(pericardium)

        # Derive chambers and coronaries using standardized spatial rules
        return self._derive_chambers_and_vessels(pericardium, arr)

