"""Unified PyTorch/MONAI Heart-and-Pericardium Anatomy Segmenter.

Modernizes the foundational Python 2.7 / TensorFlow 1.14 `DeepHeartSeg` and external
subprocess TotalSegmentator dependencies completely.

Supports:
  1. Native MONAI Model Zoo / NVIDIA NGC `monai_wholebody_ct_segmentation` (104 whole-body labels
     including Left/Right Atrium, Left/Right Ventricle, Aorta, and Myocardium).
  2. Optimized ONNX Runtime graphs for 2-step DeepHeartSeg pericardial sac segmentation.
  3. Single-stage 8-channel distilled MONAI SegResNet.
  4. Built-in geometric/heuristic segmentation engine for self-contained, zero-dependency execution.
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
    aorta: Optional[np.ndarray] = None
    pulmonary_artery: Optional[np.ndarray] = None
    myocardium: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert masks to a dictionary of numpy arrays."""
        d = {
            "pericardium": self.pericardium,
            "la": self.la,
            "lv": self.lv,
            "ra": self.ra,
            "rv": self.rv,
            "lad": self.lad,
            "lcx": self.lcx,
            "rca": self.rca,
        }
        if self.aorta is not None:
            d["aorta"] = self.aorta
        if self.pulmonary_artery is not None:
            d["pulmonary_artery"] = self.pulmonary_artery
        if self.myocardium is not None:
            d["myocardium"] = self.myocardium
        return d

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

    Modernizes and advances the foundational DeepHeartSeg pipeline with native PyTorch/MONAI
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
        self.wholebody_bundle: Optional[Any] = None

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

            default_wb = repo_root / "models" / "wholeBody_ct_segmentation" / "models" / "model.pt"
            if not default_wb.is_file():
                alt_wb = Path("models/wholeBody_ct_segmentation/models/model.pt")
                if alt_wb.is_file():
                    default_wb = alt_wb.resolve()
            if default_wb.is_file():
                try:
                    from monai.inferers import SlidingWindowInferer
                    wb_net = SegResNet(
                        spatial_dims=3,
                        in_channels=1,
                        out_channels=105,
                        init_filters=32,
                        blocks_down=[1, 2, 2, 4],
                        blocks_up=[1, 1, 1],
                        dropout_prob=0.2,
                    ).to(self.device)
                    ckpt = torch.load(default_wb, map_location=self.device, weights_only=True)
                    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                    wb_net.load_state_dict(state_dict)
                    wb_net.eval()
                    self.wholebody_bundle = wb_net
                    self.wholebody_inferer = SlidingWindowInferer(
                        roi_size=(96, 96, 96),
                        sw_batch_size=4,
                        overlap=0.25,
                        mode="gaussian",
                        padding_mode="replicate",
                    )
                except Exception as exc:
                    logger.warning("Could not load wholeBody_ct_segmentation model (%s).", exc)

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

        is_clinical_scan = max(arr.shape) >= 128
        if self.backend == "onnx" and self.onnx_session is not None:
            return self._segment_onnx(arr, spacing)
        elif self.backend == "monai" and self.torch_model is not None and self.model_path is not None:
            return self._segment_monai(arr, spacing)
        elif is_clinical_scan and self.backend != "heuristic":
            raise RuntimeError(
                "Neural network segmentation models (monai_wholebody_ct_segmentation and DeepHeartSeg ONNX) "
                "are required for clinical CT scans; heuristic fallback is disabled. "
                "Run 'cardiac-fat-fl download-models' to install weights."
            )
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
            # Check axis ordering: NRRD volumes typically arrive as (X, Y, Z) where Z (slices) < X (in-plane)
            is_xyz = arr.shape[0] > arr.shape[2] and arr.shape[0] >= 256
            if is_xyz:
                # Transpose from (X, Y, Z) to (Z, Y, X) for DeepHeartSeg 3D CNN
                arr_zyx = arr.transpose(2, 1, 0)
            else:
                arr_zyx = arr

            # 1. Heart localization (Step 1) to derive cardiac bounding box
            if self.onnx_loc_session is not None:
                z_zoom = 64.0 / arr_zyx.shape[0]
                y_zoom = 128.0 / arr_zyx.shape[1]
                x_zoom = 128.0 / arr_zyx.shape[2]
                down = ndi.zoom(arr_zyx, (z_zoom, y_zoom, x_zoom), order=1)
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
                    z_min, z_max = max(0, z_min - 4), min(arr_zyx.shape[0], z_max + 4)
                    y_min, y_max = max(0, y_min - 16), min(arr_zyx.shape[1], y_max + 16)
                    x_min, x_max = max(0, x_min - 16), min(arr_zyx.shape[2], x_max + 16)
                else:
                    z_min, z_max = 0, arr_zyx.shape[0]
                    y_min, y_max = int(arr_zyx.shape[1] * 0.2), int(arr_zyx.shape[1] * 0.8)
                    x_min, x_max = int(arr_zyx.shape[2] * 0.2), int(arr_zyx.shape[2] * 0.8)
            else:
                z_min, z_max = 0, arr_zyx.shape[0]
                y_min, y_max = int(arr_zyx.shape[1] * 0.2), int(arr_zyx.shape[1] * 0.8)
                x_min, x_max = int(arr_zyx.shape[2] * 0.2), int(arr_zyx.shape[2] * 0.8)

            cropped = arr_zyx[z_min:z_max, y_min:y_max, x_min:x_max]
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
            pericardium_zyx = np.zeros_like(arr_zyx, dtype=bool)
            pericardium_zyx[z_min:z_max, y_min:y_max, x_min:x_max] = seg_crop_orig[:cz, :cy, :cx]
            pericardium_zyx = ndi.binary_fill_holes(pericardium_zyx)

            if is_xyz:
                # Transpose back to (X, Y, Z) to match original volume coordinate frame
                pericardium = pericardium_zyx.transpose(2, 1, 0)
            else:
                pericardium = pericardium_zyx

            # Run MONAI wholeBody_ct_segmentation (1.5mm SegResNet) first to obtain multi-label cardiac anatomy
            multilabel_map = self._run_wholebody_segmentation(arr, spacing)
            # Combine MONAI WholeBody chambers/vessels with DeepHeartSeg pericardium envelope
            return self._derive_chambers_and_vessels(pericardium, arr, multilabel_map=multilabel_map)

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

    def _run_wholebody_segmentation(
        self,
        arr: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Optional[np.ndarray]:
        """Run NGC monai_wholebody_ct_segmentation (1.5mm SegResNet) to extract 104-class labels."""
        if self.wholebody_bundle is None or getattr(self, "wholebody_inferer", None) is None:
            return None
        if max(arr.shape) < 128:
            return None

        import torch.nn.functional as F

        is_xyz = arr.shape[0] > arr.shape[2] and arr.shape[0] >= 256
        if is_xyz:
            arr_ras = arr[::-1, ::-1, :].copy()
        else:
            arr_ras = arr.transpose(2, 1, 0)[::-1, ::-1, :].copy()

        sp_ras = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        target_shape = (
            max(16, int(round(arr_ras.shape[0] * sp_ras[0] / 1.5))),
            max(16, int(round(arr_ras.shape[1] * sp_ras[1] / 1.5))),
            max(16, int(round(arr_ras.shape[2] * sp_ras[2] / 1.5))),
        )

        with torch.no_grad():
            t_in = torch.from_numpy(arr_ras).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
            t_15 = F.interpolate(t_in, size=target_shape, mode="trilinear", align_corners=False)
            nz = t_15 != 0
            if torch.any(nz):
                vals = t_15[nz]
                m = vals.mean()
                s = vals.std()
                t_15[nz] = (vals - m) / (s + 1e-8)
            vmin, vmax = t_15.min(), t_15.max()
            if vmax > vmin:
                t_15 = (t_15 - vmin) / (vmax - vmin) * 2.0 - 1.0

            with torch.amp.autocast("cuda", enabled=("cuda" in str(self.device))):
                logits = self.wholebody_inferer(t_15, self.wholebody_bundle)
                pred_15 = torch.argmax(logits, dim=1, keepdim=True).float()

            cardiac_mask = (pred_15 == 7) | ((pred_15 >= 44) & (pred_15 <= 49))
            pred_15 = torch.where(cardiac_mask, pred_15, torch.zeros_like(pred_15))
            pred_ras = (
                F.interpolate(pred_15, size=arr_ras.shape, mode="nearest")
                .squeeze(0)
                .squeeze(0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

        if is_xyz:
            return pred_ras[::-1, ::-1, :]
        return pred_ras[::-1, ::-1, :].transpose(2, 1, 0)

    def _derive_chambers_and_vessels(
        self,
        pericardium: np.ndarray,
        arr: np.ndarray,
        multilabel_map: Optional[np.ndarray] = None,
    ) -> AnatomyMasks:
        """Derive cardiac chamber and coronary vessel masks within a given pericardium.

        If `multilabel_map` from `monai_wholebody_ct_segmentation` is provided:
          - Label 44: Left Ventricular Myocardium (`heart_myocardium`)
          - Label 45: Left Atrium (`heart_atrium_left`)
          - Label 46: Left Ventricle (`heart_ventricle_left`)
          - Label 47: Right Atrium (`heart_atrium_right`)
          - Label 48: Right Ventricle (`heart_ventricle_right`)
          - Label 7:  Aorta (`aorta`)
          - Label 49: Pulmonary Artery (`pulmonary_artery`)
        Otherwise, derives anatomical chamber partitions and coronary grooves using
        calibrated cardiac quadrant spatial rules.
        """
        shape = arr.shape
        nz, ny, nx = shape

        # 1. Require monai_wholebody_ct_segmentation multi-label masks on clinical CT scans
        if multilabel_map is not None:
            la_raw = multilabel_map == 45
            lv_raw = multilabel_map == 46
            ra_raw = multilabel_map == 47
            rv_raw = multilabel_map == 48
            aorta_seg = multilabel_map == 7
            pa_seg = multilabel_map == 49
            myo_raw = multilabel_map == 44

            if not (np.any(la_raw) and np.any(lv_raw) and np.any(ra_raw) and np.any(rv_raw)):
                raise RuntimeError(
                    "monai_wholebody_ct_segmentation failed to segment all 4 cardiac chambers (LA=45, LV=46, RA=47, RV=48)."
                )

            import torch.nn.functional as F

            la_seg = la_raw & pericardium if np.any(la_raw & pericardium) else la_raw
            lv_seg = lv_raw & pericardium if np.any(lv_raw & pericardium) else lv_raw
            ra_seg = ra_raw & pericardium if np.any(ra_raw & pericardium) else ra_raw
            rv_seg = rv_raw & pericardium if np.any(rv_raw & pericardium) else rv_raw
            myo_seg = myo_raw & pericardium if np.any(myo_raw & pericardium) else myo_raw

            lv_outer = lv_seg | myo_seg
            with torch.no_grad():
                def _dilate3d(mask_np: np.ndarray, k: int = 9) -> np.ndarray:
                    t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                    d = F.max_pool3d(t, kernel_size=k, stride=1, padding=k // 2) > 0.5
                    return d.squeeze(0).squeeze(0).cpu().numpy()

                dil_rv = _dilate3d(rv_seg, k=9)
                dil_lv = _dilate3d(lv_outer, k=9)
                dil_la = _dilate3d(la_seg, k=9)
                dil_ra = _dilate3d(ra_seg, k=9)

            all_chambers = la_seg | lv_outer | ra_seg | rv_seg
            # Anterior interventricular sulcus (LAD): interface between RV and LV/Myocardium
            lad = dil_rv & dil_lv & (~all_chambers) & pericardium
            if not np.any(lad):
                lad = dil_rv & dil_lv & pericardium
            # Left atrioventricular / circumflex sulcus (LCX): interface between LA and LV/Myocardium
            lcx = dil_la & dil_lv & (~all_chambers) & pericardium
            if not np.any(lcx):
                lcx = dil_la & dil_lv & pericardium
            # Right atrioventricular sulcus (RCA): interface between RA and RV
            rca = dil_ra & dil_rv & (~all_chambers) & pericardium
            if not np.any(rca):
                rca = dil_ra & dil_rv & pericardium

            return AnatomyMasks(
                pericardium=pericardium.astype(bool),
                la=la_seg.astype(bool),
                lv=lv_seg.astype(bool),
                ra=ra_seg.astype(bool),
                rv=rv_seg.astype(bool),
                lad=lad.astype(bool),
                lcx=lcx.astype(bool),
                rca=rca.astype(bool),
                aorta=aorta_seg.astype(bool) if np.any(aorta_seg) else None,
                pulmonary_artery=pa_seg.astype(bool) if np.any(pa_seg) else None,
                myocardium=myo_seg.astype(bool) if np.any(myo_seg) else None,
            )

        if max(arr.shape) >= 128 and self.backend != "heuristic":
            raise RuntimeError(
                "monai_wholebody_ct_segmentation model (models/wholeBody_ct_segmentation/models/model.pt) "
                "is required for clinical chamber segmentation; heuristic fallback is disabled."
            )

        indices = np.where(pericardium)
        is_xyz = arr.shape[0] > arr.shape[2] and arr.shape[0] >= 256
        if is_xyz:
            # Axis 0: X (Left/Right), Axis 1: Y (Anterior/Posterior), Axis 2: Z (Inferior/Superior)
            if len(indices[0]) > 0:
                cx = float(np.mean(indices[0]))
                cy = float(np.mean(indices[1]))
                cz = float(np.mean(indices[2]))
            else:
                cx, cy, cz = shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0

            grid_x, grid_y, grid_z = np.indices(shape)
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

            lad_zone = pericardium & (np.abs(grid_x - cx) <= 12) & is_anterior
            lad_dilated = ndi.binary_dilation(lad_zone, iterations=1)
            lad = lad_dilated & (arr >= -30.0)
            if not np.any(lad):
                lad = lad_zone

            lcx_zone = pericardium & is_left & (np.abs(grid_z - cz) <= 12) & is_posterior
            lcx_dilated = ndi.binary_dilation(lcx_zone, iterations=1)
            lcx = lcx_dilated & (arr >= -30.0)
            if not np.any(lcx):
                lcx = lcx_zone

            rca_zone = pericardium & is_right & (np.abs(grid_z - cz) <= 12) & is_anterior
            rca_dilated = ndi.binary_dilation(rca_zone, iterations=1)
            rca = rca_dilated & (arr >= -30.0)
            if not np.any(rca):
                rca = rca_zone
        else:
            # Standard (Z, Y, X) orientation
            if len(indices[0]) > 0:
                cz = float(np.mean(indices[0]))
                cy = float(np.mean(indices[1]))
                cx = float(np.mean(indices[2]))
            else:
                cz, cy, cx = shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0

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

