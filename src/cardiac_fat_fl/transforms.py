"""MONAI MapTransforms for Cardiac Fat Mask Computation and Quantification.

Implements GPU-accelerated and NumPy/SciPy fallback MapTransforms:
  - `ComputeEatPvatMasksd`: Computes 8-compartment EAT/PVAT binary masks from CT and anatomy.
  - `QuantifyAdiposeMetricsd`: Quantifies cm3 volume and mean HU across the 9 clinical regions.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform

from cardiac_fat_fl.anatomy_segmenter import AnatomyMasks, UnifiedAnatomySegmenter
from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    DEFAULT_PVAT_RADIAL_DISTANCE_MM,
    MEASUREMENT_CSV_COLUMNS,
    REPORTED_REGIONS,
    ReportedRegion,
    SegmentationCompartment,
    calculate_volume_cm3,
)


class ComputeEatPvatMasksd(MapTransform):
    """MONAI MapTransform to compute 8-channel EAT and PVAT masks.

    Takes a CT image and either pre-segmented anatomy masks or runs the
    UnifiedAnatomySegmenter to determine pericardium, chambers, and coronaries.
    Computes Euclidean distance transforms for chamber partition and coronary
    radial envelopes, thresholds adipose tissue within [-190, -30] HU, and
    outputs an 8-channel tensor.
    """

    def __init__(
        self,
        keys: KeysCollection = ("image",),
        ct_key: str = "image",
        anatomy_key: Optional[str] = "anatomy",
        output_key: str = "pred",
        hu_min: float = ADIPOSE_HU_MIN,
        hu_max: float = ADIPOSE_HU_MAX,
        pvat_radial_distance_mm: float = DEFAULT_PVAT_RADIAL_DISTANCE_MM,
        restrict_pvat_to_pericardium: bool = True,
        segmenter: Optional[UnifiedAnatomySegmenter] = None,
        allow_missing_keys: bool = False,
        pvat_radial_dilation_mm: Optional[float] = None,
    ) -> None:
        """Initialize transform.

        Args:
            keys: Keys to process.
            ct_key: Dictionary key containing the CT image.
            anatomy_key: Optional dictionary key containing AnatomyMasks or dict of masks.
            output_key: Key under which to store the computed 8-channel mask.
            hu_min: Lower attenuation bound (-190.0 HU).
            hu_max: Upper attenuation bound (-30.0 HU).
            pvat_radial_distance_mm: Radial dilation envelope for PVAT in mm (default: 10.0mm).
            restrict_pvat_to_pericardium: Whether PVAT must lie within pericardial envelope.
            segmenter: Optional custom UnifiedAnatomySegmenter instance.
            allow_missing_keys: If False, raises KeyError when key is missing.
            pvat_radial_dilation_mm: Backward-compatible alias for pvat_radial_distance_mm.
        """
        super().__init__(keys, allow_missing_keys)
        self.ct_key = ct_key
        self.anatomy_key = anatomy_key
        self.output_key = output_key
        self.hu_min = hu_min
        self.hu_max = hu_max
        if pvat_radial_dilation_mm is not None:
            pvat_radial_distance_mm = pvat_radial_dilation_mm
        self.pvat_radial_distance_mm = pvat_radial_distance_mm
        self.restrict_pvat_to_pericardium = restrict_pvat_to_pericardium
        self._segmenter = segmenter if segmenter is not None else UnifiedAnatomySegmenter(backend="auto")

    def _extract_array_and_spacing(self, img_obj: Any) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Extract 3D numpy array and voxel spacing from tensor or MetaTensor."""
        spacing = (1.0, 1.0, 1.0)
        if isinstance(img_obj, MetaTensor):
            arr = img_obj.detach().cpu().numpy()
            if "spacing" in img_obj.meta:
                sp = img_obj.meta["spacing"]
                spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
        elif isinstance(img_obj, torch.Tensor):
            arr = img_obj.detach().cpu().numpy()
        else:
            arr = np.asarray(img_obj)

        # Squeeze batch/channel if needed (shape -> 3D)
        while arr.ndim > 3:
            arr = arr[0]

        return arr.astype(np.float32), spacing

    def _compute_chamber_voronoi(
        self,
        la: np.ndarray,
        lv: np.ndarray,
        ra: np.ndarray,
        rv: np.ndarray,
        spacing: Tuple[float, float, float],
        pericardium: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Voronoi proximity partition among the 4 cardiac chambers."""
        # Arrays are indexed in (Z, Y, X) order, while spacing is (sx, sy, sz) = (X, Y, Z).
        # SciPy distance_transform_edt sampling must match the array axis order (sz, sy, sx).
        sampling_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))

        if pericardium is not None and np.any(pericardium):
            idx = np.where(pericardium)
            z0, z1 = int(np.min(idx[0])), int(np.max(idx[0])) + 1
            y0, y1 = int(np.min(idx[1])), int(np.max(idx[1])) + 1
            x0, x1 = int(np.min(idx[2])), int(np.max(idx[2])) + 1

            margin_z = int(np.ceil(20.0 / spacing[2]))
            margin_y = int(np.ceil(20.0 / spacing[1]))
            margin_x = int(np.ceil(20.0 / spacing[0]))

            z0, z1 = max(0, z0 - margin_z), min(la.shape[0], z1 + margin_z)
            y0, y1 = max(0, y0 - margin_y), min(la.shape[1], y1 + margin_y)
            x0, x1 = max(0, x0 - margin_x), min(la.shape[2], x1 + margin_x)

            sub_la = la[z0:z1, y0:y1, x0:x1]
            sub_lv = lv[z0:z1, y0:y1, x0:x1]
            sub_ra = ra[z0:z1, y0:y1, x0:x1]
            sub_rv = rv[z0:z1, y0:y1, x0:x1]

            d_la = ndi.distance_transform_edt(~sub_la, sampling=sampling_zyx) if np.any(sub_la) else np.full(sub_la.shape, 1e6)
            d_lv = ndi.distance_transform_edt(~sub_lv, sampling=sampling_zyx) if np.any(sub_lv) else np.full(sub_lv.shape, 1e6)
            d_ra = ndi.distance_transform_edt(~sub_ra, sampling=sampling_zyx) if np.any(sub_ra) else np.full(sub_ra.shape, 1e6)
            d_rv = ndi.distance_transform_edt(~sub_rv, sampling=sampling_zyx) if np.any(sub_rv) else np.full(sub_rv.shape, 1e6)

            stacked = np.stack([d_la, d_lv, d_ra, d_rv], axis=0)
            nearest = np.argmin(stacked, axis=0)

            zone_la = np.zeros_like(la, dtype=bool)
            zone_la[z0:z1, y0:y1, x0:x1] = nearest == 0
            zone_lv = np.zeros_like(lv, dtype=bool)
            zone_lv[z0:z1, y0:y1, x0:x1] = nearest == 1
            zone_ra = np.zeros_like(ra, dtype=bool)
            zone_ra[z0:z1, y0:y1, x0:x1] = nearest == 2
            zone_rv = np.zeros_like(rv, dtype=bool)
            zone_rv[z0:z1, y0:y1, x0:x1] = nearest == 3
            return zone_la, zone_lv, zone_ra, zone_rv

        # Full-volume fallback if no pericardium provided
        d_la = ndi.distance_transform_edt(~la, sampling=sampling_zyx) if np.any(la) else np.full(la.shape, 1e6)
        d_lv = ndi.distance_transform_edt(~lv, sampling=sampling_zyx) if np.any(lv) else np.full(lv.shape, 1e6)
        d_ra = ndi.distance_transform_edt(~ra, sampling=sampling_zyx) if np.any(ra) else np.full(ra.shape, 1e6)
        d_rv = ndi.distance_transform_edt(~rv, sampling=sampling_zyx) if np.any(rv) else np.full(rv.shape, 1e6)

        stacked = np.stack([d_la, d_lv, d_ra, d_rv], axis=0)  # Shape (4, D, H, W)
        nearest = np.argmin(stacked, axis=0)

        zone_la = nearest == 0
        zone_lv = nearest == 1
        zone_ra = nearest == 2
        zone_rv = nearest == 3

        return zone_la, zone_lv, zone_ra, zone_rv

    def _compute_pvat_envelope(
        self,
        vessel_mask: np.ndarray,
        spacing: Tuple[float, float, float],
        max_dist_mm: float,
        pericardium: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute radial distance envelope around a coronary vessel centerline/mask."""
        if not np.any(vessel_mask):
            return np.zeros_like(vessel_mask, dtype=bool)

        sampling_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))

        if pericardium is not None and np.any(pericardium):
            idx = np.where(pericardium)
            z0, z1 = int(np.min(idx[0])), int(np.max(idx[0])) + 1
            y0, y1 = int(np.min(idx[1])), int(np.max(idx[1])) + 1
            x0, x1 = int(np.min(idx[2])), int(np.max(idx[2])) + 1

            margin_z = int(np.ceil((max_dist_mm + 5.0) / spacing[2]))
            margin_y = int(np.ceil((max_dist_mm + 5.0) / spacing[1]))
            margin_x = int(np.ceil((max_dist_mm + 5.0) / spacing[0]))

            z0, z1 = max(0, z0 - margin_z), min(vessel_mask.shape[0], z1 + margin_z)
            y0, y1 = max(0, y0 - margin_y), min(vessel_mask.shape[1], y1 + margin_y)
            x0, x1 = max(0, x0 - margin_x), min(vessel_mask.shape[2], x1 + margin_x)

            sub_v = vessel_mask[z0:z1, y0:y1, x0:x1]
            if not np.any(sub_v):
                return np.zeros_like(vessel_mask, dtype=bool)

            dist = ndi.distance_transform_edt(~sub_v, sampling=sampling_zyx)
            sub_env = (dist <= max_dist_mm) & (~sub_v)

            envelope = np.zeros_like(vessel_mask, dtype=bool)
            envelope[z0:z1, y0:y1, x0:x1] = sub_env
            return envelope

        dist = ndi.distance_transform_edt(~vessel_mask, sampling=sampling_zyx)
        envelope = (dist <= max_dist_mm) & (~vessel_mask)
        return envelope

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)

        if self.ct_key not in d:
            raise KeyError(f"CT key '{self.ct_key}' not found in data dictionary.")

        ct_obj = d[self.ct_key]
        ct_arr, spacing = self._extract_array_and_spacing(ct_obj)

        # Obtain anatomy masks
        anatomy: Optional[AnatomyMasks] = None
        if self.anatomy_key is not None and self.anatomy_key in d:
            raw_anat = d[self.anatomy_key]
            if isinstance(raw_anat, AnatomyMasks):
                anatomy = raw_anat
            elif isinstance(raw_anat, dict):
                anatomy = AnatomyMasks(
                    pericardium=np.asarray(raw_anat["pericardium"], dtype=bool),
                    la=np.asarray(raw_anat["la"], dtype=bool),
                    lv=np.asarray(raw_anat["lv"], dtype=bool),
                    ra=np.asarray(raw_anat["ra"], dtype=bool),
                    rv=np.asarray(raw_anat["rv"], dtype=bool),
                    lad=np.asarray(raw_anat["lad"], dtype=bool),
                    lcx=np.asarray(raw_anat["lcx"], dtype=bool),
                    rca=np.asarray(raw_anat["rca"], dtype=bool),
                )

        if anatomy is None:
            anatomy = self._segmenter.segment(ct_arr, spacing=spacing)

        # 1. Adipose attenuation thresholding: strictly [-190.0, -30.0] HU
        adipose_voxels = (ct_arr >= self.hu_min) & (ct_arr <= self.hu_max)

        # 2. Total Epicardial Adipose Tissue (EAT_Total)
        # All adipose tissue inside the pericardial/heart mask
        eat_total = adipose_voxels & anatomy.pericardium

        # 3. Chamber-specific EAT (EAT_LA, EAT_LV, EAT_RA, EAT_RV)
        zone_la, zone_lv, zone_ra, zone_rv = self._compute_chamber_voronoi(
            anatomy.la, anatomy.lv, anatomy.ra, anatomy.rv, spacing=spacing, pericardium=anatomy.pericardium
        )
        eat_la = eat_total & zone_la
        eat_lv = eat_total & zone_lv
        eat_ra = eat_total & zone_ra
        eat_rv = eat_total & zone_rv

        # 4. Pericoronary Adipose Tissue (PVAT_LAD, PVAT_LCX, PVAT_RCA)
        # Radial Euclidean distance dilation around vessel centerline/mask
        envelope_lad = self._compute_pvat_envelope(anatomy.lad, spacing, self.pvat_radial_distance_mm, pericardium=anatomy.pericardium)
        envelope_lcx = self._compute_pvat_envelope(anatomy.lcx, spacing, self.pvat_radial_distance_mm, pericardium=anatomy.pericardium)
        envelope_rca = self._compute_pvat_envelope(anatomy.rca, spacing, self.pvat_radial_distance_mm, pericardium=anatomy.pericardium)

        pvat_lad = adipose_voxels & envelope_lad
        pvat_lcx = adipose_voxels & envelope_lcx
        pvat_rca = adipose_voxels & envelope_rca

        if self.restrict_pvat_to_pericardium:
            pvat_lad = pvat_lad & anatomy.pericardium
            pvat_lcx = pvat_lcx & anatomy.pericardium
            pvat_rca = pvat_rca & anatomy.pericardium

        # Stack into 8-channel boolean tensor (8, D, H, W)
        masks_8ch = np.stack(
            [
                eat_total,  # Channel 0: EAT_Total
                eat_la,     # Channel 1: EAT_LA
                eat_lv,     # Channel 2: EAT_LV
                eat_ra,     # Channel 3: EAT_RA
                eat_rv,     # Channel 4: EAT_RV
                pvat_lad,   # Channel 5: PVAT_LAD
                pvat_lcx,   # Channel 6: PVAT_LCX
                pvat_rca,   # Channel 7: PVAT_RCA
            ],
            axis=0,
        ).astype(np.float32)

        # Wrap in MetaTensor if input was MetaTensor
        if isinstance(ct_obj, MetaTensor):
            out_tensor = MetaTensor(
                torch.from_numpy(masks_8ch),
                affine=ct_obj.affine,
                meta=dict(ct_obj.meta),
            )
        else:
            out_tensor = torch.from_numpy(masks_8ch)

        d[self.output_key] = out_tensor
        d["anatomy_masks"] = anatomy
        return d


class QuantifyAdiposeMetricsd(MapTransform):
    """MONAI MapTransform to calculate cm3 volumes and mean HU attenuation.

    Quantifies all 9 reported regions:
      - EAT Total
      - EAT LA
      - EAT LV
      - EAT RA
      - EAT RV
      - PVAT Total (spatial union of PVAT_LAD | PVAT_LCX | PVAT_RCA)
      - PVAT LAD
      - PVAT LCX
      - PVAT RCA

    Handles empty masks gracefully by returning 0.0 cm3 and NaN mean HU.
    Populates an exact 19-column dictionary conforming to CardiacFatQuant.
    """

    def __init__(
        self,
        keys: KeysCollection = ("image",),
        ct_key: str = "image",
        mask_key: str = "pred",
        participant_id_key: str = "participant_id",
        output_metrics_key: str = "metrics",
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize quantification transform.

        Args:
            keys: Keys to process.
            ct_key: Dictionary key containing the CT image in HU.
            mask_key: Dictionary key containing the 8-channel EAT/PVAT mask.
            participant_id_key: Key for participant identifier.
            output_metrics_key: Key where quantified dictionary will be placed.
            allow_missing_keys: Whether to ignore missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        self.ct_key = ct_key
        self.mask_key = mask_key
        self.participant_id_key = participant_id_key
        self.output_metrics_key = output_metrics_key

    def _extract_array_and_spacing(self, img_obj: Any) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Extract array and voxel spacing."""
        spacing = (1.0, 1.0, 1.0)
        if isinstance(img_obj, MetaTensor):
            arr = img_obj.detach().cpu().numpy()
            if "spacing" in img_obj.meta:
                sp = img_obj.meta["spacing"]
                spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
        elif isinstance(img_obj, torch.Tensor):
            arr = img_obj.detach().cpu().numpy()
        else:
            arr = np.asarray(img_obj)

        while arr.ndim > 3:
            arr = arr[0]

        return arr.astype(np.float32), spacing

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)

        if self.ct_key not in d:
            raise KeyError(f"CT key '{self.ct_key}' not found in data.")
        if self.mask_key not in d:
            raise KeyError(f"Mask key '{self.mask_key}' not found in data.")

        ct_arr, spacing = self._extract_array_and_spacing(d[self.ct_key])
        mask_obj = d[self.mask_key]

        if isinstance(mask_obj, torch.Tensor):
            mask_np = mask_obj.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_obj)

        # Squeeze leading dimensions if shape is (1, 8, D, H, W) -> (8, D, H, W)
        if mask_np.ndim == 5 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]

        if mask_np.shape[0] != 8:
            raise ValueError(f"Expected 8-channel mask, but found shape {mask_np.shape}")

        # Channels:
        # 0: EAT_Total
        # 1: EAT_LA
        # 2: EAT_LV
        # 3: EAT_RA
        # 4: EAT_RV
        # 5: PVAT_LAD
        # 6: PVAT_LCX
        # 7: PVAT_RCA
        m_eat_total = mask_np[SegmentationCompartment.EAT_TOTAL] > 0.5
        m_eat_la = mask_np[SegmentationCompartment.EAT_LA] > 0.5
        m_eat_lv = mask_np[SegmentationCompartment.EAT_LV] > 0.5
        m_eat_ra = mask_np[SegmentationCompartment.EAT_RA] > 0.5
        m_eat_rv = mask_np[SegmentationCompartment.EAT_RV] > 0.5
        m_pvat_lad = mask_np[SegmentationCompartment.PVAT_LAD] > 0.5
        m_pvat_lcx = mask_np[SegmentationCompartment.PVAT_LCX] > 0.5
        m_pvat_rca = mask_np[SegmentationCompartment.PVAT_RCA] > 0.5

        # PVAT Total is computed as the spatial union of PVAT_LAD | PVAT_LCX | PVAT_RCA
        m_pvat_total = m_pvat_lad | m_pvat_lcx | m_pvat_rca

        region_masks: Dict[ReportedRegion, np.ndarray] = {
            ReportedRegion.EAT_TOTAL: m_eat_total,
            ReportedRegion.EAT_LA: m_eat_la,
            ReportedRegion.EAT_LV: m_eat_lv,
            ReportedRegion.EAT_RA: m_eat_ra,
            ReportedRegion.EAT_RV: m_eat_rv,
            ReportedRegion.PVAT_TOTAL: m_pvat_total,
            ReportedRegion.PVAT_LAD: m_pvat_lad,
            ReportedRegion.PVAT_LCX: m_pvat_lcx,
            ReportedRegion.PVAT_RCA: m_pvat_rca,
        }

        participant_id = str(d.get(self.participant_id_key, "participant_001"))
        metrics: Dict[str, Any] = {"participant_id": participant_id}

        for region in REPORTED_REGIONS:
            m = region_masks[region]
            voxel_count = int(np.sum(m))

            if voxel_count > 0:
                vol_cm3 = calculate_volume_cm3(voxel_count, spacing)
                hu_values = ct_arr[m]
                mean_hu = float(np.mean(hu_values))
            else:
                vol_cm3 = 0.0
                mean_hu = float("nan")

            metrics[f"{region.value} Volume (cm3)"] = vol_cm3
            metrics[f"{region.value} Mean Attenuation (HU)"] = mean_hu

        d[self.output_metrics_key] = metrics
        return d

