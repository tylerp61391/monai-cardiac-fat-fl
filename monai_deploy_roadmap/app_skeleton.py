"""Reference MONAI Deploy Application Package (MAP) Skeleton for monai-cardiac-fat-fl.

Wraps the existing `cardiac_fat_fl` 3D segmentation, anisotropic EDT morphometry,
multi-frame DICOM-SEG (1.2.840.10008.5.1.4.1.1.66.4), and DICOM-SR TID 1500
(1.2.840.10008.5.1.4.1.1.88.34) modules into a 3-node MONAI Deploy Operator DAG.

Can be executed standalone or packaged via `monai-deploy package`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from cardiac_fat_fl.anatomy_segmenter import UnifiedAnatomySegmenter
from cardiac_fat_fl.dicom_sr import export_cardiac_fat_sr
from cardiac_fat_fl.io_metatensor import (
    discover_participants,
    read_medical_image_as_metatensor,
    write_eat_pvat_seg_nrrd,
)
from cardiac_fat_fl.ohif_exporter import (
    export_ct_to_dicom_series,
    export_segmentation_to_dicom_seg,
)
from cardiac_fat_fl.transforms import ComputeEatPvatMasksd, QuantifyAdiposeMetricsd

logger = logging.getLogger(__name__)


class CardiacCTStudyLoaderOp:
    """Operator 1: Ingests a clinical DICOM CT study directory or 3D volume into a MONAI MetaTensor."""

    def compute(self, input_dir: Path) -> Dict[str, Any]:
        participants = discover_participants(input_dir)
        if participants:
            pid, scan_path = participants[0]
        else:
            pid, scan_path = input_dir.name, input_dir

        meta_ct = read_medical_image_as_metatensor(scan_path)
        return {
            "participant_id": pid,
            "scan_path": scan_path,
            "image": meta_ct,
        }


class CardiacFatQuantifierOp:
    """Operator 2: Executes NGC WholeBody 1.5mm SegResNet + DeepHeartSeg ONNX + 3D Voronoi EDT."""

    def __init__(self, models_dir: Path | None = None, pvat_radius_mm: float = 10.0) -> None:
        heart_onnx = (models_dir / "deepheartseg_heart.onnx") if models_dir else None
        loc_onnx = (models_dir / "deepheartseg_loc.onnx") if models_dir else None
        segmenter = UnifiedAnatomySegmenter(
            model_path=heart_onnx if (heart_onnx and heart_onnx.is_file()) else None,
            loc_model_path=loc_onnx if (loc_onnx and loc_onnx.is_file()) else None,
            backend="onnx",
        )
        self.mask_tx = ComputeEatPvatMasksd(
            keys=["image"],
            pvat_radial_distance_mm=pvat_radius_mm,
            segmenter=segmenter,
        )
        self.quant_tx = QuantifyAdiposeMetricsd(keys=["image"])

    def compute(self, study_dict: Dict[str, Any]) -> Dict[str, Any]:
        out = self.mask_tx(study_dict)
        out = self.quant_tx(out)
        return out


class DicomSegAndSrWriterOp:
    """Operator 3: Emits standard multi-frame DICOM-SEG (.66.4), DICOM-SR TID 1500 (.88.34), and JSON."""

    def compute(self, quant_dict: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        pid = str(quant_dict["participant_id"])
        meta_ct = quant_dict["image"]
        seg_mask_8ch = quant_dict["pred"]
        metrics = quant_dict["metrics"]
        scan_path = Path(quant_dict["scan_path"])

        # 1. Write temporary NRRD segmentation for exporter compatibility
        seg_nrrd_path = output_dir / "eat_pvat.seg.nrrd"
        spacing = tuple(meta_ct.meta.get("spacing", (1.0, 1.0, 1.0)))
        origin = tuple(meta_ct.meta.get("origin", (0.0, 0.0, 0.0))) if "origin" in meta_ct.meta else None
        write_eat_pvat_seg_nrrd(
            seg_data=seg_mask_8ch,
            output_path=seg_nrrd_path,
            reference_header=meta_ct.meta.get("nrrd_header"),
            spacing=spacing,
            origin=origin,
        )

        # 2. Ensure reference DICOM CT series is available for UID linkage
        if scan_path.is_dir() and list(scan_path.glob("*.dcm")):
            ct_dcm_dir = scan_path
        else:
            ct_dcm_dir = output_dir / "ct_dicom"
            export_ct_to_dicom_series(scan_path, ct_dcm_dir, patient_id=pid)

        # 3. Export multi-frame DICOM-SEG (1.2.840.10008.5.1.4.1.1.66.4)
        seg_dcm_path = output_dir / "eat_pvat_seg.dcm"
        export_segmentation_to_dicom_seg(
            seg_nrrd_path=seg_nrrd_path,
            ct_dicom_dir=ct_dcm_dir,
            output_seg_dcm=seg_dcm_path,
        )

        # 4. Export Comprehensive 3D DICOM-SR TID 1500 (1.2.840.10008.5.1.4.1.1.88.34)
        sr_dcm_path = output_dir / "eat_pvat_sr.dcm"
        export_cardiac_fat_sr(
            metrics=metrics,
            ct_dicom_dir=ct_dcm_dir,
            output_sr_dcm=sr_dcm_path,
            seg_dcm_path=seg_dcm_path,
        )

        # 5. Export JSON summary metrics
        json_path = output_dir / "cardiac_fat_measurements.json"
        json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return {
            "dicom_seg": seg_dcm_path,
            "dicom_sr": sr_dcm_path,
            "metrics_json": json_path,
        }


def run_map_pipeline(input_dir: Path, output_dir: Path, models_dir: Path | None = None) -> Dict[str, Path]:
    """Execute the 3-operator MONAI Deploy DAG."""
    loader = CardiacCTStudyLoaderOp()
    quantifier = CardiacFatQuantifierOp(models_dir=models_dir)
    writer = DicomSegAndSrWriterOp()

    study_data = loader.compute(input_dir)
    quant_data = quantifier.compute(study_data)
    return writer.compute(quant_data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MONAI Deploy Cardiac Fat Application Package (MAP)")
    parser.add_argument("-i", "--input", type=Path, default=Path("/var/monai/input"), help="Input DICOM study directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("/var/monai/output"), help="Output directory for SEG & SR")
    parser.add_argument("-m", "--models", type=Path, default=Path("/opt/holoscan/models"), help="Model weights directory")
    args = parser.parse_args()

    resolved_models = args.models if args.models.exists() else Path("models")
    artifacts = run_map_pipeline(args.input, args.output, resolved_models)
    print(f"MONAI Deploy MAP execution complete: {artifacts}")

