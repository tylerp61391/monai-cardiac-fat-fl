"""MONAI Deploy Application Package (MAP) implementation for cardiac fat quantification.

Provides clinical MONAI Deploy App SDK (NVIDIA Holoscan SDK) operators and application DAG
for automated 3D Epicardial (EAT) and Pericoronary (PVAT) adipose tissue segmentation,
volumetric/attenuation quantification, and standard DICOM-SEG (.66.4) and DICOM-SR TID 1500 (.88.34)
clinical archive export.

Conforms to MONAI Application Package (MAP) specification (/var/monai/input, /var/monai/output,
and /opt/holoscan/models).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pydicom

from cardiac_fat_fl.anatomy_segmenter import UnifiedAnatomySegmenter
from cardiac_fat_fl.dicom_sr import export_cardiac_fat_sr
from cardiac_fat_fl.io_metatensor import (
    discover_participants,
    read_medical_image_as_metatensor,
    write_eat_pvat_seg_nrrd,
)
from cardiac_fat_fl.measurement_schema import (
    MEASUREMENT_CSV_COLUMNS,
    validate_measurement_dataframe,
)
from cardiac_fat_fl.ohif_exporter import (
    export_ct_to_dicom_series,
    export_segmentation_to_dicom_seg,
)
from cardiac_fat_fl.transforms import ComputeEatPvatMasksd, QuantifyAdiposeMetricsd

logger = logging.getLogger(__name__)

# Check if monai-deploy-app-sdk is installed; provide seamless fallback classes if not
try:
    from monai.deploy.core import (
        Application as MonaiDeployApp,
        ExecutionContext,
        InputContext,
        Operator as MonaiDeployOperator,
        OutputContext,
        env,
    )

    _HAS_MONAI_DEPLOY_SDK = True
except ImportError:
    _HAS_MONAI_DEPLOY_SDK = False

    class MonaiDeployOperator:  # type: ignore[no-redef]
        """Fallback base operator when monai-deploy-app-sdk is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class MonaiDeployApp:  # type: ignore[no-redef]
        """Fallback base application when monai-deploy-app-sdk is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def compose(self) -> None:
            pass

        def run(self, *args: Any, **kwargs: Any) -> Any:
            pass

    def env(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        def decorator(cls: Any) -> Any:
            return cls

        return decorator


class CardiacCTStudyLoaderOp(MonaiDeployOperator):
    """Operator 1: Ingests clinical DICOM CT series or 3D volume (NRRD/NIfTI) into a MONAI MetaTensor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def compute(
        self,
        op_input: Any = None,
        op_output: Any = None,
        context: Any = None,
        input_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Load study from input path or op_input context."""
        resolved_path: Optional[Path] = None
        if input_path is not None:
            resolved_path = Path(input_path).resolve()
        elif op_input is not None and hasattr(op_input, "receive"):
            val = op_input.receive()
            if val is not None:
                resolved_path = Path(val).resolve()

        if resolved_path is None or not resolved_path.exists():
            resolved_path = Path("/var/monai/input").resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Input study directory or file does not exist: {resolved_path}")

        # Discover participant scan or handle single volume file directly
        if resolved_path.is_file():
            pid = resolved_path.parent.name if resolved_path.parent.name not in ["", ".", "/", "\\"] else resolved_path.stem
            scan_path = resolved_path
        else:
            participants = discover_participants(resolved_path)
            if participants:
                pid, scan_path = participants[0]
            else:
                pid = resolved_path.name
                scan_path = resolved_path

        meta_ct = read_medical_image_as_metatensor(scan_path)

        result = {
            "participant_id": str(pid),
            "scan_path": scan_path,
            "image": meta_ct,
        }

        if op_output is not None and hasattr(op_output, "emit"):
            op_output.emit(result)

        return result


class CardiacFatQuantifierOp(MonaiDeployOperator):
    """Operator 2: Dual-network anatomy segmentation (WholeBody + DeepHeartSeg) + 3D Voronoi EDT."""

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        pvat_radius_mm: float = 10.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pvat_radius_mm = float(pvat_radius_mm)
        self.models_dir = Path(models_dir).resolve() if models_dir else Path("models").resolve()

        # Resolve model paths
        heart_onnx = self.models_dir / "deepheartseg_heart.onnx"
        loc_onnx = self.models_dir / "deepheartseg_loc.onnx"

        # Initialize UnifiedAnatomySegmenter
        segmenter = UnifiedAnatomySegmenter(
            model_path=heart_onnx if heart_onnx.is_file() else None,
            loc_model_path=loc_onnx if loc_onnx.is_file() else None,
            backend="onnx" if heart_onnx.is_file() else "heuristic",
        )

        self.mask_transform = ComputeEatPvatMasksd(
            keys=["image"],
            pvat_radial_distance_mm=self.pvat_radius_mm,
            segmenter=segmenter,
        )
        self.quant_transform = QuantifyAdiposeMetricsd(keys=["image"])

    def compute(
        self,
        study_dict: Optional[Dict[str, Any]] = None,
        op_input: Any = None,
        op_output: Any = None,
        context: Any = None,
    ) -> Dict[str, Any]:
        """Execute 8-compartment adipose masking and metrics quantification."""
        data = study_dict
        if data is None and op_input is not None and hasattr(op_input, "receive"):
            data = op_input.receive()

        if data is None:
            raise ValueError("No study dictionary provided to CardiacFatQuantifierOp")

        # 1. 8-compartment EAT/PVAT mask computation
        out_dict = self.mask_transform(data)
        # 2. Volume & attenuation quantification
        out_dict = self.quant_transform(out_dict)

        if op_output is not None and hasattr(op_output, "emit"):
            op_output.emit(out_dict)

        return out_dict


class DicomSegAndSrWriterOp(MonaiDeployOperator):
    """Operator 3: Exports multi-frame DICOM-SEG (.66.4), DICOM-SR TID 1500 (.88.34), and JSON/CSV metrics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def compute(
        self,
        quant_dict: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        op_input: Any = None,
        op_output: Any = None,
        context: Any = None,
    ) -> Dict[str, Path]:
        """Export clinical DICOM-SEG, DICOM-SR TID 1500, and summary files."""
        data = quant_dict
        if data is None and op_input is not None and hasattr(op_input, "receive"):
            data = op_input.receive()

        if data is None:
            raise ValueError("No quantification dictionary provided to DicomSegAndSrWriterOp")

        out_path = Path(output_dir).resolve() if output_dir is not None else Path("/var/monai/output").resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        pid = str(data.get("participant_id", "PARTICIPANT_001"))
        meta_ct = data["image"]
        seg_mask_8ch = data["pred"]
        metrics = data.get("metrics", {})
        scan_path = Path(data.get("scan_path", out_path))

        # 1. Write reference eat_pvat.seg.nrrd
        seg_nrrd_path = out_path / "eat_pvat.seg.nrrd"
        ref_header = meta_ct.meta.get("nrrd_header")
        spacing = tuple(meta_ct.meta.get("spacing", (1.0, 1.0, 1.0)))
        origin = tuple(meta_ct.meta.get("origin", (0.0, 0.0, 0.0))) if "origin" in meta_ct.meta else None

        write_eat_pvat_seg_nrrd(
            seg_data=seg_mask_8ch,
            output_path=seg_nrrd_path,
            reference_header=ref_header,
            spacing=spacing,
            origin=origin,
        )

        # 2. Collect or convert DICOM CT slice instances for SOP linkage
        ct_dicom_paths: List[Path] = []
        if scan_path.is_dir():
            ct_candidates = [
                p for p in list(scan_path.glob("*.dcm")) + list(scan_path.glob("*/*.dcm"))
                if not p.name.endswith("_seg.dcm") and not p.name.startswith("eat_pvat")
            ]
            for p in ct_candidates:
                try:
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                    if getattr(ds, "Modality", None) == "CT":
                        ct_dicom_paths.append(p)
                except Exception:
                    continue

        if not ct_dicom_paths:
            # Convert CT volume to DICOM series
            ct_dest_dir = out_path / "ct_dicom"
            ct_arr = meta_ct.detach().cpu().numpy().squeeze()
            ct_dicom_paths = export_ct_to_dicom_series(
                ct_array=ct_arr,
                spacing=spacing,
                output_dir=ct_dest_dir,
                patient_id=pid,
            )

        # Ensure slices are sorted by ImagePositionPatient[2] or InstanceNumber
        def _slice_key(p: Path) -> float:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                if hasattr(ds, "ImagePositionPatient"):
                    return float(ds.ImagePositionPatient[2])
                return float(getattr(ds, "InstanceNumber", 0))
            except Exception:
                return 0.0

        ct_dicom_paths.sort(key=_slice_key)

        # 3. Export multi-frame DICOM-SEG (1.2.840.10008.5.1.4.1.1.66.4)
        dicom_seg_path = out_path / "eat_pvat_seg.dcm"
        export_segmentation_to_dicom_seg(
            seg_array=seg_mask_8ch,
            ct_dicom_paths=ct_dicom_paths,
            output_path=dicom_seg_path,
            patient_id=pid,
        )

        # 4. Export Comprehensive 3D DICOM-SR TID 1500 (1.2.840.10008.5.1.4.1.1.88.34)
        dicom_sr_path = out_path / "eat_pvat_sr.dcm"
        first_ct_dcm = ct_dicom_paths[0] if ct_dicom_paths else None
        export_cardiac_fat_sr(
            measurements=metrics,
            output_path=dicom_sr_path,
            source_ct_dcm=first_ct_dcm,
            seg_dcm=dicom_seg_path,
            participant_id=pid,
        )

        # 5. Export JSON summary metrics
        metrics_json_path = out_path / "cardiac_fat_measurements.json"
        metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # 6. Export 19-column CSV summary
        metrics_csv_path = out_path / "cardiac_fat_measurements.csv"
        csv_row = {"participant_id": pid}
        csv_row.update(metrics)
        df_out = pd.DataFrame([csv_row])
        for col in MEASUREMENT_CSV_COLUMNS:
            if col not in df_out.columns:
                df_out[col] = float("nan")
        df_out = df_out[MEASUREMENT_CSV_COLUMNS]
        validate_measurement_dataframe(df_out)
        df_out.to_csv(metrics_csv_path, index=False)

        artifacts = {
            "eat_pvat_seg_nrrd": seg_nrrd_path,
            "dicom_seg": dicom_seg_path,
            "dicom_sr": dicom_sr_path,
            "metrics_json": metrics_json_path,
            "metrics_csv": metrics_csv_path,
        }

        # 7. Export Clinical Diagnostic Report (HTML, PDF, DICOM Encapsulated PDF, and MPR Montage)
        try:
            from cardiac_fat_fl.clinical_report import export_clinical_imaging_report
            ct_vol_np = ct_meta.cpu().numpy() if hasattr(ct_meta, "cpu") else np.asarray(ct_meta)
            report_artifacts = export_clinical_imaging_report(
                output_dir=out_path,
                metrics=metrics,
                image=ct_vol_np,
                mask=seg_mask_8ch,
                participant_id=pid,
                source_ct_dcm=first_ct_dcm,
            )
            artifacts.update(report_artifacts)
        except Exception as report_err:
            logger.warning("Clinical report generation skipped or failed: %s", report_err)

        if op_output is not None and hasattr(op_output, "emit"):
            op_output.emit(artifacts)

        return artifacts


@env(
    pip_packages=[
        "torch>=2.4.0",
        "monai>=1.4.0",
        "pydicom>=2.4.0",
        "pynrrd>=1.0.0",
        "scipy>=1.11.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
    ]
)
class CardiacFatQuantApplication(MonaiDeployApp):
    """MONAI Deploy Application DAG for Cardiac Fat (EAT & PVAT) Quantification."""

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        pvat_radius_mm: float = 10.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.models_dir = Path(models_dir).resolve() if models_dir else Path("models").resolve()
        self.pvat_radius_mm = float(pvat_radius_mm)

    def compose(self) -> None:
        """Compose the 3-operator DAG in MONAI Deploy App SDK."""
        self.loader_op = CardiacCTStudyLoaderOp()
        self.quant_op = CardiacFatQuantifierOp(
            models_dir=self.models_dir,
            pvat_radius_mm=self.pvat_radius_mm,
        )
        self.writer_op = DicomSegAndSrWriterOp()

        if hasattr(self, "add_flow"):
            self.add_flow(self.loader_op, self.quant_op)
            self.add_flow(self.quant_op, self.writer_op)

    def run_standalone(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, Path]:
        """Execute the DAG programmatically without requiring the Holoscan runner daemon."""
        self.compose()
        study_data = self.loader_op.compute(input_path=input_path)
        quant_data = self.quant_op.compute(study_dict=study_data)
        artifacts = self.writer_op.compute(quant_dict=quant_data, output_dir=output_path)
        return artifacts


def run_monai_deploy_pipeline(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    models_dir: Optional[Union[str, Path]] = None,
    pvat_radius_mm: float = 10.0,
) -> Dict[str, Path]:
    """Top-level pipeline execution function for MONAI Deploy Application Package (MAP).

    Args:
        input_path: Path to input DICOM study directory, NRRD, or NIfTI scan.
        output_path: Path to output directory where DICOM-SEG, DICOM-SR, and CSV/JSON are written.
        models_dir: Directory containing deep learning model weights.
        pvat_radius_mm: Radial dilation buffer distance in mm for Pericoronary Adipose Tissue (default: 10.0 mm).

    Returns:
        Dictionary mapping artifact keys to their generated output file Paths.
    """
    app = CardiacFatQuantApplication(models_dir=models_dir, pvat_radius_mm=pvat_radius_mm)
    return app.run_standalone(input_path=input_path, output_path=output_path)
