"""MONAI Deploy Application Package (MAP) Entrypoint for Cardiac Fat Quantification.

Conforms to MONAI Deploy Application Package (MAP) specification.
Can be executed standalone or packaged via `monai-deploy package`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to sys.path if running directly from repo root
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cardiac_fat_fl.monai_deploy import (
    CardiacFatQuantApplication,
    run_monai_deploy_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("monai_deploy_cardiac_fat")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MONAI Deploy Application Package (MAP): 3D Cardiac Fat (EAT & PVAT) Quantification"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("/var/monai/input"),
        help="Input DICOM study directory, NRRD, or NIfTI scan (default: /var/monai/input)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/var/monai/output"),
        help="Output directory for DICOM-SEG, DICOM-SR TID 1500, and metrics (default: /var/monai/output)",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=Path,
        default=Path("/opt/holoscan/models"),
        help="Directory containing model weights (default: /opt/holoscan/models or local models/)",
    )
    parser.add_argument(
        "--pvat-radius",
        type=float,
        default=10.0,
        help="Radial dilation distance in mm for Pericoronary Adipose Tissue (default: 10.0 mm)",
    )

    args = parser.parse_args()

    # Fallback to local models directory if default /opt/holoscan/models doesn't exist
    resolved_models = args.models
    if not resolved_models.exists():
        local_models = repo_root / "models"
        if local_models.exists():
            resolved_models = local_models

    logger.info("Starting MONAI Deploy MAP pipeline...")
    logger.info("  Input:  %s", args.input)
    logger.info("  Output: %s", args.output)
    logger.info("  Models: %s", resolved_models)
    logger.info("  PVAT Radius: %.1f mm", args.pvat_radius)

    artifacts = run_monai_deploy_pipeline(
        input_path=args.input,
        output_path=args.output,
        models_dir=resolved_models,
        pvat_radius_mm=args.pvat_radius,
    )

    logger.info("Pipeline completed successfully! Generated artifacts:")
    for k, v in artifacts.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()

