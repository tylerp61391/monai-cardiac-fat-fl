"""Modernized MONAI & NVIDIA FLARE Cardiac Fat Quantification Package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

if TYPE_CHECKING:
    from cardiac_fat_fl.anatomy_segmenter import AnatomyMasks, UnifiedAnatomySegmenter
    from cardiac_fat_fl.clinical_report import (
        encapsulate_pdf_to_dicom,
        evaluate_clinical_risk,
        export_clinical_imaging_report,
        generate_clinical_report_html,
        generate_clinical_report_pdf,
        generate_mpr_snapshot,
    )
    from cardiac_fat_fl.cohort_research import (
        CohortDataset,
        CohortParticipant,
        categorize_calcium_score,
    )
    from cardiac_fat_fl.dashboard import create_dashboard_app, launch_dashboard
    from cardiac_fat_fl.fl_client_algo import CardiacFatClientAlgo
    from cardiac_fat_fl.fl_stats_executor import (
        CardiacFatStatsExecutor,
        aggregate_federated_statistics,
    )
    from cardiac_fat_fl.io_metatensor import (
        discover_participants,
        read_medical_image_as_metatensor,
        read_nrrd_as_metatensor,
        read_nrrd_ct,
        write_eat_pvat_seg_nrrd,
    )
    from cardiac_fat_fl.measurement_schema import (
        ADIPOSE_HU_MAX,
        ADIPOSE_HU_MIN,
        DEFAULT_PVAT_RADIAL_DISTANCE_MM,
        MEASUREMENT_CSV_COLUMNS,
        REPORTED_REGIONS,
        ReportedRegion,
        SegmentationCompartment,
        calculate_volume_cm3,
        validate_measurement_dataframe,
    )
    from cardiac_fat_fl.model import create_cardiac_fat_model, filter_weights_for_fedbn
    from cardiac_fat_fl.monai_deploy import (
        CardiacCTStudyLoaderOp,
        CardiacFatQuantApplication,
        CardiacFatQuantifierOp,
        DicomSegAndSrWriterOp,
        run_monai_deploy_pipeline,
    )
    from cardiac_fat_fl.pacs_daemon import (
        DaemonConfig,
        PACSWatchdogDaemon,
        find_pending_studies,
        is_study_settled,
        process_study,
    )
    from cardiac_fat_fl.segmentation_evaluator import (
        CohortBenchmarkReport,
        ParticipantBenchmarkResult,
        SegmentationBenchmarkEngine,
        compute_compartment_dice,
        compute_compartment_iou,
        compute_surface_distance_metrics,
        compute_volume_metrics,
    )
    from cardiac_fat_fl.transforms import ComputeEatPvatMasksd, QuantifyAdiposeMetricsd

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ADIPOSE_HU_MIN": ("cardiac_fat_fl.measurement_schema", "ADIPOSE_HU_MIN"),
    "ADIPOSE_HU_MAX": ("cardiac_fat_fl.measurement_schema", "ADIPOSE_HU_MAX"),
    "DEFAULT_PVAT_RADIAL_DISTANCE_MM": (
        "cardiac_fat_fl.measurement_schema",
        "DEFAULT_PVAT_RADIAL_DISTANCE_MM",
    ),
    "SegmentationCompartment": (
        "cardiac_fat_fl.measurement_schema",
        "SegmentationCompartment",
    ),
    "ReportedRegion": ("cardiac_fat_fl.measurement_schema", "ReportedRegion"),
    "REPORTED_REGIONS": ("cardiac_fat_fl.measurement_schema", "REPORTED_REGIONS"),
    "MEASUREMENT_CSV_COLUMNS": (
        "cardiac_fat_fl.measurement_schema",
        "MEASUREMENT_CSV_COLUMNS",
    ),
    "validate_measurement_dataframe": (
        "cardiac_fat_fl.measurement_schema",
        "validate_measurement_dataframe",
    ),
    "calculate_volume_cm3": ("cardiac_fat_fl.measurement_schema", "calculate_volume_cm3"),
    "discover_participants": ("cardiac_fat_fl.discovery", "discover_participants"),
    "read_medical_image_as_metatensor": (
        "cardiac_fat_fl.io_metatensor",
        "read_medical_image_as_metatensor",
    ),
    "read_nrrd_ct": ("cardiac_fat_fl.io_metatensor", "read_nrrd_ct"),
    "read_nrrd_as_metatensor": (
        "cardiac_fat_fl.io_metatensor",
        "read_nrrd_as_metatensor",
    ),
    "write_eat_pvat_seg_nrrd": (
        "cardiac_fat_fl.io_metatensor",
        "write_eat_pvat_seg_nrrd",
    ),
    "AnatomyMasks": ("cardiac_fat_fl.anatomy_segmenter", "AnatomyMasks"),
    "UnifiedAnatomySegmenter": (
        "cardiac_fat_fl.anatomy_segmenter",
        "UnifiedAnatomySegmenter",
    ),
    "ComputeEatPvatMasksd": ("cardiac_fat_fl.transforms", "ComputeEatPvatMasksd"),
    "QuantifyAdiposeMetricsd": ("cardiac_fat_fl.transforms", "QuantifyAdiposeMetricsd"),
    "create_cardiac_fat_model": ("cardiac_fat_fl.model", "create_cardiac_fat_model"),
    "filter_weights_for_fedbn": ("cardiac_fat_fl.model", "filter_weights_for_fedbn"),
    "CardiacFatClientAlgo": ("cardiac_fat_fl.fl_client_algo", "CardiacFatClientAlgo"),
    "CardiacFatStatsExecutor": (
        "cardiac_fat_fl.fl_stats_executor",
        "CardiacFatStatsExecutor",
    ),
    "aggregate_federated_statistics": (
        "cardiac_fat_fl.fl_stats_executor",
        "aggregate_federated_statistics",
    ),
    "CardiacCTStudyLoaderOp": ("cardiac_fat_fl.monai_deploy", "CardiacCTStudyLoaderOp"),
    "CardiacFatQuantifierOp": ("cardiac_fat_fl.monai_deploy", "CardiacFatQuantifierOp"),
    "DicomSegAndSrWriterOp": ("cardiac_fat_fl.monai_deploy", "DicomSegAndSrWriterOp"),
    "CardiacFatQuantApplication": (
        "cardiac_fat_fl.monai_deploy",
        "CardiacFatQuantApplication",
    ),
    "run_monai_deploy_pipeline": (
        "cardiac_fat_fl.monai_deploy",
        "run_monai_deploy_pipeline",
    ),
    "DaemonConfig": ("cardiac_fat_fl.pacs_daemon", "DaemonConfig"),
    "PACSWatchdogDaemon": ("cardiac_fat_fl.pacs_daemon", "PACSWatchdogDaemon"),
    "is_study_settled": ("cardiac_fat_fl.pacs_daemon", "is_study_settled"),
    "find_pending_studies": ("cardiac_fat_fl.pacs_daemon", "find_pending_studies"),
    "process_study": ("cardiac_fat_fl.pacs_daemon", "process_study"),
    "evaluate_clinical_risk": ("cardiac_fat_fl.clinical_report", "evaluate_clinical_risk"),
    "export_clinical_imaging_report": (
        "cardiac_fat_fl.clinical_report",
        "export_clinical_imaging_report",
    ),
    "generate_clinical_report_html": (
        "cardiac_fat_fl.clinical_report",
        "generate_clinical_report_html",
    ),
    "generate_clinical_report_pdf": (
        "cardiac_fat_fl.clinical_report",
        "generate_clinical_report_pdf",
    ),
    "generate_mpr_snapshot": ("cardiac_fat_fl.clinical_report", "generate_mpr_snapshot"),
    "encapsulate_pdf_to_dicom": (
        "cardiac_fat_fl.clinical_report",
        "encapsulate_pdf_to_dicom",
    ),
    "CohortDataset": ("cardiac_fat_fl.cohort_research", "CohortDataset"),
    "CohortParticipant": ("cardiac_fat_fl.cohort_research", "CohortParticipant"),
    "categorize_calcium_score": (
        "cardiac_fat_fl.cohort_research",
        "categorize_calcium_score",
    ),
    "SegmentationBenchmarkEngine": (
        "cardiac_fat_fl.segmentation_evaluator",
        "SegmentationBenchmarkEngine",
    ),
    "CohortBenchmarkReport": (
        "cardiac_fat_fl.segmentation_evaluator",
        "CohortBenchmarkReport",
    ),
    "ParticipantBenchmarkResult": (
        "cardiac_fat_fl.segmentation_evaluator",
        "ParticipantBenchmarkResult",
    ),
    "compute_compartment_dice": (
        "cardiac_fat_fl.segmentation_evaluator",
        "compute_compartment_dice",
    ),
    "compute_compartment_iou": (
        "cardiac_fat_fl.segmentation_evaluator",
        "compute_compartment_iou",
    ),
    "compute_volume_metrics": (
        "cardiac_fat_fl.segmentation_evaluator",
        "compute_volume_metrics",
    ),
    "compute_surface_distance_metrics": (
        "cardiac_fat_fl.segmentation_evaluator",
        "compute_surface_distance_metrics",
    ),
    "create_dashboard_app": ("cardiac_fat_fl.dashboard", "create_dashboard_app"),
    "launch_dashboard": ("cardiac_fat_fl.dashboard", "launch_dashboard"),
}

__all__ = [
    "ADIPOSE_HU_MIN",
    "ADIPOSE_HU_MAX",
    "DEFAULT_PVAT_RADIAL_DISTANCE_MM",
    "SegmentationCompartment",
    "ReportedRegion",
    "REPORTED_REGIONS",
    "MEASUREMENT_CSV_COLUMNS",
    "validate_measurement_dataframe",
    "calculate_volume_cm3",
    "discover_participants",
    "read_medical_image_as_metatensor",
    "read_nrrd_ct",
    "read_nrrd_as_metatensor",
    "write_eat_pvat_seg_nrrd",
    "AnatomyMasks",
    "UnifiedAnatomySegmenter",
    "ComputeEatPvatMasksd",
    "QuantifyAdiposeMetricsd",
    "create_cardiac_fat_model",
    "filter_weights_for_fedbn",
    "CardiacFatClientAlgo",
    "CardiacFatStatsExecutor",
    "aggregate_federated_statistics",
    "CardiacCTStudyLoaderOp",
    "CardiacFatQuantifierOp",
    "DicomSegAndSrWriterOp",
    "CardiacFatQuantApplication",
    "run_monai_deploy_pipeline",
    "DaemonConfig",
    "PACSWatchdogDaemon",
    "is_study_settled",
    "find_pending_studies",
    "process_study",
    "evaluate_clinical_risk",
    "export_clinical_imaging_report",
    "generate_clinical_report_html",
    "generate_clinical_report_pdf",
    "generate_mpr_snapshot",
    "encapsulate_pdf_to_dicom",
    "CohortDataset",
    "CohortParticipant",
    "categorize_calcium_score",
    "SegmentationBenchmarkEngine",
    "CohortBenchmarkReport",
    "ParticipantBenchmarkResult",
    "compute_compartment_dice",
    "compute_compartment_iou",
    "compute_volume_metrics",
    "compute_surface_distance_metrics",
    "create_dashboard_app",
    "launch_dashboard",
]


def __getattr__(name: str) -> Any:
    """Lazily import public symbols on first attribute access (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
