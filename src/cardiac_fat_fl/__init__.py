"""Modernized MONAI & NVIDIA FLARE Cardiac Fat Quantification Package."""

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

__version__ = "0.1.0"
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
