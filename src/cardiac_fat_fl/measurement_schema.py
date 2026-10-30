"""Measurement Schema and Clinical Definitions for Cardiac Fat Quantification.

This module defines the clinical attenuation thresholds, 8 segmentation compartments,
9 reported metric regions, and the exact 19-column CSV schema matching the MGH
CardiacFatQuant pipeline.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final, List, Tuple
import pandas as pd

# -----------------------------------------------------------------------------
# Clinical Attenuation Constants
# -----------------------------------------------------------------------------
# Adipose tissue attenuation window in Hounsfield Units (HU)
# Strictly -190.0 HU to -30.0 HU inclusive (-190 <= HU <= -30)
ADIPOSE_HU_MIN: Final[float] = -190.0
ADIPOSE_HU_MAX: Final[float] = -30.0

# Pericoronary radial dilation parameters (standard clinical default: 10.0 mm)
DEFAULT_PVAT_RADIAL_DISTANCE_MM: Final[float] = 10.0


class SegmentationCompartment(IntEnum):
    """The 8 target output channels written to `eat_pvat.seg.nrrd`.

    Channels 0 through 7 represent the distinct anatomical compartments:
    0: EAT_Total - all adipose tissue inside the pericardial/heart mask within [-190, -30] HU
    1: EAT_LA    - Left Atrium epicardial adipose tissue
    2: EAT_LV    - Left Ventricle epicardial adipose tissue
    3: EAT_RA    - Right Atrium epicardial adipose tissue
    4: EAT_RV    - Right Ventricle epicardial adipose tissue
    5: PVAT_LAD  - Left Anterior Descending pericoronary adipose tissue
    6: PVAT_LCX  - Left Circumflex pericoronary adipose tissue
    7: PVAT_RCA  - Right Coronary Artery pericoronary adipose tissue
    """
    EAT_TOTAL = 0
    EAT_LA = 1
    EAT_LV = 2
    EAT_RA = 3
    EAT_RV = 4
    PVAT_LAD = 5
    PVAT_LCX = 6
    PVAT_RCA = 7

    @property
    def label_name(self) -> str:
        return self.name

    @classmethod
    def get_names(cls) -> List[str]:
        return [c.name for c in cls]


SEGMENTATION_COMPARTMENTS: Final[List[SegmentationCompartment]] = list(SegmentationCompartment)


class ReportedRegion(StrEnum):
    """The 9 reported anatomical regions in `cardiac_fat_measurements.csv`.

    Note: PVAT Total is computed as the spatial union of PVAT_LAD | PVAT_LCX | PVAT_RCA.
    """
    EAT_TOTAL = "EAT Total"
    EAT_LA = "EAT LA"
    EAT_LV = "EAT LV"
    EAT_RA = "EAT RA"
    EAT_RV = "EAT RV"
    PVAT_TOTAL = "PVAT Total"
    PVAT_LAD = "PVAT LAD"
    PVAT_LCX = "PVAT LCX"
    PVAT_RCA = "PVAT RCA"


# List of reported regions in required canonical order
REPORTED_REGIONS: Final[Tuple[ReportedRegion, ...]] = (
    ReportedRegion.EAT_TOTAL,
    ReportedRegion.EAT_LA,
    ReportedRegion.EAT_LV,
    ReportedRegion.EAT_RA,
    ReportedRegion.EAT_RV,
    ReportedRegion.PVAT_TOTAL,
    ReportedRegion.PVAT_LAD,
    ReportedRegion.PVAT_LCX,
    ReportedRegion.PVAT_RCA,
)


def generate_measurement_columns() -> List[str]:
    """Generate the exact 19-column list for `cardiac_fat_measurements.csv`.

    Schema:
      participant_id,
      followed by exact pairs of `<Region> Volume (cm3)` and `<Region> Mean Attenuation (HU)`
      for all 9 reported regions.
    """
    cols: List[str] = ["participant_id"]
    for region in REPORTED_REGIONS:
        cols.append(f"{region.value} Volume (cm3)")
        cols.append(f"{region.value} Mean Attenuation (HU)")
    return cols


# Constant canonical list of 19 columns
MEASUREMENT_CSV_COLUMNS: Final[List[str]] = generate_measurement_columns()


def validate_measurement_dataframe(df: pd.DataFrame) -> bool:
    """Validate that a DataFrame strictly matches the 19-column CardiacFatQuant schema.

    Args:
        df: DataFrame to validate.

    Returns:
        True if the DataFrame columns exactly match MEASUREMENT_CSV_COLUMNS in order.

    Raises:
        ValueError: If columns or ordering do not match.
    """
    if list(df.columns) != MEASUREMENT_CSV_COLUMNS:
        missing = [c for c in MEASUREMENT_CSV_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in MEASUREMENT_CSV_COLUMNS]
        raise ValueError(
            f"Measurement DataFrame schema mismatch.\n"
            f"Expected {len(MEASUREMENT_CSV_COLUMNS)} columns: {MEASUREMENT_CSV_COLUMNS}\n"
            f"Found {len(df.columns)} columns: {list(df.columns)}\n"
            f"Missing columns: {missing}\n"
            f"Unexpected columns: {extra}"
        )
    return True


def calculate_volume_cm3(voxel_count: int, voxel_spacing_mm: Tuple[float, float, float]) -> float:
    """Calculate volume in cubic centimeters (cm3) from voxel count and spacing in mm.

    Formula:
        Volume (cm3) = voxel_count * (sx * sy * sz) / 1000.0

    Args:
        voxel_count: Number of voxels in the region.
        voxel_spacing_mm: 3-tuple (sx, sy, sz) representing voxel dimensions in millimeters.

    Returns:
        Volume in cm3 as a float.
    """
    sx, sy, sz = voxel_spacing_mm
    voxel_volume_mm3 = float(sx * sy * sz)
    volume_cm3 = (float(voxel_count) * voxel_volume_mm3) / 1000.0
    return volume_cm3

