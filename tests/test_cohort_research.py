"""Unit and Integration Tests for Cohort Research Engine."""

from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import pytest

from cardiac_fat_fl.cohort_research import (
    CohortDataset,
    CohortParticipant,
    categorize_calcium_score,
)


def test_categorize_calcium_score():
    """Verify standard Agatston CAC risk tier cutoffs."""
    assert categorize_calcium_score(None) == "Unknown"
    assert categorize_calcium_score(float("nan")) == "Unknown"
    assert categorize_calcium_score(0.0) == "Zero (0)"
    assert categorize_calcium_score(-5.0) == "Zero (0)"
    assert categorize_calcium_score(5.4) == "Minimal (1-10)"
    assert categorize_calcium_score(10.0) == "Minimal (1-10)"
    assert categorize_calcium_score(45.0) == "Mild (11-100)"
    assert categorize_calcium_score(100.0) == "Mild (11-100)"
    assert categorize_calcium_score(250.0) == "Moderate (101-400)"
    assert categorize_calcium_score(400.0) == "Moderate (101-400)"
    assert categorize_calcium_score(400.1) == "Severe (>400)"
    assert categorize_calcium_score(1250.0) == "Severe (>400)"


def test_cohort_dataset_loading_example_cohort():
    """Verify loading from example_source_dir."""
    cohort_dir = Path("example_source_dir")
    if not cohort_dir.exists():
        pytest.skip("example_source_dir not found")

    cohort = CohortDataset(cohort_dir)
    assert cohort.total_count >= 3
    assert "0506" in cohort.participants
    assert "0909" in cohort.participants
    assert "0987" in cohort.participants

    p = cohort.participants["0506"]
    assert p.seg_path is not None
    assert p.seg_path.name == "eat_pvat.seg.nrrd"
    assert p.report_pdf is not None
    assert p.report_html is not None
    assert p.report_dcm is not None
    assert p.metrics.get("EAT Total Volume (cm3)", 0.0) > 0.0

    # Summary analytics
    summary = cohort.compute_summary_analytics()
    assert summary["total_count"] >= 3
    assert summary["segmented_count"] >= 3
    assert "eat_volume_stats" in summary
    assert summary["eat_volume_stats"]["count"] >= 3

    # Scatter data
    scatter = cohort.get_scatter_data()
    assert len(scatter) >= 3
    p0 = next(s for s in scatter if s["participant_id"] == "0506")
    assert p0["eat_volume_cm3"] > 0.0


def test_cohort_dataset_with_synthetic_calcium_scores(tmp_path: Path):
    """Verify linking with scores_gated.csv and calculating correlations."""
    # Build a synthetic cohort folder with 5 cases
    p_dir = tmp_path / "study_patients"
    p_dir.mkdir()

    scores_rows = []
    meas_rows = []
    for i in range(1, 6):
        pid = f"PAT_{i:03d}"
        case_dir = p_dir / pid
        case_dir.mkdir()

        # Dummy seg
        (case_dir / "eat_pvat.seg.nrrd").write_text("dummy")

        # Fake correlation: CAC = i * 100, EAT = 80 + i * 15
        cac = i * 100
        eat_vol = 80.0 + i * 15.0
        pvat_hu = -85.0 + i * 3.0  # -82, -79, -76, -73, -70

        scores_rows.append({"PatientId": pid, "TotalScore": cac})
        meas_rows.append({
            "participant_id": pid,
            "EAT Total Volume (cm3)": eat_vol,
            "PVAT Total Mean Attenuation (HU)": pvat_hu,
        })

    # Write scores_gated.csv
    scores_df = pd.DataFrame(scores_rows)
    scores_df.to_csv(tmp_path / "scores_gated.csv", index=False)

    # Write cardiac_fat_measurements.csv
    meas_df = pd.DataFrame(meas_rows)
    meas_df.to_csv(tmp_path / "cardiac_fat_measurements.csv", index=False)

    cohort = CohortDataset(tmp_path)
    assert cohort.total_count == 5
    assert cohort.cac_scored_count == 5
    assert cohort.segmented_count == 5

    p1 = cohort.participants["PAT_001"]
    assert p1.calcium_score == 100
    assert p1.calcium_risk_tier == "Mild (11-100)"

    summary = cohort.compute_summary_analytics()
    assert "correlations" in summary
    assert "eat_vs_cac" in summary["correlations"]
    assert pytest.approx(summary["correlations"]["eat_vs_cac"]["pearson_r"], 1e-2) == 1.0

def test_cohort_dataset_natural_sorting(tmp_path: Path):
    """Verify that participant IDs are sorted naturally (0, 1, 2, ..., 10, 100) not alphabetically."""
    for pid in ["100", "0", "10", "2", "1", "20"]:
        case_dir = tmp_path / pid
        case_dir.mkdir()
        (case_dir / "scan.nrrd").write_text("dummy")

    cohort = CohortDataset(tmp_path)
    assert list(cohort.participants.keys()) == ["0", "1", "2", "10", "20", "100"]

