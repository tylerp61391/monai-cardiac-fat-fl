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

    # Test Tier Group Comparisons and Bivariate Regression
    tier_stats = cohort.compute_tier_group_comparisons()
    assert "CAC_TIERS" in tier_stats
    assert "eat_volume" in tier_stats
    assert "pvat_attenuation" in tier_stats
    assert "tiers" in tier_stats["eat_volume"]
    assert "anova" in tier_stats["eat_volume"]
    assert "kruskal_wallis" in tier_stats["eat_volume"]

    bivariate = cohort.compute_bivariate_regression()
    assert bivariate["n_paired"] == 5
    assert "pearson_r" in bivariate
    assert "slope" in bivariate
    assert "equation" in bivariate
    assert bivariate["r_squared"] >= 0.0


def test_publication_figure_generation(tmp_path: Path):
    """Verify that generate_publication_figure produces a valid 300 DPI image."""
    from cardiac_fat_fl.publication_figure import generate_publication_figure
    from PIL import Image

    # Use synthetic cohort setup
    p_dir = tmp_path / "study_patients"
    p_dir.mkdir()
    scores_rows = []
    meas_rows = []
    for i in range(1, 11):
        pid = f"PAT_{i:03d}"
        case_dir = p_dir / pid
        case_dir.mkdir()
        (case_dir / "eat_pvat.seg.nrrd").write_text("dummy")

        scores_rows.append({"PatientId": pid, "TotalScore": i * 50})
        meas_rows.append({
            "participant_id": pid,
            "EAT Total Volume (cm3)": 50.0 + i * 5.0,
            "PVAT Total Mean Attenuation (HU)": -80.0 + i * 2.0,
            "EAT LA Volume (cm3)": 10.0,
            "EAT LV Volume (cm3)": 15.0,
            "EAT RA Volume (cm3)": 8.0,
            "EAT RV Volume (cm3)": 20.0,
            "PVAT LAD Volume (cm3)": 5.0,
            "PVAT LCX Volume (cm3)": 4.0,
            "PVAT RCA Volume (cm3)": 6.0,
        })

    pd.DataFrame(scores_rows).to_csv(tmp_path / "scores_gated.csv", index=False)
    pd.DataFrame(meas_rows).to_csv(tmp_path / "cardiac_fat_measurements.csv", index=False)

    cohort = CohortDataset(tmp_path)
    out_img = tmp_path / "test_fig_300dpi.png"
    png_bytes = generate_publication_figure(cohort, output_path=out_img, width=1200, height=900, dpi=300)

    assert len(png_bytes) > 10000
    assert out_img.is_file()

    im = Image.open(out_img)
    assert im.size == (1200, 900)
    assert im.mode == "RGB"

    # Constant chamber volumes must produce finite 0.0 skewness without SciPy warnings
    summary = cohort.compute_summary_analytics()
    assert summary["chamber_volume_summaries"]["EAT LA"]["skewness"] == 0.0

def test_cohort_dataset_natural_sorting(tmp_path: Path):
    """Verify that participant IDs are sorted naturally (0, 1, 2, ..., 10, 100) not alphabetically."""
    for pid in ["100", "0", "10", "2", "1", "20"]:
        case_dir = tmp_path / pid
        case_dir.mkdir()
        (case_dir / "scan.nrrd").write_text("dummy")

    cohort = CohortDataset(tmp_path)
    assert list(cohort.participants.keys()) == ["0", "1", "2", "10", "20", "100"]


def test_cohort_dataset_temp_cache_roundtrip_and_invalidation(tmp_path: Path):
    """Verify temp JSON index cache speeds reloads without touching cohort_dir and invalidates cleanly."""
    for i in range(1, 23):
        pid = f"PAT_{i:03d}"
        cdir = tmp_path / pid
        cdir.mkdir()
        (cdir / "scan.nrrd").write_text("dummy")
        (cdir / "eat_pvat.seg.nrrd").write_text("dummy")
        if i == 1:
            (cdir / "nv_reason_narrative.txt").write_text("Normal heart size.")

    pd.DataFrame(
        [{"PatientId": "PAT_001", "TotalScore": 150.0}]
    ).to_csv(tmp_path / "scores_gated.csv", index=False)
    pd.DataFrame(
        [
            {
                "participant_id": "PAT_001",
                "EAT Total Volume (cm3)": 92.5,
                "PVAT Total Mean Attenuation (HU)": -68.4,
            }
        ]
    ).to_csv(tmp_path / "cardiac_fat_measurements.csv", index=False)

    cohort1 = CohortDataset(tmp_path)
    cache_file = cohort1._cache_file_path()
    assert cache_file.is_file()
    assert tmp_path not in cache_file.parents
    assert cohort1.total_count == 22
    assert cohort1.participants["PAT_001"].calcium_score == 150.0
    assert cohort1.participants["PAT_001"].has_nv_reason is True

    # Warm reload from user-temp cache
    cohort2 = CohortDataset(tmp_path)
    assert cohort2.total_count == 22
    assert cohort2.participants["PAT_001"].calcium_score == 150.0
    assert cohort2.participants["PAT_001"].metrics["EAT Total Volume (cm3)"] == 92.5
    assert cohort2.participants["PAT_001"].has_nv_reason is True

    # Adding a new participant folder must invalidate the cache automatically
    new_dir = tmp_path / "PAT_023"
    new_dir.mkdir()
    (new_dir / "scan.nrrd").write_text("dummy")

    cohort3 = CohortDataset(tmp_path)
    assert cohort3.total_count == 23
    assert "PAT_023" in cohort3.participants


