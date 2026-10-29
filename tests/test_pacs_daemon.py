"""Unit and integration tests for Automated Clinical PACS Watchdog & Auto-Router Daemon."""

import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cardiac_fat_fl.cli import app
from cardiac_fat_fl.pacs_daemon import (
    DaemonConfig,
    PACSWatchdogDaemon,
    find_pending_studies,
    is_study_settled,
    process_study,
)

runner = CliRunner()


def test_is_study_settled(tmp_path: Path) -> None:
    """Test file and directory settling detection based on modification timestamps."""
    test_file = tmp_path / "study.nrrd"
    test_file.write_text("dummy scan data", encoding="utf-8")

    # Brand new file (< 0.1s old) should NOT be settled with settle_seconds=2.0
    assert not is_study_settled(test_file, settle_seconds=2.0)

    # Artificially age the timestamp to 10 seconds ago
    past_time = time.time() - 10.0
    os.utime(test_file, (past_time, past_time))
    assert is_study_settled(test_file, settle_seconds=2.0)

    # Test folder settling
    study_dir = tmp_path / "study_001"
    study_dir.mkdir()
    dcm1 = study_dir / "slice1.dcm"
    dcm1.write_text("dcm1", encoding="utf-8")

    # Newly written slice in folder is not settled
    assert not is_study_settled(study_dir, settle_seconds=2.0)

    # Age folder contents
    os.utime(dcm1, (past_time, past_time))
    assert is_study_settled(study_dir, settle_seconds=2.0)


def test_find_pending_studies(tmp_path: Path) -> None:
    """Test candidate discovery and proper filtering of ignored and transient paths."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    # Valid candidates
    valid_dir = inbox / "patient_0506"
    valid_dir.mkdir()
    (valid_dir / "ct.nrrd").write_text("data", encoding="utf-8")

    valid_file = inbox / "patient_0909.nrrd"
    valid_file.write_text("data", encoding="utf-8")

    # Ignored paths
    (inbox / ".git").mkdir()
    (inbox / "temp.lock").write_text("lock", encoding="utf-8")
    (inbox / "partial.tmp").write_text("tmp", encoding="utf-8")

    outbox = inbox / "outbox"
    outbox.mkdir()
    (outbox / "result.nrrd").write_text("data", encoding="utf-8")

    candidates = find_pending_studies(inbox, ignored_subdirs={"outbox"})
    candidate_names = [c.name for c in candidates]

    assert "patient_0506" in candidate_names
    assert "patient_0909.nrrd" in candidate_names
    assert "outbox" not in candidate_names
    assert ".git" not in candidate_names
    assert "temp.lock" not in candidate_names


def test_daemon_process_study_dry_run(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test study processing in non-destructive dry-run mode."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outbox = tmp_path / "outbox"
    archive = tmp_path / "archive"

    # Copy synthetic scan to inbox
    scan_inbox = inbox / synthetic_nrrd_scan.name
    shutil.copyfile(synthetic_nrrd_scan, scan_inbox)

    config = DaemonConfig(
        inbox_dir=inbox,
        outbox_dir=outbox,
        archive_dir=archive,
        settle_seconds=0.0,
        dry_run=True,
    )

    result = process_study(scan_inbox, config)
    assert result["status"] == "success"
    assert result["pacs_result"]["status"] == "dry_run_success"

    # Verify generated artifacts
    study_outbox = outbox / scan_inbox.stem
    assert (study_outbox / "eat_pvat.seg.nrrd").exists()
    assert (study_outbox / "eat_pvat_seg.dcm").exists()
    assert (study_outbox / "eat_pvat_sr.dcm").exists()
    assert (study_outbox / "cardiac_fat_measurements.csv").exists()

    # Verify scan moved to archive
    assert not scan_inbox.exists()
    assert (archive / scan_inbox.name).exists()


def test_daemon_fault_tolerance_failed_study(tmp_path: Path) -> None:
    """Test that corrupted scans are quarantined in failed_dir with error trace."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outbox = tmp_path / "outbox"
    failed = tmp_path / "failed"

    bad_scan = inbox / "corrupt_scan.nrrd"
    bad_scan.write_text("CORRUPT_NON_MEDICAL_DATA", encoding="utf-8")

    config = DaemonConfig(
        inbox_dir=inbox,
        outbox_dir=outbox,
        failed_dir=failed,
        settle_seconds=0.0,
        dry_run=True,
    )

    result = process_study(bad_scan, config)
    assert result["status"] == "failed"
    assert "error" in result

    # Corrupt scan moved out of inbox into failed_dir
    assert not bad_scan.exists()
    assert (failed / "corrupt_scan.nrrd").exists()
    assert (failed / "corrupt_scan.error.log").exists()


def test_pacs_watchdog_daemon_single_pass(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test end-to-end execution of PACSWatchdogDaemon service loop in single-pass mode."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outbox = tmp_path / "outbox"

    scan_inbox = inbox / synthetic_nrrd_scan.name
    shutil.copyfile(synthetic_nrrd_scan, scan_inbox)

    past = time.time() - 10.0
    os.utime(scan_inbox, (past, past))

    config = DaemonConfig(
        inbox_dir=inbox,
        outbox_dir=outbox,
        settle_seconds=1.0,
        dry_run=True,
        single_pass=True,
    )

    daemon = PACSWatchdogDaemon(config)
    completed_events = []

    summary = daemon.run(progress_callback=lambda res: completed_events.append(res))

    assert summary["processed_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["termination_reason"] == "single_pass finished"
    assert len(completed_events) == 1


def test_cli_daemon_dry_run(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test Typer CLI invocation of cardiac-fat-fl daemon."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outbox = tmp_path / "outbox"

    scan_inbox = inbox / synthetic_nrrd_scan.name
    shutil.copyfile(synthetic_nrrd_scan, scan_inbox)

    past = time.time() - 10.0
    os.utime(scan_inbox, (past, past))

    res = runner.invoke(
        app,
        [
            "daemon",
            str(inbox),
            "--outbox",
            str(outbox),
            "--dry-run",
            "--single-pass",
            "--settle-seconds",
            "0.5",
        ],
    )

    assert res.exit_code == 0, f"CLI exited with {res.exit_code}: {res.output}"
    assert "Automated Clinical PACS Watchdog & Auto-Router Daemon" in res.output
    assert "DRY-RUN" in res.output
    assert "Daemon finished cleanly" in res.output

