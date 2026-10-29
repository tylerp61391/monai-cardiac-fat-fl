"""Automated Clinical PACS Watchdog & Auto-Router Daemon.

Monitors an incoming study inbox directory or local DICOM staging area for new cardiac CT
studies, executes the 3-operator MONAI Deploy Application Package (MAP) DAG, computes
volumetric/attenuation metrics, and auto-pushes multi-frame DICOM-SEG and DICOM-SR TID 1500
reports to institutional PACS archives via DICOMweb (STOW-RS) or DIMSE (C-STORE).

Safety & Clinical Invariants:
  - Strictly opt-in / dormant by default: runs only when explicitly invoked.
  - Safe settling / debounce: checks file modification times to ensure DICOM transfers are complete.
  - Non-destructive --dry-run mode: computes segmentation & metrics without pushing to PACS.
  - Fault-tolerant isolation: moves corrupted or non-conforming studies to a failed directory with
    diagnostic error traces so the daemon never enters infinite retry loops.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from cardiac_fat_fl.monai_deploy import run_monai_deploy_pipeline
from cardiac_fat_fl.pacs_client import (
    find_dicom_instances_to_push,
    push_to_dicomweb_stow,
    push_to_dimse_cstore,
)

logger = logging.getLogger("cardiac_fat_fl.pacs_daemon")


@dataclass
class DaemonConfig:
    """Configuration contract for the PACS Watchdog & Auto-Router Daemon."""

    inbox_dir: Path
    outbox_dir: Optional[Path] = None
    archive_dir: Optional[Path] = None
    failed_dir: Optional[Path] = None
    settle_seconds: float = 5.0
    poll_interval: float = 5.0
    models_dir: Optional[Path] = None
    pvat_radius_mm: float = 10.0
    pacs_url: Optional[str] = None
    pacs_host: Optional[str] = None
    pacs_port: Optional[int] = None
    called_aet: str = "ANY-SCP"
    calling_aet: str = "CARDIAC_FAT_FL"
    include_ct: bool = False
    dry_run: bool = False
    ca_cert: Optional[Union[str, Path, bool]] = None
    auth: Optional[Tuple[str, str]] = None
    token: Optional[str] = None
    single_pass: bool = False
    max_studies: Optional[int] = None

    def __post_init__(self) -> None:
        self.inbox_dir = Path(self.inbox_dir).resolve()
        if self.outbox_dir is None:
            self.outbox_dir = self.inbox_dir / "outbox"
        else:
            self.outbox_dir = Path(self.outbox_dir).resolve()

        if self.archive_dir is not None:
            self.archive_dir = Path(self.archive_dir).resolve()

        if self.failed_dir is not None:
            self.failed_dir = Path(self.failed_dir).resolve()

        if self.models_dir is not None:
            self.models_dir = Path(self.models_dir).resolve()


def is_study_settled(study_path: Path, settle_seconds: float = 5.0) -> bool:
    """Verify that file writes or network transfers to the study have ceased.

    A study is deemed settled if all files within it have modification timestamps
    older than `settle_seconds` relative to the current wall-clock time.

    Args:
        study_path: Path to the study directory or standalone volume file.
        settle_seconds: Minimum quiet duration in seconds with no timestamp change.

    Returns:
        True if all files have settled; False if writes are still active or pending.
    """
    now = time.time()
    if study_path.is_file():
        try:
            mtime = study_path.stat().st_mtime
            return (now - mtime) >= settle_seconds
        except OSError:
            return False

    # Check directory recursively
    latest_mtime = 0.0
    file_count = 0
    try:
        for entry in study_path.rglob("*"):
            if entry.is_file():
                file_count += 1
                mtime = entry.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
    except OSError:
        return False

    if file_count == 0:
        return False

    return (now - latest_mtime) >= settle_seconds


def find_pending_studies(
    inbox_dir: Path,
    ignored_subdirs: Optional[Set[str]] = None,
) -> List[Path]:
    """Discover candidate CT studies in the inbox folder waiting to be processed.

    Recognizes:
      1. Directories containing DICOM series (`.dcm`), NRRD (`.nrrd`), or NIfTI (`.nii*`).
      2. Standalone image volume files (`.nrrd`, `.nii`, `.nii.gz`).

    Args:
        inbox_dir: Inbox root directory.
        ignored_subdirs: Set of directory names to bypass (e.g., outbox, archive, failed).

    Returns:
        List of candidate study Paths in deterministic alphabetical order.
    """
    ignored = ignored_subdirs or set()
    if not inbox_dir.exists() or not inbox_dir.is_dir():
        return []

    candidates: List[Path] = []
    for item in sorted(inbox_dir.iterdir(), key=lambda p: p.name.lower()):
        # Skip hidden files or transient lock files
        name = item.name
        if name.startswith(".") or name.endswith((".tmp", ".lock", ".partial")):
            continue

        if item.is_dir():
            if name.lower() in ignored or item.resolve() in [Path(p).resolve() for p in ignored if Path(p).is_absolute()]:
                continue
            # Check if directory contains recognized medical formats
            has_images = any(
                item.glob("*.dcm")
            ) or any(
                item.glob("*.nrrd")
            ) or any(
                item.glob("*.nii*")
            ) or any(
                item.glob("*/*.dcm")
            )
            if has_images:
                candidates.append(item)
        elif item.is_file():
            suffix = item.suffix.lower()
            if suffix in [".nrrd", ".nii", ".gz"] and not name.startswith("eat_pvat"):
                candidates.append(item)

    return candidates


def process_study(
    study_path: Path,
    config: DaemonConfig,
) -> Dict[str, Any]:
    """Execute the MAP pipeline on a single CT study and route outputs to PACS.

    Args:
        study_path: Candidate study directory or volume file.
        config: Daemon configuration containing routing and directory settings.

    Returns:
        Summary dict containing execution status, generated artifact paths, and PACS push status.
    """
    study_id = study_path.stem if study_path.is_file() else study_path.name
    target_outbox = config.outbox_dir / study_id
    target_outbox.mkdir(parents=True, exist_ok=True)

    logger.info("Processing incoming study [%s] -> %s", study_id, target_outbox)

    try:
        # 1. Execute MONAI Deploy 3-operator DAG (Loader -> Quantifier -> Writer)
        artifacts = run_monai_deploy_pipeline(
            input_path=study_path,
            output_path=target_outbox,
            models_dir=config.models_dir,
            pvat_radius_mm=config.pvat_radius_mm,
        )

        pacs_result: Dict[str, Any] = {}

        # 2. PACS Transmission Routing
        if config.dry_run:
            logger.info("[DRY-RUN] Processed study [%s]. Skipping PACS transmission.", study_id)
            pacs_result = {
                "status": "dry_run_success",
                "details": f"Dry-run mode: outputs saved to {target_outbox}; no network push executed.",
            }
        else:
            files_to_push: List[Path] = []
            if artifacts.get("dicom_seg") and artifacts["dicom_seg"].exists():
                files_to_push.append(artifacts["dicom_seg"])
            if artifacts.get("dicom_sr") and artifacts["dicom_sr"].exists():
                files_to_push.append(artifacts["dicom_sr"])

            # If user wants CT included and original study is DICOM
            if config.include_ct:
                try:
                    all_instances = find_dicom_instances_to_push(study_path, include_ct=True)
                    for inst in all_instances:
                        if inst not in files_to_push:
                            files_to_push.append(inst)
                except Exception as exc:
                    logger.warning("Could not gather CT slices for [%s]: %s", study_id, exc)

            if config.pacs_url:
                logger.info("Auto-routing %d instances to STOW-RS: %s", len(files_to_push), config.pacs_url)
                pacs_result = push_to_dicomweb_stow(
                    files=files_to_push,
                    stow_url=config.pacs_url,
                    auth=config.auth,
                    token=config.token,
                    ca_cert=config.ca_cert,
                )
            elif config.pacs_host and config.pacs_port:
                logger.info(
                    "Auto-routing %d instances to DIMSE C-STORE: %s@%s:%d",
                    len(files_to_push),
                    config.called_aet,
                    config.pacs_host,
                    config.pacs_port,
                )
                pacs_result = push_to_dimse_cstore(
                    files=files_to_push,
                    host=config.pacs_host,
                    port=config.pacs_port,
                    called_aet=config.called_aet,
                    calling_aet=config.calling_aet,
                )
            else:
                pacs_result = {
                    "status": "local_only",
                    "details": "No PACS destination configured; saved locally in outbox.",
                }

        # 3. Archive incoming scan if requested
        if config.archive_dir is not None:
            config.archive_dir.mkdir(parents=True, exist_ok=True)
            dest = config.archive_dir / study_path.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(study_path), str(dest))
            logger.info("Archived input study to %s", dest)

        return {
            "study_id": study_id,
            "status": "success",
            "artifacts": {k: str(v) for k, v in artifacts.items()},
            "pacs_result": pacs_result,
        }

    except Exception as exc:
        err_msg = traceback.format_exc()
        logger.error("Failed processing study [%s]:\n%s", study_id, err_msg)

        # Move to failed directory with diagnostic trace if configured
        if config.failed_dir is not None:
            try:
                config.failed_dir.mkdir(parents=True, exist_ok=True)
                err_log_path = config.failed_dir / f"{study_id}.error.log"
                err_log_path.write_text(err_msg, encoding="utf-8")

                dest = config.failed_dir / study_path.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(study_path), str(dest))
                logger.info("Moved failed study [%s] to %s with diagnostic trace", study_id, dest)
            except Exception as move_err:
                logger.error("Could not move failed study [%s]: %s", study_id, move_err)

        return {
            "study_id": study_id,
            "status": "failed",
            "error": str(exc),
            "traceback": err_msg,
        }


class PACSWatchdogDaemon:
    """Autonomous PACS Watchdog & Auto-Router Service."""

    def __init__(self, config: DaemonConfig) -> None:
        self.config = config
        self._running = False
        self.processed_count = 0
        self.failed_count = 0

    def stop(self) -> None:
        """Signal the daemon loop to terminate cleanly."""
        self._running = False

    def run(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the watchdog service loop.

        Args:
            progress_callback: Optional callback invoked after each study finishes.

        Returns:
            Dictionary summarizing total processed, failed, and termination reason.
        """
        self._running = True
        cfg = self.config

        # Ensure directory structures
        cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
        cfg.outbox_dir.mkdir(parents=True, exist_ok=True)
        if cfg.archive_dir:
            cfg.archive_dir.mkdir(parents=True, exist_ok=True)
        if cfg.failed_dir:
            cfg.failed_dir.mkdir(parents=True, exist_ok=True)

        ignored_dirs = {
            cfg.outbox_dir.name.lower(),
        }
        if cfg.archive_dir:
            ignored_dirs.add(cfg.archive_dir.name.lower())
        if cfg.failed_dir:
            ignored_dirs.add(cfg.failed_dir.name.lower())

        logger.info(
            "Starting PACS Watchdog Daemon on [%s] (dry_run=%s, single_pass=%s)",
            cfg.inbox_dir,
            cfg.dry_run,
            cfg.single_pass,
        )

        termination_reason = "completed"

        while self._running:
            candidates = find_pending_studies(cfg.inbox_dir, ignored_subdirs=ignored_dirs)

            for study_path in candidates:
                if not self._running:
                    break

                # Settle check: ensure file writing has stopped
                if not is_study_settled(study_path, settle_seconds=cfg.settle_seconds):
                    logger.debug("Study [%s] is still settling; skipping for next cycle.", study_path.name)
                    continue

                # Process study
                res = process_study(study_path, cfg)
                if res["status"] == "success":
                    self.processed_count += 1
                else:
                    self.failed_count += 1

                if progress_callback:
                    progress_callback(res)

                if cfg.max_studies and (self.processed_count + self.failed_count) >= cfg.max_studies:
                    termination_reason = f"max_studies ({cfg.max_studies}) reached"
                    self._running = False
                    break

            if cfg.single_pass:
                termination_reason = "single_pass finished"
                break

            if self._running:
                time.sleep(cfg.poll_interval)

        logger.info(
            "PACS Watchdog Daemon stopped (%s). Total processed: %d, Failed: %d",
            termination_reason,
            self.processed_count,
            self.failed_count,
        )

        return {
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "termination_reason": termination_reason,
        }
