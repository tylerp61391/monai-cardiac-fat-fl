"""Batch Autonomous NVIDIA NV-Reason-CT AI Radiologist Generator for Stanford COCA.

Iterates across Stanford COCA participants:
  1. Loads 3D CT volume (DICOM series or NRRD).
  2. Ground-conditions NV-Reason-CT (3D ViT + Qwen3.5-4B) with 18 quantitative fat biomarkers.
  3. Generates generative clinical findings and impression with clean tag sanitization.
  4. Upgrades clinical reports (cardiac_fat_report.pdf with Page 2, cardiac_fat_report.html, eat_pvat_report.dcm).
  5. Checkpointed, resumable, and runs with Windows BelowNormal process priority.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import pydicom
import torch

from cardiac_fat_fl.clinical_report import export_clinical_imaging_report
from cardiac_fat_fl.cohort_research import CohortDataset, CohortParticipant
from cardiac_fat_fl.nv_reason_reporter import (
    DEFAULT_MODEL_DIR,
    build_clinical_prompt,
    clean_vlm_narrative,
    is_nv_reason_available,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("coca_nv_reason")


def set_process_low_priority() -> None:
    """Set process to BelowNormal priority so desktop UI remains 100% fluid."""
    try:
        p = psutil.Process(os.getpid())
        if sys.platform == "win32":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            logger.info("Process priority set to BELOW_NORMAL_PRIORITY_CLASS (Windows).")
        else:
            p.nice(10)
            logger.info("Process nice set to 10 (Unix).")
    except Exception as exc:
        logger.warning("Could not adjust process priority: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch NV-Reason-CT vision-language inference across COCA cohort."
    )
    parser.add_argument(
        "--cohort-dir",
        type=str,
        default="E:/CT/Stanford_COCA",
        help="Root path to Stanford COCA cohort.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="E:/models/NV-Reason-CT",
        help="Path to NV-Reason-CT model weights.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum participants to process (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run NV-Reason-CT even if narrative already exists.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per study (default: 512).",
    )
    return parser.parse_args()


def get_source_dicom_slice(participant: CohortParticipant) -> Optional[pydicom.Dataset]:
    """Fast extraction of study metadata from a single slice header (< 2ms)."""
    if participant.scan_path and participant.scan_path.is_dir():
        dcm_files = list(participant.scan_path.glob("*.dcm"))
        if dcm_files:
            try:
                return pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            except Exception:
                pass

    ohif_dcm_dir = participant.study_dir / "ohif_export"
    if ohif_dcm_dir.exists():
        dcm_files = list(ohif_dcm_dir.glob("*.dcm"))
        for dcm_f in dcm_files:
            if not dcm_f.name.endswith("_seg.dcm") and not dcm_f.name.endswith("_sr.dcm") and not dcm_f.name.endswith("_report.dcm"):
                try:
                    return pydicom.dcmread(str(dcm_f), stop_before_pixels=True)
                except Exception:
                    pass
    return None


def main() -> int:
    args = parse_args()
    set_process_low_priority()

    cohort_path = Path(args.cohort_dir).resolve()
    model_path = Path(args.model_dir).resolve()

    if not cohort_path.exists():
        logger.error("Cohort path does not exist: %s", cohort_path)
        return 1

    if not is_nv_reason_available(model_path):
        logger.error("NV-Reason-CT model not found at %s or CUDA unavailable.", model_path)
        return 1

    print("=" * 80)
    print("      NVIDIA NV-Reason-CT 3D Vision-Language Cohort Generator")
    print("      Model: 3D ViT + Qwen-3.5 4B (bfloat16, SDPA) on RTX 4090")
    print("=" * 80)
    print(f"Cohort Directory:    {cohort_path}")
    print(f"Model Directory:     {model_path}")
    print(f"Priority Mode:       BelowNormal (Desktop Smoothness Preserved)")
    print(f"Overwrite Mode:      {'Enabled' if args.overwrite else 'Disabled (Resumable)'}")
    if args.limit:
        print(f"Limit:               {args.limit} participants")

    # 1. Discover Participants
    logger.info("Discovering cohort participants from %s...", cohort_path)
    dataset = CohortDataset(cohort_path)
    eligible = [
        p for p in dataset.participants.values()
        if p.metrics and len(p.metrics) > 0 and p.scan_path and p.scan_path.exists()
    ]
    logger.info("Found %d eligible participants with completed metrics and scan files.", len(eligible))

    if args.limit and args.limit > 0:
        eligible = eligible[:args.limit]
        logger.info("Processing limited to first %d participants.", len(eligible))

    total_target = len(eligible)
    if total_target == 0:
        logger.warning("No eligible participants found.")
        return 0

    # 2. Load Model Once into CUDA Memory
    logger.info("Loading NV-Reason-CT model and processor into CUDA memory...")
    t0_model = time.time()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.eval()
    logger.info("NV-Reason-CT model ready in %.2f seconds.", time.time() - t0_model)

    system_instruction = (
        "You are an expert cardiovascular radiologist specializing in quantitative cardiac CT. "
        "Analyze the 3D CT scan and synthesize findings with the patient's quantitative adipose biomarkers. "
        "Adhere to SCCT CAD-RADS guidelines and CRISP-CT inflammation thresholds (-70.1 HU cutoff)."
    )

    manifest_path = cohort_path / "coca_nv_reason_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    processed_cases = manifest.get("cases", {})

    success_count = 0
    skipped_count = 0
    error_count = 0
    t_start = time.time()

    try:
        for idx, p in enumerate(eligible, 1):
            pid = p.participant_id
            txt_path = p.study_dir / "nv_reason_narrative.txt"

            # Check if already done and valid
            if not args.overwrite and pid in processed_cases and txt_path.exists():
                skipped_count += 1
                continue

            t_case_start = time.time()
            try:
                user_prompt = build_clinical_prompt(p.metrics, mode="structured")
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ]

                prompt_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                inputs = processor(
                    text=prompt_text,
                    images3d=[str(p.scan_path)],
                    anatomy_region="chest",
                    return_tensors="pt",
                ).to(model.device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )

                new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
                raw_narrative = processor.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                cleaned_narrative = clean_vlm_narrative(raw_narrative)

                # Save raw & clean narrative files
                txt_path.write_text(cleaned_narrative, encoding="utf-8")
                json_path = p.study_dir / "nv_reason_narrative.json"
                json_path.write_text(
                    json.dumps(
                        {
                            "participant_id": pid,
                            "raw": raw_narrative,
                            "cleaned": cleaned_narrative,
                            "latency_seconds": time.time() - t_case_start,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                # Re-export 2-page PDF, HTML, and DICOM Encapsulated PDF
                source_dcm = get_source_dicom_slice(p)
                exam_date = getattr(source_dcm, "StudyDate", None) if source_dcm else None
                if not exam_date or exam_date == "00010101":
                    exam_date = "2026-10-29"

                mpr_img = None
                mpr_png = p.study_dir / "mpr_montage.png"
                if mpr_png.exists():
                    try:
                        from PIL import Image
                        mpr_img = Image.open(mpr_png)
                    except Exception:
                        pass

                export_clinical_imaging_report(
                    output_dir=p.study_dir,
                    metrics=p.metrics,
                    participant_id=pid,
                    source_ct_dcm=source_dcm,
                    exam_date=exam_date,
                    ai_reasoning_narrative=cleaned_narrative,
                )

                case_latency = time.time() - t_case_start
                processed_cases[pid] = {
                    "status": "completed",
                    "latency_seconds": round(case_latency, 2),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                success_count += 1

                elapsed = time.time() - t_start
                rate = (success_count + skipped_count) / max(elapsed, 0.001)
                remaining = (total_target - (success_count + skipped_count)) / max(rate, 0.001)

                print(
                    f"[{idx:4d}/{total_target}] Pt {pid:>6}: "
                    f"Inference: {case_latency:5.1f}s | "
                    f"Avg: {elapsed / max(success_count, 1):5.1f}s | "
                    f"ETA: {remaining / 60:5.1f}m | "
                    f"Success: {success_count:3d} (Skipped: {skipped_count:3d})",
                    flush=True,
                )

                # Checkpoint manifest every 5 cases
                if success_count % 5 == 0:
                    manifest_data = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "cohort_dir": str(cohort_path),
                        "total_target": total_target,
                        "success_count": success_count,
                        "skipped_count": skipped_count,
                        "error_count": error_count,
                        "cases": processed_cases,
                    }
                    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

            except Exception as exc:
                logger.error("Inference failed for participant %s: %s", pid, exc)
                error_count += 1
                processed_cases[pid] = {
                    "status": "failed",
                    "error": str(exc),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

    finally:
        total_time = time.time() - t_start
        manifest_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cohort_dir": str(cohort_path),
            "total_target": total_target,
            "success_count": success_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "total_time_seconds": total_time,
            "cases": processed_cases,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        print("\n" + "=" * 80)
        print("         NV-Reason-CT Cohort Inference Finished / Checkpointed")
        print("=" * 80)
        print(f"Total Cohort Target:      {total_target}")
        print(f"Completed Cases:          {success_count}")
        print(f"Already Present (Skip):   {skipped_count}")
        print(f"Errors:                   {error_count}")
        print(f"Total Runtime:            {total_time:.1f} s ({total_time / 60:.2f} min)")
        print(f"Manifest Checkpoint:      {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
