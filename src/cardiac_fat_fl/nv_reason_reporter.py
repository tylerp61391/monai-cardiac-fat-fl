"""NVIDIA NV-Reason-CT 3D Vision-Language Integration Module.

Provides automated generative radiology report impressions and clinical reasoning
from 3D CT volumes and quantitative adipose tissue metrics using NVIDIA's
NV-Reason-CT model (3D ViT + Qwen3.5-4B).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch

DEFAULT_MODEL_DIR = Path(os.environ.get("NV_REASON_MODEL_DIR", r"E:\models\NV-Reason-CT"))


def is_nv_reason_available(model_dir: Optional[Path] = None) -> bool:
    """Check if NV-Reason-CT model weights and CUDA dependencies are present."""
    path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not path.exists():
        return False
    weights_file = path / "model.safetensors"
    if not weights_file.exists():
        return False
    return torch.cuda.is_available()


def build_clinical_prompt(metrics: Dict[str, Any], mode: str = "structured") -> str:
    """Format structured clinical instruction and numeric fat metrics for the VLM."""
    eat_vol = metrics.get("EAT Total Volume (cm3)", 0.0)
    eat_hu = metrics.get("EAT Total Mean Attenuation (HU)", -71.3)
    pvat_vol = metrics.get("PVAT Total Volume (cm3)", 0.0)
    pvat_hu = metrics.get("PVAT Total Mean Attenuation (HU)", -72.6)

    lad_hu = metrics.get("PVAT LAD Mean Attenuation (HU)", float("nan"))
    lcx_hu = metrics.get("PVAT LCX Mean Attenuation (HU)", float("nan"))
    rca_hu = metrics.get("PVAT RCA Mean Attenuation (HU)", float("nan"))

    lad_str = f"{lad_hu:.1f} HU" if not (torch.isnan(torch.tensor(lad_hu)) if isinstance(lad_hu, float) else False) else "N/A"
    lcx_str = f"{lcx_hu:.1f} HU" if not (torch.isnan(torch.tensor(lcx_hu)) if isinstance(lcx_hu, float) else False) else "N/A"
    rca_str = f"{rca_hu:.1f} HU" if not (torch.isnan(torch.tensor(rca_hu)) if isinstance(rca_hu, float) else False) else "N/A"

    eat_burden = "Severely Elevated" if eat_vol > 125 else ("Moderate" if eat_vol > 100 else "Normal")

    prompt = (
        f"Analyze this non-contrast cardiac CT scan.\n\n"
        f"EXTRACTED 3D QUANTITATIVE FAT BIOMARKERS:\n"
        f"- Total Epicardial Adipose Tissue (EAT) Volume: {eat_vol:.1f} cm3 ({eat_burden} burden)\n"
        f"- Total EAT Mean Attenuation: {eat_hu:.1f} HU\n"
        f"- Pericoronary Fat (PVAT) Volume: {pvat_vol:.1f} cm3, Mean Attenuation: {pvat_hu:.1f} HU\n"
        f"  * LAD PVAT: {lad_str}\n"
        f"  * LCX PVAT: {lcx_str}\n"
        f"  * RCA PVAT: {rca_str}\n\n"
        f"CRISP-CT Criterion: PVAT attenuation > -70.1 HU indicates active coronary microvascular inflammation.\n"
    )

    if mode == "structured":
        prompt += (
            "Provide a structured findings section and concise clinical impression interpreting "
            "coronary calcification, epicardial adiposity, and vessel-specific inflammation."
        )
    else:
        prompt += (
            "Provide a full reasoning analysis of this cardiac CT, evaluating coronary atherosclerotic "
            "calcification, epicardial adiposity distribution, and clinical risk recommendations."
        )

    return prompt


def generate_nv_reason_narrative(
    ct_path: Path,
    metrics: Dict[str, Any],
    model_dir: Optional[Path] = None,
    mode: str = "structured",
    max_new_tokens: int = 512,
) -> str:
    """Run NV-Reason-CT inference to generate clinical findings and impression.

    Args:
        ct_path: Path to input 3D CT volume (.nrrd, .nii, or .nii.gz).
        metrics: 19-column clinical metrics dictionary.
        model_dir: Directory containing model weights (defaults to E:\\models\\NV-Reason-CT).
        mode: "structured" (fast ~13s report) or "reasoning" (deep dictation).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text string containing clinical findings and impression.
    """
    model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not is_nv_reason_available(model_path):
        raise RuntimeError(
            f"NV-Reason-CT model not found at {model_path} or CUDA GPU unavailable."
        )

    from transformers import AutoModelForImageTextToText, AutoProcessor  # pyright: ignore[reportMissingImports]

    # Pre-clean CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model = None
    processor = None
    inputs = None

    try:
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        model.eval()

        system_instruction = (
            "You are an expert cardiovascular radiologist specializing in quantitative cardiac CT. "
            "Analyze the 3D CT scan and synthesize findings with the patient's quantitative adipose biomarkers. "
            "Adhere to SCCT CAD-RADS guidelines and CRISP-CT inflammation thresholds (-70.1 HU cutoff)."
        )
        user_prompt = build_clinical_prompt(metrics, mode=mode)

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

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(mode != "structured"),
        )

        inputs = processor(
            text=prompt,
            images3d=[str(ct_path)],
            anatomy_region="chest",
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
        narrative = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return narrative.strip()

    finally:
        # Guarantee CUDA cache flush so other tasks have full VRAM headroom
        if "model" in locals():
            del model
        if "processor" in locals():
            del processor
        if "inputs" in locals():
            del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
