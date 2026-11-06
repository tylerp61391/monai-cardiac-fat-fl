"""Phase 3: Cardiovascular Risk & Biomarker Grounded Prompt for NV-Reason-CT."""

import time
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = Path(r"E:\models\NV-Reason-CT")
ct_path = Path(r"example_source_dir\0506\img.nrrd")

print("=" * 80)
print("PHASE 3: CARDIOVASCULAR RISK & ADIPOSITY PROMPT EVALUATION")
print("Target GPU: NVIDIA GeForce RTX 4090 (24 GB VRAM)")
print("=" * 80)

t0 = time.time()
print(f"Loading processor and model from: {model_path} ...")
processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda",
)
model.eval()

t_load = time.time() - t0
print(f"Model ready in {t_load:.2f} s")

# System prompt establishing board-certified cardiovascular radiologist persona
system_instruction = (
    "You are an expert cardiovascular radiologist specializing in quantitative cardiac CT, "
    "epicardial adiposity, and pericoronary fat attenuation analysis. "
    "Analyze the provided non-contrast cardiac CT in conjunction with the patient's quantitative 3D adipose biomarkers. "
    "Adhere strictly to Society of Cardiovascular Computed Tomography (SCCT) guidelines and the CRISP-CT clinical trial "
    "thresholds: pericoronary adipose tissue (PVAT) attenuation > -70.1 HU indicates active coronary vascular inflammation "
    "and high residual inflammatory risk. Total Epicardial Adipose Tissue (EAT) volume > 100 cm3 indicates pathological "
    "visceral adiposity burden. Structure your interpretation with:\n"
    "1. QUANTITATIVE ADIPOSE & PLAQUE FINDINGS\n"
    "2. THORACIC & CARDIAC MORPHOLOGY\n"
    "3. IMPRESSION & CARDIOVASCULAR RISK STRATIFICATION\n"
    "4. CLINICAL MANAGEMENT RECOMMENDATIONS\n"
    "Be precise, factual, objective, and clinically actionable."
)

user_content = (
    "Provide a full reasoning analysis of this cardiac chest CT for Patient 0506.\n\n"
    "EXTRACTED 3D QUANTITATIVE FAT BIOMARKERS:\n"
    "- Total EAT Volume: 140.73 cm3 (Severely elevated visceral adiposity, >90th sex-matched percentile)\n"
    "- Total EAT Mean Attenuation: -77.02 HU\n"
    "- Pericoronary Fat Attenuation by Vessel:\n"
    "  * LAD PVAT: 13.12 cm3, Mean Attenuation: -78.95 HU (Reference < -70.1 HU)\n"
    "  * LCX PVAT: 3.14 cm3, Mean Attenuation: -62.02 HU (Acutely elevated inflammation > -70.1 HU, delta = +8.08 HU)\n"
    "  * RCA PVAT: 8.29 cm3, Mean Attenuation: -72.17 HU (Reference < -70.1 HU)\n"
    "  * Global PVAT Mean: -74.49 HU\n"
    "- Coronary Plaque: Dense calcified atheroma noted on CT.\n\n"
    "Evaluate coronary vascular inflammation, subepicardial fat distribution, and clinical management."
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_instruction}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_content},
        ],
    },
]

print("\nFormatting specialized cardiovascular prompt...")
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

inputs = processor(
    text=prompt,
    images3d=[str(ct_path)],
    anatomy_region="chest",
    return_tensors="pt",
).to(model.device)

print("Preprocessed volume into 3D tokens. Starting autoregressive generation (max_new_tokens=1536)...")
torch.cuda.reset_peak_memory_stats()
t_gen0 = time.time()

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1536,
        do_sample=False,
        use_cache=True,
    )
t_gen = time.time() - t_gen0

new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
num_tokens = new_tokens.shape[1]
output_text = processor.batch_decode(
    new_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
tokens_per_sec = num_tokens / t_gen if t_gen > 0 else 0

print("=" * 80)
print(f"GENERATION METRICS: {num_tokens} tokens generated in {t_gen:.2f} s ({tokens_per_sec:.2f} tokens/s)")
print(f"PEAK VRAM ALLOCATED: {peak_vram_gb:.2f} GB / 24.00 GB")
print("=" * 80)
print("SPECIALIZED RADIOLOGY REPORT & CLINICAL REASONING:")
print("=" * 80)
print(output_text)
print("=" * 80)

# Save result to report file
out_path = Path("example_source_dir/0506/nv_reason_ct_cardiac_specialized_report.txt")
out_path.write_text(output_text, encoding="utf-8")
print(f"Report saved to: {out_path}")

