"""End-to-end inference verification for NVIDIA NV-Reason-CT on RTX 4090."""

import time
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = Path(r"E:\models\NV-Reason-CT")
ct_path = Path(r"example_source_dir\0506\img.nrrd")

print("=" * 80)
print("NVIDIA NV-REASON-CT CLINICAL REPORT GENERATION INFERENCE TEST")
print("Target GPU: NVIDIA GeForce RTX 4090 (24 GB VRAM)")
print("=" * 80)

# Reset peak memory tracker
torch.cuda.reset_peak_memory_stats()
t0 = time.time()

print(f"Loading processor from: {model_path} ...")
processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

print("Loading model on CUDA (dtype=bfloat16, attn_implementation=sdpa) ...")
model = AutoModelForImageTextToText.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda",
)
model.eval()

t_load = time.time() - t0
vram_weights_gb = torch.cuda.memory_allocated() / (1024**3)
print(f"Model loaded in {t_load:.2f} s | Model Weights VRAM: {vram_weights_gb:.2f} GB")

# Construct clinical prompt with 3D fat quantification biomarkers
clinical_prompt = (
    "Analyze this non-contrast cardiac CT. "
    "Quantitative epicardial and pericoronary fat biomarkers have been extracted: "
    "Total Epicardial Adipose Tissue (EAT) volume is 140.73 cm3, mean attenuation is -77.0 HU. "
    "Pericoronary Adipose Tissue (PVAT) volume is 24.55 cm3, mean attenuation is -74.5 HU "
    "(LAD: 13.12 cm3, -78.9 HU; LCX: 3.14 cm3, -62.0 HU; RCA: 8.29 cm3, -72.2 HU). "
    "Write a structured clinical findings and impression section interpreting these cardiovascular risk markers."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": clinical_prompt},
        ],
    }
]

print("\nFormatting multimodal prompt with 3D image processor...")
t_prep0 = time.time()
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

t_prep = time.time() - t_prep0
print(f"Volume preprocessed into 3D grid tokens in {t_prep:.2f} s")
print(f"Input IDs shape: {inputs.input_ids.shape} | Pixel values shape: {inputs.pixel_values.shape}")

print("\nRunning autoregressive generation (max_new_tokens=1024)...")
t_gen0 = time.time()
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
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
reserved_vram_gb = torch.cuda.max_memory_reserved() / (1024**3)
tokens_per_sec = num_tokens / t_gen if t_gen > 0 else 0

print("=" * 80)
print("GENERATION METRICS & PERFORMANCE ON RTX 4090")
print("=" * 80)
print(f"Generated Tokens:     {num_tokens}")
print(f"Generation Latency:   {t_gen:.2f} seconds ({tokens_per_sec:.2f} tokens/s)")
print(f"Peak VRAM Allocated:  {peak_vram_gb:.2f} GB / 24.00 GB ({(peak_vram_gb/24.0)*100:.1f}%)")
print(f"Peak VRAM Reserved:   {reserved_vram_gb:.2f} GB / 24.00 GB")
print("=" * 80)
print("GENERATED CLINICAL RADIOLOGY REPORT & REASONING:")
print("=" * 80)
print(output_text)
print("=" * 80)

# Save result to output file
out_path = Path("example_source_dir/0506/nv_reason_ct_generated_report.txt")
out_path.write_text(output_text, encoding="utf-8")
print(f"Report saved to: {out_path}")

