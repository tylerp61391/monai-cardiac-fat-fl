"""Quick test to verify loading NV-Reason-CT and checking VRAM allocation on RTX 4090."""

from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = Path(r"E:\models\NV-Reason-CT")
ct_path = Path(r"example_source_dir\0506\img.nrrd")

print("================================================================================")
print("TESTING NVIDIA NV-Reason-CT INITIALIZATION ON RTX 4090")
print("================================================================================")
print(f"Loading processor from: {model_path}")
processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
print("Processor loaded successfully!")

print(f"Loading model on CUDA with bfloat16 and SDPA from: {model_path}")
model = AutoModelForImageTextToText.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda",
)
model.eval()

allocated = torch.cuda.memory_allocated() / (1024**3)
reserved = torch.cuda.memory_reserved() / (1024**3)
print(f"SUCCESS! NV-Reason-CT is loaded on {model.device}.")
print(f"Allocated VRAM: {allocated:.2f} GB | Reserved VRAM: {reserved:.2f} GB")
print("================================================================================")
