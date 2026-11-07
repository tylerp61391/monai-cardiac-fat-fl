"""Unit tests for NVIDIA NV-Reason-CT integration module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from cardiac_fat_fl.nv_reason_reporter import (
    build_clinical_prompt,
    clean_vlm_narrative,
    generate_nv_reason_narrative,
    is_nv_reason_available,
)


def test_is_nv_reason_available_missing_dir(tmp_path: Path):
    """Test availability check on non-existent directory."""
    non_existent = tmp_path / "does_not_exist"
    assert not is_nv_reason_available(non_existent)


def test_is_nv_reason_available_missing_safetensors(tmp_path: Path):
    """Test availability check on directory missing weights."""
    model_dir = tmp_path / "model_empty"
    model_dir.mkdir()
    assert not is_nv_reason_available(model_dir)


def test_is_nv_reason_available_real_or_mocked(tmp_path: Path):
    """Test availability when weights file exists."""
    model_dir = tmp_path / "mock_model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").touch()

    with patch("torch.cuda.is_available", return_value=True):
        assert is_nv_reason_available(model_dir)

    with patch("torch.cuda.is_available", return_value=False):
        assert not is_nv_reason_available(model_dir)


def test_build_clinical_prompt_formatting():
    """Test clinical prompt construction with standard 19-column metrics."""
    metrics = {
        "EAT Total Volume (cm3)": 140.73,
        "EAT Total Mean Attenuation (HU)": -77.02,
        "PVAT Total Volume (cm3)": 24.55,
        "PVAT Total Mean Attenuation (HU)": -74.49,
        "PVAT LAD Mean Attenuation (HU)": -78.95,
        "PVAT LCX Mean Attenuation (HU)": -62.02,
        "PVAT RCA Mean Attenuation (HU)": -72.17,
    }

    # Structured mode
    prompt_struct = build_clinical_prompt(metrics, mode="structured")
    assert "140.7 cm3" in prompt_struct
    assert "Severely Elevated burden" in prompt_struct
    assert "-62.0 HU" in prompt_struct
    assert "-70.1 HU" in prompt_struct
    assert "Provide a structured findings section" in prompt_struct

    # Reasoning mode
    prompt_reason = build_clinical_prompt(metrics, mode="reasoning")
    assert "Provide a full reasoning analysis" in prompt_reason


@patch("cardiac_fat_fl.nv_reason_reporter.is_nv_reason_available", return_value=True)
def test_generate_nv_reason_narrative_mocked(mock_avail: MagicMock, tmp_path: Path):
    """Test generate_nv_reason_narrative pipeline execution with mocked transformers."""
    ct_file = tmp_path / "scan.nrrd"
    ct_file.touch()

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "Formatted prompt"
    mock_inputs = MagicMock()
    mock_inputs.input_ids.shape = (1, 100)
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ["Calcified coronary arteries. Impression: Elevated EAT burden."]

    mock_model = MagicMock()
    mock_model.device = "cuda:0"
    mock_model.generate.return_value = torch.zeros((1, 150), dtype=torch.long)

    metrics = {
        "EAT Total Volume (cm3)": 120.0,
        "EAT Total Mean Attenuation (HU)": -72.0,
        "PVAT Total Volume (cm3)": 20.0,
        "PVAT Total Mean Attenuation (HU)": -73.0,
    }

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    mock_transformers.AutoModelForImageTextToText.from_pretrained.return_value = mock_model

    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        narrative = generate_nv_reason_narrative(
            ct_path=ct_file,
            metrics=metrics,
            model_dir=tmp_path,
            mode="structured",
            max_new_tokens=100,
        )

    assert "Calcified coronary arteries" in narrative
    assert mock_model.generate.called


def test_clean_vlm_narrative_sanitization():
    """Test sanitization of <answer>, <thought>, <think>, and chat template tokens."""
    raw = (
        "<|im_start|>system\nYou are a radiologist.<|im_end|>\n"
        "<thought>Patient has calcified plaques.</thought>\n"
        "IMPRESSION:\n"
        "1. Coronary atherosclerosis.\n"
        "<answer> Coronary artery wall calcification, Emphysema </answer>"
    )
    cleaned = clean_vlm_narrative(raw)
    assert "<|im_start|>" not in cleaned
    assert "<|im_end|>" not in cleaned
    assert "<thought>" not in cleaned
    assert "</thought>" not in cleaned
    assert "<answer>" not in cleaned
    assert "</answer>" not in cleaned
    assert "CLINICAL REASONING PROCESS:" in cleaned
    assert "KEY FINDINGS: Coronary artery wall calcification, Emphysema" in cleaned


