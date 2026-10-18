"""Tests for hardware auto-detection and safe concurrency configuration."""

from __future__ import annotations

from typer.testing import CliRunner

from cardiac_fat_fl.cli import app
from cardiac_fat_fl.hardware import (
    HardwareProfile,
    detect_hardware,
    display_hardware_summary,
    validate_worker_concurrency,
)

runner = CliRunner()


def test_detect_hardware() -> None:
    """Verify hardware profile detection produces valid positive metrics."""
    profile = detect_hardware()
    assert profile.logical_cpus >= 1
    assert profile.physical_cpus >= 1
    assert profile.total_ram_gb > 0.0
    assert profile.available_ram_gb > 0.0
    assert profile.recommended_workers >= 1
    assert profile.intra_op_threads >= 1
    assert profile.recommended_device in ("cuda", "cpu")


def test_validate_worker_concurrency_clamps_unsafe_requests() -> None:
    """Verify worker validation safely throttles requests exceeding RAM budget."""
    mock_profile = HardwareProfile(
        logical_cpus=16,
        physical_cpus=8,
        total_ram_gb=16.0,
        available_ram_gb=6.0,
        has_cuda=False,
        recommended_workers=2,
    )

    # Auto-detect mode (requested <= 0) returns recommended
    assert validate_worker_concurrency(0, mock_profile) == 2
    assert validate_worker_concurrency(-1, mock_profile) == 2

    # Safe request within bounds
    assert validate_worker_concurrency(2, mock_profile) == 2

    # Dangerous request (e.g. 16 or 32 workers on 6GB available RAM) is clamped to safe cap
    clamped = validate_worker_concurrency(32, mock_profile)
    assert clamped <= 4  # max safe on 6GB available RAM is 6.0 / 1.5 = 4 workers


def test_cli_hardware_command() -> None:
    """Test CLI hardware command output and formatting."""
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Hardware Auto-Detection & Safe Concurrency Profile" in result.stdout
    assert "CPUs" in result.stdout
    assert "RAM" in result.stdout
    assert "Safe Concurrency Sizing" in result.stdout
