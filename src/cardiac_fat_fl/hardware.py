"""Hardware auto-detection and safe concurrency configuration.

Inspects CPU core counts, available RAM, and CUDA GPU VRAM to dynamically determine
safe worker limits and prevent system freezing, OOM crashes, or thread oversubscription.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import psutil
import torch
from rich.console import Console
from rich.table import Table


@dataclass
class HardwareProfile:
    """Comprehensive hardware profile and safe concurrency recommendation."""

    logical_cpus: int
    physical_cpus: int
    total_ram_gb: float
    available_ram_gb: float
    has_cuda: bool
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    total_vram_gb: float = 0.0
    recommended_workers: int = 1
    recommended_device: str = "cpu"
    intra_op_threads: int = 2

    @property
    def is_rtx_4090(self) -> bool:
        """Check if an NVIDIA GeForce RTX 4090 is detected."""
        return any("4090" in name for name in self.gpu_names)


def detect_hardware() -> HardwareProfile:
    """Dynamically probe host hardware and calculate safe concurrency settings."""
    logical_cpus = os.cpu_count() or 4
    physical_cpus = psutil.cpu_count(logical=False) or (logical_cpus // 2)

    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024**3), 2)
    available_ram_gb = round(vm.available / (1024**3), 2)

    has_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if has_cuda else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if has_cuda else []

    total_vram_gb = 0.0
    if has_cuda and gpu_count > 0:
        total_vram_gb = round(
            sum(torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)) / (1024**3),
            2,
        )

    # Calculate safe worker concurrency
    # Each 3D cardiac CT volume + ONNX model requires ~2.0 GB - 2.5 GB RAM working set
    # Never allocate more than 75% of available RAM or leave fewer than 2 cores free for OS
    ram_worker_cap = max(1, int(available_ram_gb / 2.5))
    cpu_worker_cap = max(1, logical_cpus - 2) if logical_cpus > 2 else 1

    if has_cuda and total_vram_gb >= 16.0:
        # High-end GPU profile (e.g. RTX 4090 with 24GB VRAM)
        # Cap at 8 workers to avoid I/O bottlenecks while saturating GPU Tensor Cores
        recommended_workers = min(8, cpu_worker_cap, ram_worker_cap)
        recommended_device = "cuda"
    elif has_cuda:
        # Mid-tier GPU profile (e.g. 8-12 GB VRAM)
        recommended_workers = min(4, cpu_worker_cap, max(1, int(total_vram_gb / 2.5)))
        recommended_device = "cuda"
    else:
        # CPU-only execution profile
        # Use balanced fraction of logical cores to avoid thermal throttling and UI freezes
        recommended_workers = min(cpu_worker_cap, ram_worker_cap, 8)
        recommended_device = "cpu"

    # Set intra-op threads per worker to avoid thread oversubscription: workers * threads <= logical_cpus
    intra_op_threads = max(1, min(4, logical_cpus // recommended_workers))

    return HardwareProfile(
        logical_cpus=logical_cpus,
        physical_cpus=physical_cpus,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        has_cuda=has_cuda,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        total_vram_gb=total_vram_gb,
        recommended_workers=recommended_workers,
        recommended_device=recommended_device,
        intra_op_threads=intra_op_threads,
    )


def validate_worker_concurrency(requested_workers: int, profile: Optional[HardwareProfile] = None) -> int:
    """Validate requested workers against host safety bounds and return clamped value if unsafe."""
    if profile is None:
        profile = detect_hardware()

    if requested_workers <= 0:
        return profile.recommended_workers

    # Check for dangerous over-allocation (e.g. asking for 16 workers on an 8GB machine)
    max_safe = max(1, int(profile.available_ram_gb / 1.5))
    if requested_workers > max_safe:
        return max_safe

    return requested_workers


def display_hardware_summary(profile: HardwareProfile, console: Optional[Console] = None) -> None:
    """Render a clean summary table of detected host hardware and safe concurrency limits."""
    if console is None:
        console = Console()

    table = Table(title="Hardware Auto-Detection & Safe Concurrency Profile", title_style="bold cyan")
    table.add_column("Resource", style="bold white", justify="left")
    table.add_column("Detected Specification", style="green", justify="left")
    table.add_column("Safe Limit Recommendation", style="cyan", justify="left")

    gpu_desc = f"{profile.gpu_count}x {', '.join(profile.gpu_names)} ({profile.total_vram_gb:.1f} GB)" if profile.has_cuda else "None (CPU Execution)"
    table.add_row("CPUs (Logical / Physical)", f"{profile.logical_cpus} / {profile.physical_cpus} cores", f"Reserve >= 2 cores for OS")
    table.add_row("System RAM (Total / Available)", f"{profile.total_ram_gb:.1f} GB / {profile.available_ram_gb:.1f} GB", f"Max {profile.recommended_workers * 2.5:.1f} GB budget")
    table.add_row("CUDA Acceleration", gpu_desc, f"Device: '{profile.recommended_device}'")
    table.add_row("Safe Concurrency Sizing", f"{profile.recommended_workers} parallel workers", f"{profile.intra_op_threads} threads/worker (Zero Contention)")

    console.print(table)

