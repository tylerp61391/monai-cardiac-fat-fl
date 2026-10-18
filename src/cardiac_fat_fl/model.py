"""Single-Stage 3D Deep Learning Model Factory for 8-Channel Cardiac Fat Segmentation.

Provides SegResNet and SwinUNETR architectures for direct distillation of the 8-compartment
EAT/PVAT masks, with built-in support for Federated Batch Normalization (FedBN).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR, UNet


def create_cardiac_fat_model(
    model_name: str = "segresnet",
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 8,
    init_filters: int = 16,
    dropout_prob: float = 0.2,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    **kwargs: Any,
) -> nn.Module:
    """Factory creating 3D deep learning models for 8-compartment cardiac fat segmentation.

    Args:
        model_name: 'segresnet' (default), 'swinunetr', or 'unet'.
        spatial_dims: Spatial dimensionality (default: 3).
        in_channels: Input image channels (default: 1 for CT).
        out_channels: Target output channels (default: 8 for EAT/PVAT compartments).
        init_filters: Number of initial feature filters for SegResNet/UNet.
        dropout_prob: Dropout probability.
        img_size: Input spatial dimensions (required for SwinUNETR).
        **kwargs: Additional model parameters forwarded to the architecture.

    Returns:
        PyTorch nn.Module instance.
    """
    model_type = model_name.lower().strip()

    if model_type == "segresnet":
        blocks_down = kwargs.pop("blocks_down", [1, 2, 2, 4])
        blocks_up = kwargs.pop("blocks_up", [1, 1, 1])
        model = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
            **kwargs,
        )

    elif model_type == "swinunetr":
        feature_size = kwargs.pop("feature_size", 24)
        use_checkpoint = kwargs.pop("use_checkpoint", True)
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            drop_rate=dropout_prob,
            **kwargs,
        )

    elif model_type == "unet":
        channels = kwargs.pop("channels", (16, 32, 64, 128, 256))
        strides = kwargs.pop("strides", (2, 2, 2, 2))
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout_prob,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. Choose from 'segresnet', 'swinunetr', or 'unet'."
        )

    return model


def is_normalization_param(param_name: str) -> bool:
    """Check if a parameter name corresponds to a normalization layer (FedBN check).

    Matches BatchNorm, InstanceNorm, LayerNorm, and GroupNorm parameters:
    running_mean, running_var, num_batches_tracked, weight, bias of norm layers.
    """
    name_lower = param_name.lower()
    norm_keywords = ["bn", "norm", "running_mean", "running_var", "num_batches_tracked"]
    return any(kw in name_lower for kw in norm_keywords)


def filter_weights_for_fedbn(
    state_dict: Dict[str, torch.Tensor],
    exclude_norm: bool = True,
) -> Dict[str, torch.Tensor]:
    """Filter state dictionary for Federated Batch Normalization (FedBN).

    When exclude_norm is True, normalization parameters are stripped from the
    weights sent to the central FL server, keeping local scanner calibration
    intact on each hospital client.

    Args:
        state_dict: PyTorch model state_dict.
        exclude_norm: Whether to exclude normalization layers.

    Returns:
        Filtered state_dict for server aggregation.
    """
    if not exclude_norm:
        return {k: v.clone() for k, v in state_dict.items()}

    filtered: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not is_normalization_param(k):
            filtered[k] = v.clone()
    return filtered

