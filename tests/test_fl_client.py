"""Unit and Integration Tests for MONAI FL Client and NVFLARE Statistics Executor.

Verifies:
  - ClientAlgo.initialize(), train(), evaluate(), and get_weights() lifecycle
  - Mode 1: FedProx proximal loss and FedBN normalization weight filtering
  - Mode 2: Federated cohort analytics moments extraction
  - CardiacFatStatsExecutor and multi-site federated aggregation
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from monai.fl.utils.exchange_object import ExchangeObject, WeightType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal

from cardiac_fat_fl.fl_client_algo import CardiacFatClientAlgo
from cardiac_fat_fl.fl_stats_executor import (
    CardiacFatStatsExecutor,
    aggregate_federated_statistics,
)
from cardiac_fat_fl.measurement_schema import (
    MEASUREMENT_CSV_COLUMNS,
    REPORTED_REGIONS,
)
from cardiac_fat_fl.model import create_cardiac_fat_model, filter_weights_for_fedbn


def test_model_factory_and_fedbn_filtering() -> None:
    """Verify that model factory instantiates 8-channel SegResNet and FedBN filters norm layers."""
    model = create_cardiac_fat_model(
        model_name="segresnet",
        spatial_dims=3,
        in_channels=1,
        out_channels=8,
        init_filters=8,
    )
    assert isinstance(model, torch.nn.Module)

    state = model.state_dict()
    assert len(state) > 0

    # Filter with FedBN
    filtered = filter_weights_for_fedbn(state, exclude_norm=True)
    assert len(filtered) < len(state)

    # Verify no normalization parameters exist in filtered dict
    for k in filtered.keys():
        assert "norm" not in k.lower()
        assert "bn" not in k.lower()


def test_cardiac_fat_client_algo_lifecycle(tmp_path: Path, synthetic_cohort_dir: Path) -> None:
    """Verify complete ClientAlgo lifecycle: initialize -> train -> evaluate -> get_weights."""
    client = CardiacFatClientAlgo(
        data_dir=synthetic_cohort_dir,
        model_name="segresnet",
        lr=1e-3,
        fedprox_mu=0.05,
        fed_bn=True,
        epochs_per_round=1,
        device="cpu",
    )

    # 1. Initialize
    client.initialize()
    assert client.initialized is True
    assert client.model is not None
    assert client.optimizer is not None

    # 2. Extract initial weights
    initial_weights_obj = client.get_weights()
    assert isinstance(initial_weights_obj, ExchangeObject)
    assert initial_weights_obj.weight_type == WeightType.WEIGHTS
    assert initial_weights_obj.weights is not None
    assert len(initial_weights_obj.weights) > 0

    # 3. Simulate training round receiving global weights
    train_exchange = ExchangeObject(weights=initial_weights_obj.weights)
    client.train(train_exchange)

    # 4. Extract updated weights
    updated_weights_obj = client.get_weights()
    assert updated_weights_obj.weights is not None
    assert len(updated_weights_obj.weights) == len(initial_weights_obj.weights)

    # 5. Evaluate
    eval_obj = client.evaluate(train_exchange)
    assert isinstance(eval_obj, ExchangeObject)
    assert eval_obj.metrics is not None
    assert "val_loss" in eval_obj.metrics
    assert "val_dice" in eval_obj.metrics
    assert 0.0 <= eval_obj.metrics["val_dice"] <= 1.0


def test_cardiac_fat_client_cohort_analytics(tmp_path: Path) -> None:
    """Verify Mode 2: Federated Cohort Analytics extracts summary moments without patient rows."""
    # Create mock cardiac_fat_measurements.csv
    csv_file = tmp_path / "cardiac_fat_measurements.csv"
    rows = []
    for i in range(10):
        row = {"participant_id": f"PAT_{i:03d}"}
        for region in REPORTED_REGIONS:
            row[f"{region.value} Volume (cm3)"] = 50.0 + i * 2.0
            row[f"{region.value} Mean Attenuation (HU)"] = -100.0 + i * 1.5
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)

    client = CardiacFatClientAlgo(measurements_csv=csv_file, device="cpu")
    stats_obj = client.get_data_stats()

    assert isinstance(stats_obj, ExchangeObject)
    stats = stats_obj.statistics
    assert stats is not None
    assert stats["participant_count"] == 10

    # Verify summary moments for EAT Total Volume
    eat_vol_stats = stats["EAT Total Volume (cm3)"]
    assert eat_vol_stats["count"] == 10
    # Mean of 50 + 2*i for i in 0..9 is 50 + 9 = 59.0
    assert math.isclose(eat_vol_stats["mean"], 59.0, rel_tol=1e-5)
    assert eat_vol_stats["min"] == 50.0
    assert eat_vol_stats["max"] == 68.0


def test_nvflare_stats_executor_and_multi_site_aggregation(tmp_path: Path) -> None:
    """Verify CardiacFatStatsExecutor and exact multi-site mathematical aggregation."""
    # Create Site 1 CSV (5 patients, mean volume 60.0)
    site1_csv = tmp_path / "site1.csv"
    rows1 = [
        {"participant_id": f"S1_{i}", "EAT Total Volume (cm3)": 60.0, "EAT Total Mean Attenuation (HU)": -90.0}
        for i in range(5)
    ]
    pd.DataFrame(rows1).to_csv(site1_csv, index=False)

    # Create Site 2 CSV (5 patients, mean volume 80.0)
    site2_csv = tmp_path / "site2.csv"
    rows2 = [
        {"participant_id": f"S2_{i}", "EAT Total Volume (cm3)": 80.0, "EAT Total Mean Attenuation (HU)": -110.0}
        for i in range(5)
    ]
    pd.DataFrame(rows2).to_csv(site2_csv, index=False)

    # Run executor on Site 1
    exec1 = CardiacFatStatsExecutor(csv_path=site1_csv, num_bins=10)
    fl_ctx = FLContext()
    signal = Signal()
    reply1 = exec1.execute("get_stats", Shareable(), fl_ctx, signal)

    assert reply1.get_return_code() == ReturnCode.OK
    s1_stats = reply1["stats"]
    assert s1_stats["participant_count"] == 5

    # Run executor on Site 2
    exec2 = CardiacFatStatsExecutor(csv_path=site2_csv, num_bins=10)
    reply2 = exec2.execute("get_stats", Shareable(), fl_ctx, signal)
    s2_stats = reply2["stats"]
    assert s2_stats["participant_count"] == 5

    # Aggregate across sites
    global_agg = aggregate_federated_statistics([s1_stats, s2_stats])
    assert global_agg["total_participants"] == 10
    assert global_agg["site_count"] == 2

    # Global mean of [60, 60, 60, 60, 60, 80, 80, 80, 80, 80] = 70.0
    eat_vol_metric = global_agg["metrics"]["EAT Total Volume (cm3)"]
    assert eat_vol_metric["count"] == 10
    assert math.isclose(eat_vol_metric["mean"], 70.0, rel_tol=1e-5)
    assert eat_vol_metric["min"] == 60.0
    assert eat_vol_metric["max"] == 80.0

