"""NVIDIA FLARE Federated Statistics Executor for Cardiac Fat Cohort Analytics.

Aggregates cohort-level volume and attenuation statistics across distributed clinical sites
from `cardiac_fat_measurements.csv` without transmitting individual patient rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal

from cardiac_fat_fl.measurement_schema import (
    MEASUREMENT_CSV_COLUMNS,
    REPORTED_REGIONS,
    ReportedRegion,
)


class CardiacFatStatsExecutor(Executor):
    """NVFLARE Executor that computes privacy-preserving local summary statistics.

    Extracts count, mean, standard deviation, sum, sum-of-squares, and binned
    histograms for all 9 reported EAT/PVAT regions from the local CSV.
    """

    def __init__(
        self,
        csv_path: Union[str, Path] = "cardiac_fat_measurements.csv",
        num_bins: int = 20,
    ) -> None:
        """Initialize stats executor.

        Args:
            csv_path: Path to the local `cardiac_fat_measurements.csv` file.
            num_bins: Number of histogram bins for distributions.
        """
        super().__init__()
        self.csv_path = Path(csv_path).resolve()
        self.num_bins = num_bins

    def compute_local_stats(self) -> Dict[str, Any]:
        """Compute local privacy-preserving statistical moments and histograms."""
        if not self.csv_path.is_file():
            return {
                "participant_count": 0,
                "status": "error",
                "message": f"File not found: {self.csv_path}",
                "metrics": {},
            }

        df = pd.read_csv(self.csv_path)
        stats: Dict[str, Any] = {
            "participant_count": int(len(df)),
            "status": "ok",
            "metrics": {},
        }

        for region in REPORTED_REGIONS:
            for metric_type, unit in [("Volume", "cm3"), ("Mean Attenuation", "HU")]:
                col_name = f"{region.value} {metric_type} ({unit})"
                if col_name in df.columns:
                    series = pd.to_numeric(df[col_name], errors="coerce").dropna()
                    arr = series.to_numpy(dtype=np.float64)
                    n = int(len(arr))

                    if n > 0:
                        s1 = float(np.sum(arr))
                        s2 = float(np.sum(arr ** 2))
                        mean_val = float(np.mean(arr))
                        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                        min_val = float(np.min(arr))
                        max_val = float(np.max(arr))

                        # Binned histogram
                        if min_val < max_val:
                            hist_counts, bin_edges = np.histogram(arr, bins=self.num_bins)
                            hist = {
                                "counts": hist_counts.tolist(),
                                "bin_edges": bin_edges.tolist(),
                            }
                        else:
                            hist = {
                                "counts": [n],
                                "bin_edges": [min_val, min_val + 1.0],
                            }

                        stats["metrics"][col_name] = {
                            "count": n,
                            "sum": s1,
                            "sum_squares": s2,
                            "mean": mean_val,
                            "std": std_val,
                            "min": min_val,
                            "max": max_val,
                            "histogram": hist,
                        }
                    else:
                        stats["metrics"][col_name] = {
                            "count": 0,
                            "sum": 0.0,
                            "sum_squares": 0.0,
                            "mean": float("nan"),
                            "std": float("nan"),
                            "min": float("nan"),
                            "max": float("nan"),
                            "histogram": {"counts": [], "bin_edges": []},
                        }

        return stats

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Handle execution request from NVFLARE federated server."""
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        stats_data = self.compute_local_stats()

        reply = make_reply(ReturnCode.OK)
        reply["stats"] = stats_data
        return reply


def aggregate_federated_statistics(site_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate local statistical summaries from multiple clinical sites into global cohort metrics.

    Uses sum and sum-of-squares to compute mathematically exact pooled mean and variance:
      Total N = sum(N_k)
      Global Mean = sum(Sum_k) / Total N
      Global Variance = (sum(SumSquares_k) - (Total N) * (Global Mean)^2) / (Total N - 1)

    Args:
        site_statistics: List of stats dictionaries returned by sites.

    Returns:
        Aggregated global cohort statistics dictionary.
    """
    total_participants = sum(site.get("participant_count", 0) for site in site_statistics)
    aggregated: Dict[str, Any] = {
        "total_participants": total_participants,
        "site_count": len(site_statistics),
        "metrics": {},
    }

    if total_participants == 0:
        return aggregated

    # Collect all metric keys
    metric_keys = set()
    for site in site_statistics:
        metrics_dict = site.get("metrics", {})
        metric_keys.update(metrics_dict.keys())

    for key in sorted(metric_keys):
        total_n = 0
        total_sum = 0.0
        total_sum_sq = 0.0
        global_min = float("inf")
        global_max = float("-inf")

        for site in site_statistics:
            m = site.get("metrics", {}).get(key)
            if m and m.get("count", 0) > 0:
                n = m["count"]
                total_n += n
                total_sum += m["sum"]
                total_sum_sq += m["sum_squares"]
                if m["min"] < global_min:
                    global_min = m["min"]
                if m["max"] > global_max:
                    global_max = m["max"]

        if total_n > 0:
            global_mean = total_sum / total_n
            if total_n > 1:
                variance = (total_sum_sq - (total_n * (global_mean ** 2))) / (total_n - 1)
                variance = max(0.0, variance)  # Prevent numerical underflow below 0
                global_std = float(np.sqrt(variance))
            else:
                global_std = 0.0

            aggregated["metrics"][key] = {
                "count": total_n,
                "mean": float(global_mean),
                "std": float(global_std),
                "min": float(global_min),
                "max": float(global_max),
            }

    return aggregated

