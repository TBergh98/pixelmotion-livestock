"""
Analytics module for computing descriptive statistics and trends from motility data.

Provides functions to:
- Load frame-level motility records from JSONL
- Aggregate into custom time windows (intra-day)
-Compute descriptive metrics (percentiles, CV, slope, etc.)
- Build inter-day aggregates and trend metrics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from datetime import datetime
from statistics import mean, median, stdev, quantiles

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a motility window."""
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    amplitude: float  # max - min
    active_ratio: float
    coeff_variation: float | None = None  # std / mean if mean > 0
    percentiles: dict[int, float] | None = None  # e.g., {10: 0.05, 25: 0.1, ...}
    trend_slope: float | None = None  # linear regression slope over time


def compute_descriptive_stats(motility_values: list[float], percentiles_list: list[int] | None = None, include_slope: bool = False) -> DescriptiveStats:
    """
    Compute descriptive statistics for a list of motility scores.
    
    Args:
        motility_values: List of motility scores [0-1]
        percentiles_list: List of percentiles to compute (e.g., [10, 25, 50, 75, 90])
        include_slope: Whether to compute linear regression slope
        
    Returns:
        DescriptiveStats dataclass with computed metrics
    """
    if not motility_values:
        return DescriptiveStats(
            count=0, mean=0.0, median=0.0, std=0.0, min=0.0, max=0.0,
            amplitude=0.0, active_ratio=0.0, coeff_variation=None, percentiles=None, trend_slope=None
        )
    
    count = len(motility_values)
    values_array = np.array(motility_values, dtype=float)
    
    mean_val = float(np.mean(values_array))
    median_val = float(np.median(values_array))
    std_val = float(np.std(values_array))
    min_val = float(np.min(values_array))
    max_val = float(np.max(values_array))
    amplitude = max_val - min_val
    
    # Active ratio: fraction >= 0.02 (default threshold)
    active_ratio = float(np.sum(values_array >= 0.02) / count) if count > 0 else 0.0
    
    coeff_var = None
    if mean_val > 0:
        coeff_var = float(std_val / mean_val)
    
    # Compute percentiles
    percentiles_dict = None
    if percentiles_list and count > 0:
        try:
            quantile_vals = np.percentile(values_array, percentiles_list)
            percentiles_dict = {int(p): float(v) for p, v in zip(percentiles_list, quantile_vals)}
        except Exception as e:
            LOGGER.warning(f"Failed to compute percentiles: {e}")
    
    # Compute linear trend (slope)
    trend_slope = None
    if include_slope and count > 1:
        try:
            x = np.arange(count, dtype=float)
            y = values_array
            coeffs = np.polyfit(x, y, 1)
            trend_slope = float(coeffs[0])
        except Exception as e:
            LOGGER.warning(f"Failed to compute trend slope: {e}")
    
    return DescriptiveStats(
        count=count,
        mean=round(mean_val, 6),
        median=round(median_val, 6),
        std=round(std_val, 6),
        min=round(min_val, 6),
        max=round(max_val, 6),
        amplitude=round(amplitude, 6),
        active_ratio=round(active_ratio, 6),
        coeff_variation=round(coeff_var, 6) if coeff_var is not None else None,
        percentiles=percentiles_dict,
        trend_slope=round(trend_slope, 6) if trend_slope is not None else None,
    )


def load_frame_data_from_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """
    Load frame-level motility data from JSONL file.
    
    Returns list of frame records with timestamp_seconds and motility_score fields.
    """
    records = []
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("record_type") == "frame_result":
                        result = record.get("result", {})
                        if "motility_score" in result and "timestamp_seconds" in result:
                            records.append(result)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        LOGGER.error(f"Failed to load JSONL {jsonl_path}: {e}")
    
    return records


def aggregate_into_windows(frame_records: list[dict[str, Any]], window_seconds: int) -> dict[int, list[float]]:
    """
    Aggregate frame-level motility scores into time windows.
    
    Args:
        frame_records: List of frame records with timestamp_seconds and motility_score
        window_seconds: Size of each window in seconds
        
    Returns:
        Dict mapping window_index to list of motility scores
    """
    windows: dict[int, list[float]] = {}
    
    for record in frame_records:
        timestamp = record.get("timestamp_seconds", 0)
        motility = record.get("motility_score", 0)
        
        window_index = int(timestamp // window_seconds)
        if window_index not in windows:
            windows[window_index] = []
        windows[window_index].append(float(motility))
    
    return windows


@dataclass
class IntraDayMetrics:
    """Metrics for a single day, aggregated by time window."""
    recording_date: str | None
    group_id: str | None
    window_duration_seconds: int
    windows: dict[int, DescriptiveStats]  # window_index -> stats
    daily_stats: DescriptiveStats  # Aggregated across all windows


def compute_intraday_metrics(
    frame_records: list[dict[str, Any]],
    window_seconds: int,
    percentiles_list: list[int] | None = None,
    include_slope: bool = False,
    recording_date: str | None = None,
    group_id: str | None = None,
) -> IntraDayMetrics:
    """
    Compute intra-day metrics (by time window) for a set of frame records.
    """
    if percentiles_list is None:
        percentiles_list = [10, 25, 50, 75, 90]
    
    # Aggregate into windows
    windows_data = aggregate_into_windows(frame_records, window_seconds)
    
    # Compute statistics for each window
    window_stats = {}
    all_motility = []
    
    for window_idx in sorted(windows_data.keys()):
        motility_values = windows_data[window_idx]
        all_motility.extend(motility_values)
        window_stats[window_idx] = compute_descriptive_stats(
            motility_values,
            percentiles_list=percentiles_list,
            include_slope=include_slope
        )
    
    # Compute daily aggregate
    daily_stats = compute_descriptive_stats(
        all_motility,
        percentiles_list=percentiles_list,
        include_slope=False  # Don't compute slope for daily aggregate
    )
    
    return IntraDayMetrics(
        recording_date=recording_date,
        group_id=group_id,
        window_duration_seconds=window_seconds,
        windows=window_stats,
        daily_stats=daily_stats,
    )


def save_intraday_metrics_json(metrics: IntraDayMetrics, output_path: Path) -> None:
    """Save intra-day metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "recording_date": metrics.recording_date,
        "group_id": metrics.group_id,
        "window_duration_seconds": metrics.window_duration_seconds,
        "windows": {
            str(idx): asdict(stats) for idx, stats in metrics.windows.items()
        },
        "daily_aggregate": asdict(metrics.daily_stats),
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    LOGGER.info(f"Intra-day metrics saved to {output_path}")


@dataclass
class InterDayMetrics:
    """Aggregated metrics across multiple days for a group."""
    group_id: str
    primary_metric: str  # "mean" or "median"
    daily_summaries: list[dict[str, Any]]  # List of {date, mean, median, std, ...}
    trend_data: dict[str, Any]  # Overall trend stats


def compute_interday_metrics(
    daily_metrics_list: list[IntraDayMetrics],
    primary_metric: str = "median"
) -> InterDayMetrics:
    """
    Compute inter-day trend metrics from a list of daily intra-day metrics.
    """
    if not daily_metrics_list:
        return InterDayMetrics(group_id="unknown", primary_metric=primary_metric, daily_summaries=[], trend_data={})
    
    group_id = daily_metrics_list[0].group_id or "unknown"
    daily_summaries = []
    primary_values = []
    
    for intraday in sorted(daily_metrics_list, key=lambda x: x.recording_date or ""):
        daily_stats = intraday.daily_stats
        primary_val = getattr(daily_stats, primary_metric, daily_stats.mean)
        
        summary = {
            "date": intraday.recording_date,
            "mean": daily_stats.mean,
            "median": daily_stats.median,
            "std": daily_stats.std,
            "min": daily_stats.min,
            "max": daily_stats.max,
            "amplitude": daily_stats.amplitude,
            "active_ratio": daily_stats.active_ratio,
            "primary_value": primary_val,  # The main metric for trending
        }
        daily_summaries.append(summary)
        primary_values.append(primary_val)
    
    # Compute inter-day trend stats
    trend_data = {}
    if primary_values:
        trend_data = {
            "primary_metric": primary_metric,
            "daily_count": len(primary_values),
            "primary_mean": round(float(np.mean(primary_values)), 6),
            "primary_std": round(float(np.std(primary_values)), 6),
            "primary_min": round(float(np.min(primary_values)), 6),
            "primary_max": round(float(np.max(primary_values)), 6),
        }
        
        # Day-over-day change
        if len(primary_values) > 1:
            diffs = [primary_values[i+1] - primary_values[i] for i in range(len(primary_values)-1)]
            trend_data["daily_changes"] = [round(d, 6) for d in diffs]
            trend_data["avg_daily_change"] = round(float(np.mean(diffs)), 6)
    
    return InterDayMetrics(
        group_id=group_id,
        primary_metric=primary_metric,
        daily_summaries=daily_summaries,
        trend_data=trend_data,
    )


def save_interday_metrics_json(metrics: InterDayMetrics, output_path: Path) -> None:
    """Save inter-day metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "group_id": metrics.group_id,
        "primary_metric": metrics.primary_metric,
        "daily_summaries": metrics.daily_summaries,
        "trend_analysis": metrics.trend_data,
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    LOGGER.info(f"Inter-day metrics saved to {output_path}")
