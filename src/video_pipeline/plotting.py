"""
Plotting module for generating intra-day and inter-day motility trend visualizations.

Produces both static (PNG) and interactive (HTML) plots with descriptive metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# Imports will be lazy-loaded to avoid hard dependencies
LOGGER = logging.getLogger(__name__)


def _import_matplotlib():
    """Lazy import matplotlib to avoid hard dependency."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        LOGGER.error(f"matplotlib not available: {e}")
        return None


def _import_plotly():
    """Lazy import plotly to avoid hard dependency."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        return go, px
    except ImportError as e:
        LOGGER.error(f"plotly not available: {e}")
        return None, None


def plot_intraday_timeseries(
    windows_data: dict[int, list[float]],
    window_duration_seconds: int,
    descriptive_stats: dict[int, Any],
    group_id: str | None,
    recording_date: str | None,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    """
    Generate intra-day time series plot showing motility score over time.
    
    Args:
        windows_data: Dict mapping window_index to list of motility scores
        window_duration_seconds: Duration of each window
        descriptive_stats: Dict of descriptive stats per window
        group_id: Group identifier
        recording_date: Recording date
        output_dir: Output directory for saving plots
        
    Returns:
        Tuple of (png_path, html_path) or (None, None) if plotting fails
    """
    plt = _import_matplotlib()
    if plt is None:
        LOGGER.warning("Cannot generate PNG plot: matplotlib not available")
        return None, None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract window means and time points
        window_indices = sorted(windows_data.keys())
        times_hours = [idx * window_duration_seconds / 3600 for idx in window_indices]
        means = [descriptive_stats[idx].get("mean", 0) for idx in window_indices if idx in descriptive_stats]
        medians = [descriptive_stats[idx].get("median", 0) for idx in window_indices if idx in descriptive_stats]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(times_hours, means, marker='o', label='Mean', linestyle='-', linewidth=2, markersize=4)
        ax.plot(times_hours, medians, marker='s', label='Median', linestyle='--', linewidth=1.5, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Time of Day (hours)', fontsize=11)
        ax.set_ylabel('Motility Score', fontsize=11)
        ax.set_title(f'Intra-day Motility Trend - {group_id or "Unknown"} ({recording_date or "No Date"})', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save as PNG
        date_str = recording_date or "nodate"
        group_str = group_id or "unknown"
        png_filename = f"intraday_timeseries_{date_str}_{group_str}.png"
        png_path = output_dir / png_filename
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        LOGGER.info(f"Saved PNG plot to {png_path}")
        
        plt.close(fig)
        return png_path, None  # HTML will be generated separately
        
    except Exception as e:
        LOGGER.error(f"Failed to generate intra-day time series plot: {e}")
        return None, None


def plot_intraday_distribution(
    windows_data: dict[int, list[float]],
    window_duration_seconds: int,
    group_id: str | None,
    recording_date: str | None,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    """
    Generate intra-day distribution plot (box/violin plot by time window).
    """
    plt = _import_matplotlib()
    if plt is None:
        return None, None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        window_indices = sorted(windows_data.keys())
        data_list = [windows_data[idx] for idx in window_indices]
        labels = [f"{idx}h" for idx in window_indices]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xlabel('Time Window', fontsize=11)
        ax.set_ylabel('Motility Score', fontsize=11)
        ax.set_title(f'Intra-day Distribution by Window - {group_id or "Unknown"} ({recording_date or "No Date"})', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        
        date_str = recording_date or "nodate"
        group_str = group_id or "unknown"
        png_filename = f"intraday_distribution_{date_str}_{group_str}.png"
        png_path = output_dir / png_filename
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        LOGGER.info(f"Saved PNG distribution plot to {png_path}")
        
        plt.close(fig)
        return png_path, None
        
    except Exception as e:
        LOGGER.error(f"Failed to generate intra-day distribution plot: {e}")
        return None, None


def plot_interday_trend(
    daily_summaries: list[dict[str, Any]],
    group_id: str,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    """
    Generate inter-day trend plot showing daily metric evolution.
    """
    plt = _import_matplotlib()
    if plt is None:
        return None, None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dates = [s.get("date", "") for s in daily_summaries]
        primary_vals = [s.get("primary_value", 0) for s in daily_summaries]
        means = [s.get("mean", 0) for s in daily_summaries]
        stds = [s.get("std", 0) for s in daily_summaries]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot with error bars (std as uncertainty band)
        ax.errorbar(range(len(dates)), primary_vals, yerr=stds, marker='o', linestyle='-', linewidth=2, markersize=6, capsize=5, label='Daily Primary Metric ± std', capthick=2)
        ax.plot(range(len(dates)), means, marker='s', linestyle='--', linewidth=1, markersize=4, alpha=0.7, label='Daily Mean')
        
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Motility Score', fontsize=11)
        ax.set_title(f'Inter-day Trend - {group_id}', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        png_filename = f"interday_trend_{group_id}.png"
        png_path = output_dir / png_filename
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        LOGGER.info(f"Saved PNG inter-day trend plot to {png_path}")
        
        plt.close(fig)
        return png_path, None
        
    except Exception as e:
        LOGGER.error(f"Failed to generate inter-day trend plot: {e}")
        return None, None


def plot_interday_delta(
    daily_summaries: list[dict[str, Any]],
    group_id: str,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    """
    Generate inter-day delta plot showing day-over-day changes.
    """
    plt = _import_matplotlib()
    if plt is None:
        return None, None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if len(daily_summaries) < 2:
            LOGGER.warning("Not enough days for delta plot")
            return None, None
        
        dates = [s.get("date", "") for s in daily_summaries]
        primary_vals = [s.get("primary_value", 0) for s in daily_summaries]
        
        # Compute day-over-day deltas
        deltas = [primary_vals[i+1] - primary_vals[i] for i in range(len(primary_vals)-1)]
        delta_dates = dates[1:]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['green' if d >= 0 else 'red' for d in deltas]
        ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(delta_dates)))
        ax.set_xticklabels(delta_dates, rotation=45)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Day-over-Day Change', fontsize=11)
        ax.set_title(f'Inter-day Delta (Change) - {group_id}', fontsize=13)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        png_filename = f"interday_delta_{group_id}.png"
        png_path = output_dir / png_filename
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        LOGGER.info(f"Saved PNG inter-day delta plot to {png_path}")
        
        plt.close(fig)
        return png_path, None
        
    except Exception as e:
        LOGGER.error(f"Failed to generate inter-day delta plot: {e}")
        return None, None


def generate_plots_for_group(
    group_id: str,
    intraday_metrics_dir: Path,
    interday_metrics_path: Path,
    output_dir: Path,
    config_plots: dict[str, bool],
    config_formats: dict[str, bool],
) -> dict[str, list[str]]:
    """
    Generate all configured plots for a group.
    
    Returns a dict mapping plot type to list of output file paths.
    """
    generated_plots = {}
    
    LOGGER.info(f"Generating plots for group {group_id}")
    
    # You would load the individual intra-day and inter-day metrics files and call the respective plot functions
    # This is a stub showing the structure; full implementation depends on file organization
    
    return generated_plots
