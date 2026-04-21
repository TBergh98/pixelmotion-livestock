from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analytics import (
    IntraDayMetrics,
    aggregate_into_windows,
    aggregate_spatial_grids_into_windows,
    compute_interday_metrics,
    compute_intraday_metrics,
    load_frame_data_from_jsonl,
    save_interday_metrics_json,
    save_intraday_metrics_json,
)
from .config_loader import AppConfig, load_config
from .plot_collage import build_intraday_plot_collages
from .plotting import (
    plot_interday_delta,
    plot_interday_trend,
    plot_intraday_distribution,
    plot_intraday_timeseries,
    plot_spatial_heatmap,
)
from .quick_summary import generate_quick_summary_files

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupScaleContext:
    max_window_count: int
    y_limits: tuple[float, float]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _discover_result_jsonl(
    results_root: Path,
    group_filter: set[str],
    date_filter: set[str],
) -> list[tuple[str, str, Path]]:
    if not results_root.exists() or not results_root.is_dir():
        return []

    discovered: list[tuple[str, str, Path]] = []
    for group_dir in sorted([p for p in results_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        group_name = group_dir.name
        if group_filter and group_name not in group_filter:
            continue

        for day_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            day_name = day_dir.name
            if date_filter and day_name not in date_filter:
                continue

            jsonl_files = sorted(day_dir.glob("*_results.jsonl"), key=lambda p: p.name)
            for jsonl_path in jsonl_files:
                discovered.append((group_name, day_name, jsonl_path))

    return discovered


def _build_analytics_day_dir(config: AppConfig, group_name: str, day_name: str) -> Path:
    return config.output.directory / config.analytics.output_subdir / group_name / day_name


def _compute_group_scale_contexts(
    discovered: list[tuple[str, str, Path]],
    window_seconds: int,
) -> dict[str, GroupScaleContext]:
    per_group_max_windows: dict[str, int] = {}
    per_group_min_val: dict[str, float] = {}
    per_group_max_val: dict[str, float] = {}

    for group_name, _day_name, jsonl_path in discovered:
        frame_records = load_frame_data_from_jsonl(jsonl_path)
        if not frame_records:
            continue

        windows = aggregate_into_windows(frame_records, window_seconds)
        if windows:
            max_window_idx = max(windows.keys())
            per_group_max_windows[group_name] = max(
                per_group_max_windows.get(group_name, 0),
                int(max_window_idx) + 1,
            )

        values = [float(record.get("motility_score", 0.0)) for record in frame_records]
        if not values:
            continue

        vmin = min(values)
        vmax = max(values)
        per_group_min_val[group_name] = min(per_group_min_val.get(group_name, vmin), vmin)
        per_group_max_val[group_name] = max(per_group_max_val.get(group_name, vmax), vmax)

    contexts: dict[str, GroupScaleContext] = {}
    for group_name in set(per_group_max_windows) | set(per_group_max_val):
        max_windows = max(1, per_group_max_windows.get(group_name, 1))
        lower = min(0.0, per_group_min_val.get(group_name, 0.0))
        upper_raw = per_group_max_val.get(group_name, 1.0)
        upper = max(upper_raw * 1.05, lower + 1e-6)
        contexts[group_name] = GroupScaleContext(
            max_window_count=max_windows,
            y_limits=(lower, upper),
        )

    return contexts


def _generate_intraday_from_jsonl(
    config: AppConfig,
    group_name: str,
    day_name: str,
    jsonl_path: Path,
    selected_plot_types: set[str],
    include_html: bool,
    scale_context: GroupScaleContext | None,
) -> IntraDayMetrics | None:
    frame_records = load_frame_data_from_jsonl(jsonl_path)
    if not frame_records:
        LOGGER.warning("No frame records in %s", jsonl_path)
        return None

    intraday = compute_intraday_metrics(
        frame_records=frame_records,
        window_seconds=config.analytics.intraday_window_seconds,
        percentiles_list=list(config.analytics.percentiles),
        include_slope=config.analytics.include_trend_slope,
        include_coeff_variation=config.analytics.include_coeff_variation,
        include_outlier_ratio=config.analytics.include_outlier_ratio,
        recording_date=day_name,
        group_id=group_name,
    )

    day_dir = _build_analytics_day_dir(config, group_name, day_name)
    metrics_path = day_dir / "intraday_metrics.json"
    save_intraday_metrics_json(intraday, metrics_path)

    windows_data = aggregate_into_windows(frame_records, config.analytics.intraday_window_seconds)
    descriptive_stats = {idx: asdict(stats) for idx, stats in intraday.windows.items()}
    plots_dir = day_dir / "plots"

    if "intraday_timeseries" in selected_plot_types:
        plot_intraday_timeseries(
            windows_data=windows_data,
            window_duration_seconds=config.analytics.intraday_window_seconds,
            descriptive_stats=descriptive_stats,
            group_id=group_name,
            recording_date=day_name,
            output_dir=plots_dir,
            x_max_hours=(scale_context.max_window_count * config.analytics.intraday_window_seconds / 3600.0) if scale_context else None,
            y_limits=scale_context.y_limits if scale_context else None,
            generate_png=config.analytics.output_formats.get("png", True),
            generate_html=include_html and config.analytics.output_formats.get("html", False),
            annotations_enabled=config.analytics.plot_annotations_enabled,
            annotation_density=config.analytics.annotation_density,
        )

    if "intraday_distribution" in selected_plot_types:
        plot_intraday_distribution(
            windows_data=windows_data,
            window_duration_seconds=config.analytics.intraday_window_seconds,
            group_id=group_name,
            recording_date=day_name,
            output_dir=plots_dir,
            x_max_windows=scale_context.max_window_count if scale_context else None,
            y_limits=scale_context.y_limits if scale_context else None,
            generate_png=config.analytics.output_formats.get("png", True),
            generate_html=include_html and config.analytics.output_formats.get("html", False),
            annotations_enabled=config.analytics.plot_annotations_enabled,
            annotation_density=config.analytics.annotation_density,
        )

    if "spatial_heatmap" in selected_plot_types:
        spatial_grids = aggregate_spatial_grids_into_windows(
            frame_records=frame_records,
            window_seconds=config.analytics.intraday_window_seconds,
            grid_size=int(config.processing.get("spatial_grid_size", 16)),
        )
        if spatial_grids:
            plot_spatial_heatmap(
                spatial_grids=spatial_grids,
                window_duration_seconds=config.analytics.intraday_window_seconds,
                group_id=group_name,
                recording_date=day_name,
                output_dir=plots_dir,
                grid_size=int(config.processing.get("spatial_grid_size", 16)),
                generate_png=config.analytics.output_formats.get("png", True),
                annotations_enabled=config.analytics.plot_annotations_enabled,
                annotation_density=config.analytics.annotation_density,
            )
        else:
            LOGGER.info("No spatial grid data available for %s", jsonl_path)

    return intraday


def _generate_interday(
    config: AppConfig,
    selected_plot_types: set[str],
    grouped_intraday: dict[str, list[IntraDayMetrics]],
    include_html: bool,
) -> None:
    if not grouped_intraday:
        return

    for group_name, daily_metrics in grouped_intraday.items():
        if not daily_metrics:
            continue

        interday = compute_interday_metrics(
            daily_metrics_list=daily_metrics,
            primary_metric=config.analytics.primary_metric,
        )

        interday_dir = config.output.directory / config.analytics.output_subdir / group_name / "interday"
        metrics_path = interday_dir / "interday_metrics.json"
        save_interday_metrics_json(interday, metrics_path)

        plots_dir = interday_dir / "plots"
        if "interday_trend" in selected_plot_types:
            plot_interday_trend(
                daily_summaries=interday.daily_summaries,
                group_id=group_name,
                output_dir=plots_dir,
                generate_png=config.analytics.output_formats.get("png", True),
                generate_html=include_html and config.analytics.output_formats.get("html", False),
                trend_data=interday.trend_data,
                annotations_enabled=config.analytics.plot_annotations_enabled,
                annotation_density=config.analytics.annotation_density,
            )
        if "interday_delta" in selected_plot_types:
            plot_interday_delta(
                daily_summaries=interday.daily_summaries,
                group_id=group_name,
                output_dir=plots_dir,
                generate_png=config.analytics.output_formats.get("png", True),
                generate_html=include_html and config.analytics.output_formats.get("html", False),
                trend_data=interday.trend_data,
                annotations_enabled=config.analytics.plot_annotations_enabled,
                annotation_density=config.analytics.annotation_density,
            )

        if config.analytics.summary_enabled:
            generate_quick_summary_files(
                interday_metrics=interday,
                output_dir=config.output.directory / config.analytics.output_subdir / group_name,
                formats=config.analytics.summary_formats,
            )


def _default_plot_types() -> set[str]:
    return {
        "intraday_timeseries",
        "intraday_distribution",
        "spatial_heatmap",
        "interday_trend",
        "interday_delta",
    }


def run_replot(
    config: AppConfig,
    groups: list[str],
    dates: list[str],
    plot_types: list[str],
    include_html: bool,
    regenerate_collages: bool,
    collage_output_dirname: str,
    collage_max_height_px: int,
    collage_heatmap_layout: str,
) -> dict[str, Any]:
    selected_plot_types = set(plot_types) if plot_types else _default_plot_types()
    group_filter = set(groups)
    date_filter = set(dates)

    results_root = config.output.directory / "results"
    discovered = _discover_result_jsonl(results_root, group_filter, date_filter)
    scale_contexts = _compute_group_scale_contexts(
        discovered=discovered,
        window_seconds=config.analytics.intraday_window_seconds,
    )

    grouped_intraday: dict[str, list[IntraDayMetrics]] = {}
    intraday_processed = 0

    for group_name, day_name, jsonl_path in discovered:
        intraday = _generate_intraday_from_jsonl(
            config=config,
            group_name=group_name,
            day_name=day_name,
            jsonl_path=jsonl_path,
            selected_plot_types=selected_plot_types,
            include_html=include_html,
            scale_context=scale_contexts.get(group_name),
        )
        if intraday is None:
            continue

        intraday_processed += 1
        grouped_intraday.setdefault(group_name, []).append(intraday)

    if {"interday_trend", "interday_delta"} & selected_plot_types:
        _generate_interday(
            config=config,
            selected_plot_types=selected_plot_types,
            grouped_intraday=grouped_intraday,
            include_html=include_html,
        )

    intraday_collage_types = [
        plot_type
        for plot_type in ("intraday_distribution", "intraday_timeseries", "spatial_heatmap")
        if plot_type in selected_plot_types
    ]
    collage_written = 0
    if regenerate_collages and intraday_collage_types and grouped_intraday:
        analytics_root = config.output.directory / config.analytics.output_subdir
        collage_files = build_intraday_plot_collages(
            analytics_root=analytics_root,
            output_dirname=collage_output_dirname,
            groups=sorted(grouped_intraday.keys()),
            plot_types=intraday_collage_types,
            max_height_px=collage_max_height_px,
            heatmap_layout=collage_heatmap_layout,
            generate_png=True,
            generate_pdf=True,
        )
        collage_written = len(collage_files)

    return {
        "discovered_jsonl": len(discovered),
        "intraday_processed": intraday_processed,
        "groups": sorted(grouped_intraday.keys()),
        "collage_written": collage_written,
    }


def run_from_cli(args: argparse.Namespace) -> int:
    config = getattr(args, "_loaded_config", None)
    if config is None:
        config = load_config(args.config)

    groups = _parse_csv(args.groups or "")
    dates = _parse_csv(args.dates or "")
    plot_types = _parse_csv(args.plot_types or "")

    summary = run_replot(
        config=config,
        groups=groups,
        dates=dates,
        plot_types=plot_types,
        include_html=bool(getattr(args, "include_html", False)),
        regenerate_collages=not bool(getattr(args, "skip_collages", False)),
        collage_output_dirname=str(getattr(args, "collage_output_dirname", "composites")),
        collage_max_height_px=int(getattr(args, "collage_max_height_px", 20000)),
        collage_heatmap_layout=str(getattr(args, "collage_heatmap_layout", "auto")),
    )

    LOGGER.info(
        "Replot completed: discovered_jsonl=%d, intraday_processed=%d, collage_written=%d, groups=%s",
        summary["discovered_jsonl"],
        summary["intraday_processed"],
        summary["collage_written"],
        ",".join(summary["groups"]) if summary["groups"] else "none",
    )

    return 0 if summary["intraday_processed"] > 0 else 1
