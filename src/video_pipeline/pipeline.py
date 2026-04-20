from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpointing import (
    CHECKPOINT_VERSION,
    build_checkpoint_path,
    build_config_snapshot,
    load_checkpoint,
    running_stats_from_dict,
    running_stats_to_dict,
    save_checkpoint,
    snapshots_match,
)
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
from .config_loader import AppConfig
from .plotting import (
    plot_interday_delta,
    plot_interday_trend,
    plot_intraday_distribution,
    plot_intraday_timeseries,
    plot_spatial_heatmap,
)
from .processor import (
    RunningStats,
    compute_motility,
    process_frame,
    process_segment,
    summarize_running_stats,
    update_running_stats,
)
from .video_io import discover_videos, get_video_metadata, iter_video_frames

LOGGER = logging.getLogger(__name__)


@dataclass
class SegmentState:
    segment_index: int
    sampled_frames: int
    motility_stats: RunningStats


class Sampler:
    def __init__(self, every_n_frames: int | None, every_n_seconds: float | None) -> None:
        self._every_n_frames = every_n_frames
        self._every_n_seconds = every_n_seconds
        self._next_sample_at_seconds = 0.0

    def should_sample(self, frame_index: int, timestamp_seconds: float) -> bool:
        if self._every_n_frames is not None:
            return frame_index % self._every_n_frames == 0

        if self._every_n_seconds is not None:
            if timestamp_seconds >= self._next_sample_at_seconds:
                self._next_sample_at_seconds += self._every_n_seconds
                return True
            return False

        return False


def _sanitize_path_component(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback

    stripped = value.strip()
    if not stripped:
        return fallback

    return stripped.replace("/", "_").replace("\\", "_").replace(":", "-")


def _build_video_output_path(config: AppConfig, video_path: Path, group_id: str | None, recording_date: str | None) -> Path:
    group_key = _sanitize_path_component(group_id, "ungrouped")
    date_key = _sanitize_path_component(recording_date, "undated")
    return config.output.directory / "results" / group_key / date_key / f"{video_path.stem}_results.jsonl"


def _build_analytics_base_dir(config: AppConfig, group_id: str | None, recording_date: str | None) -> Path:
    group_key = _sanitize_path_component(group_id, "ungrouped")
    date_key = _sanitize_path_component(recording_date, "undated")
    return config.output.directory / config.analytics.output_subdir / group_key / date_key


def _segment_state_to_payload(segment_state: SegmentState) -> dict[str, Any]:
    return {
        "segment_index": int(segment_state.segment_index),
        "sampled_frames": int(segment_state.sampled_frames),
        "motility_stats": running_stats_to_dict(segment_state.motility_stats),
    }


def _segment_state_from_payload(payload: dict[str, Any]) -> SegmentState:
    return SegmentState(
        segment_index=int(payload.get("segment_index", 0)),
        sampled_frames=int(payload.get("sampled_frames", 0)),
        motility_stats=running_stats_from_dict(dict(payload.get("motility_stats", {}))),
    )


def _restore_previous_sampled_frame(video_path: Path, target_frame_index: int) -> object | None:
    if target_frame_index < 0:
        return None

    for frame_record in iter_video_frames(video_path):
        if frame_record.frame_index == target_frame_index:
            return frame_record.frame
        if frame_record.frame_index > target_frame_index:
            break
    return None


def _truncate_output_for_resume(output_file: Path, expected_size: int, strict_resume: bool) -> None:
    if not output_file.exists():
        if strict_resume:
            raise FileNotFoundError(
                f"Checkpoint expects output file for resume, but it does not exist: {output_file}"
            )
        return

    current_size = output_file.stat().st_size
    if current_size < expected_size:
        if strict_resume:
            raise RuntimeError(
                f"Output file {output_file} is smaller than checkpointed size "
                f"({current_size} < {expected_size})."
            )
        LOGGER.warning(
            "Output file %s is smaller than checkpointed size (%d < %d); continuing due to non-strict resume.",
            output_file,
            current_size,
            expected_size,
        )
        return

    if current_size > expected_size:
        LOGGER.info(
            "Truncating %s from %d bytes to checkpointed size %d bytes before resume.",
            output_file,
            current_size,
            expected_size,
        )
        with output_file.open("r+b") as handle:
            handle.truncate(expected_size)


def _save_running_checkpoint(
    *,
    checkpoint_path: Path,
    config_snapshot: dict[str, Any],
    video_path: Path,
    output_file: Path,
    group_id: str | None,
    recording_date: str | None,
    sampled_total: int,
    last_frame_processed: int,
    last_timestamp_processed: float,
    last_written_frame_index: int | None,
    previous_sampled_frame_index: int | None,
    current_segment: SegmentState,
    video_stats: RunningStats,
    segment_reports: list[dict[str, Any]],
    sampler: Sampler,
) -> None:
    save_checkpoint(
        checkpoint_path,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "status": "running",
            "video_path": str(video_path),
            "group_id": group_id,
            "recording_date": recording_date,
            "output_jsonl": str(output_file),
            "sampled_total": sampled_total,
            "last_frame_processed": last_frame_processed,
            "last_timestamp_processed": round(last_timestamp_processed, 6),
            "last_written_frame_index": last_written_frame_index,
            "previous_sampled_frame_index": previous_sampled_frame_index,
            "current_segment": _segment_state_to_payload(current_segment),
            "video_stats": running_stats_to_dict(video_stats),
            "segment_reports": segment_reports,
            "sampler_state": {
                "next_sample_at_seconds": float(getattr(sampler, "_next_sample_at_seconds", 0.0))
            },
            "jsonl_size_bytes": output_file.stat().st_size if output_file.exists() else 0,
            "config_snapshot": config_snapshot,
        },
    )


def _save_completed_checkpoint(
    *,
    checkpoint_path: Path,
    config_snapshot: dict[str, Any],
    video_path: Path,
    output_file: Path,
    group_id: str | None,
    recording_date: str | None,
    video_report: dict[str, Any],
) -> None:
    save_checkpoint(
        checkpoint_path,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "status": "video_completed",
            "video_path": str(video_path),
            "group_id": group_id,
            "recording_date": recording_date,
            "output_jsonl": str(output_file),
            "jsonl_size_bytes": output_file.stat().st_size if output_file.exists() else 0,
            "config_snapshot": config_snapshot,
            "video_report": video_report,
        },
    )


def _generate_intraday_artifacts(
    config: AppConfig,
    report: dict[str, Any],
) -> IntraDayMetrics | None:
    jsonl_path = Path(report["output_jsonl"])
    frame_records = load_frame_data_from_jsonl(jsonl_path)
    if not frame_records:
        LOGGER.warning("Skipping analytics for %s: no frame-level motility records.", jsonl_path)
        return None

    group_id = report.get("group_id")
    recording_date = report.get("recording_date")
    intraday = compute_intraday_metrics(
        frame_records=frame_records,
        window_seconds=config.analytics.intraday_window_seconds,
        percentiles_list=list(config.analytics.percentiles),
        include_slope=config.analytics.include_trend_slope,
        recording_date=recording_date,
        group_id=group_id,
    )

    analytics_day_dir = _build_analytics_base_dir(config, group_id, recording_date)
    metrics_path = analytics_day_dir / "intraday_metrics.json"
    save_intraday_metrics_json(intraday, metrics_path)

    report["analytics"] = {
        "intraday_metrics": str(metrics_path),
        "plots": [],
    }

    windows_data = aggregate_into_windows(frame_records, config.analytics.intraday_window_seconds)
    descriptive_stats = {idx: asdict(stats) for idx, stats in intraday.windows.items()}
    plots_dir = analytics_day_dir / "plots"

    if config.analytics.plots.get("intraday_timeseries", False):
        png_path, html_path = plot_intraday_timeseries(
            windows_data=windows_data,
            window_duration_seconds=config.analytics.intraday_window_seconds,
            descriptive_stats=descriptive_stats,
            group_id=group_id,
            recording_date=recording_date,
            output_dir=plots_dir,
            generate_png=config.analytics.output_formats.get("png", True),
            generate_html=config.analytics.output_formats.get("html", False),
        )
        if png_path is not None:
            report["analytics"]["plots"].append(str(png_path))
        if html_path is not None:
            report["analytics"]["plots"].append(str(html_path))

    if config.analytics.plots.get("intraday_distribution", False):
        png_path, html_path = plot_intraday_distribution(
            windows_data=windows_data,
            window_duration_seconds=config.analytics.intraday_window_seconds,
            group_id=group_id,
            recording_date=recording_date,
            output_dir=plots_dir,
            generate_png=config.analytics.output_formats.get("png", True),
            generate_html=config.analytics.output_formats.get("html", False),
        )
        if png_path is not None:
            report["analytics"]["plots"].append(str(png_path))
        if html_path is not None:
            report["analytics"]["plots"].append(str(html_path))

    if config.analytics.plots.get("intraday_heatmap", False):
        try:
            spatial_grids = aggregate_spatial_grids_into_windows(
                frame_records=frame_records,
                window_seconds=config.analytics.intraday_window_seconds,
                grid_size=int(config.processing.get("spatial_grid_size", 16)),
            )
            if spatial_grids:
                heatmap_path = plot_spatial_heatmap(
                    spatial_grids=spatial_grids,
                    window_duration_seconds=config.analytics.intraday_window_seconds,
                    group_id=group_id,
                    recording_date=recording_date,
                    output_dir=plots_dir,
                    grid_size=int(config.processing.get("spatial_grid_size", 16)),
                    generate_png=config.analytics.output_formats.get("png", True),
                )
                if heatmap_path is not None:
                    report["analytics"]["plots"].append(str(heatmap_path))
        except Exception as e:
            LOGGER.warning(f"Failed to generate spatial heatmap: {e}")

    return intraday


def _generate_interday_artifacts(config: AppConfig, group_key: str, daily_metrics: list[IntraDayMetrics]) -> dict[str, Any]:
    if not daily_metrics:
        return {}

    interday = compute_interday_metrics(
        daily_metrics_list=daily_metrics,
        primary_metric=config.analytics.primary_metric,
    )
    group_dir = config.output.directory / config.analytics.output_subdir / group_key / "interday"
    interday_metrics_path = group_dir / "interday_metrics.json"
    save_interday_metrics_json(interday, interday_metrics_path)

    generated_plots: list[str] = []
    plots_dir = group_dir / "plots"

    if config.analytics.plots.get("interday_trend", False):
        png_path, html_path = plot_interday_trend(
            daily_summaries=interday.daily_summaries,
            group_id=group_key,
            output_dir=plots_dir,
            generate_png=config.analytics.output_formats.get("png", True),
            generate_html=config.analytics.output_formats.get("html", False),
        )
        if png_path is not None:
            generated_plots.append(str(png_path))
        if html_path is not None:
            generated_plots.append(str(html_path))

    if config.analytics.plots.get("interday_delta", False):
        png_path, html_path = plot_interday_delta(
            daily_summaries=interday.daily_summaries,
            group_id=group_key,
            output_dir=plots_dir,
            generate_png=config.analytics.output_formats.get("png", True),
            generate_html=config.analytics.output_formats.get("html", False),
        )
        if png_path is not None:
            generated_plots.append(str(png_path))
        if html_path is not None:
            generated_plots.append(str(html_path))

    return {
        "interday_metrics": str(interday_metrics_path),
        "plots": generated_plots,
    }


def _run_analytics(config: AppConfig, reports: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_daily_metrics: dict[str, list[IntraDayMetrics]] = defaultdict(list)

    for report in reports:
        intraday = _generate_intraday_artifacts(config=config, report=report)
        if intraday is None:
            continue
        group_key = _sanitize_path_component(intraday.group_id, "ungrouped")
        grouped_daily_metrics[group_key].append(intraday)

    summary: dict[str, Any] = {"groups": {}}
    for group_key, daily_metrics in grouped_daily_metrics.items():
        summary["groups"][group_key] = _generate_interday_artifacts(
            config=config,
            group_key=group_key,
            daily_metrics=daily_metrics,
        )

    return summary


def _segment_index_for_timestamp(config: AppConfig, timestamp_seconds: float) -> int:
    if not config.segmentation.enabled:
        return 0
    return int(timestamp_seconds // config.segmentation.segment_duration_seconds)


def _flush_segment(
    handle,
    video_path: Path,
    segment_state: SegmentState,
    segment_duration_seconds: int,
) -> dict[str, Any]:
    start_seconds = segment_state.segment_index * segment_duration_seconds
    end_seconds = start_seconds + segment_duration_seconds
    motility_summary = summarize_running_stats(segment_state.motility_stats)
    record = {
        "record_type": "segment_summary",
        "video_name": video_path.name,
        "result": process_segment(
            segment_index=segment_state.segment_index,
            segment_start_seconds=start_seconds,
            segment_end_seconds=end_seconds,
            sampled_frames=segment_state.sampled_frames,
            motility_summary=motility_summary,
        ),
    }
    handle.write(json.dumps(record) + "\n")
    return {
        "segment_index": segment_state.segment_index,
        "segment_start_seconds": round(start_seconds, 3),
        "segment_end_seconds": round(end_seconds, 3),
        "sampled_frames": segment_state.sampled_frames,
        "motility": motility_summary,
    }


def _process_video(video_path: Path, output_file: Path, config: AppConfig, group_id: str | None = None, recording_date: str | None = None) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = get_video_metadata(video_path, group_id=group_id, recording_date=recording_date)
    config_snapshot = build_config_snapshot(config)

    checkpoint_path = build_checkpoint_path(
        base_dir=config.checkpoint.directory,
        video_path=video_path,
        group_id=group_id,
        recording_date=recording_date,
    )
    checkpoint_payload = None
    if config.checkpoint.enabled:
        checkpoint_payload = load_checkpoint(checkpoint_path)

    sampler = Sampler(
        every_n_frames=config.sampling.every_n_frames,
        every_n_seconds=config.sampling.every_n_seconds,
    )

    segment_duration = config.segmentation.segment_duration_seconds
    current_segment = SegmentState(segment_index=0, sampled_frames=0, motility_stats=RunningStats())
    sampled_total = 0
    previous_sampled_frame = None
    previous_sampled_frame_index: int | None = None
    segment_reports: list[dict[str, Any]] = []
    video_stats = RunningStats()
    last_frame_processed = -1
    last_timestamp_processed = 0.0
    last_written_frame_index: int | None = None
    should_resume = False

    if checkpoint_payload is not None and checkpoint_payload.get("status") == "running":
        saved_snapshot = checkpoint_payload.get("config_snapshot", {})
        if config.checkpoint.validate_config_snapshot and not snapshots_match(config_snapshot, dict(saved_snapshot)):
            message = (
                f"Checkpoint config mismatch for {video_path}. "
                "Either clear checkpoints or restore the original processing/sampling settings."
            )
            if config.checkpoint.strict_resume:
                raise RuntimeError(message)
            LOGGER.warning(message)
        else:
            expected_output = checkpoint_payload.get("output_jsonl")
            if isinstance(expected_output, str) and expected_output and Path(expected_output).resolve() != output_file.resolve():
                raise RuntimeError(
                    f"Checkpoint output mismatch for {video_path}: {expected_output} != {output_file}"
                )

            expected_size = int(checkpoint_payload.get("jsonl_size_bytes", 0))
            _truncate_output_for_resume(output_file, expected_size, config.checkpoint.strict_resume)

            sampled_total = int(checkpoint_payload.get("sampled_total", 0))
            last_frame_processed = int(checkpoint_payload.get("last_frame_processed", -1))
            last_timestamp_processed = float(checkpoint_payload.get("last_timestamp_processed", 0.0))
            last_written_frame_index = checkpoint_payload.get("last_written_frame_index")

            current_segment = _segment_state_from_payload(dict(checkpoint_payload.get("current_segment", {})))
            video_stats = running_stats_from_dict(dict(checkpoint_payload.get("video_stats", {})))

            restored_reports = checkpoint_payload.get("segment_reports", [])
            if isinstance(restored_reports, list):
                segment_reports = restored_reports

            sampler_state = checkpoint_payload.get("sampler_state", {})
            if isinstance(sampler_state, dict) and config.sampling.every_n_seconds is not None:
                sampler._next_sample_at_seconds = float(sampler_state.get("next_sample_at_seconds", 0.0))

            restored_prev_index = checkpoint_payload.get("previous_sampled_frame_index")
            if isinstance(restored_prev_index, int):
                previous_sampled_frame_index = restored_prev_index
                previous_sampled_frame = _restore_previous_sampled_frame(video_path, restored_prev_index)
                if previous_sampled_frame is None and config.checkpoint.strict_resume:
                    raise RuntimeError(
                        f"Unable to restore previous sampled frame {restored_prev_index} for {video_path}."
                    )

            should_resume = True
            LOGGER.info(
                "Resuming %s from frame %d (sampled_total=%d).",
                video_path.name,
                last_frame_processed,
                sampled_total,
            )

    diff_threshold = int(config.processing["diff_threshold"])
    blur_kernel_size = int(config.processing["blur_kernel_size"])
    active_motion_threshold = float(config.processing["active_motion_threshold"])
    compute_spatial_grid = bool(config.processing.get("compute_spatial_grid", False))
    spatial_grid_size = int(config.processing.get("spatial_grid_size", 16))
    progress_interval_seconds = float(config.logging.progress_update_seconds)

    start_time = time.monotonic()
    last_progress_log_time = start_time
    frames_processed = 0
    frames_since_checkpoint = 0

    LOGGER.info(
        "Processing video: %s | frames=%s | fps=%.3f | duration=%.1fs",
        video_path,
        metadata.frame_count if metadata.frame_count > 0 else "unknown",
        metadata.fps,
        metadata.duration_seconds,
    )

    def log_progress(force: bool = False) -> None:
        nonlocal last_progress_log_time

        now = time.monotonic()
        if not force and (now - last_progress_log_time) < progress_interval_seconds:
            return

        elapsed_seconds = max(now - start_time, 0.001)
        progress_ratio = (frames_processed / metadata.frame_count) if metadata.frame_count > 0 else None

        if progress_ratio is not None and progress_ratio > 0:
            estimated_total_seconds = elapsed_seconds / progress_ratio
            eta_seconds = max(estimated_total_seconds - elapsed_seconds, 0.0)
            percent_text = f"{progress_ratio * 100:.1f}%"
            eta_text = f"ETA {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}"
            frame_text = f"{frames_processed}/{metadata.frame_count}"
        else:
            percent_text = "n/a"
            eta_text = "ETA n/a"
            frame_text = str(frames_processed)

        LOGGER.info(
            "Progress %s | frames=%s | sampled=%d | elapsed=%s | %s | %s",
            video_path.name,
            frame_text,
            sampled_total,
            time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
            percent_text,
            eta_text,
        )

        last_progress_log_time = now

    file_mode = "a" if should_resume else "w"
    with output_file.open(file_mode, encoding="utf-8") as handle:
        for frame_record in iter_video_frames(video_path):
            frames_processed += 1

            if frame_record.frame_index <= last_frame_processed:
                continue

            segment_index = _segment_index_for_timestamp(config, frame_record.timestamp_seconds)

            if segment_index != current_segment.segment_index:
                segment_reports.append(
                    _flush_segment(
                        handle=handle,
                        video_path=video_path,
                        segment_state=current_segment,
                        segment_duration_seconds=segment_duration,
                    )
                )
                current_segment = SegmentState(
                    segment_index=segment_index,
                    sampled_frames=0,
                    motility_stats=RunningStats(),
                )
                previous_sampled_frame = None
                previous_sampled_frame_index = None

                if config.checkpoint.enabled:
                    _save_running_checkpoint(
                        checkpoint_path=checkpoint_path,
                        config_snapshot=config_snapshot,
                        video_path=video_path,
                        output_file=output_file,
                        group_id=group_id,
                        recording_date=recording_date,
                        sampled_total=sampled_total,
                        last_frame_processed=last_frame_processed,
                        last_timestamp_processed=last_timestamp_processed,
                        last_written_frame_index=last_written_frame_index,
                        previous_sampled_frame_index=previous_sampled_frame_index,
                        current_segment=current_segment,
                        video_stats=video_stats,
                        segment_reports=segment_reports,
                        sampler=sampler,
                    )
                    frames_since_checkpoint = 0

            if sampler.should_sample(frame_record.frame_index, frame_record.timestamp_seconds):
                motility = None
                if previous_sampled_frame is not None:
                    motility = compute_motility(
                        previous_sampled_frame,
                        frame_record.frame,
                        diff_threshold=diff_threshold,
                        blur_kernel_size=blur_kernel_size,
                        compute_spatial_grid=compute_spatial_grid,
                        spatial_grid_size=spatial_grid_size,
                    )
                    update_running_stats(
                        current_segment.motility_stats,
                        motility["motility_score"],
                        active_motion_threshold,
                    )
                    update_running_stats(
                        video_stats,
                        motility["motility_score"],
                        active_motion_threshold,
                    )

                frame_result = process_frame(
                    frame_index=frame_record.frame_index,
                    timestamp_seconds=frame_record.timestamp_seconds,
                    segment_index=segment_index,
                    motility=motility,
                )
                handle.write(
                    json.dumps(
                        {
                            "record_type": "frame_result",
                            "video_name": video_path.name,
                            "result": frame_result,
                        }
                    )
                    + "\n"
                )
                current_segment.sampled_frames += 1
                sampled_total += 1
                previous_sampled_frame = frame_record.frame
                previous_sampled_frame_index = frame_record.frame_index
                last_written_frame_index = frame_record.frame_index

            last_frame_processed = frame_record.frame_index
            last_timestamp_processed = frame_record.timestamp_seconds

            if config.checkpoint.enabled and config.checkpoint.save_every_frames > 0:
                frames_since_checkpoint += 1
                if frames_since_checkpoint >= config.checkpoint.save_every_frames:
                    _save_running_checkpoint(
                        checkpoint_path=checkpoint_path,
                        config_snapshot=config_snapshot,
                        video_path=video_path,
                        output_file=output_file,
                        group_id=group_id,
                        recording_date=recording_date,
                        sampled_total=sampled_total,
                        last_frame_processed=last_frame_processed,
                        last_timestamp_processed=last_timestamp_processed,
                        last_written_frame_index=last_written_frame_index,
                        previous_sampled_frame_index=previous_sampled_frame_index,
                        current_segment=current_segment,
                        video_stats=video_stats,
                        segment_reports=segment_reports,
                        sampler=sampler,
                    )
                    frames_since_checkpoint = 0

            log_progress()

        segment_reports.append(
            _flush_segment(
                handle=handle,
                video_path=video_path,
                segment_state=current_segment,
                segment_duration_seconds=segment_duration,
            )
        )

    log_progress(force=True)

    LOGGER.info("Completed %s | sampled frames: %d | output: %s", video_path.name, sampled_total, output_file)
    report = {
        "video_name": video_path.name,
        "input_path": str(video_path),
        "output_jsonl": str(output_file),
        "group_id": group_id,
        "recording_date": recording_date,
        "sampled_frames": sampled_total,
        "segments": segment_reports,
        "motility": summarize_running_stats(video_stats),
    }

    if config.checkpoint.enabled:
        _save_completed_checkpoint(
            checkpoint_path=checkpoint_path,
            config_snapshot=config_snapshot,
            video_path=video_path,
            output_file=output_file,
            group_id=group_id,
            recording_date=recording_date,
            video_report=report,
        )

    return report


def run_pipeline(config: AppConfig) -> None:
    videos = discover_videos(config.input, config.hierarchy)
    if not videos:
        LOGGER.warning("No videos found under configured input path: %s", config.input.path)
        return

    LOGGER.info("Discovered %d video(s).", len(videos))

    reports: list[dict[str, Any]] = []
    for video_path, group_id, recording_date in videos:
        output_file = _build_video_output_path(
            config=config,
            video_path=video_path,
            group_id=group_id,
            recording_date=recording_date,
        )

        if config.checkpoint.enabled:
            checkpoint_path = build_checkpoint_path(
                base_dir=config.checkpoint.directory,
                video_path=video_path,
                group_id=group_id,
                recording_date=recording_date,
            )
            checkpoint_payload = load_checkpoint(checkpoint_path)
            if checkpoint_payload is not None and checkpoint_payload.get("status") == "video_completed":
                output_jsonl = checkpoint_payload.get("output_jsonl")
                if isinstance(output_jsonl, str) and Path(output_jsonl).exists():
                    saved_report = checkpoint_payload.get("video_report")
                    if isinstance(saved_report, dict):
                        LOGGER.info("Skipping already completed video %s due to checkpoint.", video_path.name)
                        reports.append(saved_report)
                        continue

        report = _process_video(video_path=video_path, output_file=output_file, config=config, group_id=group_id, recording_date=recording_date)
        reports.append(report)

    report_payload: dict[str, Any] = {"videos": reports}
    if config.analytics.enabled:
        LOGGER.info("Analytics enabled: generating metrics and plots.")
        report_payload["analytics"] = _run_analytics(config=config, reports=reports)

    report_file = config.output.directory / "motility_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    LOGGER.info("Motility report written to: %s", report_file)
