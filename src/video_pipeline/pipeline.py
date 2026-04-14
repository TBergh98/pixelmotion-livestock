from __future__ import annotations

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_loader import AppConfig
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


def _process_video(video_path: Path, output_file: Path, config: AppConfig) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = get_video_metadata(video_path)

    sampler = Sampler(
        every_n_frames=config.sampling.every_n_frames,
        every_n_seconds=config.sampling.every_n_seconds,
    )

    segment_duration = config.segmentation.segment_duration_seconds
    current_segment = SegmentState(segment_index=0, sampled_frames=0, motility_stats=RunningStats())
    sampled_total = 0
    previous_sampled_frame = None
    segment_reports: list[dict[str, Any]] = []
    video_stats = RunningStats()

    diff_threshold = int(config.processing["diff_threshold"])
    blur_kernel_size = int(config.processing["blur_kernel_size"])
    active_motion_threshold = float(config.processing["active_motion_threshold"])
    progress_interval_seconds = float(config.logging.progress_update_seconds)

    start_time = time.monotonic()
    last_progress_log_time = start_time
    frames_processed = 0

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

    with output_file.open("w", encoding="utf-8") as handle:
        for frame_record in iter_video_frames(video_path):
            frames_processed += 1
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

            if sampler.should_sample(frame_record.frame_index, frame_record.timestamp_seconds):
                motility = None
                if previous_sampled_frame is not None:
                    motility = compute_motility(
                        previous_sampled_frame,
                        frame_record.frame,
                        diff_threshold=diff_threshold,
                        blur_kernel_size=blur_kernel_size,
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
    return {
        "video_name": video_path.name,
        "input_path": str(video_path),
        "output_jsonl": str(output_file),
        "sampled_frames": sampled_total,
        "segments": segment_reports,
        "motility": summarize_running_stats(video_stats),
    }


def run_pipeline(config: AppConfig) -> None:
    videos = discover_videos(config.input)
    if not videos:
        LOGGER.warning("No videos found under configured input path: %s", config.input.path)
        return

    LOGGER.info("Discovered %d video(s).", len(videos))

    reports: list[dict[str, Any]] = []
    for video_path in videos:
        output_file = config.output.directory / f"{video_path.stem}_results.jsonl"
        report = _process_video(video_path=video_path, output_file=output_file, config=config)
        reports.append(report)

    report_file = config.output.directory / "motility_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps({"videos": reports}, indent=2), encoding="utf-8")
    LOGGER.info("Motility report written to: %s", report_file)
