from __future__ import annotations

import json
import logging
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
from .video_io import discover_videos, iter_video_frames

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

    LOGGER.info("Processing video: %s", video_path)

    with output_file.open("w", encoding="utf-8") as handle:
        for frame_record in iter_video_frames(video_path):
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

            if not sampler.should_sample(frame_record.frame_index, frame_record.timestamp_seconds):
                continue

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

        segment_reports.append(
            _flush_segment(
                handle=handle,
                video_path=video_path,
                segment_state=current_segment,
                segment_duration_seconds=segment_duration,
            )
        )

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
