from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2


@dataclass
class RunningStats:
    count: int = 0
    sum_value: float = 0.0
    sum_square: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    active_count: int = 0


def update_running_stats(stats: RunningStats, value: float, active_threshold: float) -> None:
    stats.count += 1
    stats.sum_value += value
    stats.sum_square += value * value
    stats.min_value = value if stats.min_value is None else min(stats.min_value, value)
    stats.max_value = value if stats.max_value is None else max(stats.max_value, value)
    if value >= active_threshold:
        stats.active_count += 1


def summarize_running_stats(stats: RunningStats) -> dict[str, float | int]:
    if stats.count == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "active_ratio": 0.0,
        }

    mean = stats.sum_value / stats.count
    variance = max((stats.sum_square / stats.count) - (mean * mean), 0.0)
    std = math.sqrt(variance)
    return {
        "count": stats.count,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(stats.min_value if stats.min_value is not None else 0.0, 6),
        "max": round(stats.max_value if stats.max_value is not None else 0.0, 6),
        "active_ratio": round(stats.active_count / stats.count, 6),
    }


def compute_motility(
    previous_frame: object,
    current_frame: object,
    *,
    diff_threshold: int,
    blur_kernel_size: int,
) -> dict[str, float]:
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if blur_kernel_size > 1:
        kernel = (blur_kernel_size, blur_kernel_size)
        prev_gray = cv2.GaussianBlur(prev_gray, kernel, 0)
        curr_gray = cv2.GaussianBlur(curr_gray, kernel, 0)

    diff = cv2.absdiff(curr_gray, prev_gray)
    _, thresholded = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    total_pixels = thresholded.size
    changed_pixels = int(cv2.countNonZero(thresholded))
    active_pixel_ratio = (changed_pixels / total_pixels) if total_pixels else 0.0
    mean_diff_intensity = float(diff.mean()) / 255.0 if total_pixels else 0.0

    return {
        "motility_score": round(active_pixel_ratio, 6),
        "active_pixel_ratio": round(active_pixel_ratio, 6),
        "mean_diff_intensity": round(mean_diff_intensity, 6),
    }


def process_frame(
    *,
    frame_index: int,
    timestamp_seconds: float,
    segment_index: int,
    motility: dict[str, float] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp_seconds, 3),
        "segment_index": segment_index,
    }
    if motility is not None:
        payload.update(motility)
    return payload


def process_segment(
    *,
    segment_index: int,
    segment_start_seconds: float,
    segment_end_seconds: float,
    sampled_frames: int,
    motility_summary: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "segment_index": segment_index,
        "segment_start_seconds": round(segment_start_seconds, 3),
        "segment_end_seconds": round(segment_end_seconds, 3),
        "sampled_frames": sampled_frames,
        "motility": motility_summary,
    }
