from __future__ import annotations

from typing import Any


def process_frame(
    *,
    frame_index: int,
    timestamp_seconds: float,
    segment_index: int,
    processing_config: dict[str, Any],
) -> dict[str, Any]:
    label = str(processing_config.get("placeholder_label", "unclassified"))
    confidence = float(processing_config.get("placeholder_confidence", 0.0))

    return {
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp_seconds, 3),
        "segment_index": segment_index,
        "label": label,
        "confidence": confidence,
    }


def process_segment(
    *,
    segment_index: int,
    segment_start_seconds: float,
    segment_end_seconds: float,
    sampled_frames: int,
    processing_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "segment_index": segment_index,
        "segment_start_seconds": round(segment_start_seconds, 3),
        "segment_end_seconds": round(segment_end_seconds, 3),
        "sampled_frames": sampled_frames,
        "label": str(processing_config.get("placeholder_label", "unclassified")),
    }
