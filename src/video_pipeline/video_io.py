from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from datetime import datetime

import cv2

from .config_loader import InputConfig, HierarchyConfig


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp_seconds: float
    frame: object


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float
    duration_seconds: float
    group_id: str | None = None  # e.g., "GroupA"
    recording_date: str | None = None  # e.g., "2026-04-15" (ISO format)


def discover_videos(input_config: InputConfig, hierarchy_config: HierarchyConfig | None = None) -> list[tuple[Path, str | None, str | None]]:
    """
    Discover video files and optionally extract hierarchy metadata.
    
    Returns: list of (video_path, group_id, recording_date) tuples.
             If hierarchy is disabled, group_id and recording_date will be None.
    """
    source = input_config.path
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    # Discover raw video files
    if source.is_file():
        extension = source.suffix.lower()
        candidates = [source] if extension in input_config.allowed_extensions else []
    else:
        pattern = "**/*" if input_config.recursive else "*"
        candidates: list[Path] = []
        for path in source.glob(pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() in input_config.allowed_extensions:
                candidates.append(path)

    candidates = sorted(candidates)

    # Extract hierarchy metadata and validate no mixed groups if required
    results = []
    seen_groups = set()

    for video_path in candidates:
        group_id, recording_date = None, None
        
        if hierarchy_config is not None:
            group_id, recording_date = _parse_hierarchy_metadata(video_path, hierarchy_config)
            
            if group_id is not None:
                seen_groups.add(group_id)
                if hierarchy_config.fail_on_mixed_groups and len(seen_groups) > 1:
                    msg = f"Multiple groups detected: {seen_groups}. fail_on_mixed_groups is enabled."
                    raise ValueError(msg)
        
        results.append((video_path, group_id, recording_date))

    return results


def get_video_metadata(video_path: Path, group_id: str | None = None, recording_date: str | None = None) -> VideoMetadata:
    """
    Extract video metadata (frame count, fps, duration) from file.
    Optionally includes pre-computed group_id and recording_date from hierarchy parsing.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_seconds = (frame_count / fps) if frame_count > 0 and fps > 0 else 0.0
        return VideoMetadata(
            frame_count=frame_count,
            fps=fps,
            duration_seconds=duration_seconds,
            group_id=group_id,
            recording_date=recording_date,
        )
    finally:
        capture.release()


def _parse_hierarchy_metadata(video_path: Path, hierarchy_config: HierarchyConfig) -> tuple[str | None, str | None]:
    """
    Extract group_id and recording_date from video path based on hierarchy config.
    
    Returns: (group_id, recording_date) tuple. Both may be None if hierarchy is disabled.
    """
    if not hierarchy_config.enabled:
        return None, None

    # Get relative path parts from the video file up to the expected hierarchy depth
    parts = video_path.parent.parts
    num_levels = len(hierarchy_config.levels)

    if len(parts) < num_levels:
        msg = f"Video path {video_path} does not have enough directory levels for hierarchy {num_levels}."
        raise ValueError(msg)

    # Extract last num_levels parts of the path (representing the hierarchy)
    hierarchy_parts = parts[-num_levels:]

    group_id = None
    recording_date = None

    # Match each level to its definition and validate/extract value
    for i, level_def in enumerate(hierarchy_config.levels):
        folder_name = hierarchy_parts[i]

        if level_def.type == "string":
            if level_def.name == "group":
                group_id = folder_name
        elif level_def.type == "date":
            if level_def.name == "day":
                # Validate date format (simple check for YYYY-MM-DD pattern)
                try:
                    datetime.strptime(folder_name, "%Y-%m-%d")
                    recording_date = folder_name
                except ValueError:
                    if hierarchy_config.fail_on_invalid_date:
                        msg = f"Invalid date format '{folder_name}' in path {video_path}. Expected YYYY-MM-DD."
                        raise ValueError(msg)

    return group_id, recording_date


def iter_video_frames(video_path: Path) -> Iterator[FrameRecord]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fallback_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fallback_fps if fallback_fps and fallback_fps > 0 else 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms > 0:
                timestamp_seconds = timestamp_ms / 1000.0
            elif fps > 0:
                timestamp_seconds = frame_index / fps
            else:
                timestamp_seconds = 0.0

            yield FrameRecord(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                frame=frame,
            )
    finally:
        capture.release()
