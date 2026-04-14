from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

from .config_loader import InputConfig


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


def discover_videos(input_config: InputConfig) -> list[Path]:
    source = input_config.path
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if source.is_file():
        extension = source.suffix.lower()
        return [source] if extension in input_config.allowed_extensions else []

    pattern = "**/*" if input_config.recursive else "*"
    candidates: list[Path] = []
    for path in source.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() in input_config.allowed_extensions:
            candidates.append(path)

    return sorted(candidates)


def get_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_seconds = (frame_count / fps) if frame_count > 0 and fps > 0 else 0.0
        return VideoMetadata(frame_count=frame_count, fps=fps, duration_seconds=duration_seconds)
    finally:
        capture.release()


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
