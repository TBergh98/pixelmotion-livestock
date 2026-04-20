from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config_loader import AppConfig
from .processor import RunningStats

CHECKPOINT_VERSION = 1


def sanitize_component(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    stripped = value.strip()
    if not stripped:
        return fallback
    return stripped.replace("/", "_").replace("\\", "_").replace(":", "-")


def build_checkpoint_path(base_dir: Path, video_path: Path, group_id: str | None, recording_date: str | None) -> Path:
    group_key = sanitize_component(group_id, "ungrouped")
    date_key = sanitize_component(recording_date, "undated")
    digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:10]
    file_name = f"{video_path.stem}_{digest}.checkpoint.json"
    return base_dir / group_key / date_key / file_name


def running_stats_to_dict(stats: RunningStats) -> dict[str, Any]:
    return {
        "count": int(stats.count),
        "sum_value": float(stats.sum_value),
        "sum_square": float(stats.sum_square),
        "min_value": float(stats.min_value) if stats.min_value is not None else None,
        "max_value": float(stats.max_value) if stats.max_value is not None else None,
        "active_count": int(stats.active_count),
    }


def running_stats_from_dict(payload: dict[str, Any]) -> RunningStats:
    return RunningStats(
        count=int(payload.get("count", 0)),
        sum_value=float(payload.get("sum_value", 0.0)),
        sum_square=float(payload.get("sum_square", 0.0)),
        min_value=(float(payload["min_value"]) if payload.get("min_value") is not None else None),
        max_value=(float(payload["max_value"]) if payload.get("max_value") is not None else None),
        active_count=int(payload.get("active_count", 0)),
    )


def build_config_snapshot(config: AppConfig) -> dict[str, Any]:
    return {
        "diff_threshold": int(config.processing["diff_threshold"]),
        "blur_kernel_size": int(config.processing["blur_kernel_size"]),
        "active_motion_threshold": float(config.processing["active_motion_threshold"]),
        "compute_spatial_grid": bool(config.processing.get("compute_spatial_grid", False)),
        "spatial_grid_size": int(config.processing.get("spatial_grid_size", 16)),
        "segment_duration_seconds": int(config.segmentation.segment_duration_seconds),
        "sampling_every_n_frames": config.sampling.every_n_frames,
        "sampling_every_n_seconds": config.sampling.every_n_seconds,
    }


def snapshots_match(current: dict[str, Any], saved: dict[str, Any]) -> bool:
    return current == saved


def save_checkpoint(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None

    with checkpoint_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")

    version = int(payload.get("checkpoint_version", 0))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version} in {checkpoint_path}. "
            f"Expected version {CHECKPOINT_VERSION}."
        )

    return payload
