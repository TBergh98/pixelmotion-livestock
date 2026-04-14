from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputConfig:
    path: Path
    recursive: bool
    allowed_extensions: tuple[str, ...]


@dataclass(frozen=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True)
class SegmentationConfig:
    enabled: bool
    segment_duration_seconds: int


@dataclass(frozen=True)
class SamplingConfig:
    every_n_frames: int | None
    every_n_seconds: float | None


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    output: OutputConfig
    segmentation: SegmentationConfig
    sampling: SamplingConfig
    processing: dict[str, Any]
    logging: LoggingConfig


def _required_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid '{key}' section in config.")
    return value


def _normalize_extensions(raw_extensions: list[Any]) -> tuple[str, ...]:
    normalized: list[str] = []
    for ext in raw_extensions:
        if not isinstance(ext, str) or not ext.strip():
            raise ValueError("All input.allowed_extensions entries must be non-empty strings.")
        ext_value = ext.strip().lower()
        if not ext_value.startswith("."):
            ext_value = f".{ext_value}"
        normalized.append(ext_value)
    if not normalized:
        raise ValueError("input.allowed_extensions cannot be empty.")
    return tuple(normalized)


def _build_input_config(section: dict[str, Any], config_dir: Path) -> InputConfig:
    raw_path = section.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("input.path must be a non-empty string.")

    candidate_path = Path(raw_path).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = (config_dir / candidate_path).resolve()

    recursive = section.get("recursive", False)
    if not isinstance(recursive, bool):
        raise ValueError("input.recursive must be a boolean.")

    raw_extensions = section.get("allowed_extensions")
    if not isinstance(raw_extensions, list):
        raise ValueError("input.allowed_extensions must be a list of file extensions.")

    return InputConfig(
        path=candidate_path,
        recursive=recursive,
        allowed_extensions=_normalize_extensions(raw_extensions),
    )


def _build_output_config(section: dict[str, Any], config_dir: Path) -> OutputConfig:
    raw_directory = section.get("directory")
    if not isinstance(raw_directory, str) or not raw_directory.strip():
        raise ValueError("output.directory must be a non-empty string.")

    directory = Path(raw_directory).expanduser()
    if not directory.is_absolute():
        directory = (config_dir / directory).resolve()

    return OutputConfig(directory=directory)


def _build_segmentation_config(section: dict[str, Any]) -> SegmentationConfig:
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("segmentation.enabled must be a boolean.")

    duration = section.get("segment_duration_seconds")
    if not isinstance(duration, int) or duration <= 0:
        raise ValueError("segmentation.segment_duration_seconds must be a positive integer.")

    return SegmentationConfig(enabled=enabled, segment_duration_seconds=duration)


def _build_sampling_config(section: dict[str, Any]) -> SamplingConfig:
    every_n_frames = section.get("every_n_frames")
    every_n_seconds = section.get("every_n_seconds")

    if every_n_frames is not None and (not isinstance(every_n_frames, int) or every_n_frames <= 0):
        raise ValueError("sampling.every_n_frames must be a positive integer or null.")

    if every_n_seconds is not None and (
        not isinstance(every_n_seconds, (int, float)) or float(every_n_seconds) <= 0
    ):
        raise ValueError("sampling.every_n_seconds must be a positive number or null.")

    configured_modes = sum(
        [
            every_n_frames is not None,
            every_n_seconds is not None,
        ]
    )
    if configured_modes != 1:
        raise ValueError(
            "Configure exactly one sampling mode: "
            "sampling.every_n_frames OR sampling.every_n_seconds."
        )

    return SamplingConfig(
        every_n_frames=every_n_frames,
        every_n_seconds=float(every_n_seconds) if every_n_seconds is not None else None,
    )


def _build_processing_config(section: dict[str, Any]) -> dict[str, Any]:
    config = dict(section)

    diff_threshold = config.get("diff_threshold", 25)
    if not isinstance(diff_threshold, int) or not 0 <= diff_threshold <= 255:
        raise ValueError("processing.diff_threshold must be an integer between 0 and 255.")

    blur_kernel_size = config.get("blur_kernel_size", 5)
    if not isinstance(blur_kernel_size, int) or blur_kernel_size <= 0 or blur_kernel_size % 2 == 0:
        raise ValueError("processing.blur_kernel_size must be a positive odd integer.")

    active_motion_threshold = config.get("active_motion_threshold", 0.02)
    if not isinstance(active_motion_threshold, (int, float)) or not 0 <= float(active_motion_threshold) <= 1:
        raise ValueError("processing.active_motion_threshold must be a number between 0 and 1.")

    config["diff_threshold"] = diff_threshold
    config["blur_kernel_size"] = blur_kernel_size
    config["active_motion_threshold"] = float(active_motion_threshold)
    return config


def _build_logging_config(section: dict[str, Any]) -> LoggingConfig:
    level = section.get("level")
    if not isinstance(level, str) or not level.strip():
        raise ValueError("logging.level must be a non-empty string.")
    return LoggingConfig(level=level.strip().upper())


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Top-level config must be a YAML object.")

    config_dir = path.parent

    input_section = _required_section(payload, "input")
    output_section = _required_section(payload, "output")
    segmentation_section = _required_section(payload, "segmentation")
    sampling_section = _required_section(payload, "sampling")
    processing_section = _required_section(payload, "processing")
    logging_section = _required_section(payload, "logging")

    return AppConfig(
        input=_build_input_config(input_section, config_dir),
        output=_build_output_config(output_section, config_dir),
        segmentation=_build_segmentation_config(segmentation_section),
        sampling=_build_sampling_config(sampling_section),
        processing=_build_processing_config(processing_section),
        logging=_build_logging_config(logging_section),
    )
