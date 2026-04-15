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
    progress_update_seconds: float


@dataclass(frozen=True)
class HierarchyLevel:
    name: str
    type: str  # "string" or "date"


@dataclass(frozen=True)
class HierarchyConfig:
    enabled: bool
    levels: tuple[HierarchyLevel, ...]
    date_format: str
    fail_on_mixed_groups: bool
    fail_on_invalid_date: bool


@dataclass(frozen=True)
class AnalyticsConfig:
    enabled: bool
    intraday_window_seconds: int
    smooth_enabled: bool
    smooth_window_size: int
    descriptive_metrics_enabled: bool
    percentiles: tuple[int, ...]
    include_coeff_variation: bool
    include_trend_slope: bool
    output_formats: dict[str, bool]  # e.g., {"png": True, "html": True}
    plots: dict[str, bool]  # e.g., {"intraday_timeseries": True, ...}
    output_subdir: str
    primary_metric: str


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    output: OutputConfig
    segmentation: SegmentationConfig
    sampling: SamplingConfig
    processing: dict[str, Any]
    logging: LoggingConfig
    hierarchy: HierarchyConfig
    analytics: AnalyticsConfig


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

    progress_update_seconds = section.get("progress_update_seconds", 30)
    if not isinstance(progress_update_seconds, (int, float)) or float(progress_update_seconds) <= 0:
        raise ValueError("logging.progress_update_seconds must be a positive number.")

    return LoggingConfig(
        level=level.strip().upper(),
        progress_update_seconds=float(progress_update_seconds),
    )


def _optional_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Retrieve optional config section, returning empty dict if missing."""
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' section must be a YAML object if present.")
    return value


def _build_hierarchy_config(section: dict[str, Any]) -> HierarchyConfig:
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("hierarchy.enabled must be a boolean.")

    levels_raw = section.get("levels", [])
    if not isinstance(levels_raw, list):
        raise ValueError("hierarchy.levels must be a list.")

    levels = []
    for level_def in levels_raw:
        if not isinstance(level_def, dict):
            raise ValueError("Each hierarchy level must be a dict with 'name' and 'type'.")
        name = level_def.get("name")
        level_type = level_def.get("type")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("hierarchy level 'name' must be a non-empty string.")
        if level_type not in ("string", "date"):
            raise ValueError("hierarchy level 'type' must be 'string' or 'date'.")
        levels.append(HierarchyLevel(name=name.strip(), type=level_type))

    date_format = section.get("date_format", "YYYY-MM-DD")
    if not isinstance(date_format, str) or not date_format.strip():
        raise ValueError("hierarchy.date_format must be a non-empty string.")

    fail_on_mixed_groups = section.get("fail_on_mixed_groups", True)
    if not isinstance(fail_on_mixed_groups, bool):
        raise ValueError("hierarchy.fail_on_mixed_groups must be a boolean.")

    fail_on_invalid_date = section.get("fail_on_invalid_date", True)
    if not isinstance(fail_on_invalid_date, bool):
        raise ValueError("hierarchy.fail_on_invalid_date must be a boolean.")

    return HierarchyConfig(
        enabled=enabled,
        levels=tuple(levels),
        date_format=date_format.strip(),
        fail_on_mixed_groups=fail_on_mixed_groups,
        fail_on_invalid_date=fail_on_invalid_date,
    )


def _build_analytics_config(section: dict[str, Any]) -> AnalyticsConfig:
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("analytics.enabled must be a boolean.")

    intraday_window = section.get("intraday_window_seconds", 300)
    if not isinstance(intraday_window, int) or intraday_window < 0:
        raise ValueError("analytics.intraday_window_seconds must be a non-negative integer.")

    smooth_enabled = section.get("smooth_enabled", True)
    if not isinstance(smooth_enabled, bool):
        raise ValueError("analytics.smooth_enabled must be a boolean.")

    smooth_window = section.get("smooth_window_size", 3)
    if not isinstance(smooth_window, int) or smooth_window <= 0 or smooth_window % 2 == 0:
        raise ValueError("analytics.smooth_window_size must be a positive odd integer.")

    descriptive_metrics = section.get("descriptive_metrics", {})
    if not isinstance(descriptive_metrics, dict):
        raise ValueError("analytics.descriptive_metrics must be a dict.")

    desc_metrics_enabled = descriptive_metrics.get("enabled", True)
    if not isinstance(desc_metrics_enabled, bool):
        raise ValueError("analytics.descriptive_metrics.enabled must be a boolean.")

    percentiles_raw = descriptive_metrics.get("percentiles", [10, 25, 50, 75, 90])
    if not isinstance(percentiles_raw, list):
        raise ValueError("analytics.descriptive_metrics.percentiles must be a list.")
    percentiles = tuple(int(p) for p in percentiles_raw)

    include_coeff_var = descriptive_metrics.get("include_coeff_variation", True)
    if not isinstance(include_coeff_var, bool):
        raise ValueError("analytics.descriptive_metrics.include_coeff_variation must be a boolean.")

    include_trend = descriptive_metrics.get("include_trend_slope", True)
    if not isinstance(include_trend, bool):
        raise ValueError("analytics.descriptive_metrics.include_trend_slope must be a boolean.")

    output_formats_raw = section.get("output_formats", {"png": True, "html": True})
    if not isinstance(output_formats_raw, dict):
        raise ValueError("analytics.output_formats must be a dict.")
    output_formats = {str(k): bool(v) for k, v in output_formats_raw.items()}

    plots_raw = section.get("plots", {
        "intraday_timeseries": True,
        "intraday_distribution": True,
        "intraday_heatmap": True,
        "interday_trend": True,
        "interday_delta": True,
    })
    if not isinstance(plots_raw, dict):
        raise ValueError("analytics.plots must be a dict.")
    plots = {str(k): bool(v) for k, v in plots_raw.items()}

    output_subdir = section.get("output_subdir", "analytics")
    if not isinstance(output_subdir, str) or not output_subdir.strip():
        raise ValueError("analytics.output_subdir must be a non-empty string.")

    primary_metric = section.get("primary_metric", "median")
    if primary_metric not in ("mean", "median"):
        raise ValueError("analytics.primary_metric must be 'mean' or 'median'.")

    return AnalyticsConfig(
        enabled=enabled,
        intraday_window_seconds=intraday_window,
        smooth_enabled=smooth_enabled,
        smooth_window_size=smooth_window,
        descriptive_metrics_enabled=desc_metrics_enabled,
        percentiles=percentiles,
        include_coeff_variation=include_coeff_var,
        include_trend_slope=include_trend,
        output_formats=output_formats,
        plots=plots,
        output_subdir=output_subdir.strip(),
        primary_metric=primary_metric.strip(),
    )


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
    hierarchy_section = _optional_section(payload, "hierarchy")
    analytics_section = _optional_section(payload, "analytics")

    return AppConfig(
        input=_build_input_config(input_section, config_dir),
        output=_build_output_config(output_section, config_dir),
        segmentation=_build_segmentation_config(segmentation_section),
        sampling=_build_sampling_config(sampling_section),
        processing=_build_processing_config(processing_section),
        logging=_build_logging_config(logging_section),
        hierarchy=_build_hierarchy_config(hierarchy_section),
        analytics=_build_analytics_config(analytics_section),
    )
