from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
_GPU_FALLBACK_LOGGED = False


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


def compute_spatial_grid_array(
    thresholded: object,
    grid_size: int = 16,
) -> list[float]:
    """
    Divide a thresholded binary image into a grid and compute motion density per cell.
    
    Args:
        thresholded: Binary OpenCV image (uint8) where pixels are 0 or 255
        grid_size: Number of rows/columns in the grid (e.g., 16 for 16x16)
        
    Returns:
        List of grid_size^2 floats in range [0, 1] representing density per cell.
        Row-major order: cell[i, j] = result[i * grid_size + j]
    """
    if thresholded is None:
        return [0.0] * (grid_size * grid_size)
    
    thresholded_array = np.asarray(thresholded, dtype=np.uint8)
    height, width = thresholded_array.shape
    
    grid = []
    cell_height = max(1, height // grid_size)
    cell_width = max(1, width // grid_size)
    
    for i in range(grid_size):
        for j in range(grid_size):
            row_start = i * cell_height
            row_end = (i + 1) * cell_height if i < grid_size - 1 else height
            col_start = j * cell_width
            col_end = (j + 1) * cell_width if j < grid_size - 1 else width
            
            cell = thresholded_array[row_start:row_end, col_start:col_end]
            total_pixels = cell.size
            changed_pixels = int(np.sum(cell > 0))
            density = (changed_pixels / total_pixels) if total_pixels > 0 else 0.0
            grid.append(round(float(density), 6))
    
    return grid


def _cuda_is_available() -> bool:
    cuda_module = getattr(cv2, "cuda", None)
    if cuda_module is None:
        return False

    get_device_count = getattr(cuda_module, "getCudaEnabledDeviceCount", None)
    if not callable(get_device_count):
        return False

    try:
        return int(get_device_count()) > 0
    except Exception:
        return False


def probe_gpu_acceleration(blur_kernel_size: int) -> tuple[bool, str]:
    if not _cuda_is_available():
        return False, "No CUDA-capable OpenCV device is available."

    try:
        sample_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        gpu_frame = _cuda_gputmat()
        gpu_frame.upload(sample_frame)

        gray_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        if blur_kernel_size > 1:
            gaussian_filter = _get_cuda_gaussian_filter(blur_kernel_size)
            gray_frame = gaussian_filter.apply(gray_frame)

        _ = gray_frame.download()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _cuda_gputmat() -> Any:
    gputmat_type = getattr(cv2, "cuda_GpuMat", None)
    if gputmat_type is None:
        raise RuntimeError("OpenCV CUDA GpuMat support is unavailable.")
    return gputmat_type()


@lru_cache(maxsize=8)
def _get_cuda_gaussian_filter(blur_kernel_size: int) -> Any:
    cuda_module = getattr(cv2, "cuda", None)
    if cuda_module is None:
        raise RuntimeError("OpenCV CUDA module is unavailable.")

    create_gaussian_filter = getattr(cuda_module, "createGaussianFilter", None)
    if not callable(create_gaussian_filter):
        raise RuntimeError("OpenCV CUDA Gaussian filter support is unavailable.")

    return create_gaussian_filter(
        cv2.CV_8UC1,
        cv2.CV_8UC1,
        (blur_kernel_size, blur_kernel_size),
        0,
    )


def _finalize_motility_from_diff(
    diff: np.ndarray,
    *,
    diff_threshold: int,
    compute_spatial_grid: bool,
    spatial_grid_size: int,
) -> dict[str, Any]:
    _, thresholded = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    total_pixels = thresholded.size
    changed_pixels = int(cv2.countNonZero(thresholded))
    active_pixel_ratio = (changed_pixels / total_pixels) if total_pixels else 0.0
    mean_diff_intensity = float(diff.mean()) / 255.0 if total_pixels else 0.0

    result = {
        "motility_score": round(active_pixel_ratio, 6),
        "active_pixel_ratio": round(active_pixel_ratio, 6),
        "mean_diff_intensity": round(mean_diff_intensity, 6),
    }

    if compute_spatial_grid:
        result["spatial_grid"] = compute_spatial_grid_array(thresholded, spatial_grid_size)

    return result


def _compute_motility_cpu(
    previous_frame: object,
    current_frame: object,
    *,
    diff_threshold: int,
    blur_kernel_size: int,
    compute_spatial_grid: bool,
    spatial_grid_size: int,
) -> dict[str, Any]:
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if blur_kernel_size > 1:
        kernel = (blur_kernel_size, blur_kernel_size)
        prev_gray = cv2.GaussianBlur(prev_gray, kernel, 0)
        curr_gray = cv2.GaussianBlur(curr_gray, kernel, 0)

    diff = cv2.absdiff(curr_gray, prev_gray)
    return _finalize_motility_from_diff(
        diff,
        diff_threshold=diff_threshold,
        compute_spatial_grid=compute_spatial_grid,
        spatial_grid_size=spatial_grid_size,
    )


def _compute_motility_cuda(
    previous_frame: object,
    current_frame: object,
    *,
    diff_threshold: int,
    blur_kernel_size: int,
    compute_spatial_grid: bool,
    spatial_grid_size: int,
) -> dict[str, Any]:
    if not _cuda_is_available():
        raise RuntimeError("No CUDA-capable OpenCV device is available.")

    cuda_module = cv2.cuda
    previous_gpu = _cuda_gputmat()
    current_gpu = _cuda_gputmat()
    previous_gpu.upload(previous_frame)
    current_gpu.upload(current_frame)

    previous_gray = cuda_module.cvtColor(previous_gpu, cv2.COLOR_BGR2GRAY)
    current_gray = cuda_module.cvtColor(current_gpu, cv2.COLOR_BGR2GRAY)

    if blur_kernel_size > 1:
        gaussian_filter = _get_cuda_gaussian_filter(blur_kernel_size)
        previous_gray = gaussian_filter.apply(previous_gray)
        current_gray = gaussian_filter.apply(current_gray)

    diff_gpu = cuda_module.absdiff(current_gray, previous_gray)
    diff = diff_gpu.download()
    return _finalize_motility_from_diff(
        diff,
        diff_threshold=diff_threshold,
        compute_spatial_grid=compute_spatial_grid,
        spatial_grid_size=spatial_grid_size,
    )


def _log_gpu_fallback(reason: str) -> None:
    global _GPU_FALLBACK_LOGGED
    if _GPU_FALLBACK_LOGGED:
        return

    LOGGER.warning("GPU acceleration disabled for motility analysis: %s. Falling back to CPU.", reason)
    _GPU_FALLBACK_LOGGED = True


def compute_motility(
    previous_frame: object,
    current_frame: object,
    *,
    diff_threshold: int,
    blur_kernel_size: int,
    compute_spatial_grid: bool = False,
    spatial_grid_size: int = 16,
    gpu_acceleration: bool = False,
) -> dict[str, Any]:
    if gpu_acceleration:
        try:
            return _compute_motility_cuda(
                previous_frame,
                current_frame,
                diff_threshold=diff_threshold,
                blur_kernel_size=blur_kernel_size,
                compute_spatial_grid=compute_spatial_grid,
                spatial_grid_size=spatial_grid_size,
            )
        except Exception as exc:
            _log_gpu_fallback(str(exc))

    return _compute_motility_cpu(
        previous_frame,
        current_frame,
        diff_threshold=diff_threshold,
        blur_kernel_size=blur_kernel_size,
        compute_spatial_grid=compute_spatial_grid,
        spatial_grid_size=spatial_grid_size,
    )


def process_frame(
    *,
    frame_index: int,
    timestamp_seconds: float,
    segment_index: int,
    motility: dict[str, Any] | None,
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
