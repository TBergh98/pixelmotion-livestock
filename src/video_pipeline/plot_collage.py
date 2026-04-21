from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

DAY_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEFAULT_PLOT_TYPES = (
    "intraday_distribution",
    "intraday_timeseries",
    "spatial_heatmap",
)


@dataclass(frozen=True)
class PlotTile:
    image: np.ndarray
    day: str
    path: Path

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def _discover_groups(analytics_root: Path, requested_groups: list[str] | None) -> list[Path]:
    if requested_groups:
        paths: list[Path] = []
        for group_name in requested_groups:
            group_dir = analytics_root / group_name
            if group_dir.exists() and group_dir.is_dir():
                paths.append(group_dir)
            else:
                LOGGER.warning("Requested group not found: %s", group_dir)
        return paths

    return sorted(
        [
            p
            for p in analytics_root.iterdir()
            if p.is_dir() and p.name.lower() != "interday"
        ],
        key=lambda p: p.name,
    )


def _iter_day_dirs(group_dir: Path) -> list[Path]:
    day_dirs: list[Path] = []
    for child in group_dir.iterdir():
        if not child.is_dir():
            continue
        if not DAY_DIR_PATTERN.match(child.name):
            continue
        day_dirs.append(child)

    return sorted(day_dirs, key=lambda d: d.name)


def _find_plot_for_day(day_dir: Path, plot_type: str) -> Path | None:
    plots_dir = day_dir / "plots"
    if not plots_dir.exists() or not plots_dir.is_dir():
        return None

    pattern = f"{plot_type}_{day_dir.name}_*.png"
    matches = sorted(plots_dir.glob(pattern), key=lambda p: p.name)
    if not matches:
        return None

    return matches[0]


def _load_png(path: Path) -> np.ndarray | None:
    buffer = np.fromfile(path, dtype=np.uint8)
    if buffer.size == 0:
        return None

    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _collect_tiles(group_dir: Path, plot_type: str) -> list[PlotTile]:
    tiles: list[PlotTile] = []
    for day_dir in _iter_day_dirs(group_dir):
        candidate = _find_plot_for_day(day_dir, plot_type)
        if candidate is None:
            continue

        image = _load_png(candidate)
        if image is None:
            LOGGER.warning("Could not read image: %s", candidate)
            continue

        tiles.append(PlotTile(image=image, day=day_dir.name, path=candidate))

    return tiles


def _draw_header(canvas: np.ndarray, lines: list[str], left: int = 28, top: int = 38) -> None:
    y = top
    for idx, text in enumerate(lines):
        font_scale = 0.95 if idx == 0 else 0.75
        thickness = 2 if idx == 0 else 1
        cv2.putText(
            canvas,
            text,
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (25, 25, 25),
            thickness,
            cv2.LINE_AA,
        )
        y += 34 if idx == 0 else 28


def _draw_tile_caption(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )


def _compose_vertical(
    tiles: list[PlotTile],
    group_name: str,
    plot_type: str,
    two_columns: bool,
) -> np.ndarray:
    header_h = 96
    outer_pad = 28
    gutter = 26
    caption_h = 30

    if not two_columns or len(tiles) == 1:
        body_w = max(tile.width for tile in tiles)
        body_h = sum(caption_h + tile.height for tile in tiles) + gutter * (len(tiles) - 1)
        canvas_w = body_w + outer_pad * 2
        canvas_h = header_h + body_h + outer_pad * 2
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        date_min = tiles[0].day
        date_max = tiles[-1].day
        _draw_header(
            canvas,
            [
                f"{group_name} - {plot_type}",
                f"{len(tiles)} days, {date_min} to {date_max}",
                "Layout: vertical single column",
            ],
        )

        y = header_h + outer_pad
        for tile in tiles:
            _draw_tile_caption(canvas, tile.day, outer_pad, y + 20)
            y += caption_h
            canvas[y : y + tile.height, outer_pad : outer_pad + tile.width] = tile.image
            y += tile.height + gutter
        return canvas

    # Two columns, preserving chronological order left-to-right row by row.
    rows: list[tuple[PlotTile, PlotTile | None]] = []
    idx = 0
    while idx < len(tiles):
        left = tiles[idx]
        right = tiles[idx + 1] if idx + 1 < len(tiles) else None
        rows.append((left, right))
        idx += 2

    left_max_w = max(row[0].width for row in rows)
    right_max_w = max((row[1].width for row in rows if row[1] is not None), default=0)
    row_heights = [
        max(row[0].height, row[1].height if row[1] is not None else 0) + caption_h
        for row in rows
    ]

    body_w = left_max_w + (gutter + right_max_w if right_max_w > 0 else 0)
    body_h = sum(row_heights) + gutter * (len(rows) - 1)

    canvas_w = body_w + outer_pad * 2
    canvas_h = header_h + body_h + outer_pad * 2
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    date_min = tiles[0].day
    date_max = tiles[-1].day
    _draw_header(
        canvas,
        [
            f"{group_name} - {plot_type}",
            f"{len(tiles)} days, {date_min} to {date_max}",
            "Layout: vertical two columns (auto)",
        ],
    )

    y = header_h + outer_pad
    for row_idx, (left_tile, right_tile) in enumerate(rows):
        left_x = outer_pad
        _draw_tile_caption(canvas, left_tile.day, left_x, y + 20)
        left_img_y = y + caption_h
        canvas[left_img_y : left_img_y + left_tile.height, left_x : left_x + left_tile.width] = left_tile.image

        if right_tile is not None:
            right_x = outer_pad + left_max_w + gutter
            _draw_tile_caption(canvas, right_tile.day, right_x, y + 20)
            right_img_y = y + caption_h
            canvas[
                right_img_y : right_img_y + right_tile.height,
                right_x : right_x + right_tile.width,
            ] = right_tile.image

        y += row_heights[row_idx] + gutter

    return canvas


def _compose_horizontal(
    tiles: list[PlotTile],
    group_name: str,
    plot_type: str,
    max_canvas_px: int,
) -> np.ndarray:
    header_h = 96
    outer_pad = 28
    gutter = 26
    caption_h = 30

    total_width_one_row = sum(tile.width for tile in tiles) + gutter * (len(tiles) - 1)
    rows_count = 2 if total_width_one_row > max_canvas_px and len(tiles) >= 4 else 1

    if rows_count == 1:
        row_w = total_width_one_row
        row_h = max(tile.height for tile in tiles) + caption_h
        body_w = row_w
        body_h = row_h

        canvas_w = body_w + outer_pad * 2
        canvas_h = header_h + body_h + outer_pad * 2
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        date_min = tiles[0].day
        date_max = tiles[-1].day
        _draw_header(
            canvas,
            [
                f"{group_name} - {plot_type}",
                f"{len(tiles)} days, {date_min} to {date_max}",
                "Layout: horizontal single row",
            ],
        )

        x = outer_pad
        y = header_h + outer_pad
        for tile in tiles:
            _draw_tile_caption(canvas, tile.day, x, y + 20)
            img_y = y + caption_h
            canvas[img_y : img_y + tile.height, x : x + tile.width] = tile.image
            x += tile.width + gutter

        return canvas

    split_idx = int(math.ceil(len(tiles) / 2))
    row1 = tiles[:split_idx]
    row2 = tiles[split_idx:]

    def row_width(row: list[PlotTile]) -> int:
        return sum(tile.width for tile in row) + gutter * max(0, len(row) - 1)

    row1_w = row_width(row1)
    row2_w = row_width(row2)
    row1_h = max(tile.height for tile in row1) + caption_h if row1 else 0
    row2_h = max(tile.height for tile in row2) + caption_h if row2 else 0

    body_w = max(row1_w, row2_w)
    body_h = row1_h + row2_h + gutter

    canvas_w = body_w + outer_pad * 2
    canvas_h = header_h + body_h + outer_pad * 2
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    date_min = tiles[0].day
    date_max = tiles[-1].day
    _draw_header(
        canvas,
        [
            f"{group_name} - {plot_type}",
            f"{len(tiles)} days, {date_min} to {date_max}",
            "Layout: horizontal two rows (auto)",
        ],
    )

    y1 = header_h + outer_pad
    x1 = outer_pad
    for tile in row1:
        _draw_tile_caption(canvas, tile.day, x1, y1 + 20)
        img_y = y1 + caption_h
        canvas[img_y : img_y + tile.height, x1 : x1 + tile.width] = tile.image
        x1 += tile.width + gutter

    y2 = y1 + row1_h + gutter
    x2 = outer_pad
    for tile in row2:
        _draw_tile_caption(canvas, tile.day, x2, y2 + 20)
        img_y = y2 + caption_h
        canvas[img_y : img_y + tile.height, x2 : x2 + tile.width] = tile.image
        x2 += tile.width + gutter

    return canvas


def _infer_heatmap_layout(tiles: list[PlotTile], heatmap_layout: str) -> str:
    if heatmap_layout in {"vertical", "horizontal"}:
        return heatmap_layout

    mean_w = float(np.mean([tile.width for tile in tiles]))
    mean_h = float(np.mean([tile.height for tile in tiles]))

    # If panels are square or taller than wide, horizontal usually improves day-to-day readability.
    if len(tiles) >= 3 and mean_h >= mean_w * 0.95:
        return "horizontal"
    return "vertical"


def _use_two_columns(tiles: list[PlotTile], max_height_px: int) -> bool:
    header_h = 96
    outer_pad = 28
    gutter = 26
    caption_h = 30
    single_col_height = header_h + outer_pad * 2 + sum(caption_h + tile.height for tile in tiles) + gutter * (len(tiles) - 1)
    return single_col_height > max_height_px


def _save_png(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG for {path}")
    encoded.tofile(path)


def _save_pdf(path: Path, image_rgb: np.ndarray) -> None:
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    height_px, width_px = image_rgb.shape[:2]
    dpi = 100
    fig_w = max(width_px / dpi, 1)
    fig_h = max(height_px / dpi, 1)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image_rgb)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _parse_plot_types(value: str) -> list[str]:
    raw = [part.strip() for part in value.split(",") if part.strip()]
    if not raw:
        return list(DEFAULT_PLOT_TYPES)
    return raw


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "plot"


def build_intraday_plot_collages(
    analytics_root: Path,
    output_dirname: str,
    groups: list[str] | None,
    plot_types: list[str],
    max_height_px: int,
    heatmap_layout: str,
    generate_png: bool,
    generate_pdf: bool,
) -> list[Path]:
    if not analytics_root.exists() or not analytics_root.is_dir():
        raise FileNotFoundError(f"Analytics root not found: {analytics_root}")

    written_files: list[Path] = []
    group_dirs = _discover_groups(analytics_root, groups)
    for group_dir in group_dirs:
        for plot_type in plot_types:
            tiles = _collect_tiles(group_dir, plot_type)
            if not tiles:
                LOGGER.info("No tiles found for group=%s plot_type=%s", group_dir.name, plot_type)
                continue

            if plot_type == "spatial_heatmap":
                layout = _infer_heatmap_layout(tiles, heatmap_layout=heatmap_layout)
            else:
                layout = "vertical"

            if layout == "horizontal":
                collage = _compose_horizontal(
                    tiles=tiles,
                    group_name=group_dir.name,
                    plot_type=plot_type,
                    max_canvas_px=max_height_px,
                )
            else:
                collage = _compose_vertical(
                    tiles=tiles,
                    group_name=group_dir.name,
                    plot_type=plot_type,
                    two_columns=_use_two_columns(tiles, max_height_px=max_height_px),
                )

            target_dir = analytics_root / group_dir.name / output_dirname
            base_name = f"{_safe_name(plot_type)}_timeline_{_safe_name(group_dir.name)}"
            if generate_png:
                png_path = target_dir / f"{base_name}.png"
                _save_png(png_path, collage)
                written_files.append(png_path)
                LOGGER.info("Saved collage PNG: %s", png_path)
            if generate_pdf:
                pdf_path = target_dir / f"{base_name}.pdf"
                _save_pdf(pdf_path, collage)
                written_files.append(pdf_path)
                LOGGER.info("Saved collage PDF: %s", pdf_path)

    return written_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build timeline collages from intraday plot PNG files.",
    )
    parser.add_argument(
        "--analytics-root",
        default="data/output/analytics",
        help="Root folder containing analytics outputs grouped by group/date.",
    )
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated list of groups to include (default: all groups).",
    )
    parser.add_argument(
        "--plot-types",
        default=",".join(DEFAULT_PLOT_TYPES),
        help="Comma-separated plot types to aggregate.",
    )
    parser.add_argument(
        "--output-dirname",
        default="composites",
        help="Output subfolder name inside each group analytics folder.",
    )
    parser.add_argument(
        "--max-height-px",
        type=int,
        default=20000,
        help="Max canvas height before auto-switching to 2 columns (vertical layouts).",
    )
    parser.add_argument(
        "--heatmap-layout",
        choices=["auto", "vertical", "horizontal"],
        default="auto",
        help="Layout strategy for spatial_heatmap timelines.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Disable PNG export.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF export.",
    )
    return parser


def run_from_cli(args: argparse.Namespace) -> int:
    groups = [item.strip() for item in (args.groups or "").split(",") if item.strip()]
    plot_types = _parse_plot_types(args.plot_types)
    generate_png = not bool(args.no_png)
    generate_pdf = not bool(args.no_pdf)

    if not generate_png and not generate_pdf:
        raise ValueError("At least one output format must be enabled.")

    written_files = build_intraday_plot_collages(
        analytics_root=Path(args.analytics_root),
        output_dirname=args.output_dirname,
        groups=groups or None,
        plot_types=plot_types,
        max_height_px=int(args.max_height_px),
        heatmap_layout=args.heatmap_layout,
        generate_png=generate_png,
        generate_pdf=generate_pdf,
    )

    if not written_files:
        LOGGER.warning("No collage files were generated.")
        return 1

    LOGGER.info("Generated %d collage files.", len(written_files))
    return 0
