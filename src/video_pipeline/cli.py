from __future__ import annotations

import argparse
import logging
import sys

from .config_loader import load_config
from .logging_utils import configure_logging
from .pipeline import run_pipeline
from .plot_collage import run_from_cli as run_plot_collage
from .replot import run_from_cli as run_replot

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video processing pipeline and analytics utilities.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the main video processing pipeline.")
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml",
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate-plots",
        help="Aggregate intraday plot PNGs by group and type into timeline collages.",
    )
    aggregate_parser.add_argument(
        "--analytics-root",
        default="data/output/analytics",
        help="Root folder containing analytics outputs grouped by group/date.",
    )
    aggregate_parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated list of groups to include (default: all groups).",
    )
    aggregate_parser.add_argument(
        "--plot-types",
        default="intraday_distribution,intraday_timeseries,spatial_heatmap",
        help="Comma-separated plot types to aggregate.",
    )
    aggregate_parser.add_argument(
        "--output-dirname",
        default="composites",
        help="Output subfolder name inside each group analytics folder.",
    )
    aggregate_parser.add_argument(
        "--max-height-px",
        type=int,
        default=20000,
        help="Max canvas height before auto-switching to 2 columns (vertical layouts).",
    )
    aggregate_parser.add_argument(
        "--heatmap-layout",
        choices=["auto", "vertical", "horizontal"],
        default="auto",
        help="Layout strategy for spatial_heatmap timelines.",
    )
    aggregate_parser.add_argument(
        "--no-png",
        action="store_true",
        help="Disable PNG export.",
    )
    aggregate_parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF export.",
    )

    replot_parser = subparsers.add_parser(
        "replot",
        help="Regenerate analytics plots from existing results JSONL files (no video reprocessing).",
    )
    replot_parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml",
    )
    replot_parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated list of groups to include (default: all groups).",
    )
    replot_parser.add_argument(
        "--dates",
        default="",
        help="Comma-separated list of recording dates (YYYY-MM-DD) to include.",
    )
    replot_parser.add_argument(
        "--plot-types",
        default="intraday_timeseries,intraday_distribution,spatial_heatmap,interday_trend,interday_delta",
        help="Comma-separated plot types to regenerate.",
    )
    replot_parser.add_argument(
        "--include-html",
        action="store_true",
        help="Also regenerate HTML plots when enabled in config (disabled by default).",
    )
    replot_parser.add_argument(
        "--skip-collages",
        action="store_true",
        help="Do not regenerate aggregated intraday collage outputs.",
    )
    replot_parser.add_argument(
        "--collage-output-dirname",
        default="composites",
        help="Output subfolder name for regenerated collages inside each analytics group folder.",
    )
    replot_parser.add_argument(
        "--collage-max-height-px",
        type=int,
        default=20000,
        help="Max collage canvas height before auto-switching to 2 columns.",
    )
    replot_parser.add_argument(
        "--collage-heatmap-layout",
        choices=["auto", "vertical", "horizontal"],
        default="auto",
        help="Layout strategy for spatial heatmap collages.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    if not raw_args:
        raw_args = ["run"]
    elif raw_args[0] not in {"run", "aggregate-plots", "replot", "-h", "--help"}:
        # Backward compatibility: allow legacy invocation without explicit subcommand.
        raw_args = ["run", *raw_args]

    args = build_parser().parse_args(raw_args)

    try:
        if args.command == "aggregate-plots":
            configure_logging("INFO")
            return run_plot_collage(args)

        if args.command == "replot":
            config = load_config(args.config)
            configure_logging(config.logging.level)
            args._loaded_config = config
            return run_replot(args)

        config = load_config(args.config)
        configure_logging(config.logging.level)
        run_pipeline(config)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
