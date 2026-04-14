from __future__ import annotations

import argparse
import logging
import sys

from .config_loader import load_config
from .logging_utils import configure_logging
from .pipeline import run_pipeline

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the video processing pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        configure_logging(config.logging.level)
        run_pipeline(config)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
