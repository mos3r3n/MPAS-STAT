"""Command-line entry point for model-space graphics."""

import argparse
import yaml

from ..graphics.driver import run_graphics


def load_config(config_file):
    """Read a YAML configuration file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def main():
    """Command ``mpas-stat-graphics``: draw model-space figures from ``-c graphic.yaml``."""
    parser = argparse.ArgumentParser(
        description="Plot MPAS diagnostic statistics"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="YAML configuration file",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_graphics(cfg)


if __name__ == "__main__":
    main()
