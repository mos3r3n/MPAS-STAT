"""Command-line entry point for observation-space graphics."""

import argparse

from ..obs.plot import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw observation-space OMB/OMA/OMF figures."
    )
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()
    _main(args.config)


if __name__ == "__main__":
    main()
