"""Command-line entry point for radar statistics."""

import argparse

from ..radar.diag import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute radar composite-reflectivity statistics."
    )
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()
    _main(args.config)


if __name__ == "__main__":
    main()
