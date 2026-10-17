"""Command-line entry point for radar graphics."""

import argparse

from ..radar.plot import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw radar composite-reflectivity figures."
    )
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()
    _main(args.config)


if __name__ == "__main__":
    main()
