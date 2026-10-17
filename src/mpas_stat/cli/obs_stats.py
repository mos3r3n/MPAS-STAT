"""Command-line entry point for observation-space statistics."""

import argparse

from ..obs.diag import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute observation-space OMB/OMA/OMF statistics."
    )
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()
    _main(args.config)


if __name__ == "__main__":
    main()
