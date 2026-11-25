"""Command-line entry point for observation-space statistics."""

import argparse

from ..obs.diag import main as _main


def main() -> None:
    """Command ``mpas-stat-obs``: compute observation-space statistics files."""
    parser = argparse.ArgumentParser(
        description="Compute observation-space OMB/OMA/OMF statistics."
    )
    parser.add_argument("config", nargs="?", help="YAML configuration file")
    parser.add_argument("-c", "--config", dest="config_opt", metavar="CONFIG",
                        help="YAML configuration file (same as the positional argument)")
    args = parser.parse_args()
    config = args.config_opt or args.config
    if not config:
        parser.error("a configuration file is required (-c CONFIG)")
    _main(config)


if __name__ == "__main__":
    main()
