"""Command-line entry point: export Desroziers observation-error tables."""

import argparse

from ..obs.error_export import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write Desroziers observation-error estimates (CSV + YAML with a "
                    "UFO 'assign error' filter) from obs-space statistics files. "
                    "Uses the same configuration as mpas-stat-obs-plot.")
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
