import argparse
from mpas_stats_diag_config import MPASConfig

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Calculate MPAS forecast statistics")
    
    # Configuration file
    parser.add_argument('-c', '--config', type=str, default=None,
                       help="YAML configuration file path")
    
    # File arguments (can override config file)
    parser.add_argument('-f', '--forecast', type=str, default=None,
                       help="MPAS forecast file path (overrides config)")
    parser.add_argument('-i', '--init', type=str, default=None,
                       help="MPAS initialization file path (overrides config)")
    parser.add_argument('-o', '--output', type=str, default=None,
                       help="Output statistics file path (overrides config)")
    parser.add_argument('-n', '--n_member', type=int, default=None,
                       help="Number of ensemble members (overrides config)")

    # Processing options (can override config file)
    parser.add_argument('-v', '--variables', type=str, nargs='+', default=None,
                       help="Variables to process (overrides config)")
    parser.add_argument('--lat-bin', type=float, default=None,
                       help="Latitude bin width in degrees (overrides config)")
    parser.add_argument('--lon-bin', type=float, default=None,
                       help="Longitude bin width in degrees (overrides config)")
    parser.add_argument('--level-bin', type=int, default=None,
                       help="Vertical level bin size (overrides config)")
    
    parser.add_argument('--regional', action='store_true',
                       help="Whetehr is regional mode (overrides config)")
    parser.add_argument('--n-processes', type=int, default=None,
                       help="Number of workers for statistics tasks (overrides config)")

    # Additional flags
    parser.add_argument('--quiet', action='store_true',
                       help="Suppress verbose output (overrides config)")
    parser.add_argument('--overwrite', action='store_true',
                       help="Overwrite output file if it exists (overrides config)")
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create configuration from command line arguments"""
    # Load configuration from YAML if provided, otherwise use defaults
    if args.config:
        config = MPASConfig.from_yaml(args.config)
    else:
        config = MPASConfig()
    
    # Override config with command line arguments if provided
    if args.forecast:
        config.forecast_file = args.forecast
    if args.init:
        config.init_file = args.init
    if args.output:
        config.output_file = args.output
    if args.n_member:
        config.n_member = args.n_member
    if args.variables:
        config.variables = args.variables
    if args.lat_bin:
        config.lat_bin_width = args.lat_bin
    if args.lon_bin:
        config.lon_bin_width = args.lon_bin
    if args.level_bin:
        config.level_bin_size = args.level_bin
    if args.regional:
        config.regional = True
    if args.n_processes:
        config.n_processes = args.n_processes
    if args.quiet:
        config.verbose = False
    if args.overwrite:
        config.overwrite = True
    
    return config
