from mpas_stats_diag_args import parse_args, create_config_from_args
from mpas_stats_diag_driver import calculate_statistics

def main():
    """Main function to handle command line execution"""
    args = parse_args()
    config = create_config_from_args(args)
    
    # Run statistics calculation
    calculate_statistics(config)

if __name__ == "__main__":
    main()
