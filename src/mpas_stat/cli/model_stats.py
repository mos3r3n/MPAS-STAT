"""Command-line entry point for model-space statistics."""

from ..model.args import parse_args, create_config_from_args
from ..model.driver import calculate_statistics

def main():
    """Main function to handle command line execution"""
    args = parse_args()
    config = create_config_from_args(args)
    
    # Run statistics calculation
    calculate_statistics(config)

if __name__ == "__main__":
    main()
