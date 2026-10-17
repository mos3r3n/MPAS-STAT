from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import yaml

@dataclass
class MPASConfig:
    """Configuration class for MPAS statistics calculation"""
    # File paths
    forecast_file: str = "mpas_forecast.nc"
    init_file: str = "mpas_init.nc"
    output_file: str = "mpas_stat.nc"
    n_member: int = 1

    # Variables to process
    variables: List[str] = field(default_factory=lambda: ['U', 'V', 'T', 'Qv', 'P', 'U10', 'V10', 'T2', 'Q2', 'Ps'])
    
    # Binning configuration
    lat_bin_width: float = 10.0
    lon_bin_width: float = 10.0
    level_bin_size: int = 6
    
    # Vertical levels:
    pres_levels: list[float] = field(
        default_factory=lambda: [1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0,
                                 250.0, 200.0, 150.0, 100.0, 50.0, 20.0]
    )
    hgt_levels: list[float] = field(
        default_factory=lambda: [100.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0,
                                 5000.0, 7000.0, 10000.0, 15000.0, 20000.0]
    )

    # Region definitions (latitude bounds)
    regions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'Global': (-90, 90),
        'Tropics': (-30, 30),
        'NHem': (0, 90),
        'SHem': (-90, 0),
        'NMid': (30, 60),
        'SMid': (-60, -30),
        'NPol': (60, 90),
        'SPol': (-90, -60)
    })

    regional: bool = False

    # For parallel run
    n_processes: int = 4

    # Additional settings
    verbose: bool = True
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """Create configuration from YAML file"""
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
