"""Configuration of the model-space statistics stage (:class:`MPASConfig`), loaded from ``configs/model_input.yaml``."""

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

    # Statistics method.
    #   "weighted" (default): every product pools all samples in an output
    #       cell with area weights (areaCell) and, across model levels,
    #       layer-mass (pressure-thickness) weights; squares are pooled
    #       before the square root (2-D/level-bin RMS is the RMS over the
    #       column/bin, not of the column-mean error); ensemble spread is
    #       pooled as sqrt(mean variance); each product stores its "weight".
    #   "legacy": the pre-2026-11 kernels (unweighted, level-averaged errors
    #       before squaring, mean of spread), for comparison with old results.
    statistics_method: str = "weighted"
    area_weighting: bool = True     # weighted method: weight cells by areaCell
    mass_weighting: bool = True     # weighted method: weight model levels by layer pressure thickness

    # Ensemble spread per grid point: variance with this delta-degrees-of-
    # freedom (1 = unbiased) and, optionally, the finite-ensemble factor
    # (N+1)/N so that spread is directly comparable with the RMSE of the
    # ensemble mean (Fortin et al. 2014). Ignored by statistics_method "legacy"
    # (ddof 0, no correction).
    spread_ddof: int = 1
    spread_finite_ensemble_correction: bool = True

    # Reference for hgt_levels: "msl" (height above mean sea level, as
    # zgrid) or "agl" (height above the model terrain, zgrid[:, 0]).
    height_reference: str = "msl"

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
        config = cls(**config_data)
        config.validate()
        return config

    def validate(self):
        if self.statistics_method not in ("weighted", "legacy"):
            raise ValueError(f"statistics_method must be 'weighted' or 'legacy', got {self.statistics_method!r}")
        if self.height_reference not in ("msl", "agl"):
            raise ValueError(f"height_reference must be 'msl' or 'agl', got {self.height_reference!r}")
        if self.spread_ddof not in (0, 1):
            raise ValueError(f"spread_ddof must be 0 or 1, got {self.spread_ddof!r}")
