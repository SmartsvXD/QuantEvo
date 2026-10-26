# STANDARD USA

from src.constants import TRADING_DAYS_PER_YEAR
from src.metrics.cagr import CAGR
from src.metrics.cvar import CVaR
from src.metrics.enb import ENB
from src.metrics.enc import ENC
from src.metrics.omega import Omega

random_state = 42

# Window size in days for the in-sample period and out-of-sample period
window_size = TRADING_DAYS_PER_YEAR * 3
window_size_out_of_sample = TRADING_DAYS_PER_YEAR * 2

# Configuration for the statistical analysis
stats_config = {
    "bootstrap_size": 100,
    "bootstrap_block_size": 20,
}

# Configuration for the optimization process
optimization_config = {
    "pop_size": 300,
    "n_gen": 500,
    "n_seeds": 10,
    "do_early_stopping": True,
    "early_stopping_patience": 10,
    "estimate_ref_point": True,
    "hv_ref_point": None,
}

# Structural constraints for the portfolio optimization
structural_constraints = {
    "minimum_percentage_per_asset": 0.05,
    "minimum_number_of_assets": 3,
    "maximum_number_of_assets": 4,
}

# List of assets with their respective upper bounds for portfolio allocation
assets = [
    {"name": "VTI", "upper": 0.70},
    {"name": "VEA", "upper": 0.50},
    {"name": "VWO", "upper": 0.30},
    {"name": "BND", "upper": 0.40},
]


# List of objective functions to be minimized during the optimization process
objective_functions = [
    CAGR(),
    CVaR(),
]

# List of constraints to be satisfied during the optimization process
constraints = [
    Omega(threshold=1.1, annual_hurdle_rate=0.0, risk_measure="percentile", alpha=0.5),
    ENC(threshold=2.0, risk_measure="percentile", alpha=0.5),
    ENB(threshold=1.35, risk_measure="percentile", alpha=0.5),
]
