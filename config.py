from src.constants import TRADING_DAYS_PER_YEAR

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
    "maximum_number_of_assets": 6,
}

# List of assets with their respective upper bounds for portfolio allocation
assets = []

# List of objective functions to be maximized during the optimization process
objective_functions = []

# List of constraints to be satisfied during the optimization process
constraints = []
