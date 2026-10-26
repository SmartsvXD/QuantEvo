# SPDX-FileCopyrightText: 2026 Simone Vinciguerra
# SPDX-License-Identifier: GPL-3.0-or-later

# NO FALL

from src.constants import TRADING_DAYS_PER_YEAR
from src.metrics.cagr import CAGR
from src.metrics.cvar import CVaR
from src.metrics.drawdown import Drawdown
from src.metrics.enb import ENB
from src.metrics.enc import ENC

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
assets = [
    {"name": "VTI", "upper": 0.70},
    {"name": "VEA", "upper": 0.40},
    {"name": "VWO", "upper": 0.15},
    {"name": "BND", "upper": 0.80},
    {"name": "VGLT", "upper": 0.25},
    {"name": "VNQ", "upper": 0.15},
    {"name": "VGSH", "upper": 0.40},
]

# List of objective functions to be minimized during the optimization process
objective_functions = [
    CVaR(),
    Drawdown(),
]

# List of constraints to be satisfied during the optimization process
constraints = [
    ENC(threshold=3.8, risk_measure="percentile", alpha=0.5),
    ENB(threshold=3.5, risk_measure="percentile", alpha=0.5),
    CAGR(threshold=0.04, risk_measure="percentile", alpha=0.5),
]
