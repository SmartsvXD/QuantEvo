# STANDARD AMERICAN

from src.const import GIORNI_IN_UN_ANNO
from src.valutazione.cagr import CAGRIndice
from src.valutazione.enb import ENBIndice
from src.valutazione.enc import ENCIndice
from src.valutazione.omega import OmegaIndice
from src.valutazione.tailCVar import TailCVarIndice

random_state = 42

# Window size in days for the in-sample period and out-of-sample period
window_size = GIORNI_IN_UN_ANNO * 3
window_size_out_of_sample = GIORNI_IN_UN_ANNO * 2

# Configuration for the statistical analysis
stats_config = {
    "bootstrap_size": 100,
    "bootstrap_block_size": 20,
}

# Configuration for the optimization process
optimization_config = {
    "pop_size": 300,
    "n_gen": 50,
    "n_seeds": 10,
    "do_early_stopping": True,
    "early_stopping_patience": 10,
    "esteem_ref_point": True,
    "hv_ref_point": None,
}

# Structural constraints for the portfolio optimization
structural_constraints = {
    "minimum_percentage_per_asset": 0.05,
    "minimum_number_of_assets": 3,
    "maximum_number_of_assets": 5,
}

# List of assets with their respective upper bounds for portfolio allocation
assets = [
    {"name": "VTI", "upper": 0.70},
    {"name": "VEA", "upper": 0.50},
    {"name": "VWO", "upper": 0.30},
    {"name": "BND", "upper": 0.50},
    {"name": "BNDX", "upper": 0.50},
]


# List of objective functions to be maximized during the optimization process
objective_functions = [
    CAGRIndice(),
    TailCVarIndice(),
]

# List of constraints to be satisfied during the optimization process
constraints = [
    OmegaIndice(soglia=1.2, sogliaAnnua=0.035, metricaRischio="percentile", alpha=0.5),
    ENCIndice(soglia=3.7, metricaRischio="percentile", alpha=0.5),
    ENBIndice(soglia=1.2, metricaRischio="percentile", alpha=0.5),
]
