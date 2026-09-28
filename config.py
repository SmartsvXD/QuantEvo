from src.valutazione.cagr import CAGRIndice
from src.valutazione.enb import ENBIndice
from src.valutazione.enc import ENCIndice
from src.valutazione.omega import OmegaIndice
from src.valutazione.tailCVar import TailCVarIndice

random_state = 42

optimization_config = {
    "pop_size": 300,
    "n_gen": 500,
    "n_seeds": 5,
}


structural_constraints = {
    "minimum_percentage_per_asset": 0.05,
    "minimum_number_of_assets": 3,
    "maximum_number_of_assets": 6,
}


assets = [
    {"name": "XDEV.MI", "upper": 0.2},
    {"name": "SWDA.MI", "upper": 0.7},
    {"name": "XXSC.MI", "upper": 0.2},
    {"name": "XMME.MI", "upper": 0.2},
    {"name": "XGLE.MI", "upper": 0.05},
    {"name": "XDEQ.MI", "upper": 0.5},
    {"name": "XEON.MI", "upper": 0.05},
    {"name": "SGLD.MI", "upper": 0.05},
]

objective_functions = [
    CAGRIndice(),
    TailCVarIndice(),
]

constraints = [
    OmegaIndice(soglia=1.25, sogliaAnnua=0.035, metricaRischio="percentile", alpha=0.5),
    ENCIndice(soglia=3.8, metricaRischio="percentile", alpha=0.5),
    ENBIndice(soglia=1.2, metricaRischio="percentile", alpha=0.5),
]
