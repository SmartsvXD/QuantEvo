from src.simulazione.titoli import simulaTitoli
from src.simulazione.valutazione import valutaPortafoglio

def _test():
    from src.managerTitoli.statsStoriche import StatsTitoli
    import numpy as np
    
    stats = StatsTitoli()
    
    stats.nomi = ["SPY", ]
    stats.rendimentoGiornaliero = np.array([0.001, ])
    stats.volatilitàGiornaliera = np.array([0.001, ])
    stats.dividendiGiornalieri = np.array([0.001, ])
    stats.volatilitàDividendi = np.array([0.001, ])
    stats.L = np.array([1, ])
    stats.skew = np.array([0.001, ])
    stats.kurtosis = np.array([0.001, ])
    
    ris = simulaTitoli(stats)
    valutaPortafoglio({"SPY": 1}, ris)
    