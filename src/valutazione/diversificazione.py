import numpy as np
from src.valutazione.rischioTotale import rischioTotale


def diversificazione(portafoglio, stats):
    """
    Calcola la funzione di diversificazione intelligente (f3).

    Parametri:
    - portafoglio: array numpy, pesi del portafoglio (uno per ogni titolo in stats).
    - stats: oggetto StatsTitoli.

    Ritorna:
    - Valore di f3, che misura la deviazione dei contributi al rischio.
    """
    pesi = np.array(portafoglio)
    T = stats.L  # Matrice di covarianza completa
    risk_contributions = pesi * np.dot(T, pesi)
    N = len(pesi)
    target_contribution = rischioTotale(portafoglio, stats) / N
    return np.sum((risk_contributions - target_contribution) ** 2)
