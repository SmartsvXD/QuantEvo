import numpy as np
from src.managerTitoli.statsStoriche import StatsTitoli


def rischioTotale(portafoglio: dict, stats: StatsTitoli):
    pesi = np.array(portafoglio)
    T = stats.L  # Matrice di covarianza completa
    return np.dot(pesi.T, np.dot(T, pesi))
