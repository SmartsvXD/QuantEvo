import numpy as np
from numba import njit

from src.valutazione.indice import Indice
from src.const import (
    DEFAULT_PESO_CARD,
    DEFAULT_MIN_TITOLI_CARD,
    DEFAULT_EPSILON_CARDINALITA,
    DEFAULT_MAX_TITOLI_CARD
)


@njit(fastmath=True)
def _diversificazione(portafoglio, quadCorr):
    quadCorr = np.maximum(quadCorr, 1e-12)
    enb = 1.0 / quadCorr

    if portafoglio.shape[1] > 1:
        enb_norm = (enb - 1.0) / (portafoglio.shape[1] - 1.0)
    else:
        enb_norm = np.zeros_like(enb)

    enb_norm = np.clip(enb_norm, 0.0, 1.0)

    hhi = np.sum(portafoglio * portafoglio, axis=1)
    if portafoglio.shape[1] > 1:
        hhi_norm = (hhi - 1.0 / portafoglio.shape[1]) / (1.0 - 1.0 / portafoglio.shape[1])
    else:
        hhi_norm = np.ones_like(hhi)

    hhi_norm = np.clip(hhi_norm, 0.0, 1.0)

    return (1.0 - enb_norm) + hhi_norm

@njit(fastmath=True)
def penaltyCardinalita(
    pesi: np.ndarray,
    minTitoli: int = 3,
    epsilon: float = 0.01,
    maxTitoli: int = -1,
) -> np.ndarray:
    p = np.atleast_2d(pesi)
    counts = np.sum(p > epsilon, axis=1)

    # penalità per avere meno di minTitoli
    low_pen = np.where(counts >= minTitoli, 0.0, (minTitoli - counts) / minTitoli)

    # se maxTitoli <= 0 non applichiamo la penalità alta
    if maxTitoli <= 0:
        res = low_pen
    else:
        # penalità per avere più di maxTitoli
        high_pen = np.where(counts <= maxTitoli, 0.0, (counts - maxTitoli) / maxTitoli)
        
        res = low_pen + high_pen
        
    return res


class DiversificazioneIndice(Indice):
    def __init__(self,
        pesoCard=DEFAULT_PESO_CARD,
        cardMinTitoli=DEFAULT_MIN_TITOLI_CARD,
        cardMaxTitoli=DEFAULT_MAX_TITOLI_CARD,
        cardEpsilon=DEFAULT_EPSILON_CARDINALITA,):
        
        self.pesoCard = pesoCard
        self.cardMinTitoli = cardMinTitoli
        self.cardMaxTitoli = cardMaxTitoli
        self.cardEpsilon = cardEpsilon
        
        super().__init__()
    
    def calcola(self, portafoglio, stats, standardizza=False):
        portafoglio = np.atleast_2d(portafoglio)
        
        quadCorr = np.einsum("pi,ij,pj->p", portafoglio, stats.corr, portafoglio)
        
        val = _diversificazione(portafoglio, quadCorr)
        
        val += self.pesoCard * penaltyCardinalita(portafoglio, minTitoli=self.cardMinTitoli, maxTitoli=self.cardMaxTitoli, epsilon=self.cardEpsilon)
        
        return self.standardizza(val, standardizza=standardizza)
