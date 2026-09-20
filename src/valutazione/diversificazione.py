import numpy as np
from numba import njit

from src.const import (
    DEFAULT_EPSILON_CARDINALITA,
    DEFAULT_MAX_TITOLI_CARD,
    DEFAULT_MIN_TITOLI_CARD,
    DEFAULT_PESO_CARD,
)
from src.valutazione.bootstrap import IndiceBootstrap


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
    n_rows, n_cols = pesi.shape
    res = np.zeros(n_rows, dtype=np.float64)

    inv_min = 1.0 / minTitoli
    inv_max = 1.0 / maxTitoli if maxTitoli > 0 else 0.0

    for i in range(n_rows):
        # Conteggio inline senza allocare matrici booleane
        count = 0
        for j in range(n_cols):
            if pesi[i, j] > epsilon:
                count += 1

        pen = 0.0
        if count < minTitoli:
            pen += (minTitoli - count) * inv_min

        if maxTitoli > 0 and count > maxTitoli:
            pen += (count - maxTitoli) * inv_max

        res[i] = pen

    return res


class DiversificazioneIndice(IndiceBootstrap):
    def __init__(
        self,
        faiBootstrap=False,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        pesoCard=DEFAULT_PESO_CARD,
        cardMinTitoli=DEFAULT_MIN_TITOLI_CARD,
        cardMaxTitoli=DEFAULT_MAX_TITOLI_CARD,
        cardEpsilon=DEFAULT_EPSILON_CARDINALITA,
        inverti=False,
    ):

        self.pesoCard = pesoCard
        self.cardMinTitoli = cardMinTitoli
        self.cardMaxTitoli = cardMaxTitoli
        self.cardEpsilon = cardEpsilon

        super().__init__(
            inverti=inverti,
            faiBootstrap=faiBootstrap,
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
        )

    def _calcolaSemplice(self, portafoglio, stats):
        portafoglio = np.atleast_2d(portafoglio)

        quadCorr = np.einsum("pi,ij,pj->p", portafoglio, stats.corr, portafoglio)

        val = _diversificazione(portafoglio, quadCorr)

        val += self.pesoCard * penaltyCardinalita(
            portafoglio, minTitoli=self.cardMinTitoli, maxTitoli=self.cardMaxTitoli, epsilon=self.cardEpsilon
        )
        
        return val