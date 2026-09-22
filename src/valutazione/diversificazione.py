import numpy as np
from numba import njit

from src.const import (
    DEFAULT_EPSILON_CARDINALITA,
    DEFAULT_MAX_TITOLI_CARD,
    DEFAULT_MIN_TITOLI_CARD,
    DEFAULT_PESO_CARD,
)
from src.valutazione.indice import Indice


@njit(fastmath=True)
def penaltyCardinalita(
    pesi: np.ndarray,
    minTitoli: int = 3,
    epsilon: float = 0.01,
) -> np.ndarray:
    n_rows, n_cols = pesi.shape
    res = np.zeros(n_rows, dtype=np.float64)

    inv_min = 1.0 / minTitoli

    for i in range(n_rows):
        # Calcolo HHI per ottenere la cardinalità effettiva (ENC)
        hhi = 0.0
        count_nominale = 0
        for j in range(n_cols):
            w = pesi[i, j]
            hhi += w * w
            if w > epsilon:
                count_nominale += 1

        enc = 1.0 / hhi if hhi > 1e-12 else 1.0

        pen = 0.0
        # Penalizza se la diversificazione effettiva è sotto la soglia minima
        if enc < minTitoli:
            pen += (minTitoli - enc) * inv_min

        # Mantieni la penalità se anche nominalmente mancano titoli
        if count_nominale < minTitoli:
            pen += (minTitoli - count_nominale) * inv_min

        res[i] = pen

    return res


class DiversificazioneIndice(Indice):
    def __init__(
        self,
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
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
        )

    def _calcola(self, portafoglio: np.ndarray, stats: dict) -> np.ndarray:
        p = np.atleast_2d(portafoglio)
        quadCorr = np.einsum("pi,bij,pj->pb", p, stats["corr"], p)
        quadCorr = np.maximum(quadCorr, 1e-12)

        enb = 1.0 / quadCorr
        n_assets = p.shape[1]
        enb_norm = (enb - 1.0) / (n_assets - 1.0) if n_assets > 1 else np.zeros_like(enb)
        enb_norm = np.clip(enb_norm, 0.0, 1.0)

        hhi = np.sum(p * p, axis=1, keepdims=True)
        hhi_norm = (hhi - 1.0 / n_assets) / (1.0 - 1.0 / n_assets) if n_assets > 1 else np.ones_like(hhi)
        hhi_norm = np.clip(hhi_norm, 0.0, 1.0)

        pen = penaltyCardinalita(
            p, minTitoli=self.cardMinTitoli, maxTitoli=self.cardMaxTitoli, epsilon=self.cardEpsilon
        )[:, None]

        div_score = 0.5 * (enb_norm + (1.0 - hhi_norm))

        return div_score - self.pesoCard * pen
