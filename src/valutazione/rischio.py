import numpy as np
from numba import njit

from src.valutazione.indice import Indice
from src.const import (
    DEFAULT_PESO_CARD,
    DEFAULT_MIN_TITOLI_CARD,
    DEFAULT_EPSILON_CARDINALITA,
    DEFAULT_PESO_MD,
    DEFAULT_PESO_CVAR,
)


@njit(fastmath=True)
def _rischioTotale(portafoglio, sigma):
    return np.dot(portafoglio.T, np.dot(sigma, portafoglio))


class RischioTotaleIndice(Indice):
    def calcola(self, portafoglio, stats, standardizza=False, **kwargs):
        sigma = stats.sigma
        p = np.atleast_2d(portafoglio)
        # p shape: (nPort, nAssets)
        valore = np.einsum("pi,ij,pj->p", p, sigma, p)

        return self.standardizza(valore, standardizza=standardizza)


@njit(fastmath=True)
def maxDrawdown(rendimenti: np.ndarray):
    R = np.atleast_2d(rendimenti)
    # replace NaN with 0.0 for cumulative product
    R = np.where(np.isnan(R), 0.0, R)

    cumulative = np.empty((R.shape[0], R.shape[1]), dtype=np.float64)
    for i in range(R.shape[0]):
        prod = 1.0
        for t in range(R.shape[1]):
            prod *= 1.0 + R[i, t]
            cumulative[i, t] = prod

    runningMax = np.empty((R.shape[0], R.shape[1]), dtype=np.float64)
    for i in range(R.shape[0]):
        m = cumulative[i, 0]
        runningMax[i, 0] = m
        for t in range(1, R.shape[1]):
            if cumulative[i, t] > m:
                m = cumulative[i, t]
            runningMax[i, t] = m

    dd = np.empty((R.shape[0], R.shape[1]), dtype=np.float64)
    for i in range(R.shape[0]):
        for t in range(R.shape[1]):
            if runningMax[i, t] == 0.0:
                dd[i, t] = 0.0
            else:
                dd[i, t] = (cumulative[i, t] - runningMax[i, t]) / runningMax[i, t]

    worst = np.empty(R.shape[0], dtype=np.float64)
    for i in range(R.shape[0]):
        minv = dd[i, 0]
        for t in range(1, R.shape[1]):
            if dd[i, t] < minv:
                minv = dd[i, t]
        worst[i] = minv

    return worst


@njit(fastmath=True)
def rischioTailCVar(rendimenti: np.ndarray, alpha: float = 0.05) -> float:
    R = np.atleast_2d(rendimenti)
    # replace NaN with large positive so they won't be in tail (or ignore)
    # better: mask NaNs when sorting
    nPort, T = R.shape
    out = np.empty(nPort, dtype=np.float64)
    for i in range(nPort):
        r = R[i]
        r = r[~np.isnan(r)]
        if r.size == 0:
            out[i] = 0.0
            continue
        rendimentiOrdinati = np.sort(r)
        n = rendimentiOrdinati.size
        idxCutoff = max(1, int(n * alpha))
        rendimentiTail = rendimentiOrdinati[:idxCutoff]
        out[i] = np.mean(rendimentiTail)
    return out


@njit(fastmath=True)
def penaltyCardinalita(pesi: np.ndarray, minTitoli: int = 3, epsilon: float = 0.01) -> float:
    p = np.atleast_2d(pesi)
    counts = np.sum(p > epsilon, axis=1)
    res = np.where(counts >= minTitoli, 0.0, (minTitoli - counts) / minTitoli)
    return res.astype(np.float64)


class RischioTotaleCorrettoIndice(Indice):
    def __init__(
        self,
        pesoCard=DEFAULT_PESO_CARD,
        cardMinTitoli=DEFAULT_MIN_TITOLI_CARD,
        cardEpsilon=DEFAULT_EPSILON_CARDINALITA,
        pesoMD=DEFAULT_PESO_MD,
        pesoCVar=DEFAULT_PESO_CVAR,
    ):
        super().__init__()

        self.pesoCard = pesoCard
        self.cardMinTitoli = cardMinTitoli
        self.cardEpsilon = cardEpsilon
        self.pesoMD = pesoMD
        self.pesoCVar = pesoCVar

        self.rischioTotale = RischioTotaleIndice()

    def calcola(self, portafoglio, stats, standardizza=False, **kwargs):
        p = np.atleast_2d(portafoglio)

        rischio = self.rischioTotale.calcola(p, stats, standardizza=False, **kwargs)

        cardPenalty = penaltyCardinalita(p, minTitoli=self.cardMinTitoli, epsilon=self.cardEpsilon)

        port_returns = np.nansum(p[:, :, None] * stats.rendimenti[None, :, :], axis=1)

        md = maxDrawdown(port_returns)
        cvar = rischioTailCVar(port_returns)

        rischioCorretto = rischio + self.pesoCard * cardPenalty + self.pesoMD * md + self.pesoCVar * cvar

        return self.standardizza(rischioCorretto, standardizza=standardizza)
