import numpy as np
from numba import njit

from src.valutazione.indice import Indice
from src.const import (
    DEFAULT_PESO_MD,
    DEFAULT_PESO_CVAR,
)


class RischioTotaleIndice(Indice):
    def calcola(self, portafoglio, stats, standardizza=False):
        sigma = stats.sigma
        p = np.atleast_2d(portafoglio)
        # p shape: (nPort, nAssets)
        valore = np.einsum("pi,ij,pj->p", p, sigma, p)

        return self.standardizza(valore, standardizza=standardizza)


@njit(fastmath=True)
def maxDrawdown(rendimenti: np.ndarray):
    R = np.atleast_2d(rendimenti)
    R = np.where(np.isnan(R), 0.0, R)

    if R.shape[1] == 0:
        return np.zeros(R.shape[0], dtype=np.float64)

    cumulative = np.empty((R.shape[0], R.shape[1]), dtype=np.float64)
    for i in range(R.shape[0]):
        log_wealth = 0.0
        for t in range(R.shape[1]):
            log_wealth += R[i, t]
            cumulative[i, t] = np.exp(log_wealth)

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





class RischioTotaleCorrettoIndice(Indice):
    def __init__(
        self,
        pesoMD=DEFAULT_PESO_MD,
        pesoCVar=DEFAULT_PESO_CVAR,
    ):
        super().__init__()

        self.pesoMD = pesoMD
        self.pesoCVar = pesoCVar

        self.rischioTotale = RischioTotaleIndice()

    def calcola(self, portafoglio, stats, standardizza=False):
        p = np.atleast_2d(portafoglio)

        rischio = self.rischioTotale.calcola(p, stats, standardizza=False)

        port_returns = np.nansum(p[:, :, None] * stats.rendimenti[None, :, :], axis=1)

        md = maxDrawdown(port_returns)
        cvar = rischioTailCVar(port_returns)

        # md e cvar sono tipicamente <= 0 sui rendimenti: convertiamo in penalita positive.
        md_penalty = np.maximum(0.0, -md)
        cvar_penalty = np.maximum(0.0, -cvar)

        # Normalizza le componenti su scala comparabile per evitare mixing di unita' eterogenee.
        # Per input singolo non normalizziamo per preservare monotonicita' locale.
        if p.shape[0] > 1:
            r_mu, r_sigma = np.mean(rischio), np.std(rischio)
            md_mu, md_sigma = np.mean(md_penalty), np.std(md_penalty)
            cv_mu, cv_sigma = np.mean(cvar_penalty), np.std(cvar_penalty)

            r_sigma = max(r_sigma, 1e-12)
            md_sigma = max(md_sigma, 1e-12)
            cv_sigma = max(cv_sigma, 1e-12)

            rischio_n = (rischio - r_mu) / r_sigma
            md_n = (md_penalty - md_mu) / md_sigma
            cvar_n = (cvar_penalty - cv_mu) / cv_sigma
        else:
            rischio_n = rischio
            md_n = md_penalty
            cvar_n = cvar_penalty

        rischioCorretto = (
            rischio_n + self.pesoMD * md_n + self.pesoCVar * cvar_n
        )

        return self.standardizza(rischioCorretto, standardizza=standardizza)
