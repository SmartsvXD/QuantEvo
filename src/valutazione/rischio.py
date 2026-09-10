import numpy as np
from numba import njit, prange

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


@njit(fastmath=True, parallel=True)
def calcolaPortReturns(p: np.ndarray, rendimenti: np.ndarray) -> np.ndarray:
    n_p, n_a = p.shape
    n_t = rendimenti.shape[1]
    out = np.zeros((n_p, n_t), dtype=np.float64)

    for i in prange(n_p):
        for a in range(n_a):
            w = p[i, a]
            if w == 0.0:
                continue
            for t in range(n_t):
                val = rendimenti[a, t]
                if not np.isnan(val):
                    out[i, t] += w * val
    return out


@njit(fastmath=True, parallel=True)
def maxDrawdown(R: np.ndarray) -> np.ndarray:
    n_p, n_t = R.shape
    worst = np.zeros(n_p, dtype=np.float64)

    for i in prange(n_p):
        log_w = R[i, 0]
        max_w = np.exp(log_w)
        min_dd = 0.0

        for t in range(1, n_t):
            log_w += R[i, t]
            w = np.exp(log_w)
            if w > max_w:
                max_w = w
            elif max_w > 0.0:
                dd = (w - max_w) / max_w
                min_dd = min(min_dd, dd)
        worst[i] = min_dd

    return worst


@njit(fastmath=True, parallel=True)
def rischioTailCVar(R: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n_p, n_t = R.shape
    out = np.empty(n_p, dtype=np.float64)
    k_tail = max(1, int(n_t * alpha))

    for i in prange(n_p):
        row = R[i].copy()
        row.sort()
        out[i] = np.mean(row[:k_tail])

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
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)

        rischio = self.rischioTotale.calcola(p, stats, standardizza=False)

        # Calcolo 2D parallelo senza toccare stats e senza broadcast 3D
        port_returns = calcolaPortReturns(p, stats.rendimenti)

        md = maxDrawdown(port_returns)
        cvar = rischioTailCVar(port_returns)

        md_penalty = np.maximum(0.0, -md)
        cvar_penalty = np.maximum(0.0, -cvar)

        if p.shape[0] > 1:
            r_mu, r_sigma = np.mean(rischio), max(np.std(rischio), 1e-12)
            md_mu, md_sigma = np.mean(md_penalty), max(np.std(md_penalty), 1e-12)
            cv_mu, cv_sigma = np.mean(cvar_penalty), max(np.std(cvar_penalty), 1e-12)

            rischio_n = (rischio - r_mu) / r_sigma
            md_n = (md_penalty - md_mu) / md_sigma
            cvar_n = (cvar_penalty - cv_mu) / cv_sigma
        else:
            rischio_n, md_n, cvar_n = rischio, md_penalty, cvar_penalty

        rischioCorretto = rischio_n + self.pesoMD * md_n + self.pesoCVar * cvar_n
        return self.standardizza(rischioCorretto, standardizza=standardizza)
