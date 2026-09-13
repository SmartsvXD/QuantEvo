import numpy as np
from numba import njit, prange

from src.const import (
    DEFAULT_PESO_CVAR,
    DEFAULT_PESO_MD,
)
from src.valutazione.indice import Indice


class RischioTotaleIndice(Indice):
    def calcola(self, portafoglio, stats, standardizza=True, inverti=True):
        p = np.atleast_2d(portafoglio)
        
        # p shape: (nPort, nAssets)
        valore = np.einsum("pi,ij,pj->p", p, stats.sigma, p)
        valore = np.sqrt(valore)

        return super().calcola(portafoglio, stats, valore, standardizza, inverti)


@njit(parallel=True)
def calcolaPortReturns(p: np.ndarray, rendimenti: np.ndarray) -> np.ndarray:
    n_p, n_a = p.shape
    n_t = rendimenti.shape[1]
    out = np.zeros((n_p, n_t), dtype=np.float64)

    for i in prange(n_p):
        # Valore iniziale allocato a ciascun asset (Buy-and-Hold)
        v = np.empty(n_a, dtype=np.float64)
        for a in range(n_a):
            v[a] = p[i, a]

        for t in range(n_t):
            w_prev = 0.0
            w_curr = 0.0

            for a in range(n_a):
                v_a = v[a]
                w_prev += v_a
                r = rendimenti[a, t]

                # Se non c'è dato (NaN), il valore resta invariato (r = 0)
                if not np.isnan(r) and v_a > 0.0:
                    v_a = max(v_a * (1.0 + r), 0.0)

                v[a] = v_a
                w_curr += v_a

            if w_prev > 0.0:
                out[i, t] = (w_curr - w_prev) / w_prev
            else:
                out[i, t] = 0.0

    return out



@njit(fastmath=True, parallel=True)
def maxDrawdown(R: np.ndarray) -> np.ndarray:
    n_p, n_t = R.shape
    worst = np.zeros(n_p, dtype=np.float64)

    for i in prange(n_p):
        w = 1.0
        max_w = 1.0
        min_dd = 0.0

        for t in range(n_t):
            # Aggiornamento moltiplicativo con protezione al -100%
            r = max(R[i, t], -1.0)
            w *= 1.0 + r

            if w > max_w:
                max_w = w
            elif max_w > 0.0:
                dd = (w - max_w) / max_w
                min_dd = min(min_dd, dd)

            # Se il capitale si azzera, il drawdown tocca il limite minimo (-100%)
            if w <= 0.0:
                min_dd = -1.0
                break

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


class RiscioMaxDrawdownIndice(Indice):
    def calcola(self, portafoglio, stats, standardizza=True, inverti=True):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)

        port_returns = calcolaPortReturns(p, stats.rendimenti)
        md = maxDrawdown(port_returns)

        return super().calcola(portafoglio, stats, md, standardizza, inverti)


class RischioTailCVarIndice(Indice):
    def calcola(self, portafoglio, stats, standardizza=True, inverti=True):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)

        port_returns = calcolaPortReturns(p, stats.rendimenti)
        cvar = rischioTailCVar(port_returns)

        return super().calcola(portafoglio, stats, cvar, standardizza, inverti)


class RischioTotaleCorrettoIndice(Indice):
    def __init__(
        self,
        pesoMD=DEFAULT_PESO_MD,
        pesoCVar=DEFAULT_PESO_CVAR,
        inverti=False,
    ):
        super().__init__(inverti=inverti)

        self.pesoMD = pesoMD
        self.pesoCVar = pesoCVar

        self.rischioTotale = RischioTotaleIndice()
        self.rischioMD = RiscioMaxDrawdownIndice()
        self.rischioCVar = RischioTailCVarIndice()

    def fitStandardizza(self, portafoglio, stats):
        self.rischioMD.fitStandardizza(portafoglio, stats)
        self.rischioCVar.fitStandardizza(portafoglio, stats)
        self.rischioTotale.fitStandardizza(portafoglio, stats)
        super().fitStandardizza(portafoglio, stats)

    def calcola(self, portafoglio, stats, standardizza=True, inverti=True):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)

        rischio = self.rischioTotale.calcola(p, stats, standardizza=True)

        md = self.rischioMD.calcola(p, stats, standardizza=True)
        cvar = self.rischioCVar.calcola(p, stats, standardizza=True)

        rischioCorretto = rischio - self.pesoMD * md - self.pesoCVar * cvar
        return super().calcola(portafoglio, stats, rischioCorretto, standardizza, inverti)
