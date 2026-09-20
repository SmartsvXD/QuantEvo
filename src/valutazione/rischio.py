import numpy as np
from numba import njit, prange

from src.const import (
    DEFAULT_N_BOOSTRAP,
    DEFAULT_PESO_CVAR,
    DEFAULT_PESO_MD,
)
from src.valutazione.bootstrap import IndiceBootstrap
from src.valutazione.indice import Indice


class RischioTotaleIndice(Indice):
    def __init__(self, inverti=False):
        super().__init__(inverti=inverti)

    def _calcolaSemplice(self, portafoglio, stats):
        p = np.atleast_2d(portafoglio)
        valore = np.einsum("pi,ij,pj->p", p, stats.sigma, p)
        return np.sqrt(valore)


@njit(parallel=True, fastmath=True)
def calcolaPortReturns(p: np.ndarray, rendimenti: np.ndarray) -> np.ndarray:
    n_p, n_a = p.shape
    n_t = rendimenti.shape[1]
    out = np.empty((n_p, n_t), dtype=np.float64)

    for i in prange(n_p):
        v = p[i].copy()  # Copia diretta, evita l'allocazione + ciclo di inizializzazione

        for t in range(n_t):
            w_prev = 0.0
            w_curr = 0.0

            for a in range(n_a):
                v_a = v[a]
                w_prev += v_a
                r = rendimenti[a, t]

                # 'r == r' è un'alternativa ultra-veloce a np.isnan(r)
                if r == r and v_a > 0.0:
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
            r = max(R[i, t], -1.0)
            w *= 1.0 + r

            if w > max_w:
                max_w = w
            elif max_w > 0.0:
                dd = (w - max_w) / max_w
                min_dd = min(min_dd, dd)

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
        # np.partition sposta i k_tail elementi più piccoli all'inizio in O(N)
        # Evita l'ordinamento completo O(N log N) di row.sort()
        part = np.partition(R[i], k_tail)
        out[i] = np.mean(part[:k_tail])

    return out


class RiscioMaxDrawdownIndice(Indice):
    def _calcolaSemplice(self, portafoglio, stats, portReturns=None):
        if portReturns is None:
            portReturns = calcolaPortReturns(
                np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64), stats.rendimenti
            )
        return maxDrawdown(portReturns)


class RischioTailCVarIndice(Indice):
    def __init__(self, alpha=0.05, inverti=False):
        super().__init__(inverti=inverti)
        self.alpha = alpha

    def _calcolaSemplice(self, portafoglio, stats, portReturns=None):
        if portReturns is None:
            portReturns = calcolaPortReturns(
                np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64), stats.rendimenti
            )
        return rischioTailCVar(portReturns, alpha=self.alpha)


class RischioTotaleCorrettoIndice(IndiceBootstrap):
    def __init__(
        self,
        faiBootstrap=False,
        nBootstrap=DEFAULT_N_BOOSTRAP,
        lunghezzaBlocco=20,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        pesoMD=DEFAULT_PESO_MD,
        pesoCVar=DEFAULT_PESO_CVAR,
        alphaCVar=0.05,
        inverti=False,
    ):
        super().__init__(
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=inverti,
            faiBootstrap=faiBootstrap,
            nBootstrap=nBootstrap,
            lunghezzaBlocco=lunghezzaBlocco,
        )

        self.pesoMD = pesoMD
        self.pesoCVar = pesoCVar

        self.rischioTotale = RischioTotaleIndice()
        self.rischioMD = RiscioMaxDrawdownIndice()
        self.rischioCVar = RischioTailCVarIndice(alpha=alphaCVar)

    def fitStandardizza(self, portafogli, stats):
        self.rischioMD.fitStandardizza(portafogli, stats)
        self.rischioCVar.fitStandardizza(portafogli, stats)
        self.rischioTotale.fitStandardizza(portafogli, stats)
        super().fitStandardizza(portafogli, stats)

    def _calcolaSemplice(self, portafoglio, stats):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)

        portReturns = calcolaPortReturns(p, stats.rendimenti)

        rischio = self.rischioTotale.calcola(p, stats)
        md = self.rischioMD.calcola(p, stats, portReturns=portReturns)
        cvar = self.rischioCVar.calcola(p, stats, portReturns=portReturns)

        return rischio - self.pesoMD * md - self.pesoCVar * cvar
