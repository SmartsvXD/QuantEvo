import numpy as np
from numba import njit, prange

from src.const import DEFAULT_PESO_CVAR, DEFAULT_PESO_MD
from src.valutazione.indice import Indice
from src.valutazione.indice import Indice


class RischioTotaleIndice(Indice):
    def __init__(self, inverti=False):
        super().__init__(inverti=inverti)

    def _calcola(self, portafoglio: np.ndarray, tensori: dict):
        p = np.atleast_2d(portafoglio)
        return np.sqrt(np.einsum("pi,bij,pj->pb", p, tensori["sigma"], p))


@njit(parallel=True, fastmath=True)
def calcolaPortReturns(p: np.ndarray, rendimenti_boot: np.ndarray) -> np.ndarray:
    n_p, n_a = p.shape
    n_b, _, n_t = rendimenti_boot.shape
    out = np.empty((n_p, n_b, n_t), dtype=np.float64)

    for i in prange(n_p):
        for b in range(n_b):
            v = p[i].copy()
            for t in range(n_t):
                w_prev = 0.0
                w_curr = 0.0
                for a in range(n_a):
                    v_a = v[a]
                    w_prev += v_a
                    r = rendimenti_boot[b, a, t]
                    if r == r and v_a > 0.0:
                        v_a = max(v_a * (1.0 + r), 0.0)
                    v[a] = v_a
                    w_curr += v_a
                out[i, b, t] = (w_curr - w_prev) / w_prev if w_prev > 0.0 else 0.0

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


class RischioMaxDrawdownIndice(Indice):
    def _calcolaSemplice(self, portafoglio, stats, portReturns=None):
        if portReturns is None:
            portReturns = calcolaPortReturns(
                np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64), stats.stats["rendimenti"]
            )
        return maxDrawdown(portReturns)

    def _calcola(self, portafoglio: np.ndarray, tensori: dict, portReturns=None):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        if portReturns is None:
            portReturns = calcolaPortReturns(p, tensori["rendimenti"])
        nP, nB, nT = portReturns.shape
        flat_dd = maxDrawdown(portReturns.reshape(nP * nB, nT))
        return flat_dd.reshape(nP, nB)


class RischioTailCVarIndice(Indice):
    def __init__(self, alpha=0.05, inverti=False):
        super().__init__(inverti=inverti)
        self.alpha = alpha

    def _calcola(self, portafoglio: np.ndarray, tensori: dict, portReturns=None):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        if portReturns is None:
            portReturns = calcolaPortReturns(p, tensori["rendimenti"])
        nP, nB, nT = portReturns.shape
        flat_cvar = rischioTailCVar(portReturns.reshape(nP * nB, nT), alpha=self.alpha)
        return flat_cvar.reshape(nP, nB)


class RischioTotaleCorrettoIndice(Indice):
    def __init__(
        self,
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
        )

        self.pesoMD = pesoMD
        self.pesoCVar = pesoCVar

        self.rischioTotale = RischioTotaleIndice()
        self.rischioMD = RischioMaxDrawdownIndice()
        self.rischioCVar = RischioTailCVarIndice(alpha=alphaCVar)

    def fitStandardizza(self, portafogli, stats):
        self.rischioMD.fitStandardizza(portafogli, stats)
        self.rischioCVar.fitStandardizza(portafogli, stats)
        self.rischioTotale.fitStandardizza(portafogli, stats)
        super().fitStandardizza(portafogli, stats)

    def _calcola(self, portafoglio: np.ndarray, tensori: dict):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        portReturns = calcolaPortReturns(p, tensori["rendimenti"])

        rischio = self.rischioTotale._calcola(p, tensori)
        md = self.rischioMD._calcola(p, tensori, portReturns=portReturns)
        cvar = self.rischioCVar._calcola(p, tensori, portReturns=portReturns)

        return rischio - self.pesoMD * md - self.pesoCVar * cvar
