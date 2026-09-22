import numpy as np
from numba import njit, prange

from src.valutazione.indice import Indice
from src.valutazione.utils import calcolaPortReturns


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


class MaxDrawdownIndice(Indice):
    def _calcola(self, portafoglio: np.ndarray, tensori: dict, portReturns=None):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        if portReturns is None:
            portReturns = calcolaPortReturns(p, tensori["rendimenti"])
        nP, nB, nT = portReturns.shape
        flat_dd = maxDrawdown(portReturns.reshape(nP * nB, nT))
        return -flat_dd.reshape(nP, nB)
