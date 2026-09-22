import numpy as np
from numba import njit, prange

from src.const import GIORNI_IN_UN_ANNO
from src.valutazione.indice import Indice
from src.valutazione.utils import calcolaPortReturns


@njit(fastmath=True, parallel=True)
def volatilitaPortafoglio(R: np.ndarray) -> np.ndarray:
    n_p, n_t = R.shape
    out = np.empty(n_p, dtype=np.float64)

    for i in prange(n_p):
        mean = 0.0
        for t in range(n_t):
            mean += R[i, t]
        mean /= n_t

        var = 0.0
        for t in range(n_t):
            diff = R[i, t] - mean
            var += diff * diff

        out[i] = np.sqrt(var / (n_t - 1)) if n_t > 1 else 0.0

    return out * np.sqrt(GIORNI_IN_UN_ANNO)


class VolatilitaIndice(Indice):
    def _calcola(self, portafoglio: np.ndarray, tensori: dict, portReturns=None):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        if portReturns is None:
            portReturns = calcolaPortReturns(p, tensori["rendimenti"])
        nP, nB, nT = portReturns.shape
        flat_vol = volatilitaPortafoglio(portReturns.reshape(nP * nB, nT))
        return flat_vol.reshape(nP, nB)
