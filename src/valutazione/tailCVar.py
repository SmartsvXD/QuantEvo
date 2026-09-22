import numpy as np
from numba import njit, prange

from src.const import GIORNI_IN_UN_ANNO
from src.valutazione.indice import Indice
from src.valutazione.utils import calcolaPortReturns


@njit(fastmath=True, parallel=True)
def rischioTailCVar(R: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n_p, n_t = R.shape
    out = np.empty(n_p, dtype=np.float64)
    k_tail = max(1, int(n_t * alpha))

    for i in prange(n_p):
        part = np.partition(R[i], k_tail)
        out[i] = np.mean(part[:k_tail])

    return out * np.sqrt(GIORNI_IN_UN_ANNO)


class TailCVarIndice(Indice):
    def __init__(self, alpha=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def _calcola(self, portafoglio: np.ndarray, tensori: dict, portReturns=None):
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        if portReturns is None:
            portReturns = calcolaPortReturns(p, tensori["rendimenti"])
        nP, nB, nT = portReturns.shape
        flat_cvar = rischioTailCVar(portReturns.reshape(nP * nB, nT), alpha=self.alpha)
        return -flat_cvar.reshape(nP, nB)
