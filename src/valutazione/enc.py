import numpy as np
from numba import njit

from src.valutazione.indice import Indice


@njit(fastmath=True)
def _calcola_enc_kernel(pesi: np.ndarray) -> np.ndarray:
    n_p, n_assets = pesi.shape
    res = np.zeros((n_p, 1), dtype=np.float64)

    for i in range(n_p):
        hhi = 0.0
        for j in range(n_assets):
            w = pesi[i, j]
            hhi += w * w

        # ENC classico: inverso dell'HHI
        if hhi > 1e-12:
            res[i, 0] = 1.0 / hhi
        else:
            res[i, 0] = 0.0

    return res


class ENCIndice(Indice):
    PUO_FARE_BOOTSTRAP = False
    DA_MASSIMIZZARE = True

    def _calcola(self, portafoglio: np.ndarray, stats: dict) -> np.ndarray:
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        return _calcola_enc_kernel(pesi=p)
