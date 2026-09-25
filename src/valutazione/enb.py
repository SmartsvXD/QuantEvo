import numpy as np
from numba import njit

from src.valutazione.indice import Indice


@njit(fastmath=True)
def _calcola_enb_kernel(pesi: np.ndarray, corr: np.ndarray) -> np.ndarray:
    n_p, n_assets = pesi.shape
    n_b = corr.shape[0]
    res = np.zeros((n_p, n_b), dtype=np.float64)

    for p in range(n_p):
        for b in range(n_b):
            quad = 0.0
            for i in range(n_assets):
                w_i = pesi[p, i]
                if w_i == 0.0:
                    continue
                for j in range(n_assets):
                    quad += w_i * corr[b, i, j] * pesi[p, j]

            res[p, b] = 1.0 / quad if quad > 1e-12 else 1.0

    return res


class ENBIndice(Indice):
    DA_MASSIMIZZARE = True

    def _calcola(self, portafoglio: np.ndarray, stats: dict, *args, **kwargs) -> np.ndarray:
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        corr = np.asarray(stats["corr"], dtype=np.float64)

        if corr.ndim == 2:
            corr = corr[np.newaxis, :, :]

        corr = np.ascontiguousarray(corr)
        return _calcola_enb_kernel(pesi=p, corr=corr)
