import numpy as np
from numba import njit

from src.valutazione.indice import Indice


@njit(fastmath=True)
def _calcola_enc_kernel(
    pesi: np.ndarray,
) -> np.ndarray:
    n_p, n_assets = pesi.shape
    res = np.zeros((n_p, 1), dtype=np.float64)

    inv_assets = 1.0 / n_assets if n_assets > 0 else 1.0
    hhi_den = 1.0 - inv_assets

    for i in range(n_p):
        hhi = 0.0
        for j in range(n_assets):
            w = pesi[i, j]
            hhi += w * w

        # Normalizzazione HHI in [0, 1]
        if n_assets > 1 and hhi_den > 1e-12:
            hhi_norm = (hhi - inv_assets) / hhi_den
            if hhi_norm < 0.0:
                hhi_norm = 0.0
            elif hhi_norm > 1.0:
                hhi_norm = 1.0
        else:
            hhi_norm = 1.0

        enc_score = 1.0 - hhi_norm

        res[i, 0] = enc_score

    return res


class ENCIndice(Indice):
    PUO_FARE_BOOTSTRAP = False
    DA_MASSIMIZZARE = True

    def _calcola(self, portafoglio: np.ndarray, stats: dict) -> np.ndarray:
        p = np.ascontiguousarray(np.atleast_2d(portafoglio), dtype=np.float64)
        return _calcola_enc_kernel(pesi=p)
