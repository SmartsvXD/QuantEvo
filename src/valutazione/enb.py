import numpy as np
from numba import njit

from src.valutazione.indice import Indice


@njit(fastmath=True)
def _calcola_enb_kernel(pesi: np.ndarray, corr: np.ndarray) -> np.ndarray:
    n_p, n_assets = pesi.shape
    n_b = corr.shape[0]
    res = np.zeros((n_p, n_b), dtype=np.float64)

    # Buffer riutilizzabile per i contributi di rischio delle scommesse
    c = np.zeros(n_assets, dtype=np.float64)

    for b in range(n_b):
        # Autovalori (vals) e autovettori ortonormali (vecs)
        vals, vecs = np.linalg.eigh(corr[b])

        # Clamping per evitare piccoli autovalori negativi dovuti a errori numerici
        for i in range(n_assets):
            vals[i] = max(vals[i], 0.0)

        for p in range(n_p):
            total_var = 0.0

            # Proiezione del portafoglio sulle componenti ortogonali (bets)
            for k in range(n_assets):
                proj_k = 0.0
                for i in range(n_assets):
                    w_i = pesi[p, i]
                    if w_i != 0.0:
                        proj_k += w_i * vecs[i, k]

                # Varianza della scommessa k-esima
                var_k = vals[k] * proj_k * proj_k
                c[k] = var_k
                total_var += var_k

            # Entropia esponenziale di Shannon dei contributi percentuali p_k
            if total_var > 1e-12:
                entropy = 0.0
                for k in range(n_assets):
                    p_k = c[k] / total_var
                    if p_k > 1e-12:
                        entropy -= p_k * np.log(p_k)
                res[p, b] = np.exp(entropy)
            else:
                res[p, b] = 1.0

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
