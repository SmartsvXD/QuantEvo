import numpy as np

from src.const import GIORNI_IN_UN_ANNO
from src.valutazione.indice import Indice


class CAGRIndice(Indice):
    def _calcola(self, portafoglio: np.ndarray, tensori: dict):
        p = np.atleast_2d(portafoglio)
        rend_boot = tensori["rendimenti"]
        rend_clipped = np.clip(rend_boot, -0.999999, None)

        # Capitalizzazione composta sulle traiettorie bootstrap
        montante_asset = np.exp(np.sum(np.log1p(rend_clipped), axis=2))
        valore_finale = np.einsum("pi,bi->pb", p, montante_asset)

        esponente = GIORNI_IN_UN_ANNO / rend_boot.shape[2]
        cagr = np.where(valore_finale > 0.0, valore_finale**esponente - 1.0, -1.0)
        cagr = np.where(np.isnan(cagr), 0.0, cagr)

        return cagr
