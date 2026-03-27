import numpy as np
from numba import njit

from src.valutazione.indice import Indice


@njit(fastmath=True)
def _diversificazione(portafoglio, quadCorr):
    quadCorr = np.maximum(quadCorr, 1e-12)
    enb = 1.0 / quadCorr

    if portafoglio.shape[1] > 1:
        enb_norm = (enb - 1.0) / (portafoglio.shape[1] - 1.0)
    else:
        enb_norm = np.zeros_like(enb)

    enb_norm = np.clip(enb_norm, 0.0, 1.0)

    hhi = np.sum(portafoglio * portafoglio, axis=1)
    if portafoglio.shape[1] > 1:
        hhi_norm = (hhi - 1.0 / portafoglio.shape[1]) / (1.0 - 1.0 / portafoglio.shape[1])
    else:
        hhi_norm = np.ones_like(hhi)

    hhi_norm = np.clip(hhi_norm, 0.0, 1.0)

    return (1.0 - enb_norm) + hhi_norm


class DiversificazioneIndice(Indice):
    def calcola(self, portafoglio, stats, standardizza=False, **kwargs):
        portafoglio = np.atleast_2d(portafoglio)
        
        quadCorr = np.einsum("pi,ij,pj->p", portafoglio, stats.corr, portafoglio)
        
        val = _diversificazione(portafoglio, quadCorr)
        
        return self.standardizza(val, standardizza=standardizza)
