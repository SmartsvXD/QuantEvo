import numpy as np


class ValutatoreMultiplo:
    def __init__(self, indici: tuple):
        self._indici = indici
        self.numIndici = len(self._indici)

    def calcola(self, portafoglio, stats, inverti=True):
        portafoglio = np.atleast_2d(portafoglio)

        res = np.empty((portafoglio.shape[0], len(self._indici)), dtype=np.float64)
        for n, i in enumerate(self._indici):
            vals = i.calcola(portafoglio, stats, inverti=inverti)
            res[:, n] = vals

        return res
