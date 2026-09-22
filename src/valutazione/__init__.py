import numpy as np


class ValutatoreMultiplo:
    def __init__(self, indici: tuple):
        self.indici = indici
        self.numIndici = len(self.indici)

    def calcola(self, portafoglio, stats, valoreOriginale=False):
        portafoglio = np.atleast_2d(portafoglio)

        res = np.empty((portafoglio.shape[0], len(self.indici)), dtype=np.float64)
        for n, i in enumerate(self.indici):
            vals = i.calcola(portafoglio, stats, valoreOriginale=valoreOriginale)
            res[:, n] = vals

        return res
