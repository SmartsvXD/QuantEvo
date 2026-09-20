import numpy as np


class Indice:
    def __init__(self, inverti=False):
        self.puoStandardizzare = False
        self.inverti = inverti

    def fitStandardizza(self, listaPortafogli, stats):
        valori = self.calcola(listaPortafogli, stats, standardizza=False, inverti=False)

        self._mu = np.mean(valori)
        self._sigma = np.std(valori)
        self._sigma = max(self._sigma, 1e-6)

        self.puoStandardizzare = True

    def standardizza(self, valore):
        if self.puoStandardizzare:
            return (valore - self._mu) / self._sigma
        else:
            return valore

    def standardizzaEInverti(self, data, standardizza, inverti):
        if standardizza:
            data = self.standardizza(data)

        if self.inverti and inverti:
            return -data
        return data

    def _calcolaSemplice(self, portafoglio, stats):
        raise NotImplementedError("Metodo _calcolaSemplice() non implementato nella classe derivata.")

    def calcola(self, portafoglio, stats, standardizza=True, inverti=True, *args, **kwargs):
        res = self._calcolaSemplice(portafoglio, stats, *args, **kwargs)
        return self.standardizzaEInverti(res, standardizza, inverti)
