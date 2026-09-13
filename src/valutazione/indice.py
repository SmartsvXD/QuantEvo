import numpy as np


class Indice:
    def __init__(self, inverti=False):
        self.puoStandardizzare = False
        self.inverti = inverti

    def fitStandardizza(self, listaPortafogli, stats):
        valori = self.calcola(listaPortafogli, stats, standardizza=False, inverti=False)

        self.mu = np.mean(valori)
        self.sigma = np.std(valori)
        self.sigma = max(self.sigma, 1e-6)

        self.puoStandardizzare = True

    def standardizza(self, valore):
        if self.puoStandardizzare:
            return (valore - self.mu) / self.sigma
        else:
            return valore

    def calcola(self, portafoglio, stats, data, standardizza, inverti):
        if standardizza:
            data = self.standardizza(data)

        if self.inverti and inverti:
            return -data
        return data
