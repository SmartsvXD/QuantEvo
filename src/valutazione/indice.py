import numpy as np


class Indice:
    def __init__(self):
        self.puoStandardizzare = False

    def fitStandardizza(self, listaPortafogli, stats, **kwargs):
        valori = self.calcola(listaPortafogli, stats, **kwargs)

        self.mu = np.mean(valori)
        self.sigma = np.std(valori)

        self.puoStandardizzare = True

    def standardizza(self, valore, standardizza=True):
        if self.puoStandardizzare and standardizza:
            return (valore - self.mu) / self.sigma
        else:
            return valore

    def calcola(self, portafoglio, stats, **kwargs):
        return None
