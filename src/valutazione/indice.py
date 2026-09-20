import time

import numpy as np
from colorama import Fore


class Indice:
    def __init__(
        self,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=False,
    ):
        self.puoStandardizzare = False
        self.inverti = inverti
        self.metricaRischio = metricaRischio
        self.alpha = alpha
        self.lambdaPenalitaStabilita = lambdaPenalitaStabilita

    def fitStandardizza(self, listaPortafogli, stats):
        # Il fit usa il calcolo semplice (B=1) per velocità
        valori = self.calcola(listaPortafogli, stats, faiBootstrap=False, standardizza=False, inverti=False)

        self._mu = np.mean(valori)
        self._sigma = np.std(valori)
        self._sigma = max(self._sigma, 1e-6)

        self.puoStandardizzare = True

    def _aggregaValori(self, vals: np.ndarray):
        if self.metricaRischio == "cvar":
            q = np.quantile(vals, self.alpha, axis=1, keepdims=True)
            mask = vals <= q
            counts = np.maximum(1, np.sum(mask, axis=1))
            return np.sum(np.where(mask, vals, 0.0), axis=1) / counts

        elif self.metricaRischio == "percentile":
            return np.quantile(vals, self.alpha, axis=1)

        elif self.metricaRischio == "downside_penalty":
            media = np.mean(vals, axis=1)
            diff_negativa = np.minimum(0.0, vals - media[:, None])
            semi_dev = np.sqrt(np.mean(diff_negativa**2, axis=1))
            return media - self.lambdaPenalitaStabilita * semi_dev

        return np.mean(vals, axis=1)

    def _calcola(self, portafoglio: np.ndarray, tensori: dict, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Metodo _calcola() non implementato nella classe derivata.")

    def calcola(
        self,
        portafoglio,
        stats,
        faiBootstrap=True,
        standardizza=True,
        inverti=True,
        debug=False,
        *args,
        **kwargs,
    ):
        if debug and faiBootstrap:
            print(f"\n{Fore.MAGENTA}Calcolo l'indice {self.__class__.__name__}...{Fore.RESET}")
            t0 = time.time()

        p = np.atleast_2d(portafoglio)
        tensori = stats.ottieni(bootstrap=faiBootstrap)

        # Chiamata unificata: restituisce sempre forma (P, B)
        vals = self._calcola(p, tensori, *args, **kwargs)

        if faiBootstrap:
            data = self._aggregaValori(vals)
        else:
            data = vals[:, 0]  # Estrae la dimensione fittizia B=1 -> (P,)

        if portafoglio.ndim == 1:
            data = data[0]

        if debug and faiBootstrap:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

        if standardizza:
            data = (data - self._mu) / self._sigma

        if self.inverti and inverti:
            data = -data

        return data
