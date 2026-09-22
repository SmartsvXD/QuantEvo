import time

import numpy as np
from colorama import Fore


class Indice:
    def __init__(
        self,
        metricaRischio="cvar",
        soglia=0.0,
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        daMassimizzare=False,
        debug=False,
    ):
        self.daMassimizzare = daMassimizzare
        self.metricaRischio = metricaRischio
        self.alpha = alpha
        self.lambdaPenalitaStabilita = lambdaPenalitaStabilita
        self.soglia = soglia

        self.puoFareBootstrap = True
        self.debug = debug

    def _aggregaValori(self, vals: np.ndarray):
        if self.metricaRischio == "cvar":
            if self.daMassimizzare:
                q = np.quantile(vals, self.alpha, axis=1, keepdims=True)
                mask = vals <= q
            else:
                q = np.quantile(vals, 1.0 - self.alpha, axis=1, keepdims=True)
                mask = vals >= q

            counts = np.maximum(1, np.sum(mask, axis=1))
            return np.sum(np.where(mask, vals, 0.0), axis=1) / counts

        elif self.metricaRischio == "percentile":
            q_target = self.alpha if self.daMassimizzare else (1.0 - self.alpha)
            return np.quantile(vals, q_target, axis=1)

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
        valoreOriginale=False,
        debug=False,
        *args,
        **kwargs,
    ):
        faiBootstrap = faiBootstrap and self.puoFareBootstrap
        debug = debug or self.debug

        if debug:
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

        if debug:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s) Valore medio: {np.mean(data):.6f}")

        if not valoreOriginale:
            if self.soglia != 0.0:
                data = (data - self.soglia) / abs(self.soglia)
            if self.daMassimizzare:
                data = -data

        return data
