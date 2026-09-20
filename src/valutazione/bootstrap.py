import time

import numpy as np
from colorama import Fore

from src.valutazione.indice import Indice


class IndiceBootstrap(Indice):
    def __init__(
        self,
        faiBootstrap=False,
        metricaRischio="cvar",  # "cvar", "percentile", oppure "downside_penalty"
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=False,
    ):
        super().__init__(inverti=inverti)

        self.faiBootstrap = faiBootstrap
        self.metricaRischio = metricaRischio
        self.alpha = alpha
        self.lambdaPenalitaStabilita = lambdaPenalitaStabilita

    def _aggregaValori(self, vals: np.ndarray) -> np.ndarray:
        """
        Aggrega le valutazioni bootstrap (forma: n_portafogli, nBootstrap)
        usando metriche asimmetriche o coerenti con il rischio di coda.
        """
        if self.metricaRischio == "cvar":
            # Expected Shortfall: media condizionata delle osservazioni peggiori sotto l'alfa-quantile
            q = np.quantile(vals, self.alpha, axis=1, keepdims=True)
            mask = vals <= q
            counts = np.maximum(1, np.sum(mask, axis=1))
            return np.sum(np.where(mask, vals, 0.0), axis=1) / counts

        elif self.metricaRischio == "percentile":
            # Quantile conservativo (es. 5° percentile della distribuzione bootstrap)
            return np.quantile(vals, self.alpha, axis=1)

        elif self.metricaRischio == "downside_penalty":
            # Media penalizzata unicamente dalla semi-deviazione standard negativa
            media = np.mean(vals, axis=1)
            diff_negativa = np.minimum(0.0, vals - media[:, None])
            semi_dev = np.sqrt(np.mean(diff_negativa**2, axis=1))
            return media - self.lambdaPenalitaStabilita * semi_dev

        # Fallback neutrale alla media semplice
        return np.mean(vals, axis=1)

    def _calcolaSemplice(self, portafoglio, stats):
        raise NotImplementedError("Metodo _calcolaSemplice() non implementato nella classe derivata.")

    def _calcolaBootstrapFitness(self, portafoglio, stats, *args, **kwargs):
        campioni = getattr(stats, "listaBootstrap", [])
        if not campioni:
            return None

        if len(campioni) == 1:
            return self._calcolaSemplice(portafoglio, campioni[0], *args, **kwargs)

        p = np.atleast_2d(portafoglio)

        # FIX: passa 's', non 'stats'
        vals = np.column_stack([np.atleast_1d(self._calcolaSemplice(p, s, *args, **kwargs)).ravel() for s in campioni])

        result = self._aggregaValori(vals)
        return result if portafoglio.ndim > 1 else result[0]

    def calcola(self, portafoglio, stats, standardizza=True, inverti=True, debug=False, *args, **kwargs):
        if debug and self.faiBootstrap:
            print(f"\n{Fore.MAGENTA}Calcolo l'indice {self.__class__.__name__}...{Fore.RESET}")
            t0 = time.time()

        if self.faiBootstrap:
            ratio = self._calcolaBootstrapFitness(portafoglio, stats, *args, **kwargs)
        else:
            ratio = self._calcolaSemplice(portafoglio, stats, *args, **kwargs)

        if debug and self.faiBootstrap:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

        return super().standardizzaEInverti(ratio, standardizza, inverti)
