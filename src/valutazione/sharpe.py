import time

import numpy as np
from colorama import Fore
from numba import njit

from src.const import DEFAULT_N_BOOSTRAP, GIORNI_IN_UN_ANNO, RISK_FREE_RATE
from src.valutazione.bootstrap import IndiceBootstrap
from src.valutazione.rischio import RischioTotaleIndice


@njit(fastmath=True)
def _sharpeNumba(portafoglio, rendimentoGiornaliero, varianzaPortafoglio, RF):
    # rendimento atteso giornaliero per ciascun portafoglio
    mu_p_daily = np.dot(portafoglio, rendimentoGiornaliero)

    mu_annuo = mu_p_daily * GIORNI_IN_UN_ANNO
    std_annua = np.sqrt(np.maximum(varianzaPortafoglio, 1e-12))

    return (mu_annuo - RF) / std_annua


class SharpeIndice(IndiceBootstrap):
    def __init__(
        self,
        faiBootstrap=False,
        nBootstrap=DEFAULT_N_BOOSTRAP,
        lunghezzaBlocco=20,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=True,
    ):
        super().__init__(
            faiBootstrap=faiBootstrap,
            nBootstrap=nBootstrap,
            lunghezzaBlocco=lunghezzaBlocco,
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=inverti,
        )

        self.RF = self.calcolaRendimentoPrivoDiRischio()
        self.rischio = RischioTotaleIndice()

    def calcolaRendimentoPrivoDiRischio(self):
        # Da implementare con FRED
        return RISK_FREE_RATE

    def preparaListaBootstrap(self, stats, randomState=None, verbose=True):
        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparando la lista bootstrap per Sharpe...{Fore.RESET}")

        super().preparaListaBootstrap(stats, randomState, verbose)

        if verbose:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def _calcolaSemplice(self, portafoglio, stats):
        p = np.atleast_2d(portafoglio)
        varianzaPortafoglio = self.rischio.calcola(p, stats)
        
        res = _sharpeNumba(p, stats.rendimentoMedioGiornaliero, varianzaPortafoglio, self.RF)
        return res if portafoglio.ndim > 1 else res[0]
