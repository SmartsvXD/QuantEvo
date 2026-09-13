import time

import numpy as np
from colorama import Fore
from numba import njit

from src.const import DEFAULT_N_BOOSTRAP, GIORNI_IN_UN_ANNO, RISK_FREE_RATE
from src.valutazione.bootstrap import IndiceBootstrap
from src.valutazione.rischio import RischioTotaleIndice


@njit(fastmath=True)
def _sharpeNumba(portafoglio, rendimentoGiornaliero, varianzaPortafoglio, RF):
    portafoglio = np.atleast_2d(portafoglio)

    mu_p_daily = np.dot(portafoglio, rendimentoGiornaliero)

    mu_annuo = mu_p_daily * GIORNI_IN_UN_ANNO
    varianza_annua = varianzaPortafoglio
    std_annua = np.sqrt(varianza_annua)

    sharpe = (mu_annuo - RF) / std_annua

    return sharpe


class SharpeIndice(IndiceBootstrap):
    def __init__(
        self,
        faiBootstrap=False,
        nBootstrap=DEFAULT_N_BOOSTRAP,
        lambdaPenalitaStabilita=0.5,
        usaBlockBootstrap=True,
        lunghezzaBlocco=20,
        inverti=True,
    ):
        super().__init__(faiBootstrap, nBootstrap, lambdaPenalitaStabilita, usaBlockBootstrap, lunghezzaBlocco, inverti)

        self.RF = self.calcolaRendimentoPrivoDiRischio()
        self.rischio = RischioTotaleIndice()

    def calcolaRendimentoPrivoDiRischio(self):
        # Da implementare, utilizzeremo FRED
        return RISK_FREE_RATE

    def preparaListaBootstrap(self, stats, randomState=None, verbose=True):

        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparando la lista bootstrap per sharpe...{Fore.RESET}")

        super().preparaListaBootstrap(stats, randomState, verbose)

        if verbose:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def _calcolaSemplice(self, portafoglio, stats):
        super()._calcolaSemplice(portafoglio, stats)

        varianzaPortafoglio = self.rischio.calcola(portafoglio, stats)
        return _sharpeNumba(portafoglio, stats.rendimentoMedioGiornaliero, varianzaPortafoglio, self.RF)

    def _calcolaBootstrap(self, portafoglio, stats, randomState=None):
        return super()._calcolaBootstrap(portafoglio, stats, randomState)

    def _calcolaBootstrapFitness(self, portafoglio):
        return super()._calcolaBootstrapFitness(portafoglio)

    def calcola(self, portafoglio, stats, standardizza=True, inverti=True):
        return super().calcola(portafoglio, stats, standardizza, inverti)
