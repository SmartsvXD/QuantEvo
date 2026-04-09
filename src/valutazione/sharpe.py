import numpy as np
from numba import njit
import time
from colorama import Fore

from src.valutazione.bootstrap import IndiceBootstrap
from src.valutazione.rischio import RischioTotaleIndice
from src.const import GIORNI_IN_UN_ANNO, RISK_FREE_RATE, DEFAULT_N_BOOSTRAP


@njit(fastmath=True)
def _sharpeNumba(portafoglio, rendimentoGiornaliero, varianzaPortafoglio, RF):
    rendimentoAnnuo = rendimentoGiornaliero * GIORNI_IN_UN_ANNO

    portafoglio = np.atleast_2d(portafoglio)

    rendimentoAtteso = np.dot(portafoglio, rendimentoAnnuo)

    rischio = np.sqrt(varianzaPortafoglio)
    rischio = np.where(rischio < 1e-12, 1e-12, rischio)

    sharpe = (rendimentoAtteso - RF) / rischio
    return sharpe


class SharpeIndice(IndiceBootstrap):
    def __init__(
        self,
        faiBootstrap=False,
        nBootstrap=DEFAULT_N_BOOSTRAP,
        lambdaPenalitaStabilita=0.5,
        usaBlockBootstrap=True,
        lunghezzaBlocco=20,
    ):
        super().__init__(faiBootstrap, nBootstrap, lambdaPenalitaStabilita, usaBlockBootstrap, lunghezzaBlocco)

        self.RF = self.calcolaRendimentoPrivoDiRischio()
        self.rischio = RischioTotaleIndice()

    def calcolaRendimentoPrivoDiRischio(self):
        # Da implementare, utilizzeremo FRED
        return RISK_FREE_RATE

    def preparaListaBootstrap(self, stats, randomState=None, forzaRicalcolo=True, verbose=True):
        pathStats = "./data/statsBootstrapSharpe.data"
        pathLista = "./data/listaBootstrapSharpe.data"

        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparando la lista bootstrap per sharpe...{Fore.RESET}")

        super().preparaListaBootstrap(stats, pathStats, pathLista, randomState, forzaRicalcolo, verbose)

        if verbose:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def _calcolaSemplice(self, portafoglio, stats):
        # port_returns = np.tensordot(p, samples, axes=(1, 0))

        # rend_atteso = np.nanmean(port_returns, axis=2) * GIORNI_IN_UN_ANNO

        # var_port = np.nanvar(port_returns, axis=2, ddof=1) * GIORNI_IN_UN_ANNO

        # rischio = np.sqrt(var_port)
        # rischio[rischio < 1e-12] = 1e-12

        # sharpe_vals = (rend_atteso - self.RF) / rischio
        # sharpe_vals = np.nan_to_num(sharpe_vals, nan=-1e6, posinf=1e6, neginf=-1e6)
        
        super()._calcolaSemplice(portafoglio, stats)

        varianzaPortafoglio = self.rischio.calcola(portafoglio, stats)
        return _sharpeNumba(portafoglio, stats.rendimentoMedioGiornaliero, varianzaPortafoglio, self.RF)
    
    def _calcolaBootstrap(self, portafoglio, stats, randomState=None):
        return super()._calcolaBootstrap(portafoglio, stats, randomState)
    
    def _calcolaBootstrapFitness(self, portafoglio):
        return super()._calcolaBootstrapFitness(portafoglio)

    def calcola(self, portafoglio, stats, standardizza=False):
        return super().calcola(portafoglio, stats, standardizza)
