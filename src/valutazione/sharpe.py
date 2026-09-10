import numpy as np
from numba import njit
import time
from colorama import Fore

from src.valutazione.bootstrap import IndiceBootstrap
from src.valutazione.rischio import RischioTotaleIndice
from src.const import GIORNI_IN_UN_ANNO, RISK_FREE_RATE, DEFAULT_N_BOOSTRAP


@njit(fastmath=True)
def _sharpeNumba(portafoglio, rendimentoGiornaliero, varianzaPortafoglio, RF):
    # Here `rendimentoGiornaliero` is assumed to be daily log-returns (mu_d)
    # and `varianzaPortafoglio` is annualized log-variance (sigma2_a),
    # as produced by RischioTotaleIndice using annualized covariance.
    # We convert to annual arithmetic moments under log-normal assumption:
    # mu_a = mu_d * N, sigma2_a already annualized
    # E[AR] = exp(mu_a + 0.5*sigma2_a) - 1
    # Var(AR) = exp(2*mu_a + sigma2_a) * (exp(sigma2_a) - 1)

    N = GIORNI_IN_UN_ANNO

    portafoglio = np.atleast_2d(portafoglio)

    # portfolio daily log-mean
    mu_p_daily = np.dot(portafoglio, rendimentoGiornaliero)

    # annualize
    mu_a = mu_p_daily * N
    sigma2_a = varianzaPortafoglio

    # expected annual arithmetic return and variance under log-normal
    exp_mu = np.exp(mu_a + 0.5 * sigma2_a) - 1.0
    var_ar = np.exp(2.0 * mu_a + sigma2_a) * (np.exp(sigma2_a) - 1.0)

    std_ar = np.sqrt(var_ar)
    std_ar = np.where(std_ar < 1e-12, 1e-12, std_ar)

    sharpe = (exp_mu - RF) / std_ar

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
        super()._calcolaSemplice(portafoglio, stats)

        varianzaPortafoglio = self.rischio.calcola(portafoglio, stats)
        return _sharpeNumba(portafoglio, stats.rendimentoMedioGiornaliero, varianzaPortafoglio, self.RF)

    def _calcolaBootstrap(self, portafoglio, stats, randomState=None):
        return super()._calcolaBootstrap(portafoglio, stats, randomState)

    def _calcolaBootstrapFitness(self, portafoglio):
        return super()._calcolaBootstrapFitness(portafoglio)

    def calcola(self, portafoglio, stats, standardizza=False):
        return super().calcola(portafoglio, stats, standardizza)
