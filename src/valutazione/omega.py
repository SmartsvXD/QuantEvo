import numpy as np
from numba import njit
import time
from colorama import Fore

from src.valutazione.bootstrap import IndiceBootstrap
from src.const import GIORNI_IN_UN_ANNO, DEFAULT_N_BOOSTRAP, MIN_OMEGA_VALUE


@njit(fastmath=True)
def _omegaNumba(portafoglio, samples, soglia):
    # portafoglio: (n_portafogli, n_asset) or (n_asset,) -> at least 2d
    # samples: (n_asset, n_samples) rendimenti giornalieri
    # soglia: valore giornaliero soglia (non annualizzato)
    portafoglio = np.atleast_2d(portafoglio)

    # calcola rendimenti per tutti i portafogli in una singola operazione
    # portafoglio (n_port, n_asset) dot samples (n_asset, n_samples) -> (n_port, n_samples)
    port_returns = np.dot(portafoglio, samples)

    # differenze rispetto alla soglia (broadcast)
    diff = port_returns - soglia

    # somme dei guadagni sopra soglia e delle perdite sotto soglia per ogni portafoglio
    up = np.sum(np.maximum(diff, 0.0), axis=1)
    down = np.sum(np.maximum(-diff, 0.0), axis=1)

    # costruisci risultato vettorialmente (gestendo casi di down ~= 0)
    omega_vals = np.where(down < 1e-12, np.where(up > 0.0, 1e12, 0.0), up / down)

    return omega_vals


@njit(fastmath=True)
def _calcolaCAGR(portafoglio, rendimentoMedioGiornaliero):
    # portafoglio: (n_portafogli, n_asset) or (n_asset,) -> at least 2d
    # rendimentoMedioGiornaliero: vector of daily log-returns (n_asset,)
    portafoglio = np.atleast_2d(portafoglio)

    # rendimento medio giornaliero atteso per ogni portafoglio
    rendimento_port_daily = np.dot(portafoglio, rendimentoMedioGiornaliero)

    # CAGR: annualizziamo i log-return medi giornalieri
    cagr_vals = np.exp(rendimento_port_daily * GIORNI_IN_UN_ANNO) - 1.0

    # gestiamo possibili NaN
    cagr_vals = np.where(np.isnan(cagr_vals), 0.0, cagr_vals)

    return cagr_vals


class OmegaIndice(IndiceBootstrap):
    def __init__(
        self,
        sogliaAnnua=0.0,
        faiBootstrap=False,
        nBootstrap=DEFAULT_N_BOOSTRAP,
        lambdaPenalitaStabilita=0.5,
        usaBlockBootstrap=True,
        lunghezzaBlocco=20,
    ):
        super().__init__(faiBootstrap, nBootstrap, lambdaPenalitaStabilita, usaBlockBootstrap, lunghezzaBlocco)

        # soglia annuale (es. 0 per zero target, o risk-free rate annuale se desiderato)
        if sogliaAnnua <= -1.0:
            self._sogliaGiornaliera = sogliaAnnua / GIORNI_IN_UN_ANNO
        else:
            self._sogliaGiornaliera = np.log1p(sogliaAnnua) / GIORNI_IN_UN_ANNO

    def preparaListaBootstrap(self, stats, randomState=None, forzaRicalcolo=True, verbose=True):
        pathStats = "./data/statsBootstrapOmega.data"
        pathLista = "./data/listaBootstrapOmega.data"

        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparando la lista bootstrap per Omega...{Fore.RESET}")

        super().preparaListaBootstrap(stats, pathStats, pathLista, randomState, forzaRicalcolo, verbose)

        if verbose:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def _calcolaSemplice(self, portafoglio, stats):
        # chiamiamo eventuale comportamento comune
        super()._calcolaSemplice(portafoglio, stats)

        return _omegaNumba(portafoglio, stats.rendimenti, self._sogliaGiornaliera)

    def _calcolaBootstrap(self, portafoglio, stats, randomState=None):
        return super()._calcolaBootstrap(portafoglio, stats, randomState)

    def _calcolaBootstrapFitness(self, portafoglio):
        return super()._calcolaBootstrapFitness(portafoglio)

    def calcola(self, portafoglio, stats, standardizza=False):
        return super().calcola(portafoglio, stats, standardizza)


class CAGROmegaIndice(OmegaIndice):
    def calcola(self, portafoglio, stats, standardizza=False):
        omega = super().calcola(portafoglio, stats, standardizza=False)

        cagr = _calcolaCAGR(portafoglio, stats.rendimentoMedioGiornaliero)

        p = cagr * (omega - MIN_OMEGA_VALUE)
        penalized_mask = (cagr < 0.0) | (omega < MIN_OMEGA_VALUE)
        totale = np.where(penalized_mask, -np.abs(p), p)

        return self.standardizza(totale, standardizza)
