import numpy as np
from numba import njit
import time
from colorama import Fore

from src.valutazione.bootstrap import IndiceBootstrap
from src.const import GIORNI_IN_UN_ANNO, DEFAULT_N_BOOSTRAP, MIN_OMEGA_VALUE


@njit(fastmath=True)
def _omegaNumba(portafoglio, samples, soglia):
    portafoglio = np.atleast_2d(portafoglio)
    n_port, n_asset = portafoglio.shape
    n_samples = samples.shape[1]

    omega_vals = np.empty(n_port, dtype=np.float64)

    for i in range(n_port):
        up = 0.0
        down = 0.0
        for t in range(n_samples):
            ret = 0.0
            for a in range(n_asset):
                ret += portafoglio[i, a] * samples[a, t]

            diff = ret - soglia
            if diff > 0.0:
                up += diff
            else:
                down -= diff

        if down < 1e-12:
            omega_vals[i] = 1e12 if up > 0.0 else 0.0
        else:
            omega_vals[i] = up / down

    return omega_vals


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

        if sogliaAnnua <= -1.0:
            raise ValueError("sogliaAnnua deve essere strettamente maggiore di -1.0 (-100%).")

        # Conversione corretta della soglia composta annua in giornaliera
        self._sogliaGiornaliera = (1.0 + sogliaAnnua) ** (1.0 / GIORNI_IN_UN_ANNO) - 1.0

    def preparaListaBootstrap(self, stats, randomState=None, verbose=True):
        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparando la lista bootstrap per Omega...{Fore.RESET}")

        super().preparaListaBootstrap(stats, randomState, verbose)

        if verbose:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def _calcolaSemplice(self, portafoglio, stats):
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

        portafoglio = np.atleast_2d(portafoglio)

        # Rendimenti giornalieri storici del portafoglio (n_portafogli, n_samples)
        ret_port_daily = np.dot(portafoglio, stats.rendimenti)

        # Vero CAGR geometrico (usando log1p per stabilità ed evitare underflow numerici)
        ret_port_clipped = np.clip(ret_port_daily, -0.999999, None)
        cagr = np.expm1(np.mean(np.log1p(ret_port_clipped), axis=1) * GIORNI_IN_UN_ANNO)
        cagr = np.where(np.isnan(cagr), 0.0, cagr)

        # Differenziale rispetto alla soglia minima di Omega
        omega_diff = omega - MIN_OMEGA_VALUE

        # Punteggio base e penalizzazione monotona coerente
        valido = (cagr > 0.0) & (omega_diff > 0.0)
        p_valido = cagr * omega_diff

        # Penalità cumulativa proporzionale al deficit: chi peggiora ottiene valori sempre più negativi
        deficit_cagr = np.maximum(0.0, -cagr)
        deficit_omega = np.maximum(0.0, -omega_diff)
        penalita = -(deficit_cagr + deficit_omega)

        totale = np.where(valido, p_valido, penalita)

        # Se il portafoglio in ingresso era 1D, appiattisce il risultato
        if totale.size == 1:
            totale = totale.ravel()

        return self.standardizza(totale, standardizza)
