import time

import numpy as np
from colorama import Fore
from numba import njit

from src.const import DEFAULT_N_BOOSTRAP, GIORNI_IN_UN_ANNO, MIN_OMEGA_VALUE
from src.valutazione.bootstrap import IndiceBootstrap
from src.valutazione.indice import Indice


@njit(fastmath=True)
def _omegaNumba(portafoglio, samples, soglia):
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

        if sogliaAnnua <= -1.0:
            raise ValueError("sogliaAnnua deve essere strettamente maggiore di -1.0 (-100%).")

        self._sogliaGiornaliera = (1.0 + sogliaAnnua) ** (1.0 / GIORNI_IN_UN_ANNO) - 1.0

    def preparaListaBootstrap(self, stats, randomState=None, verbose=True):
        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparando la lista bootstrap per Omega...{Fore.RESET}")

        super().preparaListaBootstrap(stats, randomState, verbose)

        if verbose:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def _calcolaSemplice(self, portafoglio, stats):
        p = np.atleast_2d(portafoglio)
        res = _omegaNumba(p, stats.rendimenti, self._sogliaGiornaliera)
        return res if portafoglio.ndim > 1 else res[0]


class CAGROmegaIndice(Indice):
    def __init__(
        self,
        sogliaAnnua=0.0,
        faiBootstrap=False,
        nBootstrap=DEFAULT_N_BOOSTRAP,
        lunghezzaBlocco=20,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=True,
    ):
        super().__init__(inverti=inverti)

        self.omegaIndice = OmegaIndice(
            sogliaAnnua=sogliaAnnua,
            faiBootstrap=faiBootstrap,
            nBootstrap=nBootstrap,
            lunghezzaBlocco=lunghezzaBlocco,
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=False,
        )

    def preparaListaBootstrap(self, stats, randomState=None, verbose=True):
        self.omegaIndice.preparaListaBootstrap(stats, randomState, verbose)

    def calcola(self, portafoglio, stats, standardizza=True, inverti=True):
        omega = self.omegaIndice.calcola(portafoglio, stats, standardizza=False, inverti=False)

        p = np.atleast_2d(portafoglio)
        ret_port_daily = np.dot(p, stats.rendimenti)

        ret_port_clipped = np.clip(ret_port_daily, -0.999999, None)
        cagr = np.expm1(np.mean(np.log1p(ret_port_clipped), axis=1) * GIORNI_IN_UN_ANNO)
        cagr = np.where(np.isnan(cagr), 0.0, cagr)

        omega_diff = omega - MIN_OMEGA_VALUE
        valido = (cagr > 0.0) & (omega_diff > 0.0)
        p_valido = cagr * omega_diff

        deficit_cagr = np.maximum(0.0, -cagr)
        deficit_omega = np.maximum(0.0, -omega_diff)
        penalita = -(deficit_cagr + deficit_omega)

        totale = np.where(valido, p_valido, penalita)

        if portafoglio.ndim == 1:
            totale = totale[0]

        return super().calcola(portafoglio, stats, totale, standardizza, inverti)
