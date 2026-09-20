# Deprecated

import numpy as np
from numba import njit, prange

from src.const import GIORNI_IN_UN_ANNO, RISK_FREE_RATE
from src.valutazione.indice import Indice


@njit(fastmath=True)
def _sharpeNumba(portafogli, rendimentiCumulati, RF, giorni_in_un_anno):
    """
    Calcola lo Sharpe ratio Buy & Hold per uno o più portafogli.
    rendimentiCumulati: matrice (T + 1, N) con la riga 0 valorizzata a 1.0 (base iniziale).
    """
    n_portafogli = portafogli.shape[0]
    T = rendimentiCumulati.shape[0] - 1
    risultati = np.empty(n_portafogli, dtype=np.float64)

    for m in range(n_portafogli):
        w = portafogli[m]

        # Valore iniziale del portafoglio al tempo t=0
        v_prev = 0.0
        for i in range(len(w)):
            v_prev += w[i] * rendimentiCumulati[0, i]

        somma_r = 0.0
        somma_r2 = 0.0

        # Calcolo dei rendimenti giornalieri effettivi non ribilanciati
        for t in range(1, T + 1):
            v_curr = 0.0
            for i in range(len(w)):
                v_curr += w[i] * rendimentiCumulati[t, i]

            r_t = (v_curr / v_prev) - 1.0
            v_prev = v_curr

            somma_r += r_t
            somma_r2 += r_t * r_t

        # Media e varianza campionaria giornaliera
        mu_daily = somma_r / T
        var_daily = (somma_r2 - T * (mu_daily**2)) / (T - 1)
        var_daily = max(var_daily, 0.0)

        # Annualizzazione
        mu_annuo = mu_daily * giorni_in_un_anno
        std_annua = np.sqrt(max(var_daily * giorni_in_un_anno, 1e-12))

        risultati[m] = (mu_annuo - RF) / std_annua

    return risultati


@njit(parallel=True, fastmath=True)
def _sharpeBootstrapNumba(portafogli: np.ndarray, cum_boot: np.ndarray, RF: float, giorni_anno: float) -> np.ndarray:
    n_port = portafogli.shape[0]
    n_boot, n_steps, _ = cum_boot.shape
    T = n_steps - 1
    risultati = np.empty((n_port, n_boot), dtype=np.float64)

    for b in prange(n_boot):
        cum = cum_boot[b]
        for m in range(n_port):
            w = portafogli[m]
            v_prev = 0.0
            for i in range(len(w)):
                v_prev += w[i] * cum[0, i]

            somma_r = 0.0
            somma_r2 = 0.0
            for t in range(1, T + 1):
                v_curr = 0.0
                for i in range(len(w)):
                    v_curr += w[i] * cum[t, i]

                r_t = (v_curr / v_prev) - 1.0
                v_prev = v_curr
                somma_r += r_t
                somma_r2 += r_t * r_t

            mu_daily = somma_r / T
            var_daily = max((somma_r2 - T * (mu_daily**2)) / (T - 1), 0.0)
            mu_annuo = mu_daily * giorni_anno
            std_annua = np.sqrt(max(var_daily * giorni_anno, 1e-12))

            risultati[m, b] = (mu_annuo - RF) / std_annua

    return risultati


class SharpeIndice(Indice):
    def __init__(
        self,
        faiBootstrap=False,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=True,
    ):
        super().__init__(
            faiBootstrap=faiBootstrap,
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=inverti,
        )
        self.RF = self.calcolaRendimentoPrivoDiRischio()

    def calcolaRendimentoPrivoDiRischio(self):
        # Da implementare con FRED
        return RISK_FREE_RATE

    def _ottieniRendimentiCumulati(self, stats):
        N, T = stats.stats["rendimenti"].shape
        cum = np.empty((T + 1, N), dtype=np.float64)
        cum[0] = 1.0
        # Correzione assi temporali: trasposizione da (N, T) a (T, N)
        cum[1:] = np.cumprod(1.0 + stats.rendimenti.T, axis=0)
        return cum

    def _ottieniRendimentiCumulatiBoot(self, stats):
        rend_boot = stats.stats_boot["rendimenti"]
        B, N, T = rend_boot.shape
        cum_boot = np.empty((B, T + 1, N), dtype=np.float64)
        cum_boot[:, 0, :] = 1.0
        cum_boot[:, 1:, :] = np.cumprod(1.0 + np.transpose(rend_boot, (0, 2, 1)), axis=1)
        return cum_boot

    def _calcolaSemplice(self, portafoglio, stats):
        p = np.atleast_2d(portafoglio)
        cum = self._ottieniRendimentiCumulati(stats)
        return _sharpeNumba(p, cum, self.RF, GIORNI_IN_UN_ANNO)

    def _calcolaBootstrap(self, portafoglio, stats):
        p = np.atleast_2d(portafoglio)
        cum_boot = self._ottieniRendimentiCumulatiBoot(stats)
        return _sharpeBootstrapNumba(p, cum_boot, self.RF, GIORNI_IN_UN_ANNO)
