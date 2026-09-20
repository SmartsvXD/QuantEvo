import numpy as np
from numba import njit

from src.const import DEFAULT_N_BOOSTRAP, GIORNI_IN_UN_ANNO, RISK_FREE_RATE
from src.valutazione.bootstrap import IndiceBootstrap


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

    def calcolaRendimentoPrivoDiRischio(self):
        # Da implementare con FRED
        return RISK_FREE_RATE

    def _ottieniRendimentiCumulati(self, stats):
        """Precalcola e memorizza la serie dei fattori cumulati per evitare ricalcoli."""
        if not hasattr(stats, "rendimentiCumulati"):
            T, N = stats.rendimenti.shape
            cum = np.empty((T + 1, N), dtype=np.float64)
            cum[0] = 1.0
            cum[1:] = np.cumprod(1.0 + stats.rendimenti, axis=0)
            stats.rendimentiCumulati = cum
        return stats.rendimentiCumulati

    def _calcolaSemplice(self, portafoglio, stats):
        p = np.atleast_2d(portafoglio)
        rendimentiCumulati = self._ottieniRendimentiCumulati(stats)

        res = _sharpeNumba(
            p,
            rendimentiCumulati,
            self.RF,
            GIORNI_IN_UN_ANNO,
        )
        return res if portafoglio.ndim > 1 else res[0]
