import numpy as np
from numba import njit

from src.const import DEFAULT_N_BOOSTRAP, GIORNI_IN_UN_ANNO, MIN_OMEGA_VALUE
from src.valutazione.bootstrap import IndiceBootstrap


@njit(fastmath=True)
def _omegaNumba(portafoglio, samples, soglia):
    n_port, n_asset = portafoglio.shape
    n_samples = samples.shape[1]
    omega_vals = np.empty(n_port, dtype=np.float64)
    asset_values = np.empty(n_asset, dtype=np.float64)

    for i in range(n_port):
        # Inizializza il capitale allocato per singolo asset
        v_prev = 0.0
        for a in range(n_asset):
            val = portafoglio[i, a]
            asset_values[a] = val
            v_prev += val

        up = 0.0
        down = 0.0

        for t in range(n_samples):
            v_curr = 0.0
            for a in range(n_asset):
                asset_values[a] *= 1.0 + samples[a, t]
                v_curr += asset_values[a]

            # Rendimento periodale del portafoglio buy-and-hold
            ret = (v_curr - v_prev) / v_prev if abs(v_prev) > 1e-12 else 0.0
            v_prev = v_curr

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

    def _calcolaSemplice(self, portafoglio, stats):
        p = np.atleast_2d(portafoglio)
        res = _omegaNumba(p, stats.rendimenti, self._sogliaGiornaliera)
        return res if portafoglio.ndim > 1 else res[0]


class CAGROmegaIndice(IndiceBootstrap):
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

        self.omegaIndice = OmegaIndice(
            sogliaAnnua=sogliaAnnua,
            faiBootstrap=False,
            nBootstrap=nBootstrap,
            lunghezzaBlocco=lunghezzaBlocco,
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=False,
        )

    def _calcolaSemplice(self, portafoglio, stats):
        omega = self.omegaIndice.calcola(portafoglio, stats, standardizza=False, inverti=False)

        p = np.atleast_2d(portafoglio)

        rend_clipped = np.clip(stats.rendimenti, -0.999999, None)
        montante_asset = np.exp(np.sum(np.log1p(rend_clipped), axis=1))

        esponente = GIORNI_IN_UN_ANNO / stats.rendimenti.shape[1]

        # 1. Valore finale del portafoglio (combinazione lineare dei pesi iniziali)
        valore_finale = np.dot(p, montante_asset)  # Shape: (n_port,)

        # 2. Annualizzazione del rendimento geometrico
        cagr = np.where(valore_finale > 0.0, valore_finale**esponente - 1.0, -1.0)
        cagr = np.where(np.isnan(cagr), 0.0, cagr)

        # --- Penalizzazione e score finale ---
        omega_diff = omega - MIN_OMEGA_VALUE
        valido = (cagr > 0.0) & (omega_diff > 0.0)
        p_valido = cagr * omega_diff

        deficit_cagr = np.maximum(0.0, -cagr)
        deficit_omega = np.maximum(0.0, -omega_diff)
        penalita = -(deficit_cagr + deficit_omega)

        totale = np.where(valido, p_valido, penalita)

        if portafoglio.ndim == 1:
            totale = totale[0]

        return totale
