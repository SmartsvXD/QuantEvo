import numpy as np
from numba import njit, prange

from src.const import GIORNI_IN_UN_ANNO, MIN_OMEGA_VALUE
from src.valutazione.indice import Indice


@njit(parallel=True, fastmath=True)
def _omegaNumba(portafoglio: np.ndarray, samples_3d: np.ndarray, soglia: float) -> np.ndarray:
    n_port, n_asset = portafoglio.shape
    n_boot, _, n_samples = samples_3d.shape
    omega_vals = np.empty((n_port, n_boot), dtype=np.float64)

    for i in prange(n_port):
        asset_values = np.empty(n_asset, dtype=np.float64)
        for b in range(n_boot):
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
                    asset_values[a] *= 1.0 + samples_3d[b, a, t]
                    v_curr += asset_values[a]

                ret = (v_curr - v_prev) / v_prev if abs(v_prev) > 1e-12 else 0.0
                v_prev = v_curr

                diff = ret - soglia
                if diff > 0.0:
                    up += diff
                else:
                    down -= diff

            omega_vals[i, b] = (1e12 if up > 0.0 else 0.0) if down < 1e-12 else up / down

    return omega_vals


class OmegaIndice(Indice):
    def __init__(
        self,
        sogliaAnnua=0.0,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=True,
    ):
        super().__init__(
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=inverti,
        )

        if sogliaAnnua <= -1.0:
            raise ValueError("sogliaAnnua deve essere strettamente maggiore di -1.0 (-100%).")

        self._sogliaGiornaliera = (1.0 + sogliaAnnua) ** (1.0 / GIORNI_IN_UN_ANNO) - 1.0

    def _calcola(self, portafoglio: np.ndarray, stats: dict):
        p = np.atleast_2d(portafoglio)
        return _omegaNumba(p, stats["rendimenti"], self._sogliaGiornaliera)


class CAGROmegaIndice(Indice):
    def __init__(
        self,
        sogliaAnnua=0.0,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=True,
    ):
        super().__init__(
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=inverti,
        )

        self.omegaIndice = OmegaIndice(
            sogliaAnnua=sogliaAnnua,
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=False,
        )

    def _calcola(self, portafoglio: np.ndarray, tensori: dict):
        p = np.atleast_2d(portafoglio)
        omega = self.omegaIndice._calcola(p, tensori)

        rend_boot = tensori["rendimenti"]
        rend_clipped = np.clip(rend_boot, -0.999999, None)
        montante_asset = np.exp(np.sum(np.log1p(rend_clipped), axis=2))  # Shape: (B, N)
        valore_finale = np.einsum("pi,bi->pb", p, montante_asset)        # Shape: (P, B)

        esponente = GIORNI_IN_UN_ANNO / rend_boot.shape[2]
        cagr = np.where(valore_finale > 0.0, valore_finale**esponente - 1.0, -1.0)
        cagr = np.where(np.isnan(cagr), 0.0, cagr)

        omega_diff = omega - MIN_OMEGA_VALUE
        valido = (cagr > 0.0) & (omega_diff > 0.0)
        p_valido = cagr * omega_diff

        deficit_cagr = np.maximum(0.0, -cagr)
        deficit_omega = np.maximum(0.0, -omega_diff)
        penalita = -(deficit_cagr + deficit_omega)

        return np.where(valido, p_valido, penalita)
