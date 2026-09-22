import numpy as np
from numba import njit, prange

from src.const import GIORNI_IN_UN_ANNO, MIN_OMEGA_VALUE
from src.valutazione.cagr import CAGRIndice
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

            if down < 1e-12:
                omega_vals[i, b] = 10.0 if up > 0.0 else 1.0
            else:
                val = up / down
                omega_vals[i, b] = min(val, 10.0)

    return omega_vals


class OmegaIndice(Indice):
    def __init__(
        self,
        sogliaAnnua=0.035,
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
        sogliaAnnua=0.035,
        minOmegaValue=MIN_OMEGA_VALUE,
        metricaRischio="cvar",
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=True,
    ):
        self.minOmegaValue = minOmegaValue

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

        self.cagrIndice = CAGRIndice(
            metricaRischio=metricaRischio,
            alpha=alpha,
            lambdaPenalitaStabilita=lambdaPenalitaStabilita,
            inverti=False,
        )

    def _calcola(self, portafoglio: np.ndarray, tensori: dict):
        p = np.atleast_2d(portafoglio)
        omega = self.omegaIndice._calcola(p, tensori)
        cagr = self.cagrIndice._calcola(p, tensori)

        omega_diff = omega - self.minOmegaValue
        valido = (cagr > 0.0) & (omega_diff > 0.0)
        p_valido = cagr * omega_diff

        deficit_cagr = np.maximum(0.0, -cagr)
        deficit_omega = np.maximum(0.0, -omega_diff)
        penalita = -(deficit_cagr + deficit_omega)

        return np.where(valido, p_valido, penalita)
