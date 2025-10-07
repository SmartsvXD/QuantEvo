from numba import njit
import numpy as np
from settings import GIORNI_PER_MESE, ALIQUOTA_TASSE


@njit(fastmath=True)
def _runSimulation(Z, Z_div, nAsset, pesi, nSims, orizzonte, drift, coeffDiffusione, S0, mediaDividendi, volatilitàDividendi, dt, ribilancio):
    # Matrici casuali per ogni asset e simulazione
    incrementiGiornalieri = drift + coeffDiffusione * Z  # shape (nSims, orizzonte, n_asset)

    # Array per il valore degli asset in ogni simulazione e giorno
    valoriComponenti = np.zeros((nSims, orizzonte, nAsset), dtype=np.float32)
    valoriComponenti[:, 0, :] = S0 * pesi

    # Simulazione stocastica dei rendimenti da dividendo giornalieri (GBM additivo)
    simRendimentiDividendi = mediaDividendi * dt + volatilitàDividendi * np.sqrt(dt) * Z_div

    # Array per accumulare dividendi mensili (per asset)
    dividendiMensili = np.zeros((nSims, nAsset), dtype=np.float32)

    for day in range(1, orizzonte):
        valoriComponenti[:, day, :] = valoriComponenti[:, day - 1, :] * np.exp(incrementiGiornalieri[:, day, :])

        # Calcola dividendi giornalieri simulati (stocastici)
        dividendiGiornalieri = valoriComponenti[:, day, :] * simRendimentiDividendi[:, day, :]
        dividendiMensili += dividendiGiornalieri

        if day % GIORNI_PER_MESE == 0:
            # Ribilanciamento mensile: reinvesti ribilancio e dividendi asset per asset
            valoriComponenti[:, day, :] += dividendiMensili * (1 - ALIQUOTA_TASSE)
            valoriComponenti[:, day, :] += ribilancio * pesi
            # Ribilancia secondo i pesi
            totalePost = valoriComponenti[:, day, :].sum(axis=1)
            valoriComponenti[:, day, :] = (totalePost)[:, None] * pesi
            dividendiMensili[:] = 0
            
    valoriTotali = valoriComponenti.sum(axis=2)
    valoriFinali = valoriTotali[:, -1]
    mediana = np.median(valoriFinali)
    media = np.mean(valoriFinali)
    percentili = np.percentile(valoriFinali, [10, 25, 75, 90])

    return valoriTotali, valoriFinali, mediana, media, percentili