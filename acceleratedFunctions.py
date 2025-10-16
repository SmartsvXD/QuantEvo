from numba import njit
import numpy as np
from settings import GIORNI_PER_MESE, ALIQUOTA_TASSE


@njit(fastmath=True)
def _runSimulation(
    Z,
    nAsset,
    pesi,
    nSims,
    orizzonte,
    drift,
    coeffDiffusione,
    S0,
    mediaDividendi,
    dt,
    ribilancio,
):
    # Matrici casuali per ogni asset e simulazione
    incrementiGiornalieri = drift + coeffDiffusione * Z  # shape (nSims, orizzonte, n_asset)
    investimento = S0

    # Array per il valore degli asset in ogni simulazione e giorno
    valoriComponenti = np.zeros((nSims, orizzonte, nAsset), dtype=np.float32)
    valoriComponenti[:, 0, :] = S0 * pesi

    # I dividendi sono trattati come flussi deterministici (non stocastici)
    # Rendimento giornaliero medio da dividendi per ogni asset
    rendimentoDividendiGiornaliero = mediaDividendi * dt

    # Array per accumulare dividendi mensili (per asset)
    dividendiMensili = np.zeros((nSims, nAsset), dtype=np.float32)

    for day in range(1, orizzonte):
        valoriComponenti[:, day, :] = valoriComponenti[:, day - 1, :] * np.exp(incrementiGiornalieri[:, day, :])

        # Calcola dividendi giornalieri come flusso deterministico
        # (i dividendi degli ETF sono relativamente prevedibili)
        dividendiGiornalieri = valoriComponenti[:, day, :] * rendimentoDividendiGiornaliero
        dividendiMensili += dividendiGiornalieri

        if day % GIORNI_PER_MESE == 0:
            # Ribilanciamento mensile: reinvesti dividendi (dopo tasse) + nuovi apporti
            valoriComponenti[:, day, :] += dividendiMensili * (1 - ALIQUOTA_TASSE)
            valoriComponenti[:, day, :] += ribilancio * pesi
            investimento = investimento + ribilancio

            # Ribilancia secondo i pesi target
            totalePost = valoriComponenti[:, day, :].sum(axis=1)
            valoriComponenti[:, day, :] = (totalePost)[:, None] * pesi
            dividendiMensili[:] = 0

    valoriTotali = valoriComponenti.sum(axis=2)
    valoriFinali = valoriTotali[:, -1]
    mediana = np.median(valoriFinali) / investimento - 1
    media = np.mean(valoriFinali) / investimento - 1
    percentili = np.percentile(valoriFinali, [10, 25, 75, 90]) / investimento - 1

    return investimento, valoriTotali, valoriFinali, mediana, media, percentili
