from numba import njit
import numpy as np
from settings import GIORNI_PER_MESE, ALIQUOTA_TASSE, GIORNI_IN_UN_ANNO, LAMBDA_EWMA


@njit
def _ewma(data, lambda_val = LAMBDA_EWMA):
    """
    Calcola la media ponderata esponenziale, gestendo NaN.
    """
    valid = ~np.isnan(data)
    data_valid = data[valid]
    if len(data_valid) == 0:
        return np.nan
    n = len(data_valid)
    weights = np.power(lambda_val, np.arange(n - 1, -1, -1))
    weights /= weights.sum()
    return np.sum(weights * data_valid)


@njit
def _ewma_std(data, lambda_val = LAMBDA_EWMA):
    """
    Calcola la deviazione standard ponderata esponenziale, gestendo NaN.
    """
    valid = ~np.isnan(data)
    data_valid = data[valid]
    if len(data_valid) == 0:
        return np.nan
    mean_ewma = _ewma(data, lambda_val)
    n = len(data_valid)
    weights = np.power(lambda_val, np.arange(n - 1, -1, -1))
    weights /= weights.sum()
    variance = np.sum(weights * (data_valid - mean_ewma) ** 2)
    return np.sqrt(variance)


@njit
def _calcolaStatsComponenti(nomi, rendimenti, dividendi, close):
    """
    Calcola statistiche storiche per ciascun componente del portafoglio.
    Include rendimenti giornalieri e annuali, volatilità e rendimento da dividendi.
    Utilizza Numba per accelerare i calcoli.

    Parametri:
    - nomi: lista dei nomi dei titoli
    - rendimenti: array dei log-returns giornalieri per ciascun titolo
    - dividendi: array dei dividendi giornalieri per ciascun titolo
    - close: array dei prezzi di chiusura per ciascun titolo

    Ritorna:
    Tuple con rendimenti giornalieri, volatilità giornaliera, rendimenti annuali,
    volatilità annuali, rendimento da dividendi annuale, giornaliero e volatilità dividendi.
    """
    nBeni = len(nomi)

    rendimentoGiornaliero = np.zeros(nBeni)
    volatilitàGiornaliera = np.zeros(nBeni)

    rendimentoAnnuo = np.zeros(nBeni)
    volatilitàAnnua = np.zeros(nBeni)

    rendimentoAnnuoDividendi = np.zeros(nBeni)
    rendimentoGiornalieroDividendi = np.zeros(nBeni)
    volatilitàDividendi = np.zeros(nBeni)

    # Ciclo su ciascun bene per calcolare le statistiche individuali
    for i in range(nBeni):
        # Calcola la media e la deviazione standard dei log-returns giornalieri con EWMA
        rendimentoGiornaliero[i] = _ewma(rendimenti[i])
        volatilitàGiornaliera[i] = _ewma_std(rendimenti[i])

        # Calcola il rendimento annuo come annualizzazione del rendimento giornaliero EWMA
        rendimentoAnnuo[i] = rendimentoGiornaliero[i] * GIORNI_IN_UN_ANNO

        # Annualizza la volatilità giornaliera moltiplicando per sqrt(252)
        volatilitàAnnua[i] = volatilitàGiornaliera[i] * np.sqrt(GIORNI_IN_UN_ANNO)

        # Calcola il rendimento da dividendi come yield medio sul periodo storico disponibile
        # Questo approccio è più robusto rispetto all'uso di un singolo anno
        # Somma dividendi nel periodo / prezzo medio del periodo

        # Somma totale dei dividendi pagati nel periodo recente
        divPeriodo = np.nansum(dividendi[i])
        # Prezzo medio nel periodo recente
        prezzoMedio = np.nanmean(close[i])

        if divPeriodo > 0 and prezzoMedio > 0:
            # Yield annuale = dividendi medi annui / prezzo medio
            yield_annuale = (divPeriodo / (len(close[i]) / GIORNI_IN_UN_ANNO)) / prezzoMedio
            rendimentoAnnuoDividendi[i] = yield_annuale
        else:
            rendimentoAnnuoDividendi[i] = 0

        # I dividendi sono trattati come flussi deterministici (non stocastici)
        # Media giornaliera = yield annuale / giorni in un anno
        rendimentoGiornalieroDividendi[i] = rendimentoAnnuoDividendi[i] / GIORNI_IN_UN_ANNO

        # Volatilità = 0 (i dividendi sono relativamente prevedibili e stabili)
        # Non seguono un random walk come i prezzi
        volatilitàDividendi[i] = 0

    return (
        rendimentoGiornaliero,
        volatilitàGiornaliera,
        rendimentoAnnuo,
        volatilitàAnnua,
        rendimentoAnnuoDividendi,
        rendimentoGiornalieroDividendi,
        volatilitàDividendi,
    )


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
