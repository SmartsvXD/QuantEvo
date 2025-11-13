from numba import njit
import numpy as np
from settings import GIORNI_PER_MESE, ALIQUOTA_TASSE, GIORNI_IN_UN_ANNO, LAMBDA_EWMA


@njit
def _ewma(data, lambda_val=LAMBDA_EWMA):
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
def _ewma_std(data, lambda_val=LAMBDA_EWMA):
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
def _calcolaStatsTitoli(nomi, rendimenti, dividendi, close):
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


@njit
def _calcolaL(rendimenti):
    """
    Calcola la matrice di correlazione tra i rendimenti dei titoli, da cui ricava la matrice di Cholesky.
    """

    n = rendimenti.shape[0]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mask = ~(np.isnan(rendimenti[i]) | np.isnan(rendimenti[j]))
            if np.sum(mask) > 1:
                corr_matrix[i, j] = np.corrcoef(rendimenti[i, mask], rendimenti[j, mask])[0, 1]
            else:
                corr_matrix[i, j] = 0

    L = np.linalg.cholesky(corr_matrix)

    return L


@njit(fastmath=True)
def _simulaTitoli(
    Z,
    anni,
    nSims,
    drift: np.array,
    coeffDiffusione,
    S0,
    mediaDividendi,
):
    nAsset = drift.shape[0]

    # Matrici casuali per ogni asset e simulazione
    incrementiGiornalieri = drift + coeffDiffusione * Z  # shape (nSims, orizzonte, n_asset)
    investimento = S0

    # Array per il valore degli asset in ogni simulazione e giorno
    valoriTitoli = np.zeros((nSims, anni * GIORNI_IN_UN_ANNO, nAsset), dtype=np.float32)
    valoriTitoli[:, 0, :] = S0

    # I dividendi sono trattati come flussi deterministici (non stocastici)
    # Rendimento giornaliero medio da dividendi per ogni asset
    rendimentoDividendiGiornaliero = mediaDividendi * (1 / GIORNI_IN_UN_ANNO)

    # Array per accumulare dividendi mensili (per asset)
    dividendiMensili = np.zeros((nSims, nAsset), dtype=np.float32)

    for day in range(1, anni * GIORNI_IN_UN_ANNO):
        valoriTitoli[:, day, :] = valoriTitoli[:, day - 1, :] * np.exp(incrementiGiornalieri[:, day, :])

        # Calcola dividendi giornalieri come flusso deterministico
        # (i dividendi degli ETF sono relativamente prevedibili)
        dividendiGiornalieri = valoriTitoli[:, day, :] * rendimentoDividendiGiornaliero
        dividendiMensili += dividendiGiornalieri

        if day % GIORNI_PER_MESE == 0:
            # Reinvesti dividendi (dopo tasse) in ciascun asset + nuovi apporti
            valoriTitoli[:, day, :] += dividendiMensili * (1 - ALIQUOTA_TASSE)

            dividendiMensili[:] = 0

    return investimento, valoriTitoli
