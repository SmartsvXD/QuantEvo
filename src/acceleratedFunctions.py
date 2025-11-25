from numba import njit
import numpy as np
from src.const import GIORNI_IN_UN_ANNO


@njit
def _ewma_mean(data, lambdaEWMA):
    """
    Calcola la media ponderata esponenziale dei log-returns, gestendo NaN.
    Rappresenta il drift giornaliero (media del log-return).

    IMPORTANTE: Per investimenti reali, si usa una finestra più lunga (5-10 anni)
    per stimare il drift, evitando di sovrapesare le fluttuazioni recenti.
    """
    valid = ~np.isnan(data)
    data_valid = data[valid]
    if len(data_valid) == 0:
        return np.nan

    n = len(data_valid)
    if n == 0:
        return np.nan

    # Pesi EWMA: i dati più recenti hanno peso maggiore.
    # weights[k] = lambdaEWMA ** (n-1-k), con k indice su data_recent
    exps = np.arange(n - 1, -1, -1)
    weights = np.power(lambdaEWMA, exps)
    s = np.sum(weights)
    if s == 0:
        # fallback a media semplice se la somma dei pesi è nulla (molto improbabile)
        return np.sum(data_valid) / n

    weights = weights / s
    mean_ewma = np.sum(weights * data_valid)
    return mean_ewma


@njit
def _ewma_variance(data, lambdaEWMA):
    """
    Calcola la varianza ponderata esponenziale, gestendo NaN.
    Usa il metodo della varianza con pesi EWMA per robustezza.
    """
    valid = ~np.isnan(data)
    data_valid = data[valid]
    if len(data_valid) == 0:
        return np.nan

    n = len(data_valid)

    # Pesi EWMA (dati recenti hanno peso maggiore)
    weights = np.power(lambdaEWMA, np.arange(n - 1, -1, -1))
    weights /= weights.sum()

    # Media ponderata esponenzialmente
    mean_ewma = np.sum(weights * data_valid)

    # Varianza ponderata esponenzialmente: E[(X - E[X])^2]
    variance = np.sum(weights * (data_valid - mean_ewma) ** 2)
    return variance


@njit
def _calcolaStatsTitoli(nomi, rendimenti, dividendi, close, lambdaEWMA):
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
        # Calcola la media dei log-returns giornalieri con EWMA (drift giornaliero)
        rendimentoGiornaliero[i] = _ewma_mean(rendimenti[i], lambdaEWMA[i])

        # Calcola la volatilità giornaliera come sqrt della varianza EWMA
        varianza_giornaliera = _ewma_variance(rendimenti[i], lambdaEWMA[i])
        volatilitàGiornaliera[i] = np.sqrt(varianza_giornaliera)

        # Calcola il rendimento annuo come annualizzazione del rendimento giornaliero
        rendimentoAnnuo[i] = rendimentoGiornaliero[i] * GIORNI_IN_UN_ANNO

        # Annualizza la volatilità giornaliera moltiplicando per sqrt(252)
        volatilitàAnnua[i] = volatilitàGiornaliera[i] * np.sqrt(GIORNI_IN_UN_ANNO)

        # Calcola il dividend yield usando gli ultimi 3 anni di dati per una stima più attuale
        # Dividend Yield = Dividendi Annuali / Prezzo Corrente

        # Usa gli ultimi 3 anni (circa 756 giorni di trading)
        lookback_days = min(756, len(close[i]))

        # Dividendi totali negli ultimi lookback_days
        div_recenti = np.nansum(dividendi[i][-lookback_days:])

        # Prezzo medio recente (ultimi 30 giorni per stabilità)
        prezzo_recente = np.nanmean(close[i][-min(30, len(close[i])) :])

        if div_recenti > 0 and prezzo_recente > 0 and lookback_days > 0:
            # Annualizza i dividendi
            anni_periodo = lookback_days / GIORNI_IN_UN_ANNO
            div_annuali = div_recenti / anni_periodo

            # Dividend yield annuale = dividendi annuali / prezzo corrente
            yield_annuale = div_annuali / prezzo_recente
            rendimentoAnnuoDividendi[i] = yield_annuale
        else:
            rendimentoAnnuoDividendi[i] = 0.0

        # Dividend yield giornaliero (assunzione: distribuito uniformemente)
        # In realt\u00e0 i dividendi sono pagati trimestralmente, ma per semplicità usiamo questa approssimazione
        rendimentoGiornalieroDividendi[i] = rendimentoAnnuoDividendi[i] / GIORNI_IN_UN_ANNO

        # Volatilità = 0 (i dividendi sono relativamente prevedibili)
        volatilitàDividendi[i] = 0.0

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
def _calcolaStatsTitoliGradiente(nomi, rendimenti, dividendi, close, lambdaEWMA, giorni):
    if giorni <= 0:
        return

    giorni = giorni + GIORNI_IN_UN_ANNO

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

    rendimentoGiornaliero = np.zeros((giorni, nBeni))
    volatilitàGiornaliera = np.zeros((giorni, nBeni))

    rendimentoAnnuo = np.zeros((giorni, nBeni))
    volatilitàAnnua = np.zeros((giorni, nBeni))

    rendimentoAnnuoDividendi = np.zeros((giorni, nBeni))
    rendimentoGiornalieroDividendi = np.zeros((giorni, nBeni))
    volatilitàDividendi = np.zeros((giorni, nBeni))

    rendimentoAnnoSuccessivo = np.zeros((giorni, nBeni))

    # Ciclo su ciascun bene per calcolare le statistiche individuali
    for i in range(giorni):
        for j in range(nBeni):
            # Evita slice vuote: se il taglio rende una slice di lunghezza 0 o negativa,
            # usiamo l'intera serie come fallback per avere almeno dei dati.
            n_data = len(rendimenti[j])
            cut = n_data - i - 1
            if cut <= 0:
                data_slice = rendimenti[j]
            else:
                data_slice = rendimenti[j][:cut]

            # Calcola la media dei log-returns giornalieri con EWMA (drift giornaliero)
            rendimentoGiornaliero[i][j] = _ewma_mean(data_slice, lambdaEWMA[j])

            # Calcola la volatilità giornaliera come sqrt della varianza EWMA
            varianza_giornaliera = _ewma_variance(data_slice, lambdaEWMA[j])
            volatilitàGiornaliera[i][j] = np.sqrt(varianza_giornaliera)

            # Calcola il rendimento annuo come annualizzazione del rendimento giornaliero
            rendimentoAnnuo[i][j] = rendimentoGiornaliero[i][j] * GIORNI_IN_UN_ANNO

            # Annualizza la volatilità giornaliera moltiplicando per sqrt(252)
            volatilitàAnnua[i][j] = volatilitàGiornaliera[i][j] * np.sqrt(GIORNI_IN_UN_ANNO)

            # Calcola rendimento atteso nel periodo successivo di un anno (se disponibile)
            # idx_now: indice dell'ultimo valore disponibile dopo il taglio
            idx_now = len(close[j]) - i - 1
            idx_future = idx_now + GIORNI_IN_UN_ANNO
            if idx_now >= 0 and idx_future < len(close[j]):
                prezzo_now = close[j][idx_now]
                prezzo_future = close[j][idx_future]
                if prezzo_now > 0:
                    rendimentoAnnoSuccessivo[i][j] = (prezzo_future / prezzo_now) - 1.0
                else:
                    rendimentoAnnoSuccessivo[i][j] = 0.0
            else:
                rendimentoAnnoSuccessivo[i][j] = 0.0

            # Calcola il dividend yield usando gli ultimi 3 anni di dati per una stima più attuale
            # Dividend Yield = Dividendi Annuali / Prezzo Corrente

            # Usa lo stesso criterio di slice usato sopra per evitare array vuoti
            if cut <= 0:
                closeTagliato = close[j]
                dividendoTagliato = dividendi[j]
            else:
                closeTagliato = close[j][:cut]
                dividendoTagliato = dividendi[j][:cut]

            # Usa gli ultimi 3 anni (circa 756 giorni di trading)
            lookback_days = min(756, len(closeTagliato))

            # Dividendi totali negli ultimi lookback_days
            div_recenti = np.nansum(dividendoTagliato[-lookback_days:])

            # Prezzo medio recente (ultimi 30 giorni per stabilità)
            prezzo_recente = np.nanmean(closeTagliato[-min(30, len(closeTagliato)) :])

            if div_recenti > 0 and prezzo_recente > 0 and lookback_days > 0:
                # Annualizza i dividendi
                anni_periodo = lookback_days / GIORNI_IN_UN_ANNO
                div_annuali = div_recenti / anni_periodo

                # Dividend yield annuale = dividendi annuali / prezzo corrente
                yield_annuale = div_annuali / prezzo_recente
                rendimentoAnnuoDividendi[i][j] = yield_annuale
            else:
                rendimentoAnnuoDividendi[i][j] = 0.0

            # Dividend yield giornaliero (assunzione: distribuito uniformemente)
            # In realt\u00e0 i dividendi sono pagati trimestralmente, ma per semplicità usiamo questa approssimazione
            rendimentoGiornalieroDividendi[i][j] = rendimentoAnnuoDividendi[i][j] / GIORNI_IN_UN_ANNO

            # Volatilità = 0 (i dividendi sono relativamente prevedibili)
            volatilitàDividendi[i][j] = 0.0

    return (
        rendimentoGiornaliero,
        volatilitàGiornaliera,
        rendimentoAnnuo,
        volatilitàAnnua,
        rendimentoAnnuoDividendi,
        rendimentoGiornalieroDividendi,
        volatilitàDividendi,
        rendimentoAnnoSuccessivo,
    )


@njit
def _calcolaL(rendimenti):
    """
    Calcola la matrice di Cholesky della matrice di CORRELAZIONE.
    La volatilità verrà applicata separatamente nella simulazione.

    Processo:
    1. Calcola la matrice di covarianza empirica dai rendimenti
    2. Normalizza per ottenere la matrice di correlazione
    3. Applica Cholesky alla matrice di correlazione
    4. Nella simulazione: Z_correlato = Z_standard @ L.T, poi moltiplica per volatilità
    """
    n = rendimenti.shape[0]
    cov_matrix = np.zeros((n, n))

    # Passo 1: Calcola la covarianza empirica
    for i in range(n):
        for j in range(n):
            mask = ~(np.isnan(rendimenti[i]) | np.isnan(rendimenti[j]))
            valid_count = np.sum(mask)

            if valid_count > 1:
                r_i = rendimenti[i][mask]
                r_j = rendimenti[j][mask]

                mean_i = np.sum(r_i) / valid_count
                mean_j = np.sum(r_j) / valid_count

                cov = np.sum((r_i - mean_i) * (r_j - mean_j)) / (valid_count - 1)
                cov_matrix[i, j] = cov
            else:
                cov_matrix[i, j] = 1.0 if i == j else 0.0

    # Passo 2: Converti in matrice di correlazione
    std_empirica = np.zeros(n)
    for i in range(n):
        std_empirica[i] = np.sqrt(cov_matrix[i, i])

    # Evita divisione per zero
    for i in range(n):
        if std_empirica[i] < 1e-10:
            std_empirica[i] = 1e-10

    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = cov_matrix[i, j] / (std_empirica[i] * std_empirica[j])

    # Assicura che la matrice sia semi-definita positiva
    for i in range(n):
        corr_matrix[i, i] = 1.0 + 1e-10

    # Passo 3: Applica Cholesky alla matrice di correlazione (NON scalata)
    L = np.linalg.cholesky(corr_matrix)

    return L


@njit(fastmath=True)
def _simulaTitoli(
    Z_standard,
    anni,
    nSims,
    drift_price: np.array,
    coeffDiffusione,
    S0,
    dividend_yield,
):
    """
    Simula il percorso di prezzo degli asset usando GBM con correzione di Itô.

    Formula GBM corretta:
    dS_t / S_t = μ * dt + σ * dW_t

    In forma discreta (Eulero) per log-prezzi:
    ln(S_t) = ln(S_{t-1}) + (μ - σ²/2) * dt + σ * sqrt(dt) * Z_t

    Dove:
    - μ è il drift (rendimento atteso TOTALE = price appreciation + dividendi)
    - σ è la volatilità
    - (μ - σ²/2) è la correzione di Itô per garantire E[S_t] = S_0 * e^(μt)
    - Z_t è rumore gaussiano standard N(0,1), già correlato tramite Cholesky

    IMPORTANTE: Z_standard è GIÀ correlato (Z @ L.T), ma NON scalato per volatilità.
    La volatilità viene applicata qui moltiplicando per coeffDiffusione.
    """
    nAsset = drift_price.shape[0]
    nDays = anni * GIORNI_IN_UN_ANNO

    # Array per il valore degli asset
    valoriTitoli = np.zeros((nSims, nDays, nAsset), dtype=np.float32)
    valoriTitoli[:, 0, :] = S0

    # Drift totale include price appreciation + dividendi
    # Ma per il processo dei PREZZI, usiamo solo price drift
    # I dividendi saranno aggiunti come flusso di cassa separato nel calcolo del total return
    drift_adj = drift_price - 0.5 * coeffDiffusione * coeffDiffusione  # Correzione di Itô

    # Simula il percorso giornaliero
    for day in range(1, nDays):
        # Rumore scalato per volatilità: σ * Z_t (Z già correlato)
        noise = coeffDiffusione[np.newaxis, :] * Z_standard[:, day, :]

        # Log-return giornaliero = drift corretto + rumore
        log_returns = drift_adj[np.newaxis, :] + noise

        # Aggiorna il prezzo: S_t = S_{t-1} * exp(log_return)
        valoriTitoli[:, day, :] = valoriTitoli[:, day - 1, :] * np.exp(log_returns)

    return S0, valoriTitoli
