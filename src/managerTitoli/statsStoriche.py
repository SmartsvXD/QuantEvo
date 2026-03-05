from src.managerTitoli.downloadStorico import DownloadData
import time
from colorama import Fore
import numpy as np
from scipy.stats import skew, kurtosis
from numba import njit, prange, set_num_threads
from src.managerTitoli.ewma import mediaEWMA, varianzaEWMA
from src.const import GIORNI_IN_UN_ANNO
import os

set_num_threads(os.cpu_count())


class StatsTitoli:
    """
    Classe che contiene le statistiche finanziarie per titoli o portafogli.
    Memorizza dati come rendimenti, volatilità, dividendi e parametri per il modello GBM.
    """

    __slots__ = (
        "nomi",
        "rendimentoGiornaliero",
        "volatilitàGiornaliera",
        "dividendiGiornalieri",
        "volatilitàDividendi",
        "L",
        "skew",
        "kurtosis",
    )


@njit(parallel=True)
def _calcolaStatsTitoli(
    nomi: tuple, rendimenti: np.ndarray, dividendi: np.ndarray, close: np.ndarray, lambdaEWMA: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola statistiche storiche per ciascun titolo
    Utilizza Numba per accelerare i calcoli.

    :param nomi: lista dei nomi dei titoli
    :type nomi: `tuple`
    :param rendimenti: array dei log-returns giornalieri per ciascun titolo
    :type rendimenti: `np.ndarray`
    :param dividendi: array dei dividendi giornalieri per ciascun titolo
    :type dividendi: `np.ndarray`
    :param close: array dei prezzi di chiusura per ciascun titolo
    :type close: `np.ndarray`

    :return: rendimenti giornalieri, volatilità giornaliera, rendimento da dividendi giornaliero e volatilità dividendi.
    :rtype: `np.ndarray`, `np.ndarray`, `np.ndarray`, `np.ndarray`
    """
    nBeni = len(nomi)

    rendimentoGiornaliero = np.zeros(nBeni)
    volatilitàGiornaliera = np.zeros(nBeni)

    rendimentoAnnuo = np.zeros(nBeni)
    volatilitàAnnua = np.zeros(nBeni)

    dividendiGiornalieri = np.zeros(nBeni)
    volatilitàDividendi = np.zeros(nBeni)

    # Ciclo su ciascun bene per calcolare le statistiche individuali
    for i in prange(nBeni):
        # Calcola la media dei log-returns giornalieri con EWMA (drift giornaliero)
        rendimentoGiornaliero[i] = mediaEWMA(rendimenti[i], lambdaEWMA[i])

        # Calcola la volatilità giornaliera come sqrt della varianza EWMA
        varianza_giornaliera = varianzaEWMA(rendimenti[i], lambdaEWMA[i])
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
            dividendiGiornalieri[i] = yield_annuale / GIORNI_IN_UN_ANNO
        else:
            dividendiGiornalieri[i] = 0.0

        # Volatilità = 0 (i dividendi sono relativamente prevedibili)
        volatilitàDividendi[i] = 0.0

    return (
        rendimentoGiornaliero,
        volatilitàGiornaliera,
        dividendiGiornalieri,
        volatilitàDividendi,
    )


@njit(parallel=True)
def _calcolaMatriceCorrelazione(rendimenti: np.ndarray) -> np.ndarray:
    """
    Calcola la matrice di correlazione dai rendimenti.

    :param rendimenti: Array dei rendimenti di cui calcolare la correlazione
    :type rendimenti: `np.ndarray`

    :return: matrice di correlazione
    :rtype: `np.ndarray`
    """
    n = rendimenti.shape[0]
    cov_matrix = np.zeros((n, n))

    # Passo 1: Calcola la covarianza empirica
    for i in prange(n):
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
        corr_matrix[i, i] = 1.0

    # Rende la matrice simmetrica
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)

    return corr_matrix


def _calcolaL(rendimenti: np.ndarray) -> np.ndarray:
    """
    Calcola la matrice di Cholesky della matrice di correlazione.
    Gestisce il caso in cui la matrice non sia definita positiva aumentando il jitter.

    :param rendimenti: Array dei rendimenti di cui calcolare la correlazione
    :type rendimenti: `np.ndarray`

    :return: matrice di Cholesky
    :rtype: `np.ndarray`
    """
    corr_matrix = _calcolaMatriceCorrelazione(rendimenti)
    n = corr_matrix.shape[0]

    # Tenta di calcolare la decomposizione di Cholesky con jitter crescente
    jitter_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]

    for jitter in jitter_values:
        try:
            # Aggiunge jitter alla diagonale
            corr_matrix_jittered = corr_matrix + np.eye(n) * jitter
            L = np.linalg.cholesky(corr_matrix_jittered)
            return L
        except np.linalg.LinAlgError:
            continue

    # Fallback: usa una matrice identità (nessuna correlazione)
    print(f"{Fore.YELLOW}Warning: Impossibile calcolare la matrice di Cholesky. Usando matrice identità.{Fore.RESET}")
    return np.eye(n)


def calcolaStatsStoriche(data: DownloadData, lambdaEWMA: np.ndarray, verbose: int = 1) -> StatsTitoli:
    """
    Calcola le statistiche storiche per ciascun componente utilizzando la funzione accelerata.
    Include rendimenti, volatilità e rendimento da dividendi, e opzionalmente stampa i risultati.

    :param data: dati storici dei titoli
    :type data: `DownloadData`
    :param lambdaEWMA: caoefficienti della EWMA
    :type lambdaEWMA: `np.ndarray`
    :param verbose: 0 se non si vuole alcuna informazione, 1 se si vuole avere informazioni su cosa fa la funzione,
    2 per avere infromazioni sui risultati
    :type verbose: `int`

    :return: statistiche storiche dei componenti
    :rtype: `DownloadData`
    """

    # * Verifica se i dati sono validi
    if data is None:
        print(f"{Fore.RED}ERRORE! Impossibile utilizzare i dati!{Fore.RESET}")
        return None

    stats = StatsTitoli()
    stats.nomi = data.nomi

    t0 = time.time()

    if verbose >= 1:
        print(f"\n{Fore.MAGENTA}Elaborazione delle stats storiche dei titoli...{Fore.RESET}")

    # Calcola skew e kurtosi storici per ogni titolo ignorando i NaN
    skew_values = []
    kurt_values = []
    for r in data.rendimenti:
        valid = r[~np.isnan(r)]
        if len(valid) < 2:
            skew_values.append(0.0)
            kurt_values.append(0.0)
        else:
            skew_values.append(float(skew(valid)))
            kurt_values.append(float(kurtosis(valid, fisher=True)))

    stats.skew = np.array(skew_values)
    stats.kurtosis = np.array(kurt_values)

    (
        stats.rendimentoGiornaliero,
        stats.volatilitàGiornaliera,
        stats.dividendiGiornalieri,
        stats.volatilitàDividendi,
    ) = _calcolaStatsTitoli(data.nomi, data.rendimenti, data.dividendi, data.close, lambdaEWMA)

    # * Calcoliamo la matrice di Cholesky basata sulla correlazione
    stats.L = _calcolaL(data.rendimenti)

    t = time.time() - t0
    if verbose >= 1:
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({t:.2f}s)")

    # * Stampiamo le stats storiche se richieste
    if verbose == 2:
        print(f"\n{Fore.MAGENTA}Dati Storici Titoli{Fore.RESET}: ")
        for i, componente in enumerate(stats.nomi):
            # Converte i log-returns in rendimenti percentuali effettivi per una visualizzazione più intuitiva
            rendimentoAnnuo = np.exp(stats.rendimentoGiornaliero[i] * GIORNI_IN_UN_ANNO) - 1
            rendimentoGiornaliero = np.exp(stats.rendimentoGiornaliero[i]) - 1

            volatilitàGiornaliera = stats.volatilitàGiornaliera[i]
            volatilitàAnnua = stats.volatilitàGiornaliera[i] * np.sqrt(GIORNI_IN_UN_ANNO)

            print(f"    {Fore.GREEN}{componente}{Fore.RESET}:")
            print(f"        {Fore.YELLOW}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
            print(f"                               {Fore.YELLOW}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
            print(f"        {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
            print(f"                               {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")

            rendimentoAnnuoDividendi = stats.dividendiGiornalieri[i] * GIORNI_IN_UN_ANNO

            print(f"        {Fore.YELLOW}Rendimento da Dividendi{Fore.RESET}: \t{rendimentoAnnuoDividendi:06.2%}")

    return stats


@njit(parallel=True)
def _calcolaStatsTitoliGradiente(
    nomi, giorni: int, rendimenti: np.array, dividendi: np.array, close: np.array, lambdaEWMA: np.array
) -> tuple:
    """
    Calcola statistiche storiche per ciascun titolo per ogni giorno specificato
    Utilizza Numba per accelerare i calcoli.

    :param nomi: lista dei nomi dei titoli
    :param giorni: intero che specifica su quanti giorni calcolare il gradiente
    :param rendimenti: array dei log-returns giornalieri per ciascun titolo
    :param dividendi: array dei dividendi giornalieri per ciascun titolo
    :param close: array dei prezzi di chiusura per ciascun titolo

    :return: Tuple con rendimenti giornalieri, volatilità giornaliera, rendimenti annuali,
    volatilità annuali, rendimento da dividendi annuale, giornaliero e volatilità dividendi.
    """
    nBeni = len(nomi)

    rendimentoGiornaliero = np.zeros((giorni, nBeni))
    volatilitàGiornaliera = np.zeros((giorni, nBeni))

    dividendiGiornalieri = np.zeros((giorni, nBeni))
    volatilitàDividendi = np.zeros((giorni, nBeni))

    rendimentoAnnoSuccessivo = np.zeros((giorni, nBeni))

    # Ciclo su ciascun bene per calcolare le statistiche individuali
    for i in prange(giorni):
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
            rendimentoGiornaliero[i][j] = mediaEWMA(data_slice, lambdaEWMA[j])

            # Calcola la volatilità giornaliera come sqrt della varianza EWMA
            varianza_giornaliera = varianzaEWMA(data_slice, lambdaEWMA[j])
            volatilitàGiornaliera[i][j] = np.sqrt(varianza_giornaliera)

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
                dividendiGiornalieri[i][j] = yield_annuale / GIORNI_IN_UN_ANNO
            else:
                dividendiGiornalieri[i][j] = 0.0

            # Volatilità = 0 (i dividendi sono relativamente prevedibili)
            volatilitàDividendi[i][j] = 0.0

    return (
        rendimentoGiornaliero,
        volatilitàGiornaliera,
        dividendiGiornalieri,
        volatilitàDividendi,
        rendimentoAnnoSuccessivo,
    )


def calcolaStatsTitoliStoricheGradiente(
    data: DownloadData, giorni: int, lambdaEWMA: np.array, verbose: bool = 1
) -> tuple:
    if giorni <= 0:
        return

    giorni = giorni + GIORNI_IN_UN_ANNO

    if data is None:
        print(f"{Fore.RED}ERRORE! Impossibile utilizzare i dati!{Fore.RESET}")

    if verbose:
        print(f"\n{Fore.MAGENTA}Elaborazione delle stats storiche dei titoli...{Fore.RESET}")
    t0 = time.time()

    out = _calcolaStatsTitoliGradiente(data.nomi, giorni, data.rendimenti, data.dividendi, data.close, lambdaEWMA)

    # Calcola anche la matrice L (Cholesky) per la correlazione, come nella versione non-gradiente
    L = _calcolaL(data.rendimenti)

    t = time.time() - t0
    if verbose:
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({t:.2f}s)")

    return (*out, L)
