import time
from colorama import Fore
from src.managerTitoli.statsStoriche import StatsTitoli
from src.const import GIORNI_IN_UN_ANNO
import numpy as np
from src.simulazione.skewt import rskewt_multivariate
from numba import njit, prange, set_num_threads
import os

set_num_threads(os.cpu_count())


class RisultatiSimulazione:
    """
    Struttura che contiene i risultati della simulazione dei titoli
    """

    __slots__ = [
        "nomi",
        "valoriTitoli",
        "S0",
        "nSims",
        "anni",
    ]


@njit(parallel=True, fastmath=True)
def _simulaTitoli(
    Z: np.ndarray,
    giorni: int,
    nSims: int,
    drift: np.ndarray,
    volatilità: np.ndarray,
    S0: int,
    dividendiGiornalieri: np.ndarray,
    day0: int = 0,
    valoriTitoli: np.ndarray = None,
) -> np.ndarray:
    """
    Simula il percorso di prezzo degli asset usando GBM con correzione di Itô.

    :param Z: matrice casuale
    :type Z: `np.ndarray`
    :param giorni: numero di giorni da simulare
    :type giorni: `int`
    :param nSims: numero di simulazioni da eseguire
    :type nSims: `int`
    :param drift:
    :type drift: `np.ndarray`
    :param volatilità:
    :type volatilità: `np.ndarray`
    :param S0: investimento iniziale
    :type S0: `int`
    :param dividendiGiornalieri: al momento non utilizzato
    :type dividendiGiornalieri: `np.ndarray`
    :param day0: giorno iniziale da cui simulare
    :type day0: `int`
    :param valoriTitoli: valori dei titoli fino a day0
    :type valoriTitoli: `np.ndarray`
    :return: valori previsti dai titoli
    :rtype: `np.ndarray`
    """

    # Determiniamo il numero di titoli
    nTitoli = drift.shape[0]

    # * Inizializzamo il vettore che conterrà il valore degli asset
    if valoriTitoli is None:
        valoriTitoli = np.zeros((nSims, giorni, nTitoli), dtype=np.float64)
        valoriTitoli[:, 0, :] = S0

    # Drift totale include price appreciation + dividendi
    # Ma per il processo dei PREZZI, usiamo solo price drift
    # I dividendi saranno aggiunti come flusso di cassa separato nel calcolo del total return
    driftAgg = drift - 0.5 * volatilità * volatilità  # Correzione di Itô

    # * Parallelizziamo sulla dimensione delle simulazioni: ogni simulazione è indipendente
    for sim in prange(nSims):
        for day_offset in range(giorni):
            day = day0 + day_offset
            for asset in range(nTitoli):
                # Se siamo al primo giorno assoluto e il valore non è inizializzato,
                # impostiamo S0 (caso day == 0). Altrimenti usiamo il precedente giorno.
                if day == 0:
                    # Se valore già presente (da array fornito), lo preserviamo
                    if valoriTitoli is None:
                        prev = S0
                    else:
                        prev = valoriTitoli[sim, 0, asset]
                    # usa prev come valore iniziale e poi applica rumore per giorni successivi
                    valoriTitoli[sim, 0, asset] = prev
                else:
                    noise = volatilità[asset] * Z[sim, day_offset, asset]
                    log_return = driftAgg[asset] + noise
                    valoriTitoli[sim, day, asset] = valoriTitoli[sim, day - 1, asset] * np.exp(log_return)

    return valoriTitoli


def simulaTitoli(
    statsTitoli: StatsTitoli, S0: int = 1, anni: int = 1, nSims: int = 20000, verbose: int = 1, noise: str = "normal"
) -> RisultatiSimulazione:
    """
    Simula `anni` 

    :param statsTitoli: statiscitche storiche dei titoli
    :type statsTitoli: `StatsTitoli`
    :param S0: investimento iniziale
    :type S0: `int`
    :param anni: numero di anni da simulare
    :type anni: `int`
    :param nSims: numero di simulazioni da eseguire
    :type nSims: `int`
    :param verbose: 0 se non si vuole alcun tipo di stampa, 1 se si vuole sapere cosa sta facendo la funzione, 
    2 per stampare la mediana di ogni titolo
    :type verbose: `int`
    :param noise: tipo di rumore
    :type noise: `str`
    :return: i prezzi previsti per i titoli
    :rtype: `RisultatiSimulazione`
    """

    t0 = time.time()

    if verbose >= 1:
        print(f"\n{Fore.MAGENTA}Elaborazione della simulazione dei titoli...{Fore.RESET}")

    # * Specchettiamo i dati
    # Il drift è pari al rendimento giornaliero
    drift = statsTitoli.rendimentoGiornaliero
    # La volatilità è pari alla volatilità giornaliera
    volatilità = statsTitoli.volatilitàGiornaliera
    L = statsTitoli.L
    skew = statsTitoli.skew
    kurt = statsTitoli.kurtosis

    # * Otteniamo il numero di giorni e di titoli nella simulazione
    giorni = anni * GIORNI_IN_UN_ANNO
    nTitoli = len(statsTitoli.nomi)

    # * Stima df da kurtosi
    # kurtosi t = 6/(df-4) -> df = 6/kurt + 4 (se kurt>0)
    df = np.clip(6 / np.maximum(kurt, 1e-6) + 4, 5, 30)

    valoriTitoli = np.zeros((nSims, giorni, nTitoli), dtype=np.float64)
    valoriTitoli[:, 0, :] = S0

    # Imposto un seme per avere reproducibilità
    np.random.seed(49)

    # Dividiamo la simulazioni in blocchi per risparmiare memoria
    for giorno in range(0, giorni, 36):
        blocco = min(36, giorni - giorno)

        # * Determino lo skew
        if noise == "skewt":
            # Usa media 0, scale=1 (standardizzato), skew e df stimati
            # Per df: prendi la media tra asset (approssimazione multivariata)
            df_block = float(np.median(df))
            Z = rskewt_multivariate(nSims * blocco, df_block, skew, np.zeros(nTitoli), np.ones(nTitoli), L @ L.T)
            Z = Z.reshape(nSims, blocco, nTitoli)
        else:
            Z = np.random.randn(nSims, blocco, nTitoli)
            Z_shape = Z.shape
            Z_reshaped = Z.reshape(-1, Z_shape[-1])
            Z_correlated = Z_reshaped @ L.T
            Z = Z_correlated.reshape(Z_shape)

        # * Eseguiamo la simulazione del blocco
        valoriTitoli = _simulaTitoli(
            Z,
            blocco,
            nSims,
            drift,
            volatilità,
            S0,
            statsTitoli.dividendiGiornalieri,
            day0=giorno,
            valoriTitoli=valoriTitoli,
        )

    if verbose >= 1:
        simTime = time.time() - t0
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({simTime:.2f}s)")

    # * Stampo le mediane previste per ogni titolo
    mediana = np.zeros(len(statsTitoli.nomi))

    for i in range(len(statsTitoli.nomi)):
        mediana[i] = (np.median(valoriTitoli[:, -1, i]) / S0) - 1

    if verbose == 2:
        print(f"\n{Fore.MAGENTA}Performance dei Titoli{Fore.RESET}:")
        for i, nome in enumerate(statsTitoli.nomi):
            print(f"    {Fore.GREEN}{nome}{Fore.RESET}: ")
            print(f"        {Fore.YELLOW}Mediana{Fore.RESET}: {mediana[i]:.2%}")

    # * Prepara l'output
    risultato = RisultatiSimulazione()

    risultato.nomi = statsTitoli.nomi
    risultato.valoriTitoli = valoriTitoli
    risultato.S0 = S0
    risultato.nSims = nSims
    risultato.anni = anni

    return risultato
