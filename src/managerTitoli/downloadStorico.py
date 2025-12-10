import time
import os
import pickle as pk
import datetime as dt
from colorama import Fore
import numpy as np
import yfinance as yf


class DownloadData:
    """
    Struttura per salvare tutti i dati storici dei titoli
    """

    __slots__ = ("nomi", "rendimenti", "dividendi", "close")


def _downloadYFinance(nomi: tuple, anni: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Scarica i dati suando yfinance come fonte

    :param nomi: I nomi dei titoli che si vuole scaricare, deve essere una lista/tupla di nomi
    :type nomi: `tuple`
    :param anni: Specifica il numero di anni di dati che vogliamo scaricare
    :type anni: `int`
    :return: rendimenti, dividendi e close dei titoli richiesti
    :rtype: `np.ndarray`, `np.ndarray`, `np.ndarray` o `None`, `None`, `None` in caso di errore
    """

    # * Utilizziamo la libreria yfinance per scaricare le informazioni
    # auto_adjust=False per avere prezzi non aggiustati e calcolare total return manualmente
    # Questo permette di separare correttamente price appreciation da dividendi
    data = yf.download(
        nomi,
        period=f"{anni}y",
        auto_adjust=False,
        actions=True,  # per dividendi e split
    )

    # * Controlliamo quali ticker sono effettivamente presenti nel risultato.
    scaricati = []
    if "Close" in data and hasattr(data["Close"], "columns"):
        scaricati = list(data["Close"].columns)

    if len(scaricati) != len(nomi):
        mancanti = set(nomi) - set(scaricati)
        if mancanti:
            print(f"{Fore.RED}Download incompleto, mancano: {', '.join(mancanti)}{Fore.RESET}")
        return None, None, None

    # * Riordina i dati per avere lo stesso ordine di nomi
    close = data.get("Close")
    dividends = data.get("Dividends") if "Dividends" in data else None

    close = close[nomi]

    # Se i dividendi non sono disponibili, crea una matrice di zeri con lo stesso shape di close
    if dividends is None:
        dividends = close.copy()
        dividends.iloc[:, :] = 0.0
    else:
        # Riordina le colonne e riempi eventuali NaN con 0
        dividends = dividends.reindex(columns=nomi, fill_value=0.0)

    # * Filtra ticker con almeno 1260 giorni di dati originali (circa 5 anni di trading)
    giorni_minimi = 1260
    nomi_invalidi = [ticker for ticker in nomi if len(close[ticker].dropna()) <= giorni_minimi]
    if nomi_invalidi != []:
        print(f"{Fore.RED} I seguenti titoli non hanno un numero sufficiente di giorni di dati:")
        for i in nomi_invalidi:
            print(f"\t{i}")
        print(Fore.RESET)
        return None, None, None

    # * Trasforma i dati in numpy array
    # Calcola i log-returns giornalieri: ln(P_t / P_{t-1})
    rendimenti = np.array(np.log(close / close.shift(1)).dropna(how="all")).T
    # Salva i dividendi giornalieri
    dividendi = np.array(dividends).T
    # Salva i prezzi di chiusura
    close = np.array(close).T

    return rendimenti, dividendi, close


def download(
    nomi: tuple, fonte: str = "yfinance", anni: int = 50, forzaDownload: bool = 1, verbose: bool = 1
) -> DownloadData | None:
    """
    Permette di ottenere le informazioni per i titoli richiesti. Se esistono dei dati validi salvati in cache li carica da li,
    mentre se non esistono li scarica direttamente dalla fonte richiesta. Restituisce rendimenti, close, dividendi.

    :param nomi: I nomi dei titoli che si vuole scaricare, deve essere una lista/tupla di nomi
    :type nomi: `tuple`
    :param fonte: La fonte da cui dovremo scaricare i dati, deve essere una stringa
    :type fonte: `str`
    :param anni: Specifica il numero di anni da oggi di cui dobbiamo scaricare i dati
    :type anni: `int`
    :param forzaDownload: Obbliga il download dei dati dalla fonte
    :type forzaDownload: `bool`
    :param verbose: Specifica se vogliamo avere informazioni su quello che fa la funzione
    :type verbose: `bool`
    :return: I dati storici dei titoli richiesti
    :rtype: `DownloadData` o `None` in caso di errore
    """

    # Variabile booleana che determinerà se i dati devono essere scaricati o presi dal cache
    scarica = 0

    t0 = time.time()

    # * Verifichiamo se dovremmo scaricare o meno
    # Se non esiste un cache o è forzato devi scaricare
    if not (os.path.exists("./data/titoli.data")) or forzaDownload:
        scarica = 1
    else:
        # Se il file esite e il download non è forzato dovremmo verificare che il cache sia idoneo
        with open("./data/info.data", "rb") as file:
            info = pk.load(file)

        # Se il cache è più vecchio di un giorno non va bene
        if (dt.date.today() - info["data"]).days >= 1:
            scarica = 1
        # Se il cache non contiene gli stessi non va bene
        elif info["titoli"] != nomi:
            scarica = 1
        # Se la fonte è diversa da quella di ora non va bene
        elif info["fonte"] != fonte:
            scarica = 1
        else:
            scarica = 0

    # * Scarichiamo i dati se dobbiamo
    if scarica:
        if fonte == "yfinance":
            if verbose:
                print(f"\n{Fore.MAGENTA}Scaricando i dati da Yahoo Finance{Fore.RESET}:{Fore.LIGHTBLACK_EX}")

            rendimenti, dividendi, close = _downloadYFinance(nomi, anni)
        else:
            print(f"{Fore.RED}ERRORE! La fonte {fonte} non è valida!{Fore.RESET}")

        if rendimenti is None or dividendi is None or close is None:
            print(f"{Fore.RED}ERRORE! Impossibile scaricare i dati!{Fore.RESET}")
            return None

        data = DownloadData()
        data.nomi = nomi
        data.rendimenti = np.array(rendimenti)
        data.dividendi = np.array(dividendi)
        data.close = np.array(close)

        # Salviamo i nuovi dati scaricati
        if verbose:
            print(
                f"{Fore.MAGENTA}Salvando i dati in {Fore.RESET}./data/titoli.data{Fore.MAGENTA} e {Fore.RESET}./data/info.data{Fore.MAGENTA}...{Fore.RESET}"
            )

        # Salviamo le info
        info = {"data": dt.date.today(), "titoli": nomi, "fonte": fonte}
        with open("./data/info.data", "wb") as file:
            pk.dump(info, file)

        # Salviamo i dati
        with open("./data/titoli.data", "wb") as file:
            pk.dump(data, file)
    # * Carichiamo i dati dal cache
    else:
        if verbose:
            print(f"{Fore.MAGENTA}Caricando i dati da {Fore.RESET}./data/titoli.data{Fore.MAGENTA}...{Fore.RESET}")

        with open("./data/titoli.data", "rb") as file:
            data = pk.load(file)

    t = time.time() - t0
    if verbose:
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({t:.2f}s)")

    return data
