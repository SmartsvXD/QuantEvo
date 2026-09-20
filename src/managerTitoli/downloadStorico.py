import datetime as dt
import os
import pickle as pk
import time

import numpy as np
import yfinance as yf
from colorama import Fore

from src.const import GIORNI_MINIMI_VALIDITA


def _downloadYFinance(nomi: tuple, anni: int):
    """
    Scarica i dati suando yfinance come fonte

    :param nomi: I nomi dei titoli che si vuole scaricare, deve essere una lista/tupla di nomi
    :type nomi: `tuple`
    :param anni: Specifica il numero di anni di dati che vogliamo scaricare
    :type anni: `int`
    :return: rendimenti e close dei titoli richiesti
    :rtype: `np.ndarray` o `None` in caso di errore
    """

    # * Utilizziamo la libreria yfinance per scaricare le informazioni
    data = yf.download(
        nomi,
        period=f"{anni}y",
        auto_adjust=False,
        actions=True,  # per dividendi e split
    )

    # * Controlliamo quali ticker sono effettivamente presenti nel risultato.
    scaricati = []
    if "Adj Close" in data and hasattr(data["Adj Close"], "columns"):
        scaricati = list(data["Adj Close"].columns)

    if len(scaricati) != len(nomi):
        mancanti = set(nomi) - set(scaricati)
        if mancanti:
            raise Exception(f"{Fore.RED}Download incompleto, mancano: {', '.join(mancanti)}{Fore.RESET}")
        return None, None, None

    # * Riordina i dati per avere lo stesso ordine di nomi
    adjClose = data.get("Adj Close")

    adjClose = adjClose[nomi]

    # * Filtra ticker che non rispettano un limite minimo di giorni di dati
    nomi_invalidi = [ticker for ticker in nomi if len(adjClose[ticker].dropna()) <= GIORNI_MINIMI_VALIDITA]
    if nomi_invalidi != []:
        print(f"{Fore.RED} I seguenti titoli non hanno un numero sufficiente di giorni di dati:")
        for i in nomi_invalidi:
            print(f"\t{i}")
        print(Fore.RESET)
        raise Exception("Impossibile continuare, i titoli non rispettano i limiti minimi di giorni")

    # * Trasforma i dati in numpy array
    # Usa rendimenti lineari
    rendimenti = np.array(adjClose.pct_change().dropna().T, dtype=np.float64)  # shape: (n_assets, n_days)

    return rendimenti


def download(
    nomi: tuple, fonte: str = "yfinance", anni: int = 50, forzaDownload: bool = 1, verbose: bool = 1
) -> np.array | None:
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
    :rtype: `np.array` o `None` in caso di errore
    """

    # Variabile booleana che determinerà se i dati devono essere scaricati o presi dal cache
    scarica = 0

    t0 = time.time()
    OGGI = dt.datetime.now(tz=dt.timezone.utc).date()

    # * Verifichiamo se dovremmo scaricare o meno
    # Se non esiste un cache o è forzato devi scaricare
    if (
        not (os.path.exists("./data/titoliDownload.data"))
        or not (os.path.exists("./data/infoDownload.data"))
        or forzaDownload
    ):
        scarica = 1
    else:
        # Se il file esite e il download non è forzato dovremmo verificare che il cache sia idoneo
        with open("./data/infoDownload.data", "rb") as file:
            info = pk.load(file)

        # Se il cache è più vecchio di un giorn o se non contiene gli stessi titoli non va bene
        if (
            ((OGGI - info["data"]).days >= 1)
            or info["titoli"] != nomi
            or info["fonte"] != fonte
            or info["anni"] != anni
        ):
            scarica = 1
        else:
            scarica = 0

    # * Scarichiamo i dati se dobbiamo
    if scarica:
        if fonte == "yfinance":
            if verbose:
                print(f"\n{Fore.MAGENTA}Scaricando i dati da Yahoo Finance{Fore.RESET}:{Fore.LIGHTBLACK_EX}")

            rendimenti = _downloadYFinance(nomi, anni)
        else:
            print(f"{Fore.RED}ERRORE! La fonte {fonte} non è valida!{Fore.RESET}")

        if rendimenti is None:
            print(f"{Fore.RED}ERRORE! Impossibile scaricare i dati!{Fore.RESET}")
            return None

        # Salviamo i nuovi dati scaricati
        if verbose:
            print(
                f"{Fore.MAGENTA}Salvando i dati in {Fore.RESET}./data/titoliDownload.data{Fore.MAGENTA} e {Fore.RESET}./data/infoDownload.data{Fore.MAGENTA}...{Fore.RESET}"
            )

        # Salviamo le info
        info = {"data": OGGI, "titoli": nomi, "fonte": fonte, "anni": anni}
        with open("./data/infoDownload.data", "wb") as file:
            pk.dump(info, file)

        # Salviamo i dati
        with open("./data/titoliDownload.data", "wb") as file:
            pk.dump(rendimenti, file)

    # * Carichiamo i dati dal cache
    else:
        if verbose:
            print(
                f"\n{Fore.MAGENTA}Caricando i dati da {Fore.RESET}./data/titoliDownload.data{Fore.MAGENTA}...{Fore.RESET}"
            )

        with open("./data/titoliDownload.data", "rb") as file:
            rendimenti = pk.load(file)

    t = time.time() - t0
    if verbose:
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({t:.2f}s)")

    return rendimenti
