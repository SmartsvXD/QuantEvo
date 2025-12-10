from colorama import Fore
import numpy as np
from src.simulazione.titoli import RisultatiSimulazione


def andamentiPortafoglio(portafoglio: dict, risultatiSim: RisultatiSimulazione) -> np.ndarray:
    """
    Docstring per andamentiPortafoglio

    :param portafoglio: Descrizione
    :type portafoglio: `dict`
    :param risultatiSim: Descrizione
    :type risultatiSim: `RisultatiSimulazione`
    :return: valori del portafoglio previsti
    :rtype: `np.ndarray`
    """
    indici = []
    for titolo in portafoglio.keys():
        indici.append(risultatiSim.nomi.index(titolo))

    # Estrai i valori dei titoli nel portafoglio
    valoriPortafoglioTitoli = risultatiSim.valoriTitoli[:, :, indici]

    pesi = np.array(list(portafoglio.values()), dtype=np.float64)
    pesi /= np.sum(pesi)

    # Applica i pesi: ogni titolo contribuisce con il suo peso al portafoglio
    valoriPortafoglioPesati = valoriPortafoglioTitoli * pesi[np.newaxis, np.newaxis, :]

    # Somma i contributi pesati per ottenere il valore totale del portafoglio
    valoriPortafoglio = np.sum(valoriPortafoglioPesati, axis=2)

    return valoriPortafoglio


class Pagella:
    """
    Struttura che conitene i dati di valutazione di un portafoglio
    """

    __slot__ = ["crescitaPortafoglio", "mediana", "media", "p10", "p25", "p75", "p90", "probPerdita"]


def valutaPortafoglio(portafoglio: dict, risultatiSim: RisultatiSimulazione, verbose: bool = 1) -> Pagella:
    """
    Docstring per valutaPortafoglio

    :param portafoglio: dizionario che contiene come chiavi i nomi dei titoli del portafoglio e 
    come item i pesi di ogni titolo
    :type portafoglio: `dict`
    :param risultatiSim: i risultati della simulazione su cui basarsi
    :type risultatiSim: `RisultatiSimulazione`
    :param verbose: positivo se si vuole aver stampata la valutazione
    :type verbose: `bool`
    :return: valutazione del portafoglio
    :rtype: `Pagella`
    """
    valoriPortafoglio = andamentiPortafoglio(portafoglio, risultatiSim)

    pagella = Pagella()

    # Calcola il rendimento: (valore_finale - valore_iniziale) / valore_iniziale
    # valore_iniziale è S0 (già implicito nei valoriTitoli come S0 al day 0)
    # Ma valoriPortafoglio è già un valore, perciò: rendimento = valoriPortafoglio[:, -1] / S0 - 1
    pagella.crescitaPortafoglio = (valoriPortafoglio[:, -1] / risultatiSim.S0) - 1

    pagella.media = np.mean(pagella.crescitaPortafoglio)
    pagella.mediana = np.median(pagella.crescitaPortafoglio)
    pagella.p10 = np.percentile(pagella.crescitaPortafoglio, 10)
    pagella.p25 = np.percentile(pagella.crescitaPortafoglio, 25)
    pagella.p75 = np.percentile(pagella.crescitaPortafoglio, 75)
    pagella.p90 = np.percentile(pagella.crescitaPortafoglio, 90)

    pagella.probPerdita = np.mean(pagella.crescitaPortafoglio < 0)

    if verbose:
        print(f"\n{Fore.MAGENTA}Performance Portafoglio{Fore.RESET}:")
        print(f"    {Fore.YELLOW}Mediana{Fore.RESET}:\t{pagella.mediana:.2%}")
        print(f"    {Fore.YELLOW}Media{Fore.RESET}:\t{pagella.media:.2%}")
        print(f"    {Fore.YELLOW}Percentili{Fore.RESET}:")
        print(f"        {Fore.RED}10%{Fore.RESET}:\t{pagella.p10:.2%}")
        print(f"        {Fore.LIGHTRED_EX}25%{Fore.RESET}:\t{pagella.p25:.2%}")
        print(f"        {Fore.LIGHTGREEN_EX}75%{Fore.RESET}:\t{pagella.p75:.2%}")
        print(f"        {Fore.GREEN}90%{Fore.RESET}:\t{pagella.p90:.2%}")
        print(f"    {Fore.YELLOW}Probabilità di Perdita{Fore.RESET}:\t{pagella.probPerdita:.2%}")

    return pagella
