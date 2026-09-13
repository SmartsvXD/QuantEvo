import time

import numpy as np
from colorama import Fore

from src.const import N_PUNTI_BARRA_CARICAMENTO
from src.utils.generaPortafogliValidi import generaPortafogliValidi
from src.utils.vincoli import Vincoli


class ValutatoreMultiplo:
    def __init__(self, indici: tuple):
        self._indici = indici
        self.numIndici = len(self._indici)

    def fitStandardizza(self, stats, vincoli: Vincoli):
        print(f"\n{Fore.MAGENTA}Fitto la standardizzazione dei valutatori...{Fore.RESET}")
        t0 = time.time()

        listaPortafogli = generaPortafogliValidi(len(stats.nomi), vincoli, n=10000)

        for n, i in enumerate(self._indici):
            # Barra di caricamento
            filled = round(n / len(self._indici) * N_PUNTI_BARRA_CARICAMENTO)
            empty = N_PUNTI_BARRA_CARICAMENTO - filled
            print(f"\r{Fore.LIGHTBLACK_EX}{n}/{len(self._indici)} [{'*' * filled}{' ' * empty}]{Fore.RESET}", end="")

            # Vera e propria standardizzazione
            i.fitStandardizza(listaPortafogli, stats)

        print(
            f"\r{Fore.LIGHTBLACK_EX}{len(self._indici)}/{len(self._indici)} [{'*' * N_PUNTI_BARRA_CARICAMENTO}]{Fore.RESET}"
        )
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def calcola(self, portafoglio, stats, standardizza=False, inverti=False):
        portafoglio = np.atleast_2d(portafoglio)

        res = np.empty((portafoglio.shape[0], len(self._indici)), dtype=np.float64)
        for n, i in enumerate(self._indici):
            vals = i.calcola(portafoglio, stats, standardizza=standardizza, inverti=inverti)
            res[:, n] = vals

        return res
