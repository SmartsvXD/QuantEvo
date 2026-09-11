import numpy as np
from colorama import Fore
import time

from src.managerTitoli import StatsTitoli
from src.const import N_PUNTI_BARRA_CARICAMENTO


class ValutatoreMultiplo:
    def __init__(self, indici: tuple, inverti: tuple | None = None):
        self._indici = indici
        if inverti is None:
            self._inverti = np.ones(len(self._indici))
        else:
            if len(self._indici) != len(inverti):
                raise Exception("Il numero di indici e di pesi deve essere lo stesso")
            self._inverti = np.ones(len(self._indici)) - 2 * np.array(inverti, bool)

        self.numIndici = len(self._indici)

    def fitStandardizza(self, listaPortafogli, stats):
        print(f"\n{Fore.MAGENTA}Fitto la standardizzazione dei valutatori...{Fore.RESET}")
        t0 = time.time()

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

    def calcola(self, portafoglio, stats, inverti=False, standardizza=False):
        portafoglio = np.atleast_2d(portafoglio)

        res = np.empty((portafoglio.shape[0], len(self._indici)), dtype=np.float64)
        for n, i in enumerate(self._indici):
            vals = i.calcola(portafoglio, stats, standardizza=standardizza)
            res[:, n] = vals

        if inverti:
            res *= self._inverti

        return res
