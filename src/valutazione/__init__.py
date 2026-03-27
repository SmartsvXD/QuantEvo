import numpy as np
from colorama import Fore
import time

from src.managerTitoli import StatsTitoli


class ValutatoreMultiplo:
    def __init__(self, indici: tuple, pesi: tuple):
        self.indici = indici
        self.pesi = pesi  # Al momento non utilizzato

        if len(self.indici) != len(self.pesi):
            raise Exception("Il numero di indici e di pesi deve essere lo stesso")

    def fitStandardizza(self, listaPortafogli, stats, **kwargs):
        N_PUNTI = 30

        print(f"\n{Fore.MAGENTA}Fitto la standardizzazione dei valutatori{Fore.RESET}...")
        t0 = time.time()

        for n, i in enumerate(self.indici):
            filled = round((n + 1) / len(self.indici) * N_PUNTI)
            empty = N_PUNTI - filled
            print(f"\r{Fore.LIGHTBLACK_EX}{n + 1}/{len(self.indici)} [{'*' * filled}{' ' * empty}]{Fore.RESET}", end="")

            i.fitStandardizza(listaPortafogli, stats, **kwargs)

        print(f"\n{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def calcola(self, portafoglio, stats, standardizza=False, **kwargs):
        portafoglio = np.atleast_2d(portafoglio)

        res = np.empty((portafoglio.shape[0], len(self.indici)), dtype=np.float64)
        for n, i in enumerate(self.indici):
            vals = i.calcola(portafoglio, stats, standardizza=standardizza, **kwargs)
            res[:, n] = vals
            
        return res


def _test():
    import numpy as np

    from src.valutazione.sharpe import SharpeIndice
    from src.valutazione.rischio import RischioTotaleCorrettoIndice
    from src.valutazione.diversificazione import DiversificazioneIndice
    from src.utils.generaPortafogliValidi import generaPortafoliValidi
    from src.utils.vincoli import applicaVincoli

    titoli = ["CSUS.MI", "SWDA.MI", "SGLD.MI"]
    pesi = np.array([0.2, 0.3, 0.5])

    s = StatsTitoli(titoli, verbose=1)
    s.calcolaStats(verbose=1)

    print(f"\n{Fore.MAGENTA}Genero portafogli validi{Fore.RESET}...")
    t0 = time.time()
    portafogli = generaPortafoliValidi(titoli, vincoli=applicaVincoli, n=10000)
    print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    diversificazione = DiversificazioneIndice()
    sharpe = SharpeIndice()
    rischioCorretto = RischioTotaleCorrettoIndice()

    v = ValutatoreMultiplo((diversificazione, sharpe, rischioCorretto), (1, -1, 1))
    v.fitStandardizza(portafogli, s)

    ris = v.calcola(pesi, s, standardizza=True)

    print(ris)

    ris = v.calcola(pesi, s, standardizza=False)

    print(ris)
