from settings import GIORNI_IN_UN_ANNO
from colorama import Fore
from dataManager import Portafoglio, Titoli
import numpy as np
from acceleratedFunctions import _simulaTitoli
import time
import os
import pickle as pk


class Pagella:
    def __init__(self):
        self.nSims = None

        self.anni = None

        self.investimentoTotale = None
        self.S0 = None

        self.crescitaPortafoglio = None
        self.valoriPortafoglio = None
        self.valoriInvestimenti = None

        self.mediana = None
        self.media = None
        self.p10 = None
        self.p25 = None
        self.p75 = None
        self.p90 = None

        self.probPerdita = None


class MonteCarlo:
    def __init__(self, titoli: Titoli, S0=1.0, ribilancio=1.0, anni=1, nSims=20000):
        self.S0 = S0
        self.ribilancio = ribilancio
        self.anni = anni
        self.nSims = nSims
        self.titoli = titoli

    def simulaTitoli(self, forzaElaborazione=1, verbose=0):
        t0 = time.time()

        print()
        if forzaElaborazione or not (os.path.exists("./data/sim.data")) or self.titoli.download:
            print(f"{Fore.MAGENTA}Elaborazione della simulazione dei titoli...{Fore.RESET}")

            # Il drift del GBM per log-rendimenti è: (μ - 0.5σ²)dt
            drift = (self.titoli.stats.rendimentoAnnuo - 0.5 * self.titoli.stats.volatilitàAnnua**2) / GIORNI_IN_UN_ANNO

            coeffDiffusione = self.titoli.stats.volatilitàAnnua * np.sqrt(1 / GIORNI_IN_UN_ANNO)

            L = self.titoli.L

            # Genera rumore indipendente
            # Nota: usiamo un solo array e scriviamo il risultato della correlazione nello
            # stesso buffer per evitare la copia temporanea (Z_ind -> Z) che raddoppia il picco memoria.
            Z = np.random.randn(self.nSims, self.anni * GIORNI_IN_UN_ANNO, len(self.titoli.stats.nomi))
            # Applica correlazione usando decomposizione di Cholesky e scrive il risultato nello stesso array
            # (riduce il picco di memoria temporanea)
            np.einsum("ijk,kl->ijl", Z, L, out=Z)

            (self.investimento, self.valoriTitoli) = _simulaTitoli(
                Z,
                self.anni,
                self.nSims,
                drift,
                coeffDiffusione,
                self.S0,
                self.titoli.stats.rendimentoGiornalieroDividendi,
            )

            print(f"{Fore.MAGENTA}Salvando i dati in {Fore.RESET}./data/sim.data{Fore.MAGENTA}...")

            data = {
                "investimento": self.investimento,
                "valoriTitoli": self.valoriTitoli,
            }

            with open("./data/sim.data", "wb") as file:
                pk.dump(data, file)

        else:
            print(
                f"{Fore.RED}ATTENZIONE! Caricare i dati della simulazione dei titoli è altamente sconsigliato!{Fore.RESET}"
            )
            print(f"{Fore.MAGENTA}Caricando i dati da {Fore.RESET}./data/sim.data{Fore.MAGENTA}...")

            with open("./data/sim.data", "rb") as file:
                data = pk.load(file)

            self.investimento = data["investimento"]
            self.valoriTitoli = data["valoriTitoli"]

        simTime = time.time() - t0
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({simTime:.2f}s)")

        if verbose:
            print(f"\n{Fore.MAGENTA}Performance dei Titoli{Fore.RESET}:")
            for i, nome in enumerate(self.titoli.stats.nomi):
                mediana = (np.median(self.valoriTitoli[:, -1, i]) / self.investimento) - 1

                print(f"    {Fore.GREEN}{nome}{Fore.RESET}: ")
                print(f"        {Fore.YELLOW}Mediana{Fore.RESET}: {mediana:.2%}")

    def valutaPortafoglio(self, portafoglio: Portafoglio, altro=0, verbose=1):
        valoriPortafoglioTitoli = self.valoriTitoli[:, :, portafoglio.indiciTitoli]
        valoriPortafoglioPesati = valoriPortafoglioTitoli * portafoglio.pesi[np.newaxis, np.newaxis, :]
        valoriPortafoglio = np.sum(valoriPortafoglioPesati, axis=2)

        # Prende il valore finale (ultimo giorno) per tutte le simulazioni
        crescitaPortafoglio = valoriPortafoglio[:, -1] - 1

        mediana = np.median(crescitaPortafoglio)
        media = np.mean(crescitaPortafoglio)

        p10 = np.percentile(crescitaPortafoglio, 10)
        p25 = np.percentile(crescitaPortafoglio, 25)
        p75 = np.percentile(crescitaPortafoglio, 75)
        p90 = np.percentile(crescitaPortafoglio, 90)

        probPerdita = np.mean(crescitaPortafoglio < 0)

        pagella = Pagella()

        pagella.nSims = self.nSims

        pagella.anni = self.anni

        pagella.investimentoTotale = self.S0
        pagella.S0 = self.S0

        pagella.crescitaPortafoglio = crescitaPortafoglio
        if altro:
            pagella.valoriPortafoglio = valoriPortafoglio.astype(np.float32)
        pagella.valoriInvestimenti = np.array([self.S0] * self.anni * GIORNI_IN_UN_ANNO)

        pagella.media = media
        pagella.mediana = mediana
        pagella.p10 = p10
        pagella.p25 = p25
        pagella.p75 = p75
        pagella.p90 = p90

        pagella.probPerdita = probPerdita

        if verbose:
            print(f"\n{Fore.MAGENTA}Performance Portafoglio{Fore.RESET}:")
            print(f"    {Fore.YELLOW}Mediana{Fore.RESET}:\t{mediana:.2%}")
            print(f"    {Fore.YELLOW}Media{Fore.RESET}:\t{media:.2%}")
            print(f"    {Fore.YELLOW}Percentili{Fore.RESET}:")
            print(f"        {Fore.RED}10%{Fore.RESET}:\t{p10:.2%}")
            print(f"        {Fore.LIGHTRED_EX}25%{Fore.RESET}:\t{p25:.2%}")
            print(f"        {Fore.LIGHTGREEN_EX}75%{Fore.RESET}:\t{p75:.2%}")
            print(f"        {Fore.GREEN}90%{Fore.RESET}:\t{p90:.2%}")
            print(f"    {Fore.YELLOW}Probabilità di Perdita{Fore.RESET}:\t{probPerdita:.2%}")

        return pagella
