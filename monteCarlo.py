from settings import GIORNI_IN_UN_ANNO
from colorama import Fore, Style
from dataManager import Portafoglio, Titoli
import numpy as np
from acceleratedFunctions import _simulaTitoli


class MonteCarloResults:
    def __init__(self):
        self.investimentoTotale = None
        self.incrementoTotale = None
        self.valoriDelPortafoglio = None
        self.valoriDegliInvestimenti = None

        self.valoriFinali = None
        self.mediana = None
        self.media = None

        self.p10 = None
        self.p25 = None
        self.p75 = None
        self.p90 = None

        self.probPerdita = None

        self.anni = None
        self.orizzonte = None
        self.nSims = None
        self.simTime = None
        self.S0 = None


class MonteCarlo:
    def __init__(self, titoli: Titoli, S0=1.0, ribilancio=1.0, anni=1, nSims=20000):
        self.S0 = S0
        self.ribilancio = ribilancio
        self.anni = anni
        self.nSims = nSims
        self.titoli = titoli

        self.orizzonte = GIORNI_IN_UN_ANNO * anni
        self.dt = 1 / GIORNI_IN_UN_ANNO

        # Il drift del GBM per log-rendimenti è: (μ - 0.5σ²)dt
        self.drift = (self.titoli.stats.rendimentoAnnuo - 0.5 * self.titoli.stats.volatilitàAnnua**2) * self.dt

        self.coeffDiffusione = self.titoli.stats.volatilitàAnnua * np.sqrt(self.dt)

        self.L = np.linalg.cholesky(self.titoli.corr)

    def simulaTitoli(self, verbose=1):
        # t0 = time.time()
        # Genera rumore indipendente
        Z_ind = np.random.randn(self.nSims, self.orizzonte, len(self.titoli.stats.nomi))
        # Applica correlazione usando decomposizione di Cholesky
        Z = np.einsum("ijk,kl->ijl", Z_ind, self.L)

        (self.investimento, self.valoriTitoli) = _simulaTitoli(
            Z,
            self.nSims,
            self.orizzonte,
            self.drift,
            self.coeffDiffusione,
            self.S0,
            self.titoli.stats.rendimentoGiornalieroDividendi,
            self.dt,
        )

        # simTime = time.time() - t0
        if verbose:
            print(f"\n{Fore.MAGENTA}Performance dei Titoli{Fore.RESET}:")
            for i, nome in enumerate(self.titoli.stats.nomi):
                mediana = (np.median(self.valoriTitoli[:, -1, i]) / self.investimento) - 1

                print(f"    {Fore.GREEN}{nome}{Fore.RESET}: ")
                print(f"        {Fore.YELLOW}Mediana{Fore.RESET}: {mediana:.2%}")

    def valutaPortafoglio(self, portafoglio: Portafoglio, verbose=1):
        valoriPortafoglioTitoli = self.valoriTitoli[portafoglio.indiciTitoli]
        valoriPortafoglioPesati = valoriPortafoglioTitoli * portafoglio.pesi[:, np.newaxis, np.newaxis]
        valoriPortafoglio = np.sum(valoriPortafoglioPesati, axis=0)

        valoriPortafoglioFinale = valoriPortafoglio[-1] - 1

        mediana = np.median(valoriPortafoglioFinale)
        media = np.mean(valoriPortafoglioFinale)
        
        p10 = np.percentile(valoriPortafoglioFinale, 10)
        p25 = np.percentile(valoriPortafoglioFinale, 25)
        p75 = np.percentile(valoriPortafoglioFinale, 75)
        p90 = np.percentile(valoriPortafoglioFinale, 90)

        probPerdita = np.mean(valoriPortafoglioFinale < 0)

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
