from settings import GIORNI_IN_UN_ANNO
from colorama import Fore, Style
from dataManager import Portafoglio
import numpy as np
from acceleratedFunctions import _runSimulation
import time


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
    def __init__(self, portafoglio: Portafoglio, S0=1.0, ribilancio=1.0, anni=3, nSims=20000):
        self.S0 = S0
        self.ribilancio = ribilancio
        self.anni = anni
        self.orizzonte = GIORNI_IN_UN_ANNO * anni
        self.nSims = nSims
        self.dt = 1 / GIORNI_IN_UN_ANNO

        self.portafoglio: Portafoglio = portafoglio

        # Il drift del GBM per log-rendimenti è: (μ - 0.5σ²)dt
        self.drift = (
            self.portafoglio.statsComponenti.muGBM.values
            - 0.5 * self.portafoglio.statsComponenti.volatilitàAnnua.values**2
        ) * self.dt

        self.coeffDiffusione = self.portafoglio.statsComponenti.volatilitàAnnua.values * np.sqrt(self.dt)

    def runSimulation(self, verbose=0):
        t0 = time.time()
        nAsset = len(self.portafoglio.pesi)
        Z = np.random.randn(self.nSims, self.orizzonte, nAsset)

        results = MonteCarloResults()

        (
            results.investimentoTotale,
            results.valoriDelPortafoglio,
            results.valoriFinali,
            results.mediana,
            results.media,
            percentili,
        ) = _runSimulation(
            Z,
            nAsset,
            self.portafoglio.pesi,
            self.nSims,
            self.orizzonte,
            self.drift,
            self.coeffDiffusione,
            self.S0,
            self.portafoglio.statsComponenti.rendimentoMedioDividendi,
            self.dt,
            self.ribilancio,
        )

        results.p10, results.p25, results.p75, results.p90 = percentili
        results.probPerdita = np.mean(results.valoriFinali < results.investimentoTotale)

        results.valoriDegliInvestimenti = (
            self.S0 + (np.arange(self.orizzonte) // int(GIORNI_IN_UN_ANNO / 12)) * self.ribilancio
        )

        results.anni = self.anni
        results.orizzonte = self.orizzonte
        results.nSims = self.nSims
        results.S0 = self.S0

        results.simTime = time.time() - t0

        if verbose:
            print(f"\n{Fore.MAGENTA}Risultati{Fore.RESET}:")
            print(f"    {Fore.CYAN}Investimento{Fore.RESET}: \t€{results.investimentoTotale:05.2f}")
            print(f"    {Fore.YELLOW}Media finale{Fore.RESET}: \t{results.media:06.2%}")
            print(f"    {Style.BRIGHT}{Fore.YELLOW}Mediana finale{Style.RESET_ALL}: \t{results.mediana:06.2%}")

            print(f"    Percentili, {Style.BRIGHT}{Fore.RED}10%{Style.RESET_ALL}: \t{results.p10:06.2%}")
            print(f"                {Fore.LIGHTRED_EX}25%{Fore.RESET}: \t{results.p25:06.2%}")
            print(f"                {Fore.LIGHTGREEN_EX}75%{Fore.RESET}: \t{results.p75:06.2%}")
            print(f"                {Style.BRIGHT}{Fore.GREEN}90%{Style.RESET_ALL}: \t{results.p90:06.2%}")

            print(
                f"    {Style.BRIGHT}{Fore.LIGHTBLACK_EX}Prob. perdita{Style.RESET_ALL}: \t{results.probPerdita:06.2%}"
            )
            print(f"\n{Fore.BLUE}Tempo di simulazione{Fore.RESET}: {results.simTime:.2f}s")

        return results
