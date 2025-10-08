from settings import GIORNI_IN_UN_ANNO
from dataManager import Portafoglio
import numpy as np
from acceleratedFunctions import _runSimulation
import time


class MonteCarloResults:
    def __init__(self):
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


class MonteCarlo:
    def __init__(self, portafoglio: Portafoglio, S0=1.0, ribilancio=1.0, anni=3, nSims=20000):
        self.S0 = S0
        self.ribilancio = ribilancio
        self.anni = anni
        self.orizzonte = GIORNI_IN_UN_ANNO * anni  # giornate totali
        self.nSims = nSims  # numero di simulazioni
        self.dt = 1 / GIORNI_IN_UN_ANNO

        self.portafoglio: Portafoglio = portafoglio

        # generiamo tutte le simulazioni in modo vettoriale

        # parte deterministica (trend medio)
        # dice “quanto ci si aspetta che il log-rendimento cresca mediamente” in un singolo passo temporale.
        self.drift = (
            self.portafoglio.statsComponenti.muGBM.values
            - 0.5 * self.portafoglio.statsComponenti.volatilitàAnnua.values**2
        ) * self.dt

        # coefficiente della parte stocastica rappresenta la variazione casuale dovuta al rumore di mercato.
        self.coeffDiffusione = self.portafoglio.statsComponenti.volatilitàAnnua.values * np.sqrt(self.dt)

    def runSimulation(self, verbose=0):
        t0 = time.time()
        nAsset = len(self.portafoglio.pesi)
        Z = np.random.randn(self.nSims, self.orizzonte, nAsset)
        Z_div = np.random.randn(self.nSims, self.orizzonte, nAsset)

        results = MonteCarloResults()

        results.valoriDelPortafoglio, results.valoriFinali, results.mediana, results.media, percentili = _runSimulation(
            Z,
            Z_div,
            nAsset,
            self.portafoglio.pesi,
            self.nSims,
            self.orizzonte,
            self.drift,
            self.coeffDiffusione,
            self.S0,
            self.portafoglio.statsComponenti.mediaDividendi,
            self.portafoglio.statsComponenti.volatilitàDividendi,
            self.dt,
            self.ribilancio,
        )

        results.p10, results.p25, results.p75, results.p90 = percentili

        capitale_investito_totale = self.S0 + self.ribilancio * self.anni * 12
        results.probPerdita = np.mean(results.valoriFinali < capitale_investito_totale)

        results.valoriDegliInvestimenti = (
            self.S0 + (np.arange(self.orizzonte) // int(GIORNI_IN_UN_ANNO / 12)) * self.ribilancio
        )

        results.anni = self.anni
        results.orizzonte = self.orizzonte
        results.nSims = self.nSims
        
        results.simTime = time.time() - t0

        if verbose:
            print(f"\nRisultati ({self.anni} anni, investimenti mensili, ribilanciamento e reinvestimento dividendi):")
            print(f"Media valore finale: {results.media:.4f}")
            print(f"Mediana valore finale: {results.mediana:.4f}")
            print(
                f"10%-25%-75%-90% percentili: {results.p10:.4f}, {results.p25:.4f}, {results.p75:.4f}, {results.p90:.4f}"
            )
            print(f"Probabilità di perdere capitale investito: {results.probPerdita:.2%}")
            print(f"Tempo di simulazione: {results.simTime}")

        return results
