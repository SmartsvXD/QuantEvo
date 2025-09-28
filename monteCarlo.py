from settings import GIORNI_IN_UN_ANNO
from dataManager import Portafoglio
import numpy as np


class MonteCarloResults:
    incrementoTotale = None
    valoriDelPortafoglio = None
    valoriDegliInvestimenti = None

    valoriFinali = None
    mediana = None
    media = None

    p10 = None
    p25 = None
    p75 = None
    p90 = None

    probPerdita = None

    anni = None
    orizzonte = None
    nSims = None


class MonteCarlo:
    def __init__(self, portafoglio: Portafoglio, S0=1.0, ribilancio = 1.0,  anni=10, nSims=20000):
        self.S0 = S0
        self.ribilancio = ribilancio
        self.anni = anni
        self.orizzonte = GIORNI_IN_UN_ANNO * anni  # giornate totali
        self.nSims = nSims  # numero di simulazioni
        self.dt = 1 / GIORNI_IN_UN_ANNO

        self.portafoglio = portafoglio

        # generiamo tutte le simulazioni in modo vettoriale

        # parte deterministica (trend medio)
        # dice “quanto ci si aspetta che il log-rendimento cresca mediamente” in un singolo passo temporale.
        self.drift = (
            self.portafoglio.statsComponenti.muGBM.values
            - 0.5 * self.portafoglio.statsComponenti.volatilitàAnnua.values**2
        ) * self.dt

        # coefficiente della parte stocastica rappresenta la variazione casuale dovuta al rumore di mercato.
        self.coeffDiffusione = (
            self.portafoglio.statsComponenti.volatilitàAnnua.values * np.sqrt(self.dt)
        )

    def runSimulation(self, verbose=0):
        # Parametri asset
        n_asset = len(self.portafoglio.pesi)

        # Matrici casuali per ogni asset e simulazione
        Z = np.random.randn(self.nSims, self.orizzonte, n_asset)
        incrementiGiornalieri = (
            self.drift + self.coeffDiffusione * Z
        )  # shape (nSims, orizzonte, n_asset)

        # Array per il valore totale del portafoglio
        valoriPortafoglio = np.zeros((self.nSims, self.orizzonte))
        valoriPortafoglio[:, 0] = self.S0

        # Simulazione con investimento mensile e ribilanciamento parziale
        giorni_per_mese = int(GIORNI_IN_UN_ANNO / 12)
        mesi_totali = self.anni * 12

        for sim in range(self.nSims):
            # stato iniziale: allocazione secondo pesi
            asset_values = self.S0 * self.portafoglio.pesi.copy()

            for day in range(1, self.orizzonte):
                # aggiorna valore di ciascun asset
                asset_values = asset_values * np.exp(incrementiGiornalieri[sim, day, :])

                # ogni mese: aggiungi nuovi fondi e ribilancia
                if day % giorni_per_mese == 0:
                    asset_values = (sum(asset_values) + self.ribilancio) * self.portafoglio.pesi
                    # ribilanciamento parziale: investo solo i nuovi fondi secondo pesi target

                valoriPortafoglio[sim, day] = asset_values.sum()

        results = MonteCarloResults()
        results.valoriDelPortafoglio = valoriPortafoglio
        results.valoriFinali = valoriPortafoglio[:, -1]
        results.mediana = np.median(results.valoriFinali)
        results.media = np.mean(results.valoriFinali)
        results.p10, results.p25, results.p75, results.p90 = np.percentile(
            results.valoriFinali, [10, 25, 75, 90]
        )

        # capitale investito totale
        capitale_investito_totale = self.S0 + self.ribilancio * mesi_totali
        results.probPerdita = np.mean(results.valoriFinali < capitale_investito_totale)

        results.valoriDegliInvestimenti = (
            self.S0 + (np.arange(self.orizzonte) // giorni_per_mese) * self.ribilancio
        )

        results.anni = self.anni
        results.orizzonte = self.orizzonte
        results.nSims = self.nSims

        if verbose:
            print(
                f"\nRisultati ({self.anni} anni, investimenti mensili e ribilanciamento parziale):"
            )
            print(f"Media valore finale: {results.media:.4f}")
            print(f"Mediana valore finale: {results.mediana:.4f}")
            print(
                f"10%-25%-75%-90% percentili: {results.p10:.4f}, {results.p25:.4f}, {results.p75:.4f}, {results.p90:.4f}"
            )
            print(
                f"Probabilità di perdere capitale investito: {results.probPerdita:.2%}"
            )

        return results
