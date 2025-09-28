from settings import GIORNI_IN_UN_ANNO
import numpy as np

class MonteCarloResults:
    incrementoTotale = None
    valoriDelPortafoglio = None
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
    def __init__(self, portafoglio, S0=1.0, anni=10, nSims=20000):
        self.S0 = S0
        self.anni = anni
        self.orizzonte = GIORNI_IN_UN_ANNO * anni  # giornate totali
        self.nSims = nSims  # numero di simulazioni
        self.dt = 1 / GIORNI_IN_UN_ANNO
        
        self.portafoglio = portafoglio

        # generiamo tutte le simulazioni in modo vettoriale

        # parte deterministica (trend medio)
        # dice “quanto ci si aspetta che il log-rendimento cresca mediamente” in un singolo passo temporale.
        self.drift = (
            portafoglio.muGBM - 0.5 * portafoglio.volatilitàAnnuaAttesaStorica**2
        ) * self.dt

        # coefficiente della parte stocastica rappresenta la variazione casuale dovuta al rumore di mercato.
        self.coeffDiffusione = portafoglio.volatilitàAnnuaAttesaStorica * np.sqrt(
            self.dt
        )

    def runSimulation(self, verbose = 0):
        # matrice Z, parte causale
        self.Z = np.random.randn(self.nSims, self.orizzonte)

        # calcola log-variations e la serie cumulativa
        self.incrementiGiornalieri = (
            self.drift + self.coeffDiffusione * self.Z
        )  # shape (n_sims, horizon)
        
        results = MonteCarloResults()
        
        results.incrementoTotale = np.cumsum(
            self.incrementiGiornalieri, axis=1
        )  # log(S_t/S0)
        
        results.valoriDelPortafoglio = self.S0 * np.exp(
            results.incrementoTotale
        )  # valori del portafoglio

        results.valoriFinali = results.valoriDelPortafoglio[:, -1]
        results.mediana = np.median(results.valoriFinali)
        results.media = np.mean(results.valoriFinali)
        results.p10, results.p25, results.p75, results.p90 = np.percentile(
            results.valoriFinali, [10, 25, 75, 90]
        )
        
        # percentuale di scenari dove il capitale < 1 (perdita netta)
        results.probPerdita = np.mean(results.valoriFinali < 1.0)
        
        results.anni = self.anni
        results.orizzonte = self.orizzonte
        results.nSims = self.nSims
        
        if verbose:
            print(f"\nRisultati ({self.anni} anni):")
            print(f"Media valore finale: {results.media:.4f}")
            print(f"Mediana valore finale: {results.mediana:.4f}")
            print(f"10%-25%-75%-90% percentili: {results.p10:.4f}, {results.p25:.4f}, {results.p75:.4f}, {results.p90:.4f}")
            print(f"Probabilità di perdere capitale: {results.probPerdita:.2%}")

        return results
