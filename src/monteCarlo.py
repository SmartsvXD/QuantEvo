from src.const import GIORNI_IN_UN_ANNO
from colorama import Fore
from src.dataManager import Portafoglio, Stats
import numpy as np
from src.acceleratedFunctions import _simulaTitoli
import time


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
    def __init__(self, S0=1.0, ribilancio=1.0, anni=1):
        self.S0 = S0
        self.ribilancio = ribilancio
        self.anni = anni

    def simulaTitoli(self, statsTitoli: Stats, nSims=20000, verbose=0):
        if verbose >= 1:
            t0 = time.time()

            print()
            print(f"{Fore.MAGENTA}Elaborazione della simulazione dei titoli...{Fore.RESET}")

        # GBM con correzione di Itô:
        # dS_t / S_t = μ * dt + σ * dW_t
        # In forma discreta: ln(S_t) = ln(S_{t-1}) + (μ - σ²/2) * dt + σ * sqrt(dt) * Z

        # Drift: rendimento medio giornaliero del PREZZO (senza dividendi qui)
        # I dividendi verranno trattati separatamente come flusso di cassa
        drift_price = statsTitoli.rendimentoGiornaliero

        # Volatilità giornaliera
        volatility = statsTitoli.volatilitàGiornaliera

        # Matrice di Cholesky per correlazione
        L = statsTitoli.L

        nDays = self.anni * GIORNI_IN_UN_ANNO
        nTitoli = len(statsTitoli.nomi)

        self.valoriTitoli = np.zeros((nSims, nDays, nTitoli), dtype=np.float64)
        self.valoriTitoli[:, 0, :] = self.S0

        # Genera rumore gaussiano standardizzato N(0, 1)
        np.random.seed(49)
        for giorno in range(0, self.anni * GIORNI_IN_UN_ANNO, 36):
            Z = np.random.randn(nSims, 36, len(statsTitoli.nomi))

            # Applica correlazione: Z_corr = Z @ L.T
            # L è la matrice di Cholesky della matrice di CORRELAZIONE
            Z_shape = Z.shape
            Z_reshaped = Z.reshape(-1, Z_shape[-1])  # (nSims * nGiorni, nAsset)
            Z_correlated = Z_reshaped @ L.T  # Applica correlazione
            Z = Z_correlated.reshape(Z_shape)  # Ripristina forma

            # Esegui simulazione
            self.valoriTitoli = _simulaTitoli(
                Z,
                nDays,
                nSims,
                drift_price,
                volatility,
                self.S0,
                statsTitoli.rendimentoGiornalieroDividendi,
                day0=giorno,
                valoriTitoli=self.valoriTitoli,
            )
            self.nSims = nSims

        if verbose >= 1:
            simTime = time.time() - t0
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({simTime:.2f}s)")

        mediana = np.zeros(len(statsTitoli.nomi))

        for i in range(len(statsTitoli.nomi)):
            mediana[i] = (np.median(self.valoriTitoli[:, -1, i]) / self.S0) - 1

        if verbose == 2:
            print(f"\n{Fore.MAGENTA}Performance dei Titoli{Fore.RESET}:")
            for i, nome in enumerate(statsTitoli.nomi):
                print(f"    {Fore.GREEN}{nome}{Fore.RESET}: ")
                print(f"        {Fore.YELLOW}Mediana{Fore.RESET}: {mediana[i]:.2%}")

        return mediana

    def valutaPortafoglio(self, portafoglio: Portafoglio, altro=0, verbose=1):
        # Estrai i valori dei titoli nel portafoglio
        valoriPortafoglioTitoli = self.valoriTitoli[:, :, portafoglio.indiciTitoli]

        # Applica i pesi: ogni titolo contribuisce con il suo peso al portafoglio
        valoriPortafoglioPesati = valoriPortafoglioTitoli * portafoglio.pesi[np.newaxis, np.newaxis, :]

        # Somma i contributi pesati per ottenere il valore totale del portafoglio
        valoriPortafoglio = np.sum(valoriPortafoglioPesati, axis=2)

        # Calcola il rendimento: (valore_finale - valore_iniziale) / valore_iniziale
        # valore_iniziale è S0 (già implicito nei valoriTitoli come S0 al day 0)
        # Ma valoriPortafoglio è già un valore, perciò: rendimento = valoriPortafoglio[:, -1] / S0 - 1
        crescitaPortafoglio = (valoriPortafoglio[:, -1] / self.S0) - 1

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
        # valoriInvestimenti: traccia il valore dell'investimento iniziale nel tempo (baseline per confronto)
        # Se non ribilanciato, è semplicemente una linea orizzontale a S0
        pagella.valoriInvestimenti = np.full(self.anni * GIORNI_IN_UN_ANNO, self.S0, dtype=np.float32)

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
