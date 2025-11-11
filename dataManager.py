from settings import GIORNI_IN_UN_ANNO
from colorama import Fore
import yfinance as yf
import numpy as np
from acceleratedFunctions import _calcolaStatsTitoli, _calcolaCorrelazione


class Stats:
    """
    Classe che contiene le statistiche finanziarie per titoli o portafogli.
    Memorizza dati come rendimenti, volatilità, dividendi e parametri per il modello GBM.
    """

    def __init__(self):
        self.nomi = []

        self.close = None
        self.dividendi = None
        self.rendimenti = None

        self.rendimentoGiornaliero = None
        self.volatilitàGiornaliera = None
        self.rendimentoAnnuo = None
        self.volatilitàAnnua = None

        self.rendimentoGiornalieroDividendi = None
        self.rendimentoAnnuoDividendi = []
        self.volatilitàDividendi = []


class Titoli:
    """
    Classe che gestisce i titoli del portafoglio, scaricando dati da Yahoo Finance
    e calcolando statistiche storiche per ciascun asset.
    """

    def __init__(self, nomi):
        self.stats = Stats()
        self.stats.nomi = nomi

        self.corr = None

    def downloadYFinance(self, verbose=1):
        """
        Scarica i dati storici da Yahoo Finance per i titoli specificati.
        Filtra i titoli con dati insufficienti e prepara i dati per il calcolo delle statistiche.
        """
        if verbose:
            print(f"{Fore.GREEN}Scaricando i dati da Yahoo Finance{Fore.LIGHTBLACK_EX}:")

        # auto_adjust=False per avere prezzi non aggiustati e calcolare total return manualmente
        # Questo permette di separare correttamente price appreciation da dividendi
        data = yf.download(
            self.stats.nomi,
            period="50y",
            auto_adjust=False,
            actions=True,  # per dividendi e split
        )

        if verbose:
            print(Fore.RESET)

        # Estrae le serie principali: prezzi di chiusura e dividendi
        close = data["Close"]
        dividends = data["Dividends"] if "Dividends" in data else None

        # Assicura che l'ordine delle colonne corrisponda a self.nomi
        close = close[self.stats.nomi]
        if dividends is not None:
            dividends = dividends[self.stats.nomi]

        # Filtra ticker con almeno 1260 giorni di dati originali (circa 5 anni di trading)
        giorni_minimi = 1260
        nomi_invalidi = [ticker for ticker in self.stats.nomi if len(close[ticker].dropna()) <= giorni_minimi]
        if nomi_invalidi != []:
            print(f"{Fore.RED} I seguenti titoli non hanno un numero sufficiente di giorni di dati:")
            for i in nomi_invalidi:
                print(f"\t{i}")
            exit()

        # Calcola i log-returns giornalieri: ln(P_t / P_{t-1})
        self.stats.rendimenti = np.array(np.log(close / close.shift(1)).dropna(how="all")).T
        # Salva i dividendi giornalieri
        self.stats.dividendi = np.array(dividends).T
        # Salva i prezzi di chiusura
        self.stats.close = np.array(close).T

    def stampaDatiStorici(self):
        print(f"\n{Fore.MAGENTA}Dati Storici Titoli{Fore.RESET}: ")
        for i, componente in enumerate(self.stats.nomi):
            # Converte i log-returns in rendimenti percentuali effettivi per una visualizzazione più intuitiva
            rendimentoAnnuo = np.exp(self.stats.rendimentoAnnuo[i]) - 1
            rendimentoGiornaliero = np.exp(self.stats.rendimentoGiornaliero[i]) - 1

            volatilitàGiornaliera = self.stats.volatilitàGiornaliera[i]
            volatilitàAnnua = self.stats.volatilitàAnnua[i]

            print(f"    {Fore.GREEN}{componente}{Fore.RESET}:")
            print(f"        {Fore.YELLOW}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
            print(f"                               {Fore.YELLOW}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
            print(f"        {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
            print(f"                               {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")

            print(
                f"        {Fore.YELLOW}Rendimento da Dividendi{Fore.RESET}: \t{self.stats.rendimentoAnnuoDividendi[i]:06.2%}"
            )

    def calcolaStatsStoriche(self, verbose=1):
        """
        Calcola le statistiche storiche per ciascun componente utilizzando la funzione accelerata.
        Include rendimenti, volatilità e rendimento da dividendi, e opzionalmente stampa i risultati.
        """
        (
            self.stats.rendimentoGiornaliero,
            self.stats.volatilitàGiornaliera,
            self.stats.rendimentoAnnuo,
            self.stats.volatilitàAnnua,
            self.stats.rendimentoAnnuoDividendi,
            self.stats.rendimentoGiornalieroDividendi,
            self.stats.volatilitàDividendi,
        ) = _calcolaStatsTitoli(self.stats.nomi, self.stats.rendimenti, self.stats.dividendi, self.stats.close)

        self.corr = _calcolaCorrelazione(self.stats.rendimenti)

        if verbose:
            self.stampaDatiStorici()


class Portafoglio:
    """
    Classe che rappresenta un portafoglio di investimento, calcolando statistiche aggregate
    basate sui pesi dei titoli e sulle loro statistiche individuali.
    """

    def __init__(self, nomiEPesi: dict, titoli: Titoli):
        pesi = np.array(list(nomiEPesi.values()))
        pesi = pesi / pesi.sum()
        self.pesi = pesi
        self.titoli = titoli

        self.stats = Stats()
        self.stats.nomi = list(nomiEPesi.keys())

        self.nomiEPesi = nomiEPesi

        self.indiciTitoli = []
        for i in self.stats.nomi:
            self.indiciTitoli.append(self.titoli.stats.nomi.index(i))

    def stampaDatiStorici(self, portafoglio=1, titoli=0):
        """
        Stampa le statistiche del portafoglio e/o dei titoli in formato leggibile.
        Converte i log-returns in percentuali effettive per una migliore comprensione.
        """
        if titoli:
            print(f"\n{Fore.MAGENTA}Statistiche Titoli{Fore.RESET}: ")
            for i, componente in enumerate(self.stats.nomi):
                # Converte i log-returns in rendimenti percentuali effettivi per una visualizzazione più intuitiva
                idx = self.indiciTitoli[i]
                rendimentoAnnuo = np.exp(self.titoli.stats.rendimentoAnnuo[idx]) - 1
                rendimentoGiornaliero = np.exp(self.titoli.stats.rendimentoGiornaliero[idx]) - 1

                volatilitàGiornaliera = self.titoli.stats.volatilitàGiornaliera[idx]
                volatilitàAnnua = self.titoli.stats.volatilitàAnnua[idx]

                print(f"    {Fore.GREEN}{componente}{Fore.RESET}:")
                print(f"        {Fore.YELLOW}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
                print(f"                               {Fore.YELLOW}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
                print(f"        {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
                print(f"                               {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")

                print(
                    f"             {Fore.YELLOW}Rendimento da Dividendi{Fore.RESET}: \t{self.titoli.stats.rendimentoAnnuoDividendi[idx]:06.2%}"
                )

        if portafoglio:
            # Converte i log-returns annuali in rendimenti percentuali effettivi
            rendimentoAnnuo = np.exp(self.stats.rendimentoAnnuo) - 1
            rendimentoGiornaliero = np.exp(self.stats.rendimentoGiornaliero) - 1

            # La volatilità dei log-returns è già in termini relativi, quindi non richiede conversione aggiuntiva
            # Per log-returns, la std è già in scala logaritmica, quindi la conversione diretta è:
            volatilitàGiornaliera = self.stats.volatilitàGiornaliera
            volatilitàAnnua = self.stats.volatilitàAnnua

            print(f"\n{Fore.MAGENTA}Statistiche Portafoglio{Fore.RESET}:")
            print(f"    {Fore.YELLOW}Titoli Portafoglio{Fore.RESET}:")
            for i, nome in enumerate(self.stats.nomi):
                print(f"         {Fore.GREEN}{nome}{Fore.RESET}: {self.pesi[i]:.1%}")
            print(f"    {Fore.YELLOW}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
            print(f"                           {Fore.YELLOW}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
            print(f"    {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
            print(f"                           {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")

    def calcolaStatsStoriche(self, verbose=1):
        """
        Calcola le statistiche aggregate del portafoglio basate sui pesi e sulle statistiche dei titoli.
        Include calcolo della matrice di covarianza, annualizzazione e rendimento da dividendi.
        """
        # Calcola rendimento storico giornaliero del portafoglio
        self.stats.rendimenti = self.titoli.stats.rendimenti[self.indiciTitoli].T @ self.pesi

        # Statistiche giornaliere - uso TUTTA la storia disponibile per stime robuste
        # Media dei log-returns giornalieri (rappresenta il drift giornaliero)
        self.stats.rendimentoGiornaliero = float(
            np.dot(self.titoli.stats.rendimentoGiornaliero[self.indiciTitoli], self.pesi)
        )

        # Matrice di covarianza dei log-returns per calcolare la volatilità del portafoglio
        rendimenti = self.titoli.stats.rendimenti[self.indiciTitoli]
        n = rendimenti.shape[0]
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Calcola la covarianza tra i rendimenti di i e j, ignorando i valori NaN
                mask = ~(np.isnan(rendimenti[i]) | np.isnan(rendimenti[j]))
                if np.sum(mask) > 1:
                    cov_matrix[i, j] = np.cov(rendimenti[i, mask], rendimenti[j, mask], ddof=1)[0, 1]
                else:
                    cov_matrix[i, j] = 0

        # Volatilità giornaliera del portafoglio (std dei log-returns pesati)
        self.stats.volatilitàGiornaliera = float(np.sqrt(self.pesi.T @ cov_matrix @ self.pesi))

        # Annualizzazione: per log-returns la media si scala linearmente, la std si scala con sqrt(T)
        self.stats.rendimentoAnnuo = self.stats.rendimentoGiornaliero * GIORNI_IN_UN_ANNO

        self.stats.volatilitàAnnua = self.stats.volatilitàGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)

        # rendimento dividendi del portafoglio = media pesata
        self.stats.rendimentoAnnuoDividendi = float(
            np.dot(self.pesi, self.titoli.stats.rendimentoAnnuoDividendi[self.indiciTitoli])
        )

        if verbose:
            self.stampaDatiStorici()
