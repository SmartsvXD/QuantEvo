from settings import GIORNI_IN_UN_ANNO
from colorama import Fore
import yfinance as yf
import numpy as np
import pandas as pd


def downloadYFinance(tickers):
    print(f"{Fore.GREEN}Scaricando i dati da Yahoo Finance{Fore.LIGHTBLACK_EX}:")
    # auto_adjust=False per avere prezzi non aggiustati e calcolare total return manualmente
    # Questo permette di separare correttamente price appreciation da dividendi
    data = yf.download(
        tickers,
        period="10y",
        auto_adjust=False,
        actions=True,  # per dividendi e split
    )
    print(Fore.RESET)
    # Estraggo le serie principali
    close = data["Close"]
    dividends = data["Dividends"] if "Dividends" in data else None

    # Calcolo rendimenti storici su prezzi AGGIUSTATI (include dividendi reinvestiti)
    # Questo dà il "total return" che è ciò che vogliamo per il GBM
    rendimentiStorici = np.log(close / close.shift(1)).dropna()

    # Restituisco tutto in un dizionario
    return {
        "close": close,
        "dividends": dividends,
        "returns": rendimentiStorici,
    }

class Stats:
    def __init__(self):
        self.close = None
        self.dividendi = None
        self.storico = None

        self.rendimentoGiornaliero = None
        self.volatilitàGiornaliera = None
        self.rendimentoAnnuo = None
        self.volatilitàAnnua = None

        self.rendimentoTotaleDividendi = None
        self.rendimentoDividendi = []
        self.mediaDividendi = []

        self.muGBM = None


class Portafoglio:
    def __init__(self, tickers, pesi):
        self.tickers = tickers
        pesi = np.array(pesi)
        pesi = pesi / pesi.sum()
        self.pesi = pesi
        self.statsPortafoglio = Stats()
        self.statsComponenti = Stats()

    def calcolaStatsStoriche(self, verbose=1):
        # Scarica tutti i dati storici
        dati = downloadYFinance(self.tickers)

        # Salva i dati principali
        self.statsComponenti.close = dati["close"]
        self.statsComponenti.dividendi = dati["dividends"]
        self.statsComponenti.rendimenti = dati["returns"]

        # Calcola rendimento storico giornaliero del portafoglio
        self.statsPortafoglio.rendimenti = (self.statsComponenti.rendimenti @ self.pesi).values

        # Statistiche giornaliere - uso TUTTA la storia disponibile per stime robuste
        # Media dei log-returns giornalieri (rappresenta il drift giornaliero)
        # IMPORTANTE: riordina le colonne secondo l'ordine dei ticker originali
        self.statsComponenti.rendimentoGiornaliero = self.statsComponenti.rendimenti[self.tickers].mean()
        self.statsPortafoglio.rendimentoGiornaliero = float(
            np.dot(self.pesi, self.statsComponenti.rendimentoGiornaliero.values)
        )

        # Volatilità (deviazione standard dei log-returns giornalieri)
        self.statsComponenti.volatilitàGiornaliera = self.statsComponenti.rendimenti[self.tickers].std()

        # Matrice di covarianza dei log-returns per calcolare la volatilità del portafoglio
        # Riordina per avere la matrice allineata con i ticker
        cov_matrix = self.statsComponenti.rendimenti[self.tickers].cov().values

        # Volatilità giornaliera del portafoglio (std dei log-returns pesati)
        self.statsPortafoglio.volatilitàGiornaliera = float(np.sqrt(self.pesi.T @ cov_matrix @ self.pesi))

        # Annualizzazione: per log-returns la media si scala linearmente, la std si scala con sqrt(T)
        self.statsPortafoglio.rendimentoAnnuo = self.statsPortafoglio.rendimentoGiornaliero * GIORNI_IN_UN_ANNO
        self.statsComponenti.rendimentoAnnuo = self.statsComponenti.rendimentoGiornaliero * GIORNI_IN_UN_ANNO

        self.statsPortafoglio.volatilitàAnnua = self.statsPortafoglio.volatilitàGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)
        self.statsComponenti.volatilitàAnnua = self.statsComponenti.volatilitàGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)

        # Calcola i dividend yields per ogni asset
        # I dividendi sono eventi relativamente prevedibili (non stocastici come i prezzi)
        yields_per_ticker = []

        if self.statsComponenti.dividendi is not None:
            for ticker in self.tickers:
                div_series = (
                    self.statsComponenti.dividendi[ticker].dropna()
                    if ticker in self.statsComponenti.dividendi
                    else pd.Series(dtype=float)
                )
                close_series = self.statsComponenti.close[ticker]

                if len(div_series) > 0 and len(close_series) > 0:
                    # Calcola yield medio su TUTTO il periodo storico disponibile (più robusto)
                    # Somma dividendi degli ultimi 3-5 anni / prezzo medio del periodo
                    anni_storia = min(5, len(close_series) / GIORNI_IN_UN_ANNO)
                    data_inizio = close_series.index[-1] - pd.Timedelta(days=int(anni_storia * 365))

                    # Dividendi nel periodo
                    div_periodo = div_series[div_series.index > data_inizio].sum()
                    # Prezzo medio nel periodo (più stabile del solo prezzo corrente)
                    prezzo_medio = close_series[close_series.index > data_inizio].mean()

                    if prezzo_medio > 0 and anni_storia > 0:
                        # Yield annuale = dividendi medi annui / prezzo medio
                        yield_annuale = (div_periodo / anni_storia) / prezzo_medio
                    else:
                        yield_annuale = 0.0
                else:
                    yield_annuale = 0.0

                yields_per_ticker.append(yield_annuale)
        else:
            yields_per_ticker = [0.0] * len(self.tickers)

        self.statsComponenti.rendimentoDividendi = np.array(yields_per_ticker)

        # I dividendi sono trattati come flussi deterministici (non stocastici)
        # Media giornaliera = yield annuale / giorni in un anno
        self.statsComponenti.rendimentoMedioDividendi = self.statsComponenti.rendimentoDividendi / GIORNI_IN_UN_ANNO

        # Volatilità = 0 (i dividendi sono relativamente prevedibili e stabili)
        # Non seguono un random walk come i prezzi
        self.statsComponenti.volatilitàDividendi = np.zeros(len(self.tickers))

        # rendimento dividendi del portafoglio = media pesata
        self.statsPortafoglio.rendimentoDividendi = float(np.dot(self.pesi, self.statsComponenti.rendimentoDividendi))

        # drift per GBM (mu annuale)
        # Il drift include sia l'apprezzamento del prezzo (rendimento annuo) che i dividendi
        # Per il GBM dei prezzi, il drift è basato sul total return (capital gains + dividends)
        # Nota: i dividendi sono già reinvestiti separatamente nella simulazione,
        # quindi il drift qui rappresenta solo l'apprezzamento del capitale
        self.statsPortafoglio.muGBM = self.statsPortafoglio.rendimentoAnnuo
        self.statsComponenti.muGBM = self.statsComponenti.rendimentoAnnuo

        if verbose:
            # Conversione da log-return annuo a rendimento percentuale effettivo annuo
            rendimentoAnnuo = np.exp(self.statsPortafoglio.rendimentoAnnuo) - 1
            rendimentoGiornaliero = np.exp(self.statsPortafoglio.rendimentoGiornaliero) - 1

            # La volatilità dei log-returns si converte approssimando per valori piccoli
            # Per log-returns, la std è già in scala logaritmica, quindi la conversione diretta è:
            volatilitàGiornaliera = self.statsPortafoglio.volatilitàGiornaliera
            volatilitàAnnua = self.statsPortafoglio.volatilitàAnnua

            print(f"{Fore.MAGENTA}Statistiche Portafoglio{Fore.RESET}:")
            print(f"    {Fore.BLUE}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
            print(f"                           {Fore.BLUE}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
            print(f"    {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
            print(f"                           {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")
            print(f"    {Fore.CYAN}mu (GBM Drift Annuo){Fore.RESET}: \t\t{self.statsPortafoglio.muGBM:06.2%}")

            print(f"\n{Fore.MAGENTA}Statistiche Componenti{Fore.RESET}: ")
            for i, componente in enumerate(self.tickers):
                # Voglio visualizzare i dati non in log-return, ma linearmente
                rendimentoAnnuo = np.exp(self.statsComponenti.rendimentoAnnuo.iloc[i]) - 1
                rendimentoGiornaliero = np.exp(self.statsComponenti.rendimentoGiornaliero.iloc[i]) - 1

                volatilitàGiornaliera = self.statsComponenti.volatilitàGiornaliera.iloc[i]
                volatilitàAnnua = self.statsComponenti.volatilitàAnnua.iloc[i]

                print(f"    {Fore.GREEN}{componente}{Fore.RESET}:")
                print(f"        {Fore.BLUE}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
                print(f"                               {Fore.BLUE}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
                print(f"        {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
                print(f"                               {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")

                print(
                    f"        {Fore.LIGHTRED_EX}Rendimento da Dividendi{Fore.RESET}: \t{self.statsComponenti.rendimentoDividendi[i]:06.2%}"
                )
