from settings import GIORNI_IN_UN_ANNO
import yfinance as yf
import numpy as np
import pandas as pd


def downloadYFinance(tickers) -> pd.DataFrame:
    data = yf.download(tickers, period="10y", auto_adjust=True)["Close"]
    rendimentiStorici = np.log(data / data.shift(1)).dropna()

    return rendimentiStorici


class Stats:
    storico = (None,)

    rendimentoGiornaliero = None
    volatilitàGiornaliera = None

    rendimentoAnnuo = None
    volatilitàAnnua = None

    muGBM = None


class Portafoglio:
    def __init__(self, tickers, pesi):
        self.tickers = tickers

        pesi = np.array(pesi)
        pesi = pesi / pesi.sum()
        self.pesi = pesi
        
        self.statsPortafoglio = Stats()
        self.statsComponenti = Stats()

    def calcolaStatsStoriche(self, verbose=1):
        # Rendimenti storici di ogni singolo titolo
        self.statsComponenti.storico = downloadYFinance(self.tickers)

        # Calcola il rendimento storico giornaliero del self
        # (moltiplicando la matrice dei pesi per la matrice dei rendimenti storici gioenalieri di ogni bene)
        self.statsPortafoglio.storico = (
            self.statsComponenti.storico @ self.pesi
        ).values

        # statistiche giornaliere
        self.statsComponenti.rendimentoGiornaliero = self.statsComponenti.storico.mean()
        self.statsPortafoglio.rendimentoGiornaliero = np.mean(
            self.statsPortafoglio.storico
        )

        self.statsComponenti.volatilitàGiornaliera = self.statsComponenti.storico.std(
            ddof=1
        )
        cov_matrix = self.statsComponenti.storico.cov()
        self.statsPortafoglio.volatilitàGiornaliera = np.sqrt(
            self.pesi.T @ cov_matrix.values @ self.pesi
        )

        # annualizzo
        self.statsPortafoglio.rendimentoAnnuo = (
            self.statsPortafoglio.rendimentoGiornaliero * GIORNI_IN_UN_ANNO
        )
        self.statsComponenti.rendimentoAnnuo = (
            self.statsComponenti.rendimentoGiornaliero * GIORNI_IN_UN_ANNO
        )

        self.statsPortafoglio.volatilitàAnnua = (
            self.statsPortafoglio.volatilitàGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)
        )
        self.statsComponenti.volatilitàAnnua = (
            self.statsComponenti.volatilitàGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)
        )

        # drift per GBM (mu annuale)
        self.statsPortafoglio.muGBM = (
            self.statsPortafoglio.rendimentoAnnuo
            + 0.5 * self.statsPortafoglio.volatilitàAnnua**2
        )
        self.statsComponenti.muGBM = (
            self.statsComponenti.rendimentoAnnuo
            + 0.5 * self.statsComponenti.volatilitàAnnua**2
        )

        if verbose:
            print(
                f"Rendimento Giornaliero Medio Storico: {self.statsPortafoglio.rendimentoGiornaliero * 100:.3f}%"
            )
            print(
                f"Volatilità Giornaliera Attesa Storica: {self.statsPortafoglio.volatilitàGiornaliera * 100:.3f}%"
            )
            print(
                f"Rendimento Annuo Medio Storico: {self.statsPortafoglio.rendimentoAnnuo * 100:.3f}%"
            )
            print(
                f"Voltatilità Annua Media: {self.statsPortafoglio.volatilitàAnnua * 100:.3f}%"
            )
            print(f"mu (GBM Drift Annuo): {self.statsPortafoglio.muGBM * 100:.3f}%")
