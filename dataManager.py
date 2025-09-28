from settings import GIORNI_IN_UN_ANNO
import yfinance as yf
import numpy as np
import pandas as pd


def downloadYFinance(tickers) -> pd.DataFrame:
    data = yf.download(
        tickers,
        period="10y",
        auto_adjust=True,
        actions=True,  # per dividendi e split
    )
    # Estraggo le serie principali
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]
    dividends = data["Dividends"] if "Dividends" in data else None
    splits = data["Stock Splits"] if "Stock Splits" in data else None

    # Calcolo rendimenti storici su prezzi di chiusura
    rendimentiStorici = np.log(close / close.shift(1)).dropna()

    # Dividend yield storico (media annuale)
    dividend_yield = None
    if dividends is not None:
        dividend_yield = {}
        for ticker in tickers:
            if ticker in dividends and ticker in close:
                div_annuali = dividends[ticker].resample("Y").sum()
                prezzo_medio_annuo = close[ticker].resample("Y").mean()
                yield_annuo = (div_annuali / prezzo_medio_annuo).mean()
                dividend_yield[ticker] = yield_annuo

    # Restituisco tutto in un dizionario
    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "dividends": dividends,
        "splits": splits,
        "returns": rendimentiStorici,
        "dividend_yield": dividend_yield,
    }


class Stats:
    storico = (None,)
    rendimentoGiornaliero = None
    volatilitàGiornaliera = None
    rendimentoAnnuo = None
    volatilitàAnnua = None
    muGBM = None
    dividend_yield = None
    # beta rimosso
    splits = None
    # expense_ratio rimosso


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
        self.statsComponenti.open = dati["open"]
        self.statsComponenti.high = dati["high"]
        self.statsComponenti.low = dati["low"]
        self.statsComponenti.volume = dati["volume"]
        self.statsComponenti.dividends = dati["dividends"]
        self.statsComponenti.splits = dati["splits"]
        self.statsComponenti.storico = dati["returns"]
        self.statsComponenti.dividend_yield = dati["dividend_yield"]
        # beta rimosso
        # expense_ratio rimosso

        # Calcola rendimento totale (prezzo + dividendi)
        if self.statsComponenti.dividends is not None:
            rendimenti_div = self.statsComponenti.storico.copy()
            for ticker in self.tickers:
                if ticker in self.statsComponenti.dividends:
                    div = self.statsComponenti.dividends[
                        ticker
                    ] / self.statsComponenti.close[ticker].shift(1)
                    rendimenti_div[ticker] += div.fillna(0)
            self.statsComponenti.rendimenti_totali = rendimenti_div
        else:
            self.statsComponenti.rendimenti_totali = self.statsComponenti.storico

        # Calcola rendimento storico giornaliero del portafoglio (inclusi dividendi)
        self.statsPortafoglio.storico = (
            self.statsComponenti.rendimenti_totali @ self.pesi
        ).values

        # statistiche giornaliere
        self.statsComponenti.rendimentoGiornaliero = (
            self.statsComponenti.rendimenti_totali.mean()
        )
        self.statsPortafoglio.rendimentoGiornaliero = np.mean(
            self.statsPortafoglio.storico
        )

        self.statsComponenti.volatilitàGiornaliera = (
            self.statsComponenti.rendimenti_totali.std(ddof=1)
        )
        cov_matrix = self.statsComponenti.rendimenti_totali.cov()
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
            - 0.5 * self.statsPortafoglio.volatilitàAnnua**2
        )
        self.statsComponenti.muGBM = (
            self.statsComponenti.rendimentoAnnuo
            - 0.5 * self.statsComponenti.volatilitàAnnua**2
        )

        if verbose:
            print(
                f"Rendimento Giornaliero Medio Storico (totale): {self.statsPortafoglio.rendimentoGiornaliero * 100:.3f}%"
            )
            print(
                f"Volatilità Giornaliera Attesa Storica (totale): {self.statsPortafoglio.volatilitàGiornaliera * 100:.3f}%"
            )
            print(
                f"Rendimento Annuo Medio Storico (totale): {self.statsPortafoglio.rendimentoAnnuo * 100:.3f}%"
            )
            print(
                f"Volatilità Annua Media (totale): {self.statsPortafoglio.volatilitàAnnua * 100:.3f}%"
            )
            print(f"mu (GBM Drift Annuo): {self.statsPortafoglio.muGBM * 100:.3f}%")
            # Expense Ratio rimosso
            if self.statsComponenti.dividend_yield:
                print("Dividend Yield storico:")
                for t, y in self.statsComponenti.dividend_yield.items():
                    print(f"  {t}: {y * 100:.2f}%")
            # beta rimosso
            if self.statsComponenti.splits is not None:
                print("Split azionari (ultimi 10 anni):")
                for t in self.tickers:
                    if t in self.statsComponenti.splits:
                        n_split = self.statsComponenti.splits[t].sum()
                        print(f"  {t}: {n_split} split")
