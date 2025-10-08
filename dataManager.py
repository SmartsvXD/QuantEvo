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
    }


class Stats:
    def __init__(self):
        self.close = None
        self.open = None
        self.high = None
        self.low = None
        self.volume = None
        self.dividendi = None
        self.splits = None
        self.storico = None

        self.rendimentoGiornaliero = None
        self.volatilitàGiornaliera = None
        self.rendimentoAnnuo = None
        self.volatilitàAnnua = None

        self.rendimentoTotaleDividendi = None
        self.rendimentoDividendi = []
        self.mediaDividendi = []
        self.volatilitàDividendi = []

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
        self.statsComponenti.open = dati["open"]
        self.statsComponenti.high = dati["high"]
        self.statsComponenti.low = dati["low"]
        self.statsComponenti.volume = dati["volume"]
        self.statsComponenti.dividendi = dati["dividends"]
        self.statsComponenti.splits = dati["splits"]
        self.statsComponenti.storico = dati["returns"]

        # Calcola rendimento storico giornaliero del portafoglio
        self.statsPortafoglio.storico = (self.statsComponenti.storico @ self.pesi).values

        # statistiche giornaliere, pesando per dare più importanza a dati più recenti
        self.statsComponenti.rendimentoGiornaliero = self.statsComponenti.storico.ewm(span=75).mean().mean()
        self.statsPortafoglio.rendimentoGiornaliero = np.mean(self.statsPortafoglio.storico)

        self.statsComponenti.volatilitàGiornaliera = self.statsComponenti.storico.ewm(span=75).std().mean()
        cov_matrix = self.statsComponenti.storico.cov()
        self.statsPortafoglio.volatilitàGiornaliera = np.sqrt(self.pesi.T @ cov_matrix.values @ self.pesi)

        # annualizzo
        self.statsPortafoglio.rendimentoAnnuo = self.statsPortafoglio.rendimentoGiornaliero * GIORNI_IN_UN_ANNO
        self.statsComponenti.rendimentoAnnuo = self.statsComponenti.rendimentoGiornaliero * GIORNI_IN_UN_ANNO

        self.statsPortafoglio.volatilitàAnnua = self.statsPortafoglio.volatilitàGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)
        self.statsComponenti.volatilitàAnnua = self.statsComponenti.volatilitàGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)

        # Calcola la serie storica dei rendimenti da dividendo giornalieri per ogni asset
        rendimentoDividendi = []
        mediaDividendi = []
        volatilitaDividendi = []
        if self.statsComponenti.dividendi is not None:
            for ticker in self.tickers:
                if ticker in self.statsComponenti.dividendi:
                    # Calcola il rendimento dei dividendi giornaliero
                    rendDiv = (
                        self.statsComponenti.dividendi[ticker] / self.statsComponenti.close[ticker].shift(1)
                    ).fillna(0)
                    rendimentoDividendi.append(rendDiv.values)

                    # Calola la totalità dei dividendi ed il prezzo medio
                    dividendiTotali = self.statsComponenti.dividendi[ticker].sum()
                    prezzoMedio = self.statsComponenti.close[ticker].mean()

                    # Calcola gli anni di esistenza del componente
                    anni = (
                        self.statsComponenti.close[ticker].index[-1] - self.statsComponenti.close[ticker].index[0]
                    ).days / 365

                    # Calcola il rendimento dei dividendi medio
                    if anni > 0:
                        rendDivTot = (dividendiTotali / prezzoMedio) / anni
                    else:
                        rendDivTot = 0.0
                    mediaDividendi.append(rendDivTot)

                    # Calcola la voltatilità dei dividendi
                    volatilitaDividendi.append(rendDiv.std(ddof=1))
                else:
                    rendimentoDividendi.append(np.array([]))
                    mediaDividendi.append(0.0)
                    volatilitaDividendi.append(0.0)
        else:
            for _ in self.tickers:
                rendimentoDividendi.append(np.array([]))
                mediaDividendi.append(0.0)
                volatilitaDividendi.append(0.0)

        self.statsComponenti.rendimentoDividendi = np.array(rendimentoDividendi, dtype=object)
        self.statsComponenti.mediaDividendi = np.array(mediaDividendi, dtype=np.float32)
        self.statsComponenti.volatilitàDividendi = np.array(volatilitaDividendi, dtype=np.float32)

        # drift per GBM (mu annuale)
        self.statsPortafoglio.muGBM = (
            self.statsPortafoglio.rendimentoAnnuo - 0.5 * self.statsPortafoglio.volatilitàAnnua**2
        )
        self.statsComponenti.muGBM = (
            self.statsComponenti.rendimentoAnnuo - 0.5 * self.statsComponenti.volatilitàAnnua**2
        )

        if verbose:
            print(f"Rendimento Giornaliero Medio Storico: {self.statsPortafoglio.rendimentoGiornaliero * 100:.3f}%")
            print(f"Volatilità Giornaliera Attesa Storica: {self.statsPortafoglio.volatilitàGiornaliera * 100:.3f}%")
            print(f"Rendimento Annuo Medio Storico: {self.statsPortafoglio.rendimentoAnnuo * 100:.3f}%")
            print(f"Volatilità Annua Media: {self.statsPortafoglio.volatilitàAnnua * 100:.3f}%")
            print(f"mu (GBM Drift Annuo): {self.statsPortafoglio.muGBM * 100:.3f}%")

            print("\nDividend Yield storico:")
            for y, t in enumerate(self.tickers):
                print(f"  {t}: {self.statsComponenti.mediaDividendi[y] * 100:.2f}%")

            if self.statsComponenti.splits is not None:
                print("\nSplit azionari (ultimi 10 anni):")
                for t in self.tickers:
                    if t in self.statsComponenti.splits:
                        n_split = self.statsComponenti.splits[t].sum()
                        print(f"  {t}: {n_split} split")
