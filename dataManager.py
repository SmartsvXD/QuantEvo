from settings import GIORNI_IN_UN_ANNO
import yfinance as yf
import numpy as np


def downloadYFinance(tickers):
    data = yf.download(tickers, period="10y", auto_adjust=True)["Close"]
    rendimentiStorici = np.log(data / data.shift(1)).dropna()

    return rendimentiStorici


class Portafoglio:
    def __init__(self, tickers, pesi):
        self.tickers = tickers

        pesi = np.array(pesi)
        pesi = pesi / pesi.sum()
        self.pesi = pesi

        (
            self.rendimentiStoriciTitoli,
            self.storicoRendimentiPortafoglio,
            self.rendimentoGiornalieroMedioAttesoStorico,
            self.volatilitàGiornalieraAttesaStorica,
            self.rendimentoAnnuoMedioAttesoStorico,
            self.volatilitàAnnuaAttesaStorica,
            self.muGBM,
        ) = [None]*7

    def calcolaStatsStoriche(self, verbose=1):
        # Rendimenti storici di ogni singolo titolo
        self.rendimentiStoriciTitoli = downloadYFinance(self.tickers)

        # Calcola il rendimento storico giornaliero del self
        # (moltiplicando la matrice dei pesi per la matrice dei rendimenti storici gioenalieri di ogni bene)
        self.storicoRendimentiPortafoglio = (
            self.rendimentiStoriciTitoli @ self.pesi
        ).values

        # statistiche giornaliere
        self.rendimentoGiornalieroMedioAttesoStorico = np.mean(
            self.storicoRendimentiPortafoglio
        )
        self.volatilitàGiornalieraAttesaStorica = np.std(
            self.storicoRendimentiPortafoglio, ddof=1
        )

        # annualizzo
        self.rendimentoAnnuoMedioAttesoStorico = (
            self.rendimentoGiornalieroMedioAttesoStorico * GIORNI_IN_UN_ANNO
        )
        self.volatilitàAnnuaAttesaStorica = (
            self.volatilitàGiornalieraAttesaStorica * np.sqrt(GIORNI_IN_UN_ANNO)
        )

        # drift per GBM (mu annuale)
        self.muGBM = (
            self.rendimentoAnnuoMedioAttesoStorico
            + 0.5 * self.volatilitàAnnuaAttesaStorica**2
        )

        print(
            f"Rendimento Giornaliero Medio Storico: {self.rendimentoGiornalieroMedioAttesoStorico * 100:.3f}%"
        )
        print(
            f"Volatilità Giornaliera Attesa Storica: {self.volatilitàGiornalieraAttesaStorica * 100:.3f}%"
        )
        print(
            f"Rendimento Annuo Medio Storico: {self.rendimentoAnnuoMedioAttesoStorico * 100:.3f}%"
        )
        print(
            f"Voltatilità Annua Media: {self.volatilitàAnnuaAttesaStorica * 100:.3f}%"
        )
        print(f"mu (GBM Drift Annuo): {self.muGBM * 100:.3f}%")
