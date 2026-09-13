import time

import numpy as np
from colorama import Fore

from src.const import DEFAULT_LAMBDA_EWMA, DEFAULT_MEAN_SHRINKAGE, GIORNI_IN_UN_ANNO
from src.managerTitoli.downloadStorico import download
from src.managerTitoli.ewma import mediaEWMA, stdEWMA
from src.managerTitoli.matriceCorrelazione import _calcolaMatriceCorrelazione


def _confronta_array(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return np.allclose(a, b, rtol=1e-8, atol=1e-12, equal_nan=True)


class StatsTitoliSemplice:
    def __init__(self, rendimenti, lambdaEWMA=DEFAULT_LAMBDA_EWMA, shrink=DEFAULT_MEAN_SHRINKAGE):
        self._lambdaEWMA = lambdaEWMA
        self._shrink = shrink

        self.rendimentoMedioGiornaliero = None
        self.stdRendimentiGiornaliera = None
        self.rendimenti = rendimenti
        self.corr = None
        self.sigma = None

    def calcolaStats(self):
        self.rendimentoMedioGiornaliero = mediaEWMA(self.rendimenti, self._lambdaEWMA)
        self.stdRendimentiGiornaliera = stdEWMA(self.rendimenti, self._lambdaEWMA)

        self.rendimentoMedioGiornaliero = self._shrink * self.rendimentoMedioGiornaliero + (
            1.0 - self._shrink
        ) * np.nanmean(self.rendimentoMedioGiornaliero)

        self.corr = _calcolaMatriceCorrelazione(self.rendimenti)

        volatilitaAnnua = self.stdRendimentiGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)
        self.sigma = np.outer(volatilitaAnnua, volatilitaAnnua) * self.corr
        

    def __eq__(self, other):
        if not isinstance(other, StatsTitoliSemplice):
            return False

        try:
            return (
                self._lambdaEWMA == other._lambdaEWMA
                and self._shrink == other._shrink
                and _confronta_array(
                    self.rendimentoMedioGiornaliero,
                    other.rendimentoMedioGiornaliero,
                )
                and _confronta_array(self.stdRendimentiGiornaliera, other.stdRendimentiGiornaliera)
                and _confronta_array(self.corr, other.corr)
                and _confronta_array(self.sigma, other.sigma)
                and np.array_equal(self.rendimenti, other.rendimenti, equal_nan=True)
            )
        except Exception as e:
            print(f"Errore durante __eq__: {e}")
            return False


class StatsTitoli(StatsTitoliSemplice):
    def __init__(
        self, nomi: tuple, lambdaEWMA=DEFAULT_LAMBDA_EWMA, shrink=DEFAULT_MEAN_SHRINKAGE, forzaDownload=True, verbose=1
    ):
        self.nomi = nomi

        rendimenti = download(nomi, fonte="yfinance", anni=50, forzaDownload=forzaDownload, verbose=verbose)

        super().__init__(rendimenti, lambdaEWMA=lambdaEWMA, shrink=shrink)

    def calcolaStats(self, verbose=1):
        t0 = time.time()

        if verbose >= 1:
            print(f"\n{Fore.MAGENTA}Elaborazione delle stats storiche dei titoli...{Fore.RESET}")

        super().calcolaStats()

        t = time.time() - t0
        if verbose >= 1:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({t:.2f}s)")

        if verbose == 2:
            print(f"\n{Fore.MAGENTA}Dati Storici Titoli{Fore.RESET}: ")
            for i, componente in enumerate(self.nomi):
                rendimentoAnnuo = self.rendimentoMedioGiornaliero[i] * GIORNI_IN_UN_ANNO
                rendimentoGiornaliero = self.rendimentoMedioGiornaliero[i]

                volatilitàGiornaliera = self.stdRendimentiGiornaliera[i]
                volatilitàAnnua = self.stdRendimentiGiornaliera[i] * np.sqrt(GIORNI_IN_UN_ANNO)

                print(f"    {Fore.GREEN}{componente}{Fore.RESET}:")
                print(f"        {Fore.YELLOW}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
                print(f"                               {Fore.YELLOW}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
                print(f"        {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
                print(f"                               {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")

    def __eq__(self, other):
        return super().__eq__(other) and self.nomi == other.nomi
