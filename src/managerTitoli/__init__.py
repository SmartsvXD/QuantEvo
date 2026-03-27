import numpy as np
import time
from colorama import Fore
from sklearn.covariance import LedoitWolf

from src.const import GIORNI_IN_UN_ANNO, DEFAULT_LAMBDA_EWMA, DEFAULT_MEAN_SHRINKAGE
from src.managerTitoli.downloadStorico import download
from src.managerTitoli.preprocessing import preprocessaRendimenti
from src.managerTitoli.ewma import mediaEWMA, stdEWMA


def _calcolaMatriceCorrelazione(rendimenti: np.ndarray) -> np.ndarray:
    n = rendimenti.shape[0]

    if rendimenti.ndim == 2:
        T = rendimenti.shape[1]
    else:
        return np.eye(n)

    data_mat = np.empty((T, n), dtype=np.float64)

    for j in range(n):
        col = rendimenti[j]
        mask = ~np.isnan(col)
        if np.any(mask):
            mean = np.nanmean(col)
            filled = np.where(np.isnan(col), mean, col)
        else:
            filled = np.zeros_like(col)
        data_mat[:, j] = filled

    lw = LedoitWolf().fit(data_mat)
    cov = lw.covariance_

    std_empirica = np.sqrt(np.diag(cov))
    std_empirica[std_empirica < 1e-10] = 1e-10

    corr_matrix = cov / (std_empirica[:, None] * std_empirica[None, :])

    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix


class StatsTitoli:
    def __init__(self, nomi: tuple, lambdaEWMA=DEFAULT_LAMBDA_EWMA, shrink=DEFAULT_MEAN_SHRINKAGE, verbose=1):
        self.nomi = nomi

        self._lambdaEWMA = lambdaEWMA
        self._shrink = shrink

        self.rendimenti = download(nomi, fonte="yfinance", anni=50, forzaDownload=False, verbose=verbose)

    def calcolaStatsDaSample(self, rendimenti, verbose=1):
        t0 = time.time()

        if verbose >= 1:
            print(f"\n{Fore.MAGENTA}Elaborazione delle stats storiche dei titoli...{Fore.RESET}")

        rendimentiPreprocessati = preprocessaRendimenti(rendimenti)

        rendimentoMedioGiornaliero = mediaEWMA(rendimentiPreprocessati, self._lambdaEWMA)
        stdRendimentiGiornaliera = stdEWMA(rendimentiPreprocessati, self._lambdaEWMA)
        
        rendimentoMedioGiornaliero = self._shrink * rendimentoMedioGiornaliero + (1.0 - self._shrink) * np.mean(
            rendimentoMedioGiornaliero
        )

        # * Calcoliamo la matrice di correlazione
        corr = _calcolaMatriceCorrelazione(rendimentiPreprocessati)

        volatilitaAnnua = stdRendimentiGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)
        sigma = np.outer(volatilitaAnnua, volatilitaAnnua) * corr

        t = time.time() - t0
        if verbose >= 1:
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({t:.2f}s)")

        # * Stampiamo le stats storiche se richieste
        if verbose == 2:
            print(f"\n{Fore.MAGENTA}Dati Storici Titoli{Fore.RESET}: ")
            for i, componente in enumerate(self.nomi):
                rendimentoAnnuo = np.exp(rendimentoMedioGiornaliero[i] * GIORNI_IN_UN_ANNO) - 1
                rendimentoGiornaliero = np.exp(rendimentoMedioGiornaliero[i]) - 1

                volatilitàGiornaliera = stdRendimentiGiornaliera[i]
                volatilitàAnnua = stdRendimentiGiornaliera[i] * np.sqrt(GIORNI_IN_UN_ANNO)

                print(f"    {Fore.GREEN}{componente}{Fore.RESET}:")
                print(f"        {Fore.YELLOW}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
                print(f"                               {Fore.YELLOW}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
                print(f"        {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
                print(f"                               {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")

                rendimentoAnnuoDividendi = self.dividendiGiornalieri[i] * GIORNI_IN_UN_ANNO

                print(f"        {Fore.YELLOW}Rendimento da Dividendi{Fore.RESET}: \t{rendimentoAnnuoDividendi:06.2%}")

        return rendimentoMedioGiornaliero, stdRendimentiGiornaliera, corr, sigma

    def calcolaStats(self, verbose=1):
        self.rendimentoMedioGiornaliero, self.stdRendimentiGiornaliera, self.corr, self.sigma = (
            self.calcolaStatsDaSample(self.rendimenti, verbose=verbose)
        )


def _test():
    s = StatsTitoli(["CSUS.MI", "SWDA.MI", "SGLD.MI"], verbose=1)
    s.calcolaStats(verbose=1)
