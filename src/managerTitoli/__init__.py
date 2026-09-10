import numpy as np
import time
from colorama import Fore
from sklearn.covariance import LedoitWolf
from numba import njit

from src.const import GIORNI_IN_UN_ANNO, DEFAULT_LAMBDA_EWMA, DEFAULT_MEAN_SHRINKAGE
from src.managerTitoli.downloadStorico import download
from src.managerTitoli.preprocessing import preprocessaRendimenti
from src.managerTitoli.ewma import mediaEWMA, stdEWMA


@njit
def _calcolaMatriceCorrelazioneA(rendimenti: np.ndarray):
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

    return data_mat


@njit(fastmath=True)
def _calcolaMatriceCorrelazioneB(cov: np.ndarray):
    std_empirica = np.sqrt(np.diag(cov))
    std_empirica[std_empirica < 1e-10] = 1e-10

    corr_matrix = cov / (std_empirica[:, None] * std_empirica[None, :])

    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix


def _calcolaMatriceCorrelazione(rendimenti: np.ndarray) -> np.array:
    data_mat = _calcolaMatriceCorrelazioneA(rendimenti)

    lw = LedoitWolf().fit(data_mat)
    cov = lw.covariance_

    return _calcolaMatriceCorrelazioneB(cov)


@njit
def _calcolaStatsA(rendimentiPreprocessati, _lambdaEWMA, _shrink):
    rendimentoMedioGiornaliero = mediaEWMA(rendimentiPreprocessati, _lambdaEWMA)
    stdRendimentiGiornaliera = stdEWMA(rendimentiPreprocessati, _lambdaEWMA)

    rendimentoMedioGiornaliero = _shrink * rendimentoMedioGiornaliero + (1.0 - _shrink) * np.mean(
        rendimentoMedioGiornaliero
    )

    return rendimentoMedioGiornaliero, stdRendimentiGiornaliera


@njit(fastmath=True)
def _calcolaStatsB(stdRendimentiGiornaliera, corr):
    volatilitaAnnua = stdRendimentiGiornaliera * np.sqrt(GIORNI_IN_UN_ANNO)
    sigma = np.outer(volatilitaAnnua, volatilitaAnnua) * corr

    return sigma


class StatsTitoliSemplice:
    def __init__(self, rendimenti, lambdaEWMA=DEFAULT_LAMBDA_EWMA, shrink=DEFAULT_MEAN_SHRINKAGE):
        self._lambdaEWMA = lambdaEWMA
        self._shrink = shrink

        self.rendimentoMedioGiornaliero = None
        self.stdRendimentiGiornaliera = None
        self.rendimenti = rendimenti

    def calcolaStats(self):
        rendimentiPreprocessati = preprocessaRendimenti(self.rendimenti)

        self.rendimentoMedioGiornaliero, self.stdRendimentiGiornaliera = _calcolaStatsA(
            rendimentiPreprocessati, self._lambdaEWMA, self._shrink
        )

        self.corr = _calcolaMatriceCorrelazione(rendimentiPreprocessati)

        self.sigma = _calcolaStatsB(self.stdRendimentiGiornaliera, self.corr)

    def __eq__(self, other):
        if not isinstance(other, StatsTitoliSemplice):
            return False

        try:
            cmp_rend_med = (
                self.rendimentoMedioGiornaliero is None and other.rendimentoMedioGiornaliero is None
            ) or np.allclose(self.rendimentoMedioGiornaliero, other.rendimentoMedioGiornaliero, rtol=1e-8, atol=1e-12)

            cmp_std = (
                self.stdRendimentiGiornaliera is None and other.stdRendimentiGiornaliera is None
            ) or np.allclose(self.stdRendimentiGiornaliera, other.stdRendimentiGiornaliera, rtol=1e-8, atol=1e-12)

            cmp_corr = (
                self.corr is None and other.corr is None
            ) or np.allclose(self.corr, other.corr, rtol=1e-8, atol=1e-12)

            cmp_sigma = (
                self.sigma is None and other.sigma is None
            ) or np.allclose(self.sigma, other.sigma, rtol=1e-8, atol=1e-12)

            cmp_rendimenti = np.array_equal(self.rendimenti, other.rendimenti)

            return (
                self._lambdaEWMA == other._lambdaEWMA
                and self._shrink == other._shrink
                and cmp_rend_med
                and cmp_std
                and cmp_corr
                and cmp_sigma
                and cmp_rendimenti
            )
        except Exception:
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
                rendimentoAnnuo = np.exp(self.rendimentoMedioGiornaliero[i] * GIORNI_IN_UN_ANNO) - 1
                rendimentoGiornaliero = np.exp(self.rendimentoMedioGiornaliero[i]) - 1

                volatilitàGiornaliera = self.stdRendimentiGiornaliera[i]
                volatilitàAnnua = self.stdRendimentiGiornaliera[i] * np.sqrt(GIORNI_IN_UN_ANNO)

                print(f"    {Fore.GREEN}{componente}{Fore.RESET}:")
                print(f"        {Fore.YELLOW}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
                print(f"                               {Fore.YELLOW}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
                print(f"        {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
                print(f"                               {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")

    def __eq__(self, other):
        return super().__eq__(other) and self.nomi == other.nomi


def _test():
    s = StatsTitoli(["CSUS.MI", "SWDA.MI", "SGLD.MI"], verbose=1)
    s.calcolaStats(verbose=1)
