import time

import numpy as np
from colorama import Fore
from sklearn.covariance import LedoitWolf

from src.const import (
    DEFAULT_LAMBDA_EWMA,
    DEFAULT_MEAN_SHRINKAGE,
    DEFAULT_N_BOOSTRAP,
    N_PUNTI_BARRA_CARICAMENTO,
)
from src.managerTitoli.downloadStorico import download


class StatsTitoliSemplice:
    def __init__(self, rendimenti, lambdaEWMA=DEFAULT_LAMBDA_EWMA, shrink=DEFAULT_MEAN_SHRINKAGE):
        self._lambdaEWMA = lambdaEWMA
        self._shrink = shrink

        self.rendimenti = rendimenti
        self.corr = None
        self.sigma = None

        self.listaBootstrap = []

    def calcolaMatriceCorrelazioneECovarianza(self, rendimenti):
        n, T = rendimenti.shape
        data_mat = np.empty((T, n), dtype=np.float64)

        for j in range(n):
            col = rendimenti[j]
            mask = ~np.isnan(col)
            if np.any(mask):
                # Vincolo al -100% sui dati validi lineari
                valid = np.maximum(col[mask], -1.0)
                mean = np.mean(valid)

                filled = np.empty_like(col)
                filled[mask] = valid
                filled[~mask] = mean
            else:
                filled = np.zeros_like(col)

            data_mat[:, j] = filled

        # Stima shrinkage della covarianza
        lw = LedoitWolf().fit(data_mat)
        cov = lw.covariance_

        # Normalizzazione in matrice di correlazione
        std_empirica = np.sqrt(np.diag(cov))
        std_empirica = np.maximum(std_empirica, 1e-10)

        corr_matrix = cov / (std_empirica[:, None] * std_empirica[None, :])

        # Simmetrizzazione, clipping numerico e diagonale unitaria
        corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)

        return corr_matrix, cov

    def calcolaStats(self):
        # Diversificazione, Rischio
        self.corr, self.sigma = self.calcolaMatriceCorrelazioneECovarianza(self.rendimenti)

    def _generaIndiciBlocchi(self, T: int, nBootstrap: int, lunghezzaBlocco: int, rng: np.random.Generator):
        blocco = max(2, min(int(lunghezzaBlocco), T))
        n_blocchi = int(np.ceil(T / blocco))
        max_start = max(1, T - blocco + 1)
        start = rng.integers(0, max_start, size=(nBootstrap, n_blocchi))
        offset = np.arange(blocco, dtype=np.int64)
        idx = (start[:, :, None] + offset[None, None, :]).reshape(nBootstrap, -1)
        return idx[:, :T]

    def preparaListaBootstrap(
        self,
        nBootstrap: int = DEFAULT_N_BOOSTRAP,
        lunghezzaBlocco: int = 20,
        randomState=None,
        verbose: bool = True,
    ):
        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparando la lista bootstrap per {self.__class__.__name__}...{Fore.RESET}")

        self.listaBootstrap = []
        T = self.rendimenti.shape[1]

        if T < 2 or nBootstrap <= 0:
            return

        rng = np.random.default_rng(randomState)
        idxs = self._generaIndiciBlocchi(T, nBootstrap, lunghezzaBlocco, rng)

        for i in range(len(idxs)):
            sample = self.rendimenti[:, idxs[i]]
            s = StatsTitoliSemplice(sample, lambdaEWMA=self._lambdaEWMA, shrink=self._shrink)
            s.calcolaStats()
            self.listaBootstrap.append(s)

            if verbose and i % 10 == 0:
                filled = round(i / len(idxs) * N_PUNTI_BARRA_CARICAMENTO)
                empty = N_PUNTI_BARRA_CARICAMENTO - filled
                print(f"\r{Fore.LIGHTBLACK_EX}{i}/{len(idxs)} [{'*' * filled}{' ' * empty}]{Fore.RESET}", end="")

        if verbose:
            print(f"\r{Fore.LIGHTBLACK_EX}{len(idxs)}/{len(idxs)} [{'*' * N_PUNTI_BARRA_CARICAMENTO}]{Fore.RESET}")
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")


class StatsTitoli(StatsTitoliSemplice):
    def __init__(
        self, nomi: tuple, lambdaEWMA=DEFAULT_LAMBDA_EWMA, shrink=DEFAULT_MEAN_SHRINKAGE, forzaDownload=True, verbose=1
    ):
        self.nomi = nomi

        rendimenti = download(nomi, fonte="yfinance", anni=50, forzaDownload=forzaDownload, verbose=verbose)

        super().__init__(rendimenti, lambdaEWMA=lambdaEWMA, shrink=shrink)

    def calcolaStats(self, verbose=1):
        # t0 = time.time()

        if verbose >= 1:
            print(f"\n{Fore.MAGENTA}Elaborazione delle stats storiche dei titoli...{Fore.RESET}")

        super().calcolaStats()

        # t = time.time() - t0
        # if verbose >= 1:
        #     print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({t:.2f}s)")

        # if verbose == 2:
        #     print(f"\n{Fore.MAGENTA}Dati Storici Titoli{Fore.RESET}: ")
        #     for i, componente in enumerate(self.nomi):
        #         rendimentoAnnuo = self.rendimentoMedioGiornaliero[i] * GIORNI_IN_UN_ANNO
        #         rendimentoGiornaliero = self.rendimentoMedioGiornaliero[i]

        #         volatilitàGiornaliera = self.stdRendimentiGiornaliera[i]
        #         volatilitàAnnua = self.stdRendimentiGiornaliera[i] * np.sqrt(GIORNI_IN_UN_ANNO)

        #         print(f"    {Fore.GREEN}{componente}{Fore.RESET}:")
        #         print(f"        {Fore.YELLOW}Rendimento Medio Giornaliero{Fore.RESET}: \t{rendimentoGiornaliero:06.2%}")
        #         print(f"                               {Fore.YELLOW}Annuo{Fore.RESET}: \t{rendimentoAnnuo:06.2%}")
        #         print(f"        {Fore.YELLOW}Volatilità Media Giornaliera{Fore.RESET}: \t{volatilitàGiornaliera:06.2%}")
        #         print(f"                               {Fore.YELLOW}Annua{Fore.RESET}: \t{volatilitàAnnua:06.2%}")
