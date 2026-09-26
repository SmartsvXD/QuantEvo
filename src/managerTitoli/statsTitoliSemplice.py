import time

import numpy as np
from colorama import Fore
from sklearn.covariance import LedoitWolf

from src.const import (
    DEFAULT_N_BOOSTRAP,
    N_PUNTI_BARRA_CARICAMENTO,
)


class StatsTitoliSemplice:
    def __init__(self, rendimenti):
        raw_stats = self.calcolaStats(rendimenti)
        self.stats = {
            k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) else v
            for k, v in raw_stats.items()
        }
        
        self.stats_boot = {}
        
        self.nTitoli = rendimenti.shape[0]

    def calcolaStats(self, rendimenti):
        stats = {}
        stats["rendimenti"] = rendimenti

        # Diversificazione, Rischio
        stats["corr"] = self.calcolaMatriceCorrelazioneECovarianza(rendimenti)

        return stats

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

        return corr_matrix

    @staticmethod
    def _generaIndiciBlocchi(T: int, nBootstrap: int, lunghezzaBlocco: int, rng: np.random.Generator):
        blocco = max(2, min(int(lunghezzaBlocco), T))
        n_blocchi = int(np.ceil(T / blocco))
        max_start = max(1, T - blocco + 1)
        start = rng.integers(0, max_start, size=(nBootstrap, n_blocchi))
        offset = np.arange(blocco, dtype=np.int64)
        idx = (start[:, :, None] + offset[None, None, :]).reshape(nBootstrap, -1)
        return idx[:, :T]

    def preparaBootstrap(
        self,
        nBootstrap: int = DEFAULT_N_BOOSTRAP,
        lunghezzaBlocco: int = 20,
        randomState=None,
        verbose: bool = True,
    ):
        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparando dizionario bootstrap...{Fore.RESET}")

        rend_base = self.stats["rendimenti"][0]  # Rimuove l'asse B=1 per ottenere (N, T)
        T = rend_base.shape[1]
        if T < 2 or nBootstrap <= 0:
            return

        rng = np.random.default_rng(randomState)
        idxs = self._generaIndiciBlocchi(T, nBootstrap, lunghezzaBlocco, rng)

        # Pre-alloca (nBootstrap, ...) preservando dimensioni e dtype di ogni metrica registrata
        self.stats_boot = {
            k: np.empty((nBootstrap, *arr.shape[1:]), dtype=arr.dtype)
            for k, arr in self.stats.items()
            if isinstance(arr, np.ndarray)
        }

        for i in range(nBootstrap):
            sample = np.ascontiguousarray(rend_base[:, idxs[i]])
            sample_stats = self.calcolaStats(sample)
            for k, val in sample_stats.items():
                self.stats_boot[k][i] = val

            if verbose and i % 10 == 0:
                filled = round(i / nBootstrap * N_PUNTI_BARRA_CARICAMENTO)
                empty = N_PUNTI_BARRA_CARICAMENTO - filled
                print(f"\r{Fore.LIGHTBLACK_EX}{i}/{nBootstrap} [{'*' * filled}{' ' * empty}]{Fore.RESET}", end="")

        if verbose:
            print(f"\r{Fore.LIGHTBLACK_EX}{nBootstrap}/{nBootstrap} [{'*' * N_PUNTI_BARRA_CARICAMENTO}]{Fore.RESET}")
            print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")
            
    def ottieni(self, bootstrap: bool = True) -> dict:
        if bootstrap:
            if not self.stats_boot:
                raise ValueError("Bootstrap non preparato. Esegui prima preparaBootstrap().")
            return self.stats_boot
        return self.stats
