from numba import njit
import numpy as np
import pickle as pk
import os
from colorama import Fore

from src.valutazione.indice import Indice
from src.managerTitoli import StatsTitoli, StatsTitoliSemplice
from src.const import N_PUNTI_BARRA_CARICAMENTO, DEFAULT_N_BOOSTRAP


@njit(fastmath=True)
def _generaIndiciBlockBootstrap(T: int, rng: np.random.Generator, lunghezzaBlocco, nBootstrap):
    blocco = max(2, min(lunghezzaBlocco, T))
    n_blocchi = int(np.ceil(T / blocco))
    start = rng.integers(0, T, size=(nBootstrap, n_blocchi))
    offset = np.arange(blocco, dtype=np.int64)

    idx = (start[:, :, None] + offset[None, None, :]) % T
    idx = idx.reshape(nBootstrap, n_blocchi * blocco)

    return idx[:, :T]


class IndiceBootstrap(Indice):
    def __init__(
        self,
        faiBootstrap=False,
        nBootstrap=DEFAULT_N_BOOSTRAP,
        lambdaPenalitaStabilita=0.4,
        usaBlockBootstrap=True,
        lunghezzaBlocco=20,
    ):
        super().__init__()

        self.faiBootstrap = faiBootstrap
        self.nBootstrap = nBootstrap
        self.lambdaPenalitaStabilita = lambdaPenalitaStabilita
        self.usaBlockBootstrap = usaBlockBootstrap
        self.lunghezzaBlocco = max(2, int(lunghezzaBlocco))

        self.listaBootstrap = []
    
    def preparaListaBootstrap(
        self,
        stats: StatsTitoli,
        pathStats,
        pathLista,
        randomState=None,
        forzaRicalcolo=True,
        verbose=True,
    ):
        self.listaBootstrap = []

        ricalcola = True
        if not forzaRicalcolo and os.path.exists(pathStats) and os.path.exists(pathLista):
            with open(pathStats, "rb") as file:
                s = pk.load(file)

            if s == stats:
                ricalcola = False

        if not ricalcola:
            if verbose:
                print(f"{Fore.LIGHTBLACK_EX}Carico il risultato da {pathLista} e {pathStats}...{Fore.RESET}")

            with open(pathLista, "rb") as file:
                self.listaBootstrap = pk.load(file)
        else:
            T = stats.rendimenti.shape[1]

            if T < 2 or self.nBootstrap <= 0:
                return

            rng = np.random.default_rng(randomState)

            if self.usaBlockBootstrap:
                idxs = _generaIndiciBlockBootstrap(T, rng, self.lunghezzaBlocco, self.nBootstrap)
            else:
                idxs = rng.integers(0, T, size=(self.nBootstrap, T))

            for i in range(len(idxs)):
                sample = stats.rendimenti[:, idxs[i]]
                s = StatsTitoliSemplice(sample, lambdaEWMA=stats._lambdaEWMA, shrink=stats._shrink)
                s.calcolaStats()
                self.listaBootstrap.append(s)

                if verbose and i % 10 == 0:
                    filled = round(i / len(idxs) * N_PUNTI_BARRA_CARICAMENTO)
                    empty = N_PUNTI_BARRA_CARICAMENTO - filled
                    print(f"\r{Fore.LIGHTBLACK_EX}{i}/{len(idxs)} [{'*' * filled}{' ' * empty}]{Fore.RESET}", end="")

            if verbose:
                print(f"\r{Fore.LIGHTBLACK_EX}{len(idxs)}/{len(idxs)} [{'*' * N_PUNTI_BARRA_CARICAMENTO}]{Fore.RESET}")
                print(f"{Fore.LIGHTBLACK_EX}Salvo il risultato in {pathLista} e {pathStats}...{Fore.RESET}")

            with open(pathStats, "wb") as file:
                pk.dump(stats, file)

            with open(pathLista, "wb") as file:
                pk.dump(self.listaBootstrap, file)

    def _calcolaSemplice(self, portafoglio, stats):
        return 0

    def _calcolaBootstrap(self, portafoglio: np.ndarray, stats: StatsTitoli, randomState=None):
        T = stats.rendimenti.shape[1]

        if T < 2 or self.nBootstrap <= 0:
            return self._sharpeSemplice(portafoglio, stats)

        rng = np.random.default_rng(randomState)
        if self.usaBlockBootstrap:
            idx = _generaIndiciBlockBootstrap(T, rng, self.lunghezzaBlocco, self.nBootstrap)
        else:
            idx = rng.integers(0, T, size=(self.nBootstrap, T))

        samples = stats.rendimenti[:, idx]

        p = np.atleast_2d(portafoglio)

        vals = self._calcolaSemplice(p, samples)

        media = np.mean(vals, axis=1)
        deviazione = np.std(vals, axis=1)
        result = media - self.lambdaPenalitaStabilita * deviazione

        return result

    def _calcolaBootstrapFitness(self, portafoglio):
        if len(self.listaBootstrap) == 1:
            return self._sharpeSemplice(portafoglio, self.listaBootstrap[0])

        fitness_values = np.array([self._calcolaSemplice(portafoglio, s) for s in self.listaBootstrap])

        media = np.mean(fitness_values, axis=0)
        deviazione = np.std(fitness_values, axis=0)
        return media - self.lambdaPenalitaStabilita * deviazione
    
    def calcola(self, portafoglio, stats, standardizza=False):
        if self.faiBootstrap:
            if self.listaBootstrap != []:
                ratio = self._calcolaBootstrapFitness(portafoglio)
            else:
                ratio = self._calcolaBootstrap(portafoglio, stats)
        else:
            ratio = self._calcolaSemplice(portafoglio, stats)

        return self.standardizza(ratio, standardizza=standardizza)