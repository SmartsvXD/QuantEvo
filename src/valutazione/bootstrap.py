import numpy as np
from colorama import Fore

from src.const import DEFAULT_N_BOOSTRAP, N_PUNTI_BARRA_CARICAMENTO
from src.managerTitoli import StatsTitoli, StatsTitoliSemplice
from src.valutazione.indice import Indice


class IndiceBootstrap(Indice):
    def __init__(
        self,
        faiBootstrap=False,
        nBootstrap=DEFAULT_N_BOOSTRAP,
        lunghezzaBlocco=20,
        metricaRischio="cvar",  # "cvar", "percentile", oppure "downside_penalty"
        alpha=0.05,
        lambdaPenalitaStabilita=0.4,
        inverti=False,
    ):
        super().__init__(inverti=inverti)

        self.faiBootstrap = faiBootstrap
        self.nBootstrap = nBootstrap
        self.lunghezzaBlocco = max(2, int(lunghezzaBlocco))
        self.metricaRischio = metricaRischio
        self.alpha = alpha
        self.lambdaPenalitaStabilita = lambdaPenalitaStabilita

        self.listaBootstrap = []

    def _generaIndiciBlocchi(self, T: int, rng: np.random.Generator) -> np.ndarray:
        """Genera indici con Moving Block Bootstrap senza wrap-around circolare."""
        blocco = max(2, min(self.lunghezzaBlocco, T))
        n_blocchi = int(np.ceil(T / blocco))

        # Start limitato a T - blocco + 1 per evitare sovrapposizioni spurie tra fine e inizio
        max_start = max(1, T - blocco + 1)
        start = rng.integers(0, max_start, size=(self.nBootstrap, n_blocchi))
        offset = np.arange(blocco, dtype=np.int64)

        idx = (start[:, :, None] + offset[None, None, :]).reshape(self.nBootstrap, -1)
        return idx[:, :T]

    def _aggregaValori(self, vals: np.ndarray) -> np.ndarray:
        """
        Aggrega le valutazioni bootstrap (forma: n_portafogli, nBootstrap)
        usando metriche asimmetriche o coerenti con il rischio di coda.
        """
        if self.metricaRischio == "cvar":
            # Expected Shortfall: media condizionata delle osservazioni peggiori sotto l'alfa-quantile
            q = np.quantile(vals, self.alpha, axis=1, keepdims=True)
            mask = vals <= q
            counts = np.maximum(1, np.sum(mask, axis=1))
            return np.sum(np.where(mask, vals, 0.0), axis=1) / counts

        elif self.metricaRischio == "percentile":
            # Quantile conservativo (es. 5° percentile della distribuzione bootstrap)
            return np.quantile(vals, self.alpha, axis=1)

        elif self.metricaRischio == "downside_penalty":
            # Media penalizzata unicamente dalla semi-deviazione standard negativa
            media = np.mean(vals, axis=1)
            diff_negativa = np.minimum(0.0, vals - media[:, None])
            semi_dev = np.sqrt(np.mean(diff_negativa**2, axis=1))
            return media - self.lambdaPenalitaStabilita * semi_dev

        # Fallback neutrale alla media semplice
        return np.mean(vals, axis=1)

    def preparaListaBootstrap(
        self,
        stats: StatsTitoli,
        randomState=None,
        verbose=True,
    ):
        self.listaBootstrap = []
        T = stats.rendimenti.shape[1]

        if T < 2 or self.nBootstrap <= 0:
            return

        rng = np.random.default_rng(randomState)
        idxs = self._generaIndiciBlocchi(T, rng)

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

    def _calcolaSemplice(self, portafoglio, stats):
        return None

    def _calcolaBootstrap(self, portafoglio: np.ndarray, stats: StatsTitoli, randomState=None):
        T = stats.rendimenti.shape[1]

        if T < 2 or self.nBootstrap <= 0:
            return self._calcolaSemplice(portafoglio, stats)

        rng = np.random.default_rng(randomState)
        idxs = self._generaIndiciBlocchi(T, rng)

        p = np.atleast_2d(portafoglio)
        n_port = p.shape[0]
        vals = np.empty((n_port, len(idxs)), dtype=np.float64)

        for i in range(len(idxs)):
            sample = stats.rendimenti[:, idxs[i]]
            s = StatsTitoliSemplice(sample, lambdaEWMA=stats._lambdaEWMA, shrink=stats._shrink)
            s.calcolaStats()

            res_i = np.atleast_1d(self._calcolaSemplice(p, s))
            vals[:, i] = res_i.ravel()

        result = self._aggregaValori(vals)
        return result if portafoglio.ndim > 1 else result[0]

    def _calcolaBootstrapFitness(self, portafoglio):
        if not self.listaBootstrap:
            return None

        if len(self.listaBootstrap) == 1:
            return self._calcolaSemplice(portafoglio, self.listaBootstrap[0])

        p = np.atleast_2d(portafoglio)
        # Valuta i portafogli su ciascun campione della lista fissa (n_port, nBootstrap)
        vals = np.column_stack([
            np.atleast_1d(self._calcolaSemplice(p, s)).ravel()
            for s in self.listaBootstrap
        ])

        result = self._aggregaValori(vals)
        return result if portafoglio.ndim > 1 else result[0]

    def calcola(self, portafoglio, stats, standardizza=True, inverti=True):
        if self.faiBootstrap:
            if self.listaBootstrap:
                ratio = self._calcolaBootstrapFitness(portafoglio)
            else:
                ratio = self._calcolaBootstrap(portafoglio, stats)
        else:
            ratio = self._calcolaSemplice(portafoglio, stats)

        return super().calcola(portafoglio, stats, ratio, standardizza, inverti)
