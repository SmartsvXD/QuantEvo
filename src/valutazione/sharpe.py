import numpy as np
from numba import njit

from src.managerTitoli import StatsTitoli
from src.valutazione.indice import Indice
from src.valutazione.rischio import RischioTotaleIndice
from src.const import GIORNI_IN_UN_ANNO, RISK_FREE_RATE, DEFAULT_N_BOOSTRAP


@njit(fastmath=True)
def _sharpeNumba(portafoglio, rendimentoGiornaliero, varianzaPortafoglio, RF):
    rendimentoAnnuo = rendimentoGiornaliero * GIORNI_IN_UN_ANNO

    portafoglio = np.atleast_2d(portafoglio)

    rendimentoAtteso = np.dot(portafoglio, rendimentoAnnuo)

    rischio = np.sqrt(varianzaPortafoglio)
    rischio = np.where(rischio < 1e-12, 1e-12, rischio)

    sharpe = (rendimentoAtteso - RF) / rischio
    return sharpe


def creaListaBootstrapSharpe(
    stats: StatsTitoli,
    nBootstrap: int = 50,
    randomState=None,
):
    T = stats.rendimenti.shape[1]

    if T < 2 or nBootstrap <= 0:
        return []

    rng = np.random.default_rng(randomState)
    lista = []

    class _SimpleStats:
        __slots__ = ("rendimentoMedioGiornaliero", "stdRendimentiGiornaliera", "corr", "sigma")

    for _ in range(nBootstrap):
        idx = rng.integers(0, T, size=T)
        sample = stats.rendimenti[:, idx]
        rendimentoMedioGiornaliero, stdRendimentiGiornaliera, corr, sigma = stats.calcolaStatsDaSample(
            sample, verbose=0
        )

        s = _SimpleStats()
        s.rendimentoMedioGiornaliero = rendimentoMedioGiornaliero
        s.stdRendimentiGiornaliera = stdRendimentiGiornaliera
        s.corr = corr
        s.sigma = sigma

        lista.append(s)

    return lista


class SharpeIndice(Indice):
    def __init__(self, faiBootstrap=False, nBootstrap=DEFAULT_N_BOOSTRAP):
        super().__init__()

        self.faiBootstrap = faiBootstrap
        self.nBootstrap = nBootstrap
        self.RF = self.calcolaRendimentoPrivoDiRischio()

        self.rischio = RischioTotaleIndice()

    def calcolaRendimentoPrivoDiRischio(self):
        # Da implementare, utilizzeremo FRED
        return RISK_FREE_RATE

    def _sharpeSemplice(self, portafoglio, stats):
        varianzaPortafoglio = self.rischio.calcola(portafoglio, stats)
        return _sharpeNumba(portafoglio, stats.rendimentoMedioGiornaliero, varianzaPortafoglio, self.RF)

    def _sharpeBootstrap(self, portafoglio: np.ndarray, stats: StatsTitoli, randomState=None):
        T = stats.rendimenti.shape[1]

        if T < 2 or self.nBootstrap <= 0:
            return self._sharpeSemplice(portafoglio, stats)

        rng = np.random.default_rng(randomState)
        # Fast vectorized bootstrap: compute sample means and covariances
        # for all bootstrap samples at once using numpy operations.
        # This avoids expensive repeated preprocessing and LedoitWolf fits.
        idx = rng.integers(0, T, size=(self.nBootstrap, T))

        # samples shape: (nAssets, nBootstrap, T)
        samples = stats.rendimenti[:, idx]

        # rendimento medio giornaliero per bootstrap: (nAssets, nBootstrap)
        # rendimento_sample = np.mean(samples, axis=2)

        # demeaned: (nAssets, nBootstrap, T)
        # demeaned = samples - rendimento_sample[:, :, None]

        # Fast path: compute portfolio returns per bootstrap sample directly
        p = np.atleast_2d(portafoglio)

        # portfolio returns shape: (nPort, nBootstrap, T)
        port_returns = np.tensordot(p, samples, axes=(1, 0))

        # expected portfolio return (annualized) per bootstrap: (nPort, nBootstrap)
        rend_atteso = np.mean(port_returns, axis=2) * GIORNI_IN_UN_ANNO

        # portfolio variance (daily) per bootstrap, annualized
        var_port = np.var(port_returns, axis=2, ddof=1) * GIORNI_IN_UN_ANNO

        rischio = np.sqrt(var_port)
        rischio[rischio < 1e-12] = 1e-12

        sharpe_vals = (rend_atteso - self.RF) / rischio

        # mean over bootstrap axis -> (nPort,)
        result = np.mean(sharpe_vals, axis=1)

        return result

    def _sharpeBootstrapFitness(self, portfolio, statsList):
        if len(statsList) == 0:
            return 0.0

        if len(statsList) == 1:
            return self._sharpeSemplice(portfolio, statsList[0])

        fitness_values = self._sharpeSemplice(portfolio, statsList)

        return np.mean(fitness_values)

    def calcola(self, portafoglio, stats, standardizza=False, **kwargs):
        if self.faiBootstrap:
            if (
                "bootstrapList" in kwargs.keys()
                and kwargs["bootstrapList"] is not None
                and len(kwargs["bootstrapList"]) > 0
            ):
                ratio = self._sharpeBootstrapFitness(portafoglio, kwargs["bootstrapList"])
            else:
                ratio = self._sharpeBootstrap(portafoglio, stats)

        else:
            ratio = self._sharpeSemplice(portafoglio, stats)

        return self.standardizza(ratio, standardizza=standardizza)
