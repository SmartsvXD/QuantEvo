from src.valutazione import valuta
from src.genetica.utils import applicaVincoli
from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
import numpy as np


# Sampling custom per generare solo portafogli validi
class ValidPortfolioSampling(Sampling):
    def __init__(self, n_var, percentualeMinima=0.05):
        super().__init__()
        self.n_var = n_var
        self.percentualeMinima = percentualeMinima

    def _do(self, problem, n_samples, **kwargs):
        X = []
        for _ in range(n_samples):
            pesi = np.random.rand(self.n_var)
            # Applica il vincolo iterativamente
            pesi = self._applica_vincoli(pesi)
            X.append(pesi)
        return np.array(X)

    def _applica_vincoli(self, pesi):
        return applicaVincoli(pesi, percentualeMinima=self.percentualeMinima)


class PortfolioRepair(Repair):
    def __init__(self, percentualeMinima=0.05):
        super().__init__()
        self.percentualeMinima = percentualeMinima

    def _do(self, problem, X, **kwargs):
        X_repaired = []
        for ind in X:
            # Applica il vincolo iterativamente
            ind_repaired = self._applica_vincoli(ind)
            X_repaired.append(ind_repaired)
        return np.array(X_repaired)

    def _applica_vincoli(self, pesi):
        return applicaVincoli(pesi, percentualeMinima=self.percentualeMinima)


class PortfolioProblem(Problem):
    def __init__(self, statsStoriche, n_var, percentualeMinima=0.05):
        super().__init__(n_var=n_var, n_obj=3, xl=0.0, xu=1.0)
        self.statsStoriche = statsStoriche
        self.percentualeMinima = percentualeMinima

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([valuta(ind, self.statsStoriche) for ind in X])