import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV

from src.managerTitoli import StatsTitoli
from src.utils.vincoli import Vincoli
from src.valutazione import ValutatoreMultiplo


class ValidPortfolioSampling(Sampling):
    def __init__(self, nValutazioni, vincoli: Vincoli):
        super().__init__()
        self.nValutazioni = nValutazioni
        self.vincoli = vincoli

    def _do(self, problem, n_samples, **kwargs):
        X = []
        for _ in range(n_samples):
            pesi = np.random.rand(self.nValutazioni)
            pesi = self._applica_vincoli(pesi)
            X.append(pesi)
        return np.array(X)

    def _applica_vincoli(self, pesi):
        return self.vincoli.applica(pesi)


class PortfolioRepair(Repair):
    def __init__(self, vincoli: Vincoli):
        super().__init__()
        self.vincoli = vincoli

    def _do(self, problem, X, **kwargs):
        X_repaired = []
        for ind in X:
            ind_repaired = self._applica_vincoli(ind)
            X_repaired.append(ind_repaired)
        return np.array(X_repaired)

    def _applica_vincoli(self, pesi):
        return self.vincoli.applica(pesi)


class ProblemaPortafoglio(Problem):
    def __init__(
        self,
        statsStoriche: StatsTitoli,
        nValutazioni,
        valutatore: ValutatoreMultiplo,
    ):
        super().__init__(n_var=nValutazioni, n_obj=valutatore.numIndici, xl=0.0, xu=1.0)

        self.statsStoriche = statsStoriche
        self.valutatore = valutatore

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.valutatore.calcola(X, self.statsStoriche, inverti=True)


class EarlyStoppingCallback(Callback):
    """
    Ferma l'algoritmo genetico se non ci sono miglioramenti significativi
    nel fronte di Pareto per un certo numero di generazioni.
    """

    def __init__(self, patience=30, min_improvement=1e-4, verbose=False, ref_point=None):
        """
        :param patience: Numero di generazioni senza miglioramento prima di fermare
        :param min_improvement: Miglioramento minimo richiesto per resettare il counter
        :param verbose: Se True, stampa informazioni sulla convergenza
        """
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.verbose = verbose
        self.ref_point = None if ref_point is None else np.asarray(ref_point, dtype=np.float64)
        self.counter = 0
        self.best_hv = None  # Hypervolume del miglior fronte trovato
        self.generation = 0
        self._hv_indicator = None

    def notify(self, algorithm):
        """
        Chiamato ad ogni generazione dall'algoritmo.
        """
        self.generation += 1

        if algorithm.pop is not None and len(algorithm.pop) > 0:
            F = algorithm.pop.get("F")
            if F is not None and len(F) > 0:
                F = np.asarray(F, dtype=np.float64)
                if self.ref_point is None:
                    max_f = np.max(F, axis=0)
                    self.ref_point = max_f + 0.1 * np.maximum(np.abs(max_f), 1.0)
                    self._hv_indicator = HV(ref_point=self.ref_point)
                elif self._hv_indicator is None:
                    self._hv_indicator = HV(ref_point=self.ref_point)

                current_hv = float(self._hv_indicator(F))

                if self.best_hv is None:
                    self.best_hv = current_hv
                    if self.verbose:
                        print(f"Gen {self.generation}: Baseline HV = {current_hv:.6f}")
                else:
                    improvement = current_hv - self.best_hv  # Positivo se miglioramento

                    if improvement > self.min_improvement:
                        # C'è stato un miglioramento significativo
                        self.best_hv = current_hv
                        self.counter = 0
                        if self.verbose:
                            print(f"Gen {self.generation}: Miglioramento HV {improvement:.6f}, hv = {current_hv:.6f}")
                    else:
                        # Nessun miglioramento significativo
                        self.counter += 1
                        if self.verbose and self.counter % 10 == 0:
                            print(f"Gen {self.generation}: No improvement for {self.counter} gens")

                        if self.counter >= self.patience:
                            if self.verbose:
                                print(f"\nEarly stopping at generation {self.generation}")
                                print(f"No improvement for {self.patience} generations")
                            algorithm.termination.force_termination = True
                            return
