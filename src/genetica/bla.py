"""
Callback personalizzato per early stopping e convergence detection nell'algoritmo genetico.
Riduce l'overfitting fermando l'algoritmo quando le performance non migliorano più.
"""

from pymoo.core.callback import Callback
import numpy as np

try:
    from pymoo.indicators.hv import HV
except Exception:  # pragma: no cover - compat fallback su versioni pymoo vecchie
    from pymoo.factory import get_performance_indicator

    class HV:  # type: ignore
        def __init__(self, ref_point):
            self._hv = get_performance_indicator("hv", ref_point=np.asarray(ref_point, dtype=np.float64))

        def __call__(self, F):
            return float(self._hv.do(F))


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

        # Calcola hypervolume reale del fronte corrente (problema di minimizzazione).
        if algorithm.pop is not None and len(algorithm.pop) > 0:
            F = algorithm.pop.get("F")
            if F is not None and len(F) > 0:
                F = np.asarray(F, dtype=np.float64)
                if self.ref_point is None:
                    max_f = np.max(F, axis=0)
                    # ref point leggermente peggiore del fronte corrente.
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


class DiversityMonitorCallback(Callback):
    """
    Monitora la diversità genetica della popolazione.
    Utile per diagnosticare convergenza prematura.
    """

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.diversities = []

    def notify(self, algorithm):
        if algorithm.pop is not None and len(algorithm.pop) > 0:
            X = algorithm.pop.get("X")
            if X is not None and len(X) > 0:
                # Calcola diversità come std delle soluzioni
                diversity = np.mean(np.std(X, axis=0))
                self.diversities.append(diversity)

                if self.verbose and len(self.diversities) % 50 == 0:
                    print(f"Gen {len(self.diversities)}: Diversity = {diversity:.6f}")
