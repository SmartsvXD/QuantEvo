from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import numpy as np
from colorama import Fore
import time

from src.genetica.pymoo import ProblemaPortafoglio, ValidPortfolioSampling, PortfolioRepair, EarlyStoppingCallback
from src.genetica.utils import ordina
from src.managerTitoli import StatsTitoli
from src.valutazione import ValutatoreMultiplo
from src.const import PERCENTUALE_MINIMA_TITOLO
from pymoo.termination import get_termination


def run(
    statsStoriche: StatsTitoli,
    valutatore: ValutatoreMultiplo,
    upper=None,
    percentualeMinima=PERCENTUALE_MINIMA_TITOLO,
    popSize=100,
    nGen=500,
    seeds=None,
    faiEarlyStopping=True,
    pazienzaEarlyStopping=40,
    minTitoli=3,
    hv_ref_point=None,
    verbose=1,
):
    t0 = time.time()

    if verbose >= 1:
        print(f"\n{Fore.MAGENTA}Esecuzione dell'algoritmo evolutivo...{Fore.LIGHTBLACK_EX}")

    nValutazioni = len(statsStoriche.nomi)
    seeds = seeds if seeds is not None else [1]

    all_F, all_X = [], []

    for s in seeds:
        seed_int = int(s)
        np.random.seed(seed_int)

        problem = ProblemaPortafoglio(statsStoriche, nValutazioni, valutatore=valutatore)

        sampling = ValidPortfolioSampling(
            nValutazioni, percentualeMinima=percentualeMinima, upper=upper, minTitoli=minTitoli
        )
        repair = PortfolioRepair(percentualeMinima=percentualeMinima, upper=upper, minTitoli=minTitoli)

        algorithm = NSGA2(
            pop_size=popSize,  # Corretto da popSize a pop_size
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob=0.15, eta=25),
            repair=repair,
        )

        termination = get_termination("n_gen", nGen)

        callback = None
        if faiEarlyStopping:
            callback = EarlyStoppingCallback(
                patience=pazienzaEarlyStopping,
                min_improvement=1e-5,
                verbose=max(0, verbose - 1),
                ref_point=hv_ref_point,
            )

        res = minimize(problem, algorithm, termination, seed=seed_int, verbose=bool(verbose), callback=callback)

        if res.X is not None and len(res.X) > 0:
            all_F.append(np.atleast_2d(res.F))
            all_X.append(np.atleast_2d(res.X))

    if not all_F:
        return [], []

    F_concat = np.vstack(all_F)
    X_concat = np.vstack(all_X)

    ris, port = ordina(F_concat, X_concat, upper=upper, percentualeMinima=percentualeMinima, minTitoli=minTitoli)

    if verbose >= 1:
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    return port, ris
