import time

import numpy as np
from colorama import Fore
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.genetica.pymoo import EarlyStoppingCallback, PortfolioRepair, ProblemaPortafoglio, ValidPortfolioSampling
from src.genetica.utils import ordina
from src.managerTitoli import StatsTitoli
from src.utils.vincoli import Vincoli
from src.valutazione import ValutatoreMultiplo


def run(
    statsStoriche: StatsTitoli,
    valutatore: ValutatoreMultiplo,
    vincoli: Vincoli,
    popSize=100,
    nGen=500,
    seeds=None,
    faiEarlyStopping=True,
    pazienzaEarlyStopping=40,
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
            nValutazioni, vincoli=vincoli
        )
        repair = PortfolioRepair(vincoli=vincoli)

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

    ris, port = ordina(F_concat, X_concat, vincoli=vincoli)

    if verbose >= 1:
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    return port, ris
