import time

import numpy as np
from colorama import Fore
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.genetica.pymoo import EarlyStoppingCallback, PortfolioRepair, ProblemaPortafoglio, ValidPortfolioSampling
from src.genetica.repair import PortfolioConstraints
from src.genetica.utils import stima_hv_ref_point, trova_knee_point
from src.managerTitoli import StatsTitoli
from src.valutazione import ValutatoreMultiplo


def run(
    statsStoriche: StatsTitoli,
    valutatore: ValutatoreMultiplo,
    constraints: PortfolioConstraints,
    vincoli: ValutatoreMultiplo = None,
    popSize=200,
    nGen=500,
    seeds=None,
    faiEarlyStopping=True,
    pazienzaEarlyStopping=40,
    stima_ref_point=True,
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

        problem = ProblemaPortafoglio(statsStoriche, nValutazioni, valutatore=valutatore, vincoli=vincoli)

        sampling = ValidPortfolioSampling(nValutazioni, constraints=constraints)
        repair = PortfolioRepair(constraints=constraints)

        algorithm = NSGA2(
            pop_size=popSize,
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob=1.0 / nValutazioni, eta=25),
            repair=repair,
        )

        termination = get_termination("n_gen", nGen)

        callback = None
        if faiEarlyStopping:
            if stima_ref_point and hv_ref_point is None:
                hv_ref_point = stima_hv_ref_point(statsStoriche, valutatore, constraints)

            callback = EarlyStoppingCallback(
                patience=pazienzaEarlyStopping,
                min_improvement=1e-5,
                verbose=max(0, verbose - 1),
                ref_point=hv_ref_point,
            )

        res = minimize(problem, algorithm, termination, seed=seed_int, verbose=bool(verbose), callback=callback)

        if res.X is not None and len(res.X) > 0:
            if res.CV is not None:
                cv = np.asarray(res.CV).reshape(len(res.X), -1)
                ammissibili = np.all(cv <= 1e-6, axis=1)
            else:
                ammissibili = np.ones(len(res.X), dtype=bool)

            if np.any(ammissibili):
                all_F.append(np.atleast_2d(res.F)[ammissibili])
                all_X.append(np.atleast_2d(res.X)[ammissibili])
            else:
                if verbose >= 1:
                    print(
                        f"{Fore.YELLOW}Attenzione: nessuna soluzione ammissibile trovata per il seed {seed_int}.{Fore.RESET}"
                    )

    if not all_F:
        return [], []

    F_concat = np.vstack(all_F)
    X_concat = np.vstack(all_X)

    F_concat, X_concat = constraints.rimuoviInvalidi(F=F_concat, X=X_concat)
    ris, port = trova_knee_point(F_concat, X_concat)

    if verbose >= 1:
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    return port, ris
