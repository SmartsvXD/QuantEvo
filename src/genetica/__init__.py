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


def run(
    statsStoriche: StatsTitoli,
    valutatore: ValutatoreMultiplo,
    upper=None,
    percentualeMinima=PERCENTUALE_MINIMA_TITOLO,
    popSize=600,
    nGen=1500,
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

    all_F = []
    all_X = []

    if seeds is None:
        seeds = [1]

    for s in seeds:
        problem = ProblemaPortafoglio(statsStoriche, nValutazioni, valutatore=valutatore)

        sampling = ValidPortfolioSampling(
            nValutazioni, percentualeMinima=percentualeMinima, upper=upper, minTitoli=minTitoli
        )

        repair = PortfolioRepair(percentualeMinima=percentualeMinima, upper=upper, minTitoli=minTitoli)

        crossover = SBX(prob=0.9, eta=20)
        mutation = PM(prob=0.15, eta=25)

        algorithm = NSGA2(popSize=popSize, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair)

        termination = MaximumGenerationTermination(nGen)

        callback = None
        if faiEarlyStopping:
            callback = EarlyStoppingCallback(
                patience=pazienzaEarlyStopping,
                min_improvement=1e-5,
                verbose=verbose - 1,
                ref_point=hv_ref_point,
            )

            res = minimize(problem, algorithm, termination, seeds=int(s), verbose=verbose, callback=callback)
        else:
            res = minimize(problem, algorithm, termination, seeds=int(s), verbose=verbose)

        all_F.append(res.F)
        all_X.append(res.X)

    if len(all_F) == 0:
        return [], []

    F_concat = np.vstack(all_F)
    X_concat = np.vstack(all_X)

    ris, port = ordina(F_concat, X_concat, upper=upper, percentualeMinima=percentualeMinima, minTitoli=minTitoli)

    if verbose >= 1:
        print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.2f}s)")

    return port, ris


def _test():
    import json

    from src.valutazione.sharpe import SharpeIndice
    from src.valutazione.rischio import RischioTotaleCorrettoIndice
    from src.valutazione.diversificazione import DiversificazioneIndice
    from src.utils.generaPortafogliValidi import generaPortafogliValidi
    from src.utils.vincoli import applicaVincoli

    with open("titoli.json", "r") as file:
        titoli = [x[0] for x in json.load(file)]

    s = StatsTitoli(titoli, forzaDownload=False, verbose=1)
    s.calcolaStats(verbose=1)

    sharpe = SharpeIndice(faiBootstrap=True)
    rischio = RischioTotaleCorrettoIndice()
    div = DiversificazioneIndice()

    sharpe.preparaListaBootstrap(s, forzaRicalcolo=False)

    v = ValutatoreMultiplo((div, sharpe, rischio), (False, True, False))

    portafogli = generaPortafogliValidi(titoli, vincoli=applicaVincoli)
    v.fitStandardizza(portafogli, s)

    print(run(s, v)[1][0])
