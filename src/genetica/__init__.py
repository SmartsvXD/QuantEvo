from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from src.genetica.pymoo import PortfolioProblem, ValidPortfolioSampling, PortfolioRepair
from src.genetica.utils import ordina


# Esempio di utilizzo:
def run(statsStoriche, pop_size=600, n_gen=1500, seed=1):
    # Prova a ricavare n_var in modo robusto
    n_var = len(statsStoriche.nomi)

    problem = PortfolioProblem(statsStoriche, n_var)
    sampling = ValidPortfolioSampling(n_var, percentualeMinima=0.05)
    repair = PortfolioRepair(percentualeMinima=0.05)
    crossover = SBX(prob=0.9, eta=15)
    mutation = PM(prob=0.2, eta=20)
    algorithm = NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair)
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=True)
    
    ris, port = ordina(res.F, res.X)
    
    return ris, port


# Esempio:
def _test():
    import src.managerTitoli as titoli
    from colorama import Fore
    import json
    import numpy as np
    from src.genetica.utils import ordina

    with open("./titoli.json", "r") as file:
        nomiTitoli = json.load(file)
    storico = titoli.download(nomiTitoli, forzaDownload=0)
    statsStoriche = titoli.calcolaStatsStoriche(storico, [0.98 for _ in nomiTitoli])

    result = run(statsStoriche)

    validiPort = []
    validiRis = []
    for i, port in enumerate(result.X):
        if not np.any(port >= 0.2):
            validiPort.append(port)
            validiRis.append(result.F[i])

    print("\n\n\n")
    print("")

    ris, port = ordina(validiRis, validiPort)
    print(f"{Fore.MAGENTA}Top 5{Fore.RESET}")
    for i in reversed(range(5)):
        print(f"\t{Fore.GREEN}N.{i + 1}{Fore.RESET}")
        print(f"\t\tDiversificazione: {ris[i][0]:.2%}²; Sharpe: {-ris[i][1]:.2f}; Rischio Totale: {ris[i][2]:.2%}²")
        for j in range(len(statsStoriche.nomi)):
            if port[i][j] != 0:
                print(f"\t\t{Fore.YELLOW}{statsStoriche.nomi[j]}{Fore.RESET}: {port[i][j]:.2%}")
