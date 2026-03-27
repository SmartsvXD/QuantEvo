import src.managerTitoli as titoli
import json
from colorama import Fore
from src.genetica import run
from src.const import DEFAULT_LAMBDA_EWMA
import numpy as np

if __name__ == "__main__":
    with open("./titoli.json", "r") as file:
        nomiTitoli = json.load(file)

    # Support JSON entries of the form [ticker, upper] or just "ticker"
    nomi = []
    upper = []
    for x in nomiTitoli:
        if isinstance(x, (list, tuple)):
            nomi.append(x[0])
            upper.append(x[1])
        else:
            nomi.append(x)
            upper.append(1.0)

    # Convert upper bounds to numpy array for numba-compiled functions
    upper = np.asarray(upper, dtype=np.float64)

    storico = titoli.download(nomi, forzaDownload=0)
    # Ensure lambdaEWMA is a numpy array (numba-compiled functions expect ndarray)
    lambda_arr = np.asarray([DEFAULT_LAMBDA_EWMA for _ in nomi], dtype=np.float64)
    statsStoriche = titoli.calcolaStatsStoriche(storico, lambda_arr)

    ris, port = run(statsStoriche, n_gen=250, upper=upper, percentualeMinima=0.07)

    print("\n\n\n")
    print("")

    for i in reversed(range(5)):
        print(f"\t{Fore.GREEN}N.{i + 1}{Fore.RESET}")
        print(f"\t\tVarianza Totale: {ris[i][0]:.2%}; Sharpe: {-ris[i][1]:.2f}; Diversificazione: {ris[i][2]:.2%}")
        for j in range(len(statsStoriche.nomi)):
            if port[i][j] != 0:
                print(f"\t\t{Fore.YELLOW}{statsStoriche.nomi[j]}{Fore.RESET}: {port[i][j]:.2%}")
