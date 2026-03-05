import src.managerTitoli as titoli
import json
from colorama import Fore
from src.genetica import run

if __name__ == "__main__":
    with open("./titoli.json", "r") as file:
        nomiTitoli = json.load(file)
    storico = titoli.download(nomiTitoli, forzaDownload=0)
    statsStoriche = titoli.calcolaStatsStoriche(storico, [0.99 for _ in nomiTitoli])

    ris, port = run(statsStoriche)

    print("\n\n\n")
    print("")

    for i in reversed(range(5)):
        print(f"\t{Fore.GREEN}N.{i + 1}{Fore.RESET}")
        print(f"\t\tDiversificazione: {ris[i][0]:.2%}²; Sharpe: {-ris[i][1]:.2f}; Rischio Totale: {ris[i][2]:.2%}²")
        for j in range(len(statsStoriche.nomi)):
            if port[i][j] != 0:
                print(f"\t\t{Fore.YELLOW}{statsStoriche.nomi[j]}{Fore.RESET}: {port[i][j]:.2%}")
