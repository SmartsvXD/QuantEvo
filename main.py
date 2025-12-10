import src.managerTitoli as titoli
import src.simulazione as sim
import src.plt as plt
import json

portafoglio = {"SPY": 0.5, "EFA": 0.3, "AGG": 0.2}

if __name__ == "__main__":
    with open("./titoli.json", "r") as file:
        nomiTitoli = json.load(file)

    with open("./data/lambdaEWMA.json", "r") as file:
        lambdaEWMA = json.load(file)

    storico = titoli.download(nomiTitoli, forzaDownload=0)

    statsStoriche = titoli.calcolaStatsStoriche(storico, [lambdaEWMA[titolo] for titolo in nomiTitoli])

    risultati = sim.simulaTitoli(statsStoriche, noise="skewt")

    pagella = sim.valutaPortafoglio(portafoglio, risultati)

    plt.plot(portafoglio, risultati)
