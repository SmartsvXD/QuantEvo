import src.dataManager as dm
import src.monteCarlo as mt
import src.plt as plt
import json

with open("titoli.json", "r") as file:
    nomi = json.load(file)
titoli = {"SPY5.MI": 0.5, "XMWO.MI": 0.3, "SWDA.MI": 0.2}


def main():
    etf = dm.Titoli(nomi)
    etf.calcolaStatsStoriche(forzaDownload=1, forzaElaborazione=0)

    port = dm.Portafoglio(titoli, etf)
    # port.calcolaStatsStoriche(verbose=0)

    sim = mt.MonteCarlo(etf, nSims=20000)
    sim.simulaTitoli(forzaElaborazione=0)
    pagella = sim.valutaPortafoglio(port, altro=1)

    plt.plot(pagella)


if __name__ == "__main__":
    main()
    input()
