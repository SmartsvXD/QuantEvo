import dataManager as dm
import monteCarlo as mt
import plt
import json

with open("titoli.json", "r") as file:
    nomi = json.load(file)

titoli = {"SPY5.MI": 0.5, "XMWO.MI": 0.3, "SWDA.MI": 0.2}

etf = dm.Titoli(nomi)
etf.calcolaStatsStoriche(forzaDownload=0, forzaElaborazione=0)

port = dm.Portafoglio(titoli, etf)
# port.calcolaStatsStoriche(verbose=0)

sim = mt.MonteCarlo(etf, nSims=20000)
sim.simulaTitoli(forzaElaborazione=0)
pagella = sim.valutaPortafoglio(port, altro=0)

plt.plot(pagella)