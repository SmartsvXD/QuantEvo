import src.dataManager as dm
import src.monteCarlo as mt
import src.plt as plt

titoli = {"SPY": 0.5, "EFA": 0.3, "AGG": 0.2}

if __name__ == "__main__":
    etf = dm.Titoli()
    etf.calcolaStatsStoriche()

    port = dm.Portafoglio(titoli, etf)
    # port.calcolaStatsStoriche(verbose=0)

    sim = mt.MonteCarlo()
    sim.simulaTitoli(etf.stats, nSims=20000, verbose=1)
    pagella = sim.valutaPortafoglio(port, verbose=1, altro=1)

    plt.plot(pagella)
