"""
Obiettivo della simulazione:

Questo codice esegue una simulazione Monte Carlo per valutare la probabilità di diversi risultati futuri di un portafoglio d’investimento reale, composto da ETF liquidi (SPY, AGG, EFA) con pesi prefissati.
Utilizza dati storici di rendimenti per stimare media e volatilità, e applica il modello di moto browniano geometrico (GBM) per simulare l’evoluzione del valore del portafoglio su un orizzonte di 10 anni.

L’obiettivo è fornire una stima quantitativa del rischio e del rendimento atteso, calcolando:
- Probabilità di perdita del capitale
- Distribuzione dei risultati finali (media, mediana, percentili)
- Range di possibili traiettorie del portafoglio

La simulazione è pensata come strumento di supporto alle decisioni di investimento, per valutare la robustezza e la sostenibilità del portafoglio nel tempo, in condizioni di mercato normali.
Per una valutazione ancora più realistica, si consiglia di integrare ribilanciamenti periodici, costi, tasse, scenari di crisi e volatilità variabile.
L'obbiettivo ultimo è utilizzare questo codice per prendere deciioni d'investimento reali, con l'obbiettivo di massimizzare i profitti.
"""

import dataManager
import monteCarlo
import plt

tickers = ["SPY", "AGG", "EFA"]
pesi = [0.5, 0.3, 0.2]
nSims = 20000

portafoglio = dataManager.Portafoglio(tickers, pesi)
portafoglio.calcolaStatsStoriche()

sim = monteCarlo.MonteCarlo(portafoglio, nSims=nSims)

results = sim.runSimulation(verbose=1)

plot = plt.Plotter()

plot.setup()
plot.plot(results)
