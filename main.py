"""
Progetto MonteCarlo: Simulazione quantitativa di portafoglio ETF

Questo progetto Python implementa una simulazione Monte Carlo avanzata per analizzare l'evoluzione futura di un portafoglio d'investimento reale, composto da ETF liquidi (es. SPY, AGG, EFA) con pesi prefissati.
Utilizza dati storici per stimare media e volatilità dei rendimenti, applicando il modello di moto browniano geometrico (GBM) per simulare l'andamento del portafoglio su un orizzonte di 10 anni.

Funzionalità principali:
- Download automatico dei dati storici degli ETF tramite Yahoo Finance
- Calcolo di statistiche storiche (media, volatilità, drift GBM)
- Simulazione Monte Carlo vettorializzata con investimenti mensili e ribilanciamento periodico
- Analisi quantitativa dei risultati: media, mediana, percentili, probabilità di perdita
- Visualizzazione grafica delle traiettorie simulate e dei percentili

Obiettivi e utilizzo:
- Supporto alle decisioni di investimento reale, valutando robustezza e sostenibilità del portafoglio
- Stima quantitativa del rischio e del rendimento atteso
- Possibilità di estensione per includere costi, tasse, scenari di crisi, volatilità variabile e ribilanciamenti più sofisticati

Il progetto è pensato come strumento pratico per investitori e analisti quantitativi che vogliono massimizzare i profitti e minimizzare i rischi, basandosi su dati storici e modelli matematici robusti.
L'obbiettivo ultimo è utilizzare questo codice per prendere deciioni d'investimento reali, con l'obbiettivo di massimizzare i profitti.
"""

# Ricordati di verificare se un ETF è a distribuzione o ad accumulazione


import dataManager
import monteCarlo
import plt

tickers = ["SPY", "AGG", "EFA"]
pesi = [0.5, 0.3, 0.2]
nSims = 10000

portafoglio = dataManager.Portafoglio(tickers, pesi)
portafoglio.calcolaStatsStoriche()

sim = monteCarlo.MonteCarlo(portafoglio, nSims=nSims)
results = sim.runSimulation(verbose=1)

plot = plt.Plotter()
plot.setup()
plot.plot(results)
