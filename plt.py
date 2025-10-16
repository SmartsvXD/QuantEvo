import matplotlib.pyplot as plt
import numpy as np
from monteCarlo import MonteCarloResults
from settings import GIORNI_IN_UN_ANNO


class Plotter:
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize

    def setup(self):
        plt.figure(figsize=self.figsize)
        plt.xlabel("Anni")

    def plot(self, results: MonteCarloResults):
        # plot di alcune traiettorie a caso (per leggibilità)
        for i in np.random.choice(results.nSims, size=50, replace=False):
            plt.plot(
                results.valoriDelPortafoglio[i],
                color="lightgray",
                alpha=0.6,
                linewidth=0.8,
            )

        # calcola percentili nel tempo
        p10_t = np.percentile(results.valoriDelPortafoglio, 10, axis=0)
        p50_t = np.percentile(results.valoriDelPortafoglio, 50, axis=0)
        p90_t = np.percentile(results.valoriDelPortafoglio, 90, axis=0)
        # t = np.arange(1, results.orizzonte + 1) / GIORNI_IN_UN_ANNO

        plt.plot(p50_t, color="red", linewidth=2, label="Mediana")
        plt.plot(p10_t, color="orange", linestyle="--", label="10° percentile")
        plt.plot(p90_t, color="green", linestyle="--", label="90° percentile")
        plt.plot(results.valoriDegliInvestimenti, color = "blue", linewidth = 2, label = "Soldi Investiti")
        
        plt.xticks(
            np.arange(0, results.orizzonte + 1, GIORNI_IN_UN_ANNO),
            np.arange(0, results.anni + 1),
        )
        plt.ylabel(f"Valore del portafoglio (S0={results.S0})")
        plt.title(f"Simulazioni GBM del portafoglio ({results.anni} anni)")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()
