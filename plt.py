import matplotlib.pyplot as plt
import numpy as np
from monteCarlo import Pagella
from settings import GIORNI_IN_UN_ANNO



def plot( pagella: Pagella, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.xlabel("Anni")
    
    # plot di alcune traiettorie a caso (per leggibilità)
    for i in np.random.choice(pagella.nSims, size=50, replace=False):
        plt.plot(
            pagella.valoriPortafoglio[i],
            color="lightgray",
            alpha=0.6,
            linewidth=0.8,
        )

    # calcola percentili nel tempo
    p10_t = np.percentile(pagella.valoriPortafoglio, 10, axis=0)
    p50_t = np.percentile(pagella.valoriPortafoglio, 50, axis=0)
    p90_t = np.percentile(pagella.valoriPortafoglio, 90, axis=0)
    # t = np.arange(1, pagella.orizzonte + 1) / GIORNI_IN_UN_ANNO

    plt.plot(p50_t, color="red", linewidth=2, label="Mediana")
    plt.plot(p10_t, color="orange", linestyle="--", label="10° percentile")
    plt.plot(p90_t, color="green", linestyle="--", label="90° percentile")
    plt.plot(pagella.valoriInvestimenti, color="blue", linewidth=2, label="Soldi Investiti")

    plt.xticks(
        np.arange(0, pagella.anni * GIORNI_IN_UN_ANNO + 1, GIORNI_IN_UN_ANNO),
        np.arange(0, pagella.anni + 1),
    )
    plt.ylabel(f"Valore del portafoglio (S0={pagella.S0})")
    plt.title(f"Simulazioni GBM del portafoglio ({pagella.anni} anni)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
