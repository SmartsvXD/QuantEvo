import matplotlib.pyplot as plt
import numpy as np
from src.simulazione.valutazione import andamentiPortafoglio
from src.simulazione.titoli import RisultatiSimulazione
from src.const import GIORNI_IN_UN_ANNO


def plot(portafoglio: dict, risultatiSim: RisultatiSimulazione, figsize=(12, 6)) -> None:
    """
    Data un portafoglio, permette di visualizzare l'andamento delle varie simulazioni del portafoglio nel tempo,
    la mediana, il 10ecimo e 90esimo percentile.

    :param portafoglio: Titoli e pesi del portafoglio
    :type portafoglio: `dict`
    :param risultatiSim: Risultati della simulazione
    :type RisultatiSimulazione:
    :param figsize: La dimensione della finestra
    """

    valoriPortafoglio = andamentiPortafoglio(portafoglio, risultatiSim)

    plt.figure(figsize=figsize)
    plt.xlabel("Anni")

    # plot di alcune traiettorie a caso (per leggibilità)
    for i in np.random.choice(risultatiSim.nSims, size=50, replace=False):
        plt.plot(
            valoriPortafoglio[i],
            color="lightgray",
            alpha=0.6,
            linewidth=0.8,
        )

    # calcola percentili nel tempo
    p10_t = np.percentile(valoriPortafoglio, 10, axis=0)
    p50_t = np.percentile(valoriPortafoglio, 50, axis=0)
    p90_t = np.percentile(valoriPortafoglio, 90, axis=0)

    plt.plot(p50_t, color="red", linewidth=2, label="Mediana")
    plt.plot(p10_t, color="orange", linestyle="--", label="10° percentile")
    plt.plot(p90_t, color="green", linestyle="--", label="90° percentile")
    plt.plot(
        np.full(GIORNI_IN_UN_ANNO * risultatiSim.anni, risultatiSim.S0), color="blue", linewidth=2, label="Investimento"
    )

    plt.xticks(
        np.arange(0, risultatiSim.anni * GIORNI_IN_UN_ANNO + 1, GIORNI_IN_UN_ANNO),
        np.arange(0, risultatiSim.anni + 1),
    )
    plt.ylabel(f"Valore del portafoglio (S0={risultatiSim.S0})")
    plt.title(f"Simulazioni GBM del portafoglio ({risultatiSim.anni} anni)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
