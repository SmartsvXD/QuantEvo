import json

from src.const import PERCENTUALE_MINIMA_TITOLO
from src.genetica import run
from src.managerTitoli import StatsTitoli
from src.utils.repair import PortfolioConstraints
from src.valutazione import ValutatoreMultiplo
from src.valutazione.cagr import CAGRIndice
from src.valutazione.enb import ENBIndice
from src.valutazione.enc import ENCIndice
from src.valutazione.maxDrawdown import MaxDrawdownIndice
from src.valutazione.omega import OmegaIndice
from src.valutazione.tailCVar import TailCVarIndice

if __name__ == "__main__":
    with open("titoli.json", "r") as file:
        data = json.load(file)

    titoli = [x[0] for x in data]
    # Al momento non utilizziamo upper, ma lo lasciamo per eventuali sviluppi futuri
    # Voglio verificare come si comporta l'algoritmo senza un vincolo di upper, quindi lo imposto a None per ora
    upper = [x[1] for x in data]

    s = StatsTitoli(titoli, forzaDownload=0, verbose=1)
    s.preparaBootstrap()

    constraints = PortfolioConstraints(percentualeMinima=PERCENTUALE_MINIMA_TITOLO, upper=upper, minTitoli=3)

    cagr = CAGRIndice(daMassimizzare=True)
    maxdrawdown = MaxDrawdownIndice()
    cvar = TailCVarIndice()

    v = ValutatoreMultiplo((cagr, maxdrawdown, cvar))
    vincoli = ValutatoreMultiplo(
        (
            OmegaIndice(daMassimizzare=True, soglia=1.25, sogliaAnnua=0.035, metricaRischio="percentile", alpha=0.5),
            ENCIndice(daMassimizzare=True, soglia=0.7, metricaRischio="percentile", alpha=0.5),
            ENBIndice(daMassimizzare=True, soglia=1.6, metricaRischio="percentile", alpha=0.5),
        )
    )

    port, ris = run(s, v, constraints, vincoli=vincoli, seeds=None)

    if len(port) == 0:
        print("Nessun portafoglio valido trovato.")
    else:
        print("\n")
        for i in range(len(titoli)):
            print(f"{titoli[i]}: {port[i]:.1%}")

        print()
        for i in v.indici:
            print(f"{i.__class__.__name__}: {i.calcola(port, s, valoreOriginale=True, faiBootstrap=False):.4f}")

        print()
        for i in vincoli.indici:
            print(f"{i.__class__.__name__}: {i.calcola(port, s, valoreOriginale=True, faiBootstrap=False):.4f}")
