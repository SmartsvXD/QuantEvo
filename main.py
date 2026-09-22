import json

from src.const import PERCENTUALE_MINIMA_TITOLO
from src.genetica import run
from src.managerTitoli import StatsTitoli
from src.utils.vincoli import Vincoli
from src.valutazione import ValutatoreMultiplo
from src.valutazione.cagr import CAGRIndice
from src.valutazione.diversificazione import DiversificazioneIndice
from src.valutazione.omega import CAGROmegaIndice, OmegaIndice
from src.valutazione.rischio import RischioTotaleCorrettoIndice, RischioTailCVarIndice, RischioMaxDrawdownIndice, RischioTotaleIndice

if __name__ == "__main__":
    with open("titoli.json", "r") as file:
        data = json.load(file)

    titoli = [x[0] for x in data]
    # Al momento non utilizziamo upper, ma lo lasciamo per eventuali sviluppi futuri
    # Voglio verificare come si comporta l'algoritmo senza un vincolo di upper, quindi lo imposto a None per ora
    upper = [x[1] for x in data]

    s = StatsTitoli(titoli, forzaDownload=0, verbose=1)
    s.preparaBootstrap()

    vincoli = Vincoli(percentualeMinima=PERCENTUALE_MINIMA_TITOLO, upper=upper, minTitoli=3)

    omega = CAGRIndice(inverti=True)
    rischio = RischioMaxDrawdownIndice()

    v = ValutatoreMultiplo((omega, rischio))

    port, ris = run(s, v, vincoli, seeds=None)

    print("\n")
    for i in range(len(titoli)):
        print(f"{titoli[i]}: {port[i]:.1%}")

    print(f"\n{v.calcola(port[0], s)}")
