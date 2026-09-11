import json

from src.valutazione.omega import OmegaIndice
from src.valutazione.rischio import RischioTotaleCorrettoIndice
from src.valutazione.diversificazione import DiversificazioneIndice
from src.utils.generaPortafogliValidi import generaPortafogliValidi
from src.genetica import run
from src.valutazione import ValutatoreMultiplo
from src.managerTitoli import StatsTitoli
from src.const import PERCENTUALE_MINIMA_TITOLO

if __name__ == "__main__":
    with open("titoli.json", "r") as file:
        data = json.load(file)

    titoli = [x[0] for x in data]
    upper = [x[1] for x in data]

    s = StatsTitoli(titoli, forzaDownload=0, verbose=1)
    s.calcolaStats(verbose=1)

    omega = OmegaIndice(faiBootstrap=1)
    rischio = RischioTotaleCorrettoIndice()
    div = DiversificazioneIndice()

    omega.preparaListaBootstrap(s)

    v = ValutatoreMultiplo((div, omega, rischio), (False, True, False))

    portafogli = generaPortafogliValidi(
        titoli, n=1000, percentualeMinima=PERCENTUALE_MINIMA_TITOLO, upper=upper, minTitoli=3
    )
    v.fitStandardizza(portafogli, s)

    port, ris = run(s, v, upper=upper, seeds=None)

    print("\n")
    for i in range(len(titoli)):
        print(f"{titoli[i]}: {port[0][i]:.1%}")

    print(f"\n{v.calcola(port[0], s)}")
