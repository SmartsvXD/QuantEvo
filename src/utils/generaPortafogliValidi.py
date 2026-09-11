import numpy as np

from src.utils.vincoli import applicaVincoli
from src.const import PERCENTUALE_MINIMA_TITOLO


def generaPortafogliValidi(titoli, n=100, percentualeMinima=PERCENTUALE_MINIMA_TITOLO, upper=None, minTitoli=3):
    portafogli = np.random.dirichlet(np.ones(len(titoli)), size=n)

    portafogli = np.unique(portafogli, axis=0)

    upper=np.asarray(upper, dtype=np.float64) if upper is not None else None
    portafogli = applicaVincoli(portafogli, percentualeMinima=percentualeMinima, upper=upper, minTitoli=minTitoli)

    portafogli = np.unique(portafogli, axis=0)

    return portafogli
