import numpy as np

from src.utils.vincoli import Vincoli


def generaPortafogliValidi(nTitoli, vincoli: Vincoli, n=10000):
    portafogli = np.random.dirichlet(np.ones(nTitoli), size=n)

    portafogli = np.unique(portafogli, axis=0)

    portafogli = vincoli.applica(portafogli)

    portafogli = np.unique(portafogli, axis=0)

    return portafogli
