import numpy as np


def generaPortafogliValidi(titoli, n=100, vincoli=None):
    portafogli = np.random.dirichlet(np.ones(len(titoli)), size=n)

    portafogli = vincoli(portafogli)

    portafogli = np.round(portafogli, 2)

    portafogli = np.unique(portafogli, axis=0)

    return portafogli
