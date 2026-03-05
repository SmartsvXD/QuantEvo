from numba import njit
import numpy as np
from src.const import PESI_VALUTAZIONE


@njit(fastmath=True)
def applicaVincoli(pesi, percentualeMinima=0.05):
    """
    Applica i vincoli ad un vettore di percentuali.

    Vincoli:
    - somma = 1
    - pesi >= 0
    - pesi = 0 oppure >= p_min
    - lower <= pesi <= upper

    Parameters
    ----------
    pesi : np.ndarray
        vettore dei pesi
    lower : np.ndarray | None
        limiti inferiori
    upper : np.ndarray | None
        limiti superiori
    p_min : float
        percentuale minima se il peso è non nullo

    Returns
    -------
    np.ndarray
    """
    n = pesi.size

    #! TEMPORANEO
    upper = np.ones(n) * 0.3
    #!

    # 1 clamp tra 0 e max (NON imponiamo minimo qui, altrimenti tutti vengono portati al minimo)
    pesi = np.clip(pesi, 0.0, upper)

    # 2 rimuove pesi sotto soglia minima (0 oppure >= percentualeMinima)
    mask = (pesi > 0) & (pesi < percentualeMinima)
    pesi[mask] = 0.0

    # 3 normalizzazione
    s = pesi.sum()

    if s == 0:
        # distribuzione uniforme valida
        pesi[:] = 1.0 / n
        return pesi

    pesi /= s

    # 4 fix limiti dopo normalizzazione (upper + riapplica minimo)
    over = pesi > upper
    under = (pesi > 0) & (pesi < percentualeMinima)

    if np.any(over) or np.any(under):
        # Rimuovi pesi sotto soglia
        pesi[under] = 0.0
        # Clippa quelli sopra max
        pesi = np.clip(pesi, 0.0, upper)

        # Rinormalizza
        residuo = 1.0 - pesi.sum()

        if abs(residuo) > 1e-12:
            # Distribuisci il residuo sui pesi liberi (non a zero, non al max)
            liberi = (pesi > percentualeMinima) & (pesi < upper)

            if np.any(liberi):
                pesi[liberi] += residuo / liberi.sum()
            else:
                # Se non ci sono liberi, distribuisci uniformemente su tutti i non-zero
                non_zero = pesi > 0
                if np.any(non_zero):
                    pesi[non_zero] += residuo / non_zero.sum()

    return pesi


# Funzione per ordinare valutazioni e pesi rispetto alla media delle valutazioni
def ordina(F, X):
    """
    Ordina F (valutazioni) e X (pesi) rispetto alla media delle valutazioni.
    Ritorna: F_ordinato, X_ordinato
    """
    F = np.array(F)
    X = np.array(X)
    w = np.array(PESI_VALUTAZIONE)
    indici = np.argsort(np.sum(F * w, axis=1))
    return F[indici], X[indici]
