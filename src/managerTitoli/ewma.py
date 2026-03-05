from numba import njit
import numpy as np


@njit
def mediaEWMA(data: np.ndarray, lambdaEWMA: np.ndarray) -> float:
    """
    Calcola la media ponderata esponenziale dei log-returns, gestendo NaN.
    Rappresenta il drift giornaliero (media del log-return).

    :param data: dati su cui applicare la media
    :type data: `np.ndarray`
    :param lambdaEWMA: i coefficenti
    :type lambdaEWMA: `np.ndarray`
    :return: media EWMA
    :rtype: `float`
    """

    n = len(data)

    # Pesi EWMA
    exps = np.arange(n - 1, -1, -1)
    pesi = np.power(lambdaEWMA, exps)

    # Escludi i NaN dai calcoli
    valid = ~np.isnan(data)
    pesi_valid = pesi * valid
    somma = np.sum(pesi_valid)

    if somma == 0.0:
        return 0.0

    pesi_valid /= somma
    clean_data = np.where(valid, data, 0.0)
    media = np.sum(pesi_valid * clean_data)

    return media


@njit
def varianzaEWMA(data: np.ndarray, lambdaEWMA: np.ndarray) -> float:
    """
    Calcola la varianza ponderata esponenziale, gestendo NaN.
    Usa il metodo della varianza con pesi EWMA per robustezza.

    :param data: dati su cui applicare la media
    :type data: `np.ndarray`
    :param lambdaEWMA: i coefficenti
    :type lambdaEWMA: `np.ndarray`
    :return: varianza ottenuta tramite lambda EWMA
    :rtype: `float`
    """

    n = len(data)

    pesi = np.power(lambdaEWMA, np.arange(n - 1, -1, -1))

    valid = ~np.isnan(data)
    pesi_valid = pesi * valid
    somma = np.sum(pesi_valid)

    if somma == 0.0:
        return 0.0

    pesi_valid /= somma
    clean_data = np.where(valid, data, 0.0)

    media = np.sum(pesi_valid * clean_data)

    diff = clean_data - media
    varianza = np.sum(pesi_valid * diff * diff)

    return varianza
