from numba import njit
import numpy as np


@njit
def mediaEWMA(data: np.ndarray, lambdaEWMA: float) -> np.ndarray:
    """
    Accetta solo array 2D e restituisce un array 1D con la media EWMA per
    ogni riga (sottarray). I NaN vengono ignorati.

    Implementazione: calcola i pesi EWMA tramite numpy e poi per ogni riga
    applica i pesi ignorando i NaN (loop sulle righe per compatibilità con
    numba `@njit`).
    """

    nrows = data.shape[0]
    ncols = data.shape[1]

    # pesi EWMA: (1 - λ) * λ^(ncols-1), ..., (1 - λ) * λ^0
    pesi = (1.0 - lambdaEWMA) * np.power(lambdaEWMA, np.arange(ncols - 1, -1, -1))

    out = np.empty(nrows, dtype=np.float64)
    for i in range(nrows):
        somma = 0.0
        weighted_sum = 0.0
        for j in range(ncols):
            x = data[i, j]
            if not np.isnan(x):
                w = pesi[j]
                somma += w
                weighted_sum += w * x
        if somma == 0.0:
            out[i] = 0.0
        else:
            out[i] = weighted_sum / somma

    return out


@njit
def stdEWMA(data: np.ndarray, lambdaEWMA: float) -> np.ndarray:
    """
    Accetta solo array 2D e restituisce un array 1D con la deviazione
    standard ponderata EWMA per ogni riga. I NaN vengono ignorati.
    """
    if data.ndim != 2:
        return np.empty(0, dtype=np.float64)

    nrows = data.shape[0]
    ncols = data.shape[1]

    # Pesi EWMA: (1 - λ) * λ^(ncols-1), ..., (1 - λ) * λ^0
    pesi = (1.0 - lambdaEWMA) * np.power(lambdaEWMA, np.arange(ncols - 1, -1, -1))

    out = np.empty(nrows, dtype=np.float64)
    for i in range(nrows):
        # somma dei pesi validi
        somma = 0.0
        for j in range(ncols):
            if not np.isnan(data[i, j]):
                somma += pesi[j]

        if somma == 0.0:
            out[i] = 0.0
            continue

        # media pesata normalizzata
        mu = 0.0
        for j in range(ncols):
            x = data[i, j]
            if not np.isnan(x):
                w = pesi[j] / somma
                mu += w * x

        # varianza pesata
        var = 0.0
        for j in range(ncols):
            x = data[i, j]
            if not np.isnan(x):
                w = pesi[j] / somma
                diff = x - mu
                var += w * diff * diff

        out[i] = np.sqrt(var)

    return out
