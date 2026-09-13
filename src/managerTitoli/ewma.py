import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def mediaEWMA(data: np.ndarray, lambdaEWMA: float) -> np.ndarray:
    """Calcola la media ponderata EWMA per ogni riga su rendimenti lineari (vincolo >= -1.0)."""
    if data.ndim != 2:
        return np.empty(0, dtype=np.float64)

    nrows, ncols = data.shape
    pesi = (1.0 - lambdaEWMA) * np.power(lambdaEWMA, np.arange(ncols - 1, -1, -1))
    out = np.empty(nrows, dtype=np.float64)

    for i in prange(nrows):
        somma = 0.0
        weighted_sum = 0.0
        for j in range(ncols):
            x = data[i, j]
            if not np.isnan(x):
                x = max(x, -1.0)  # Protezione limite teorico -100%
                w = pesi[j]
                somma += w
                weighted_sum += w * x

        out[i] = (weighted_sum / somma) if somma > 0.0 else 0.0

    return out


@njit(fastmath=True, parallel=True)
def stdEWMA(data: np.ndarray, lambdaEWMA: float) -> np.ndarray:
    """Calcola la deviazione standard EWMA per ogni riga su rendimenti lineari (vincolo >= -1.0)."""
    if data.ndim != 2:
        return np.empty(0, dtype=np.float64)

    nrows, ncols = data.shape
    pesi = (1.0 - lambdaEWMA) * np.power(lambdaEWMA, np.arange(ncols - 1, -1, -1))
    out = np.empty(nrows, dtype=np.float64)

    for i in prange(nrows):
        somma = 0.0
        weighted_sum = 0.0
        # Passaggio 1: somma pesi e media
        for j in range(ncols):
            x = data[i, j]
            if not np.isnan(x):
                x = max(x, -1.0)
                w = pesi[j]
                somma += w
                weighted_sum += w * x

        if somma == 0.0:
            out[i] = 0.0
            continue

        mu = weighted_sum / somma

        # Passaggio 2: varianza pesata
        var_sum = 0.0
        for j in range(ncols):
            x = data[i, j]
            if not np.isnan(x):
                x = max(x, -1.0)
                diff = x - mu
                var_sum += pesi[j] * diff * diff

        out[i] = np.sqrt(var_sum / somma)

    return out
