import numpy as np
from numba import njit

@njit(fastmath=True)
def trovaOutlierMAD(returns: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    Rileva outliers usando Median Absolute Deviation (MAD), più robusto della std.

    :param returns: Array 2D dei rendimenti (n_assets, n_days)
    :param threshold: Numero di MAD oltre il quale un valore è considerato outlier
    :return: Array booleano con True per gli outliers
    """
    n_assets, n_days = returns.shape
    outliers = np.zeros((n_assets, n_days), dtype=np.bool_)

    for i in range(n_assets):
        valid_mask = ~np.isnan(returns[i])
        valid = returns[i][valid_mask]

        if len(valid) < 10:
            continue

        median = np.median(valid)
        abs_dev = np.abs(valid - median)
        mad = np.median(abs_dev)

        if mad < 1e-10: 
            continue

        for j in range(n_days):
            if valid_mask[j]:
                if np.abs(returns[i, j] - median) > threshold * mad * 1.4826:  # 1.4826 per consistenza con std
                    outliers[i, j] = True

    return outliers


@njit(fastmath=True)
def preprocessaRendimenti(returns: np.ndarray, threshold=5) -> np.ndarray:
    outliers = trovaOutlierMAD(returns, threshold)
    clipped = returns.copy()

    for i in range(returns.shape[0]):
        valid = returns[i][~np.isnan(returns[i])]
        if len(valid) < 10:
            continue
        median = np.median(valid)
        abs_dev = np.abs(valid - median)
        mad = np.median(abs_dev)
        if mad < 1e-10:
            continue
        bound = threshold * mad * 1.4826

        for j in range(returns.shape[1]):
            if outliers[i, j]:
                if returns[i, j] > median:
                    clipped[i, j] = median + bound
                else:
                    clipped[i, j] = median - bound

    return clipped
