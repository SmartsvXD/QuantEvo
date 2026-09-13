import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def preprocessaRendimenti(returns: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    n_assets, n_days = returns.shape
    clipped = returns.copy()

    for i in prange(n_assets):
        row = returns[i]
        valid_mask = ~np.isnan(row)
        valid = row[valid_mask]

        if len(valid) < 10:
            continue

        median = np.median(valid)
        mad = np.median(np.abs(valid - median))

        if mad < 1e-10:
            continue

        bound = threshold * mad * 1.4826

        # Vincolo a -1.0 per il rendimento semplice (perdita massima 100%)
        lower_bound = max(-1.0, median - bound)
        upper_bound = median + bound

        for j in range(n_days):
            if valid_mask[j]:
                val = row[j]
                if val > upper_bound:
                    clipped[i, j] = upper_bound
                elif val < lower_bound:
                    clipped[i, j] = lower_bound

    return clipped
