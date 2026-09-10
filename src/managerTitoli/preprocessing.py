from numba import njit, prange
import numpy as np

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

        for j in range(n_days):
            val = row[j]
            if valid_mask[j]:
                if val > median + bound:
                    clipped[i, j] = median + bound
                elif val < median - bound:
                    clipped[i, j] = median - bound

    return clipped
