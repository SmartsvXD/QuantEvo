import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def calculate_portfolio_returns(p: np.ndarray, returns_boot: np.ndarray) -> np.ndarray:
    """Simulates portfolio wealth evolution across bootstrap return trajectories."""
    n_p, n_a = p.shape
    n_b, _, n_t = returns_boot.shape
    out = np.empty((n_p, n_b, n_t), dtype=np.float64)

    for i in prange(n_p):
        for b in range(n_b):
            # Track buy-and-hold portfolio wealth trajectory across time steps
            v = p[i].copy()
            for t in range(n_t):
                w_prev = 0.0
                w_curr = 0.0
                for a in range(n_a):
                    v_a = v[a]
                    w_prev += v_a
                    r = returns_boot[b, a, t]
                    if not np.isnan(r) and v_a > 0.0:
                        v_a = max(v_a * (1.0 + r), 0.0)
                    v[a] = v_a
                    w_curr += v_a
                out[i, b, t] = (w_curr - w_prev) / w_prev if w_prev > 0.0 else 0.0

    return out
