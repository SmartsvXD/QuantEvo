import numpy as np
from numba import njit, prange

from src.metrics.metric import Metric
from src.metrics.utils import calculate_portfolio_returns


@njit(fastmath=True, parallel=True)
def max_drawdown(R: np.ndarray) -> np.ndarray:
    """Computes the maximum peak-to-trough decline over each return trajectory."""
    n_p, n_t = R.shape
    worst = np.zeros(n_p, dtype=np.float64)

    for i in prange(n_p):
        w = 1.0
        max_w = 1.0
        min_dd = 0.0

        # Track cumulative wealth compounding and peak-to-trough decline
        for t in range(n_t):
            r = max(R[i, t], -1.0)
            w *= 1.0 + r

            if w > max_w:
                max_w = w
            elif max_w > 0.0:
                dd = (w - max_w) / max_w
                min_dd = min(min_dd, dd)

            if w <= 0.0:
                min_dd = -1.0
                break

        worst[i] = min_dd

    return worst


class Drawdown(Metric):
    """Maximum drawdown risk metric measuring worst cumulative peak-to-trough drop."""
    def _calculate(
        self, portfolio: np.ndarray, tensors: dict, port_returns: np.ndarray | None = None
    ) -> np.ndarray:
        p = np.ascontiguousarray(np.atleast_2d(portfolio), dtype=np.float64)
        if port_returns is None:
            port_returns = calculate_portfolio_returns(p, tensors["returns"])
        nP, nB, nT = port_returns.shape
        flat_dd = max_drawdown(port_returns.reshape(nP * nB, nT))
        # Invert sign so maximum drawdown is represented as a positive loss
        return -flat_dd.reshape(nP, nB)
