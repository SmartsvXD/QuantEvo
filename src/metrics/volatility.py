import numpy as np
from numba import njit, prange

from src.constants import TRADING_DAYS_PER_YEAR
from src.metrics.metric import Metric
from src.metrics.utils import calculate_portfolio_returns


@njit(fastmath=True, parallel=True)
def portfolio_volatility(R: np.ndarray) -> np.ndarray:
    """Calculates annualized standard deviation of returns across simulated paths."""
    n_p, n_t = R.shape
    out = np.empty(n_p, dtype=np.float64)

    for i in prange(n_p):
        mean = 0.0
        for t in range(n_t):
            mean += R[i, t]
        mean /= n_t

        var = 0.0
        for t in range(n_t):
            diff = R[i, t] - mean
            var += diff * diff

        # Sample standard deviation annualized by square root of trading days
        out[i] = np.sqrt(var / (n_t - 1)) if n_t > 1 else 0.0

    return out * np.sqrt(TRADING_DAYS_PER_YEAR)


class Volatility(Metric):
    """Annualized volatility portfolio risk metric."""
    def _calculate(
        self, portfolio: np.ndarray, tensors: dict, port_returns: np.ndarray | None = None
    ) -> np.ndarray:
        p = np.ascontiguousarray(np.atleast_2d(portfolio), dtype=np.float64)
        if port_returns is None:
            port_returns = calculate_portfolio_returns(p, tensors["returns"])
        nP, nB, nT = port_returns.shape
        flat_vol = portfolio_volatility(port_returns.reshape(nP * nB, nT))
        return flat_vol.reshape(nP, nB)
