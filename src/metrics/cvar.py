# SPDX-FileCopyrightText: 2026 Simone Vinciguerra
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from numba import njit, prange

from src.metrics.metric import Metric
from src.metrics.utils import calculate_portfolio_returns


@njit(fastmath=True, parallel=True)
def tail_cvar_risk(R: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Computes the Conditional Value-at-Risk (Expected Shortfall) at confidence alpha."""
    n_p, n_t = R.shape
    out = np.empty(n_p, dtype=np.float64)
    k_tail = max(1, int(n_t * alpha))

    for i in prange(n_p):
        # Average returns in the worst alpha-quantile tail (Expected Shortfall)
        part = np.partition(R[i], k_tail)
        out[i] = np.mean(part[:k_tail])

    return out


class CVaR(Metric):
    """Conditional Value-at-Risk (Expected Shortfall) portfolio risk metric."""

    def __init__(self, alpha: float = 0.05, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("Parameter alpha must be between 0 and 1 (exclusive).")

        self.alpha = alpha

    def _calculate(
        self, portfolio: np.ndarray, tensors: dict, port_returns: np.ndarray | None = None
    ) -> np.ndarray:
        p = np.ascontiguousarray(np.atleast_2d(portfolio), dtype=np.float64)
        if port_returns is None:
            port_returns = calculate_portfolio_returns(p, tensors["returns"])
        nP, nB, nT = port_returns.shape
        flat_cvar = tail_cvar_risk(port_returns.reshape(nP * nB, nT), alpha=self.alpha)
        # Invert sign to represent tail risk as a positive loss
        return -flat_cvar.reshape(nP, nB)
