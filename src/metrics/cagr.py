import numpy as np

from src.constants import TRADING_DAYS_PER_YEAR
from src.metrics.metric import Metric


class CAGR(Metric):
    """Compound Annual Growth Rate (CAGR) performance metric."""

    TO_MAXIMIZE = True

    def _calculate(self, portfolio: np.ndarray, tensors: dict) -> np.ndarray:
        p = np.atleast_2d(portfolio)
        returns_boot = tensors["returns"]
        returns_clipped = np.clip(returns_boot, -0.999999, None)

        # Compounded capitalization over bootstrap trajectories
        asset_growth = np.exp(np.sum(np.log1p(returns_clipped), axis=2))
        final_wealth = np.einsum("pi,bi->pb", p, asset_growth)

        exponent = TRADING_DAYS_PER_YEAR / returns_boot.shape[2]
        cagr = np.where(final_wealth > 0.0, final_wealth**exponent - 1.0, -1.0)
        cagr = np.where(np.isnan(cagr), 0.0, cagr)

        return cagr
