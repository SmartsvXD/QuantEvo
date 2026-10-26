# SPDX-FileCopyrightText: 2026 Simone Vinciguerra
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from numba import njit, prange

from src.constants import TRADING_DAYS_PER_YEAR
from src.metrics.metric import Metric


@njit(parallel=True, fastmath=True)
def _omega_numba(portfolio: np.ndarray, samples_3d: np.ndarray, threshold: float) -> np.ndarray:
    """Computes monthly Omega ratio of gains to losses exceeding a hurdle rate."""
    n_port, n_asset = portfolio.shape
    n_boot, _, n_samples = samples_3d.shape
    omega_vals = np.empty((n_port, n_boot), dtype=np.float64)

    for i in prange(n_port):
        asset_values = np.empty(n_asset, dtype=np.float64)
        for b in range(n_boot):
            v_start_month = 0.0
            for a in range(n_asset):
                val = portfolio[i, a]
                asset_values[a] = val
                v_start_month += val

            up = 0.0
            down = 0.0
            day_counter = 0

            for t in range(n_samples):
                v_curr = 0.0
                for a in range(n_asset):
                    asset_values[a] *= 1.0 + samples_3d[b, a, t]
                    v_curr += asset_values[a]

                day_counter += 1
                if day_counter == TRADING_DAYS_PER_YEAR // 12:
                    # Calculate monthly return and upside/downside relative to hurdle rate
                    month_ret = (v_curr - v_start_month) / v_start_month if abs(v_start_month) > 1e-12 else 0.0
                    v_start_month = v_curr
                    day_counter = 0

                    diff = month_ret - threshold
                    if diff > 0.0:
                        up += diff
                    else:
                        down -= diff

            # Omega ratio of upside to downside with numerical bounds
            if down < 1e-12:
                omega_vals[i, b] = 10.0 if up > 0.0 else 1.0
            else:
                val = up / down
                omega_vals[i, b] = min(val, 10.0)

    return omega_vals


class Omega(Metric):
    """Omega ratio performance metric evaluating returns against a benchmark hurdle."""

    TO_MAXIMIZE = True

    def __init__(self, annual_hurdle_rate: float = 0.035, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if annual_hurdle_rate <= -1.0:
            raise ValueError("annual_hurdle_rate must be strictly greater than -1.0 (-100%).")

        self._monthly_hurdle_rate = (1.0 + annual_hurdle_rate) ** (1.0 / 12) - 1.0

    def _calculate(self, portfolio: np.ndarray, stats: dict) -> np.ndarray:
        p = np.atleast_2d(portfolio)
        return _omega_numba(p, stats["returns"], self._monthly_hurdle_rate)
