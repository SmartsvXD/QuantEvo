# SPDX-FileCopyrightText: 2026 Simone Vinciguerra
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from numba import njit

from src.metrics.metric import Metric


@njit(fastmath=True)
def _calculate_enc_kernel(weights: np.ndarray) -> np.ndarray:
    """Computes Effective Number of Constituents (ENC) as inverse Herfindahl index."""
    n_p, n_assets = weights.shape
    res = np.zeros((n_p, 1), dtype=np.float64)

    for i in range(n_p):
        hhi = 0.0
        for j in range(n_assets):
            w = weights[i, j]
            hhi += w * w

        # Classic ENC: inverse of HHI
        if hhi > 1e-12:
            res[i, 0] = 1.0 / hhi
        else:
            res[i, 0] = 0.0

    return res


class ENC(Metric):
    """Effective Number of Constituents (ENC) concentration metric."""

    CAN_BOOTSTRAP = False
    TO_MAXIMIZE = True

    def _calculate(self, portfolio: np.ndarray, stats: dict) -> np.ndarray:
        p = np.ascontiguousarray(np.atleast_2d(portfolio), dtype=np.float64)
        return _calculate_enc_kernel(weights=p)
