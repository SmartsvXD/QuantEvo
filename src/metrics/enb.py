# SPDX-FileCopyrightText: 2026 Simone Vinciguerra
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from numba import njit

from src.metrics.metric import Metric


@njit(fastmath=True)
def _calculate_enb_kernel(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Computes Effective Number of Bets (ENB) via principal components decomposition."""
    n_p, n_assets = weights.shape
    n_b = cov.shape[0]
    res = np.zeros((n_p, n_b), dtype=np.float64)

    # Reusable buffer for risk contributions of bets
    c = np.zeros(n_assets, dtype=np.float64)

    for b in range(n_b):
        # Eigenvalues (vals) and orthonormal eigenvectors (vecs)
        vals, vecs = np.linalg.eigh(cov[b])

        # Clamping to avoid small negative eigenvalues due to numerical errors
        for i in range(n_assets):
            vals[i] = max(vals[i], 0.0)

        for p in range(n_p):
            total_var = 0.0

            # Portfolio projection on orthogonal components (bets)
            for k in range(n_assets):
                proj_k = 0.0
                for i in range(n_assets):
                    w_i = weights[p, i]
                    if w_i != 0.0:
                        proj_k += w_i * vecs[i, k]

                # Variance of k-th bet
                var_k = vals[k] * proj_k * proj_k
                c[k] = var_k
                total_var += var_k

            # Exponential Shannon entropy of percentage contributions p_k
            if total_var > 1e-12:
                entropy = 0.0
                for k in range(n_assets):
                    p_k = c[k] / total_var
                    if p_k > 1e-12:
                        entropy -= p_k * np.log(p_k)
                res[p, b] = np.exp(entropy)
            else:
                res[p, b] = 1.0

    return res


class ENB(Metric):
    """Effective Number of Bets (ENB) diversification metric based on Shannon entropy."""

    TO_MAXIMIZE = True

    def _calculate(self, portfolio: np.ndarray, stats: dict) -> np.ndarray:
        p = np.ascontiguousarray(np.atleast_2d(portfolio), dtype=np.float64)
        cov = np.asarray(stats["cov"], dtype=np.float64)

        if cov.ndim == 2:
            cov = cov[np.newaxis, :, :]

        cov = np.ascontiguousarray(cov)
        return _calculate_enb_kernel(weights=p, cov=cov)
