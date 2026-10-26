# SPDX-FileCopyrightText: 2026 Simone Vinciguerra
# SPDX-License-Identifier: GPL-3.0-or-later

import time

import numpy as np
from colorama import Fore

from src.constants import DEBUG_METRICS
from src.data.statistics import SampleStatistics


class Metric:
    """
    Abstract base class for portfolio performance and risk metrics.

    Handles bootstrap evaluation, aggregation across resampled trajectories,
    risk adjustments (CVaR, percentile, downside penalty), and sign inversion
    for genetic algorithm minimization.
    """

    CAN_BOOTSTRAP = True
    TO_MAXIMIZE = False

    def __init__(
        self,
        risk_measure: str = "cvar",
        threshold: float = 0.0,
        alpha: float = 0.05,
        stability_penalty_lambda: float = 0.4,
        debug: bool = DEBUG_METRICS,
    ) -> None:
        self.risk_measure = risk_measure
        self.alpha = alpha
        self.stability_penalty_lambda = stability_penalty_lambda
        self.threshold = threshold
        self.debug = debug

    def _aggregate_values(self, vals: np.ndarray) -> np.ndarray:
        if self.risk_measure == "cvar":
            if self.TO_MAXIMIZE:
                q = np.quantile(vals, self.alpha, axis=1, keepdims=True)
                mask = vals <= q
            else:
                q = np.quantile(vals, 1.0 - self.alpha, axis=1, keepdims=True)
                mask = vals >= q

            counts = np.maximum(1, np.sum(mask, axis=1))
            return np.sum(np.where(mask, vals, 0.0), axis=1) / counts

        elif self.risk_measure == "percentile":
            q_target = self.alpha if self.TO_MAXIMIZE else (1.0 - self.alpha)
            return np.quantile(vals, q_target, axis=1)

        elif self.risk_measure == "downside_penalty":
            mean = np.mean(vals, axis=1)
            negative_diff = np.minimum(0.0, vals - mean[:, None])
            semi_dev = np.sqrt(np.mean(negative_diff**2, axis=1))
            return mean - self.stability_penalty_lambda * semi_dev

        return np.mean(vals, axis=1)

    def _calculate(self, portfolio: np.ndarray, tensors: dict, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Method _calculate() not implemented in derived class.")

    def calculate(
        self,
        portfolio: np.ndarray,
        stats: SampleStatistics,
        do_bootstrap: bool = True,
        raw_scale: bool = False,
        debug: bool = False,
        *args,
        **kwargs,
    ) -> np.ndarray | float:
        """
        Computes metric values for given portfolio allocations.

        :param portfolio: 1D or 2D array of portfolio weights (shape (N,) or (P, N)).
        :param stats: SampleStatistics container providing historical or bootstrap data.
        :param do_bootstrap: Whether to calculate metric over bootstrap distributions.
        :param raw_scale: If True, returns natural economic values; if False, returns
                          standardized/sign-inverted values for minimization.
        :param debug: If True, prints execution time and metric summary.
        :return: Metric values as a scalar (for 1D portfolio) or 1D array (for 2D portfolios).
        """
        do_bootstrap = do_bootstrap and self.CAN_BOOTSTRAP
        debug = debug or self.debug

        if debug:
            print(f"\n{Fore.MAGENTA}Calculating metric {self.__class__.__name__}...{Fore.RESET}")
            t0 = time.time()

        p = np.atleast_2d(portfolio)
        tensors = stats.get(bootstrap=do_bootstrap)

        # Unified call: always returns shape (P, B)
        vals = self._calculate(p, tensors, *args, **kwargs)

        if do_bootstrap:
            data = self._aggregate_values(vals)
        else:
            data = vals[:, 0]  # Extracts dummy dimension B=1 -> (P,)

        if portfolio.ndim == 1:
            data = data[0]

        if debug:
            print(f"{Fore.MAGENTA}Done!{Fore.RESET} ({time.time() - t0:.2f}s) Mean value: {np.mean(data):.6f}")

        if not raw_scale:
            if self.threshold != 0.0:
                data = (data - self.threshold) / abs(self.threshold)
            if self.TO_MAXIMIZE:
                data = -data

        return data
