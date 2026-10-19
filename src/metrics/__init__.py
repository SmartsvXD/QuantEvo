import numpy as np

from src.data.statistics import SampleStatistics


class MetricSuite:
    """
    Manages an ensemble of portfolio metrics evaluated jointly.

    Used to define multi-objective vector functions and multi-constraint functions
    for genetic portfolio optimization.
    """

    def __init__(self, metrics: tuple) -> None:
        self.metrics = metrics
        self.num_metrics = len(self.metrics)

    def evaluate(self, portfolio: np.ndarray, stats: SampleStatistics, raw_scale: bool = False) -> np.ndarray:
        """
        Evaluates all metrics in the suite on a population of portfolios.

        :param portfolio: 1D or 2D array of portfolio weights (shape (P, N)).
        :param stats: SampleStatistics object holding return distributions.
        :param raw_scale: If True, computes natural values; if False, applies optimization scale.
        :return: 2D array of shape (P, num_metrics) containing evaluated values.
        """
        portfolio = np.atleast_2d(portfolio)

        res = np.empty((portfolio.shape[0], len(self.metrics)), dtype=np.float64)
        for n, m in enumerate(self.metrics):
            vals = m.calculate(portfolio, stats, raw_scale=raw_scale)
            res[:, n] = vals

        return res
