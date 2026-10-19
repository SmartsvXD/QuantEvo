import time

import numpy as np
from colorama import Fore
from sklearn.covariance import LedoitWolf

from src.constants import PROGRESS_BAR_TICKS, TRADING_DAYS_PER_YEAR
from src.data.download import download


class SampleStatistics:
    """
    Container for historical asset returns, covariance, and bootstrap samples.
    """

    def __init__(self, returns: np.ndarray) -> None:
        raw_stats = self.calculate_stats(returns)
        self.stats = {k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) else v for k, v in raw_stats.items()}

        self.stats_boot = {}

        self.n_assets = returns.shape[0]

    def calculate_stats(self, returns: np.ndarray) -> dict:
        """Computes returns matrix along with Ledoit-Wolf correlation and covariance."""
        stats = {}
        stats["returns"] = returns

        stats["corr"], stats["cov"] = self.calculate_correlation_and_covariance_matrix(returns)

        return stats

    def calculate_correlation_and_covariance_matrix(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, T = returns.shape
        data_mat = np.empty((T, n), dtype=np.float64)

        for j in range(n):
            col = returns[j]
            mask = ~np.isnan(col)
            if np.any(mask):
                # Floor at -100% for valid linear returns
                valid = np.maximum(col[mask], -1.0)
                mean = np.mean(valid)

                filled = np.empty_like(col)
                filled[mask] = valid
                filled[~mask] = mean
            else:
                filled = np.zeros_like(col)

            data_mat[:, j] = filled

        # Covariance shrinkage estimation
        lw = LedoitWolf().fit(data_mat)
        cov = lw.covariance_

        # Normalization to correlation matrix
        empirical_std = np.sqrt(np.diag(cov))
        empirical_std = np.maximum(empirical_std, 1e-10)

        corr_matrix = cov / (empirical_std[:, None] * empirical_std[None, :])

        # Symmetrization, numerical clipping, and unit diagonal
        corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)

        return corr_matrix, cov

    @staticmethod
    def _generate_block_indices(
        T: int, n_bootstrap: int, block_length: int, rng: np.random.Generator
    ) -> np.ndarray:
        block = max(2, min(int(block_length), T))
        n_blocks = int(np.ceil(T / block))
        max_start = max(1, T - block + 1)
        start = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
        offset = np.arange(block, dtype=np.int64)
        idx = (start[:, :, None] + offset[None, None, :]).reshape(n_bootstrap, -1)
        return idx[:, :T]

    def prepare_bootstrap(
        self,
        n_bootstrap: int = 100,
        block_length: int = 20,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Generates block-bootstrap scenarios preserving short-term autocorrelation.

        :param n_bootstrap: Number of bootstrap sample paths to generate.
        :param block_length: Length of contiguous blocks in days.
        :param random_state: Seed for reproducible random sampling.
        :param verbose: Whether to display a progress indicator.
        """
        if verbose:
            t0 = time.time()
            print(f"\n{Fore.MAGENTA}Preparing bootstrap dictionary...{Fore.RESET}")

        base_returns = self.stats["returns"][0]  # Remove B=1 axis to obtain (N, T)
        T = base_returns.shape[1]
        if T < 2 or n_bootstrap <= 0:
            return

        rng = np.random.default_rng(random_state)
        idxs = self._generate_block_indices(T, n_bootstrap, block_length, rng)

        # Pre-allocate (n_bootstrap, ...) preserving dimensions and dtype of each recorded metric
        self.stats_boot = {
            k: np.empty((n_bootstrap, *arr.shape[1:]), dtype=arr.dtype)
            for k, arr in self.stats.items()
            if isinstance(arr, np.ndarray)
        }

        for i in range(n_bootstrap):
            sample = np.ascontiguousarray(base_returns[:, idxs[i]])
            sample_stats = self.calculate_stats(sample)
            for k, val in sample_stats.items():
                self.stats_boot[k][i] = val

            if verbose and i % 10 == 0:
                filled = round(i / n_bootstrap * PROGRESS_BAR_TICKS)
                empty = PROGRESS_BAR_TICKS - filled
                print(f"\r{Fore.LIGHTBLACK_EX}{i}/{n_bootstrap} [{'*' * filled}{' ' * empty}]{Fore.RESET}", end="")

        if verbose:
            print(f"\r{Fore.LIGHTBLACK_EX}{n_bootstrap}/{n_bootstrap} [{'*' * PROGRESS_BAR_TICKS}]{Fore.RESET}")
            print(f"{Fore.MAGENTA}Done!{Fore.RESET} ({time.time() - t0:.2f}s)")

    def get(self, bootstrap: bool = True) -> dict:
        """Returns statistical tensors (either historical baseline or bootstrap dictionary)."""
        if bootstrap:
            if not self.stats_boot:
                raise ValueError("Bootstrap not prepared. Run prepare_bootstrap() first.")
            return self.stats_boot
        return self.stats


class HistoricalDataset:
    """
    Manages historical price downloads and partitions data into analysis windows.
    """

    def __init__(
        self,
        tickers: tuple,
        in_sample_window: int = 3 * TRADING_DAYS_PER_YEAR,
        out_of_sample_window: int = 2 * TRADING_DAYS_PER_YEAR,
        force_download: bool = True,
        verbose: bool = True,
    ) -> None:
        self.tickers = tickers
        self.returns, self.dates = download(
            tickers,
            source="yfinance",
            years=50,
            force_download=force_download,
            verbose=verbose,
        )

        self.in_sample_window = in_sample_window
        self.out_of_sample_window = out_of_sample_window

    def prepare_single_window(self) -> SampleStatistics:
        """Extracts the most recent in-sample time window into a SampleStatistics instance."""
        return SampleStatistics(self.returns[:, -self.in_sample_window :])

    def prepare_walk_forward_windows(
        self,
    ) -> tuple[list[SampleStatistics], list[SampleStatistics], list[dict], int]:
        """
        Partitions history into rolling walk-forward in-sample and out-of-sample periods.

        :return: Tuple containing (in_sample_list, out_of_sample_list, date_intervals, num_windows).
        """
        in_sample = []
        out_of_sample = []
        intervals = []

        T = self.returns.shape[1]
        block_size = self.in_sample_window + self.out_of_sample_window

        for oos_end in range(T, block_size - 1, -self.out_of_sample_window):
            oos_start = oos_end - self.out_of_sample_window
            is_end = oos_start
            is_start = is_end - self.in_sample_window

            returns_is = self.returns[:, is_start:is_end]
            returns_oos = self.returns[:, oos_start:oos_end]

            in_sample.append(SampleStatistics(returns_is))
            out_of_sample.append(SampleStatistics(returns_oos))

            intervals.append(
                {
                    "InSampleStartDate": str(self.dates[is_start].date()),
                    "InSampleEndDate": str(self.dates[is_end - 1].date()),
                    "OutOfSampleStartDate": str(self.dates[oos_start].date()),
                    "OutOfSampleEndDate": str(self.dates[oos_end - 1].date()),
                }
            )

        in_sample.reverse()
        out_of_sample.reverse()
        intervals.reverse()

        return in_sample, out_of_sample, intervals, len(in_sample)
