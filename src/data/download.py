import datetime as dt
import os
import pickle as pk
import time

import numpy as np
import pandas as pd
import yfinance as yf
from colorama import Fore

from src.constants import MINIMUM_VALID_DAYS
from src.data.historical_data_expcetion import HistoricalDataException


def _download_yfinance(tickers: tuple, years: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Downloads data using yfinance as source.

    :param tickers: The names of the assets to download, must be a list/tuple of names.
    :type tickers: `tuple`
    :param years: Specifies the number of years of data to download.
    :type years: `int`
    :return: returns and dates of the requested assets.
    :rtype: `tuple[np.ndarray, pd.DatetimeIndex]`
    """

    # Download raw historical data via yfinance
    data = yf.download(
        tickers,
        period=f"{years}y",
        auto_adjust=False,
        actions=True,  # for dividends and splits
    )

    # Reorder data columns to match ticker order
    adj_close = data.get("Adj Close")
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])
    else:
        adj_close = adj_close[list(tickers)]

    # Filter tickers that do not meet the minimum trading days threshold
    invalid_tickers = [ticker for ticker in tickers if len(adj_close[ticker].dropna()) <= MINIMUM_VALID_DAYS]
    if invalid_tickers != []:
        print(f"{Fore.RED} The following assets do not have a sufficient number of days of data:")
        for i in invalid_tickers:
            print(f"\t{i}")
        print(Fore.RESET)
        raise HistoricalDataException("Cannot proceed, assets do not satisfy the minimum days limit.")

    # Calculate percentage returns and align dates
    rets_df = adj_close.pct_change().dropna(how="any")

    if rets_df.empty:
        raise ValueError("The requested assets have no historical dates in common.")

    dates = rets_df.index
    returns = np.atleast_2d(rets_df.T.to_numpy(dtype=np.float64))

    return returns, dates


def download(
    tickers: tuple,
    source: str = "yfinance",
    years: int = 50,
    force_download: bool = True,
    verbose: bool = True,
    end_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Retrieves information for the requested assets. If valid cached data exists, loads from there,
    otherwise downloads directly from the requested source. Returns returns and dates.

    :param tickers: The names of the assets to download, must be a list/tuple of names.
    :type tickers: `tuple`
    :param source: The source from which to download data, must be a string.
    :type source: `str`
    :param years: Specifies the number of years from today to download.
    :type years: `int`
    :param force_download: Forces downloading data from source.
    :type force_download: `bool`
    :param verbose: Specifies whether to print information on progress.
    :type verbose: `bool`
    :param end_date: Optional cutoff date. Drops all data subsequent to this date.
    :type end_date: `str | dt.date | dt.datetime | pd.Timestamp | None`
    :return: Historical data of the requested assets.
    :rtype: `tuple[np.ndarray, pd.DatetimeIndex]`
    """

    # Boolean flag determining whether data must be downloaded or loaded from cache
    download_data = False

    t0 = time.time()
    TODAY = dt.datetime.now(tz=dt.UTC).date()

    os.makedirs("./data", exist_ok=True)

    cache_data_file = "./data/titoliDownload.data"
    cache_info_file = "./data/infoDownload.data"

    # Check cache existence and validity
    if (
        not (os.path.exists(cache_data_file))
        or not (os.path.exists(cache_info_file))
        or force_download
    ):
        download_data = True
    else:
        # If cache exists and download is not forced, verify cache freshness and parameters
        with open(cache_info_file, "rb") as file:
            info = pk.load(file)

        info_date = info.get("date", info.get("data"))
        info_tickers = tuple(info.get("tickers", info.get("titoli", [])))
        info_source = info.get("source", info.get("fonte"))
        info_years = info.get("years", info.get("anni"))

        # Invalidate cache if older than one day or if parameters differ
        if (
            ((TODAY - info_date).days >= 1)
            or info_tickers != tuple(tickers)
            or info_source != source
            or info_years != years
        ):
            download_data = True
        else:
            download_data = False

    # Download data from source if cache is missing or invalidated
    if download_data:
        if source == "yfinance":
            if verbose:
                print(f"\n{Fore.MAGENTA}Downloading data from Yahoo Finance{Fore.RESET}:{Fore.LIGHTBLACK_EX}")

            returns, dates = _download_yfinance(tickers, years)
        else:
            raise HistoricalDataException(f"Source {source} not supported, cannot proceed.")

        if returns is None:
            raise HistoricalDataException("Error downloading data, cannot proceed.")

        # Persist downloaded data and metadata to disk
        if verbose:
            print(
                f"{Fore.MAGENTA}Saving data to {Fore.RESET}{cache_data_file}{Fore.MAGENTA} and {Fore.RESET}{cache_info_file}...{Fore.RESET}"
            )

        info = {"date": TODAY, "tickers": list(tickers), "source": source, "years": years}
        with open(cache_info_file, "wb") as file:
            pk.dump(info, file)

        with open(cache_data_file, "wb") as file:
            pk.dump((returns, dates), file)

    # Load data from existing valid cache
    else:
        if verbose:
            print(
                f"\n{Fore.MAGENTA}Loading data from {Fore.RESET}{cache_data_file}{Fore.MAGENTA}...{Fore.RESET}"
            )

        with open(cache_data_file, "rb") as file:
            returns, dates = pk.load(file)

    t = time.time() - t0
    if verbose:
        print(f"{Fore.MAGENTA}Done!{Fore.RESET} ({t:.2f}s)")

    target_date = end_date
    if target_date is not None:
        target_dt = pd.to_datetime(target_date)
        if dates.tz is not None and target_dt.tz is None:
            target_dt = target_dt.tz_localize(dates.tz)
        elif dates.tz is None and target_dt.tz is not None:
            target_dt = target_dt.tz_localize(None)
        mask = dates <= target_dt
        returns = returns[:, mask]
        dates = dates[mask]

    return returns, dates
