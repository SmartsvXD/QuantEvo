import argparse
import importlib
import time
from pathlib import Path

import pandas as pd

import config
from src.data import HistoricalDataset
from src.metrics import MetricSuite
from src.optimization import PortfolioConstraints, optimize_portfolio


def benchmark(config_module: object = config, output: str = "output") -> None:
    """
    Executes walk-forward backtesting across rolling in-sample and out-of-sample windows.

    Trains portfolios on each historical in-sample block, evaluates out-of-sample
    performance, computes summary metrics, and exports results to CSV files.

    :param config_module: Configuration module specifying assets, constraints, and windows.
    :param output: Destination directory for exported CSV benchmark reports.
    """
    tickers = [x.get("name", "") for x in config_module.assets]
    upper = [x.get("upper", 1.0) for x in config_module.assets]

    constraints = PortfolioConstraints(
        min_allocation=config_module.structural_constraints.get("minimum_percentage_per_asset", 0.05),
        upper=upper,
        min_assets=config_module.structural_constraints.get("minimum_number_of_assets", 3),
        max_assets=config_module.structural_constraints.get("maximum_number_of_assets", 10),
    )

    stats = HistoricalDataset(
        tickers,
        in_sample_window=config_module.window_size,
        out_of_sample_window=config_module.window_size_out_of_sample,
        force_download=False,
        verbose=True,
    )
    in_sample, out_of_sample, periods, n = stats.prepare_walk_forward_windows()

    obj_func = MetricSuite(config_module.objective_functions)
    constraints_evaluator = MetricSuite(config_module.constraints)

    metrics = []
    weights = []

    # Iterate walk-forward windows for out-of-sample backtesting
    for w in range(n):
        in_sample[w].prepare_bootstrap(
            random_state=config_module.random_state,
            n_bootstrap=config_module.stats_config.get("bootstrap_size", 100),
            block_length=config_module.stats_config.get("bootstrap_block_size", 20),
        )
        out_of_sample[w].prepare_bootstrap(
            random_state=config_module.random_state,
            n_bootstrap=config_module.stats_config.get("bootstrap_size", 100),
            block_length=config_module.stats_config.get("bootstrap_block_size", 20),
        )

        port, _ = optimize_portfolio(
            in_sample[w],
            obj_func,
            constraints,
            constraint_evaluator=constraints_evaluator,
            seeds=list(
                range(
                    config_module.random_state,
                    config_module.random_state + config_module.optimization_config.get("n_seeds", 10),
                )
            ),
            pop_size=config_module.optimization_config.get("pop_size", 200),
            n_gen=config_module.optimization_config.get("n_gen", 500),
            do_early_stopping=config_module.optimization_config.get("do_early_stopping", True),
            early_stopping_patience=config_module.optimization_config.get("early_stopping_patience", 10),
            estimate_ref_point=config_module.optimization_config.get("estimate_ref_point", True),
            hv_ref_point=config_module.optimization_config.get("hv_ref_point", None),
        )

        if len(port) == 0:
            print("No valid portfolio found.")

            row = {"Window": w + 1}
            for i in range(len(tickers)):
                row[tickers[i]] = 0.0
            weights.append(row)

            for m in obj_func.metrics + constraints_evaluator.metrics:
                metrics.append(
                    {
                        "Window": w + 1,
                        "Metric": m.__class__.__name__,
                        "InSample": None,
                        "OutOfSample": None,
                        "Difference": None,
                    }
                )
        else:
            print(
                "\nOOS Window:",
                w + 1,
                f"(from {periods[w]['OutOfSampleStartDate']} to {periods[w]['OutOfSampleEndDate']})",
            )
            print()

            row = {}
            row["Window"] = w + 1
            for i in range(len(tickers)):
                print(f"{tickers[i]}: {port[i]:.1%}")
                row[tickers[i]] = port[i]
            weights.append(row)

            print()

            for m in obj_func.metrics + constraints_evaluator.metrics:
                is_m = m.calculate(port, in_sample[w], raw_scale=True, do_bootstrap=False)
                oos_m = m.calculate(port, out_of_sample[w], raw_scale=True, do_bootstrap=False)

                metrics.append(
                    {
                        "Window": w + 1,
                        "Metric": m.__class__.__name__,
                        "InSample": is_m,
                        "OutOfSample": oos_m,
                        "Difference": oos_m - is_m,
                    }
                )

                print(f"{m.__class__.__name__} (In-Sample):     {is_m:.4f}")
                print(f"{m.__class__.__name__} (Out-Of-Sample): {oos_m:.4f}")
                print(f"{m.__class__.__name__} (Difference):    {oos_m - is_m:.4f}")
                print()

            time.sleep(3)

    # Calculate summary statistics across all walk-forward windows
    weights.append(
        {"Window": "Mean", **{tickers[i]: sum(weights[w][tickers[i]] for w in range(n)) / n for i in range(len(tickers))}}
    )

    weights.append(
        {
            "Window": "Standard Deviation",
            **{
                tickers[i]: (
                    sum((weights[w][tickers[i]] - sum(weights[w][tickers[i]] for w in range(n)) / n) ** 2 for w in range(n)) / n
                )
                ** 0.5
                for i in range(len(tickers))
            },
        }
    )

    # Export benchmark results to CSV files
    Path(f"./{output}").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics)
    df.to_csv(f"./{output}/metrics.csv", index=False)

    df = pd.DataFrame(periods)
    df.to_csv(f"./{output}/window_periods.csv", index=False)

    df = pd.DataFrame(weights)
    df.to_csv(f"./{output}/weights.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the benchmark.")
    parser.add_argument(
        "--config", type=str, default="config", help="Name of the configuration module (without .py extension)."
    )
    parser.add_argument("--output", type=str, default="output", help="Name of the output folder.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.suffix == ".py" or "/" in args.config:
        spec = importlib.util.spec_from_file_location("dynamic_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
    else:
        config_module = importlib.import_module(args.config)

    benchmark(config_module=config_module, output=args.output)
