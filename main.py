import argparse
import importlib
from pathlib import Path

import config
from src.data import HistoricalDataset
from src.metrics import MetricSuite
from src.optimization import PortfolioConstraints, optimize_portfolio


def main(config_module: object = config) -> None:
    # 1. Initialize constraints and historical dataset
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
    stats = stats.prepare_single_window()
    stats.prepare_bootstrap(
        random_state=config_module.random_state,
        n_bootstrap=config_module.stats_config.get("bootstrap_size", 100),
        block_length=config_module.stats_config.get("bootstrap_block_size", 20),
    )

    # 2. Run multi-objective optimization
    obj_func = MetricSuite(config_module.objective_functions)
    constraints_evaluator = MetricSuite(config_module.constraints)

    port, _ = optimize_portfolio(
        stats,
        obj_func,
        constraints,
        constraint_evaluator=constraints_evaluator,
        seeds=list(
            range(
                config_module.random_state,
                config_module.random_state + config_module.optimization_config.get("n_seeds", 1),
            )
        ),
        pop_size=config_module.optimization_config.get("pop_size", 200),
        n_gen=config_module.optimization_config.get("n_gen", 500),
        do_early_stopping=config_module.optimization_config.get("do_early_stopping", True),
        early_stopping_patience=config_module.optimization_config.get("early_stopping_patience", 10),
        estimate_ref_point=config_module.optimization_config.get("estimate_ref_point", True),
        hv_ref_point=config_module.optimization_config.get("hv_ref_point", None),
    )

    # 3. Display portfolio weights and metrics
    if len(port) == 0:
        print("No valid portfolio found.")
    else:
        print("\n")
        for i in range(len(tickers)):
            print(f"{tickers[i]}: {port[i]:.1%}")

        print()
        for i in obj_func.metrics:
            print(f"{i.__class__.__name__}: {i.calculate(port, stats, raw_scale=True, do_bootstrap=False):.4f}")

        print()
        for i in constraints_evaluator.metrics:
            print(f"{i.__class__.__name__}: {i.calculate(port, stats, raw_scale=True, do_bootstrap=False):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the optimization.")
    parser.add_argument(
        "--config", type=str, default="config", help="Name of the configuration module (without .py extension)."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.suffix == ".py" or "/" in args.config:
        spec = importlib.util.spec_from_file_location("dynamic_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
    else:
        config_module = importlib.import_module(args.config)

    main(config_module=config_module)
