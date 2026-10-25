import time

import numpy as np
from colorama import Fore
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.data.statistics import SampleStatistics
from src.metrics import MetricSuite
from src.optimization.pymoo import EarlyStoppingCallback, PortfolioProblem, PortfolioRepair, ValidPortfolioSampling
from src.optimization.repair import PortfolioConstraints
from src.optimization.utils import estimate_hv_ref_point, find_knee_point


def optimize_portfolio(
    historical_stats: SampleStatistics,
    evaluator: MetricSuite,
    constraints: PortfolioConstraints,
    constraint_evaluator: MetricSuite | None = None,
    pop_size: int = 200,
    n_gen: int = 500,
    seeds: list[int] | None = None,
    do_early_stopping: bool = True,
    early_stopping_patience: int = 10,
    estimate_ref_point: bool = True,
    hv_ref_point: np.ndarray | None = None,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Executes multi-objective portfolio optimization using the NSGA-II genetic algorithm.

    Runs optimization across one or more independent random seeds, enforces structural
    and metric constraints, filters feasible solutions, and identifies the best compromise
    knee point from the aggregated Pareto front.

    :param historical_stats: Asset returns and covariance statistics for evaluation.
    :param evaluator: MetricSuite representing objective functions to optimize.
    :param constraints: PortfolioConstraints specifying bounds, cardinality, and allocations.
    :param constraint_evaluator: Optional MetricSuite representing nonlinear metric constraints.
    :param pop_size: Population size for the genetic algorithm.
    :param n_gen: Maximum number of generations per seed.
    :param seeds: List of random seeds for independent optimization runs.
    :param do_early_stopping: Whether to terminate early when hypervolume plateaus.
    :param early_stopping_patience: Generations without improvement before early stopping.
    :param estimate_ref_point: Whether to calculate a reference point if none is supplied.
    :param hv_ref_point: Reference point for hypervolume calculations.
    :param verbose: Verbosity level (0 = silent, 1 = progress, 2 = detailed).
    :return: Tuple containing (optimal_portfolio_weights, optimal_metric_values).
    """
    t0 = time.time()

    if verbose:
        print(f"\n{Fore.MAGENTA}Executing evolutionary algorithm...{Fore.LIGHTBLACK_EX}")

    seeds = seeds if seeds is not None else [1]

    all_F, all_X, all_G = [], [], []

    # Execute NSGA-II optimization across independent random seeds
    for s in seeds:
        seed_int = int(s)
        np.random.seed(seed_int)

        problem = PortfolioProblem(
            historical_stats,
            historical_stats.n_assets,
            evaluator=evaluator,
            constraints_evaluator=constraint_evaluator,
        )

        sampling = ValidPortfolioSampling(historical_stats.n_assets, constraints=constraints)
        repair = PortfolioRepair(constraints=constraints)

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob=1.0 / historical_stats.n_assets, eta=25),
            repair=repair,
        )

        termination = get_termination("n_gen", n_gen)

        callback = None
        if do_early_stopping:
            if estimate_ref_point and hv_ref_point is None:
                hv_ref_point = estimate_hv_ref_point(historical_stats, evaluator, constraints)

            callback = EarlyStoppingCallback(
                patience=early_stopping_patience,
                min_improvement=1e-5,
                verbose=max(0, verbose - 1),
                ref_point=hv_ref_point,
            )

        res = minimize(problem, algorithm, termination, seed=seed_int, verbose=bool(verbose), callback=callback)
        if constraint_evaluator and res.pop is not None:
            all_G.append(res.pop.get("G"))

        # Collect feasible non-dominated individuals from the final population
        if res.X is not None and len(res.X) > 0:
            if res.CV is not None:
                cv = np.asarray(res.CV).reshape(len(res.X), -1)
                feasible = np.all(cv <= 1e-6, axis=1)
            else:
                feasible = np.ones(len(res.X), dtype=bool)

            if np.any(feasible):
                all_F.append(np.atleast_2d(res.F)[feasible])
                all_X.append(np.atleast_2d(res.X)[feasible])
            else:
                if verbose:
                    print(
                        f"{Fore.YELLOW}Warning: no feasible solution found for seed {seed_int}.{Fore.RESET}"
                    )

    if not all_F:
        if constraint_evaluator and all_G:
            G_all = np.vstack(all_G)
            best_idx = np.argmin(np.sum(np.maximum(0, G_all), axis=1))
            unsatisfied = [j for j in range(len(constraint_evaluator.metrics)) if np.min(G_all[:, j]) > 1e-6] or [j for j in range(len(constraint_evaluator.metrics)) if G_all[best_idx, j] > 1e-6]
            for j in unsatisfied:
                m = constraint_evaluator.metrics[j]
                g = np.min(G_all[:, j]) if np.min(G_all[:, j]) > 1e-6 else G_all[best_idx, j]
                best_val = m.threshold + (-g if m.TO_MAXIMIZE else g) * (abs(m.threshold) or 1.0)
                print(f"{Fore.YELLOW}Warning: constraint {m.__class__.__name__} not satisfied, best value: {best_val:.4f} (threshold: {m.threshold}){Fore.RESET}")
        return [], []

    # Aggregate solutions across all seeds and find best compromise (knee point)
    F_concat = np.vstack(all_F)
    X_concat = np.vstack(all_X)

    F_concat, X_concat = constraints.remove_infeasible(F=F_concat, X=X_concat, verbose=verbose)
    res, port = find_knee_point(F_concat, X_concat)

    if verbose:
        print(f"{Fore.MAGENTA}Done!{Fore.RESET} ({time.time() - t0:.2f}s)")

    return port, res
