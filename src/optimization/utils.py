import numpy as np

from src.data.statistics import SampleStatistics
from src.metrics import MetricSuite
from src.optimization import PortfolioConstraints


def _pareto_mask(F: np.ndarray) -> np.ndarray:
    """Returns a boolean mask with True for non-dominated points (minimization)."""
    n_points = F.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)

    for i, f in enumerate(F):
        if is_efficient[i]:
            # Point j is dominated by i if F[j] >= F[i] on all objectives and F[j] > F[i] on at least one
            dominated = np.all(F >= f, axis=1) & np.any(F > f, axis=1)
            is_efficient[dominated] = False

    return is_efficient


def find_knee_point(F: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identifies the knee point on the non-dominated Pareto front."""
    if len(F) != len(X):
        raise ValueError("F and X must have the same number of rows")

    # 1. Filter only non-dominated solutions (global Pareto front)
    mask_pareto = _pareto_mask(F)
    F_pareto = F[mask_pareto]
    X_pareto = X[mask_pareto]

    n_points, n_obj = F_pareto.shape
    if n_points <= n_obj:
        # If non-dominated points are insufficient, fallback to first point or direct TOPSIS
        if n_points == 0:
            return [], []
        return F_pareto[0], X_pareto[0]

    # 2. Min-max normalization only on non-dominated points
    f_min = np.min(F_pareto, axis=0)
    f_max = np.max(F_pareto, axis=0)
    delta = f_max - f_min
    delta[delta == 0.0] = 1.0
    F_norm = (F_pareto - f_min) / delta

    knee_idx = None

    # Case 2 OBJECTIVES: Perpendicular distance from line connecting the two extremes
    if n_obj == 2:
        idx_p1 = int(np.argmin(F_norm[:, 0]))
        idx_p2 = int(np.argmin(F_norm[:, 1]))

        p1 = F_norm[idx_p1]
        p2 = F_norm[idx_p2]

        v = p2 - p1
        v_norm = np.linalg.norm(v)

        if v_norm > 1e-8:
            numerator = np.abs(v[0] * (p1[1] - F_norm[:, 1]) - (p1[0] - F_norm[:, 0]) * v[1])
            distances = numerator / v_norm
            knee_idx = int(np.argmax(distances))

    # Case 3 OBJECTIVES: Perpendicular distance from plane passing through 3 extremes
    elif n_obj == 3:
        idx_p1 = int(np.argmin(F_norm[:, 0]))
        idx_p2 = int(np.argmin(F_norm[:, 1]))
        idx_p3 = int(np.argmin(F_norm[:, 2]))

        p1 = F_norm[idx_p1]
        p2 = F_norm[idx_p2]
        p3 = F_norm[idx_p3]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)

        if norm_len > 1e-8:
            unit_normal = normal / norm_len
            distances = np.abs(np.dot(F_norm - p1, unit_normal))
            knee_idx = int(np.argmax(distances))

    # Fallback TOPSIS (minimum Euclidean distance from ideal origin [0, 0, ...])
    if knee_idx is None:
        knee_idx = int(np.argmin(np.linalg.norm(F_norm, axis=1)))

    return F_pareto[knee_idx], X_pareto[knee_idx]


def estimate_hv_ref_point(
    historical_stats: SampleStatistics,
    evaluator: MetricSuite,
    constraints: PortfolioConstraints,
    n_samples: int = 50,
    margin: float = 0.2,
) -> np.ndarray:
    """
    Computes a priori a safe ref_point by evaluating a set of sample portfolios.

    :param margin: Percentage margin beyond worst value found.
    """
    samples = []

    # 1. 100% single-asset portfolios (if allowed by constraints, or repaired)
    for i in range(historical_stats.n_assets):
        w = np.zeros(historical_stats.n_assets)
        w[i] = 1.0
        samples.append(constraints.apply(w))

    # 2. Equal-weighted portfolio
    samples.append(constraints.apply(np.ones(historical_stats.n_assets) / historical_stats.n_assets))

    # 3. Random portfolios satisfying constraints
    for _ in range(n_samples):
        w_rand = np.random.rand(historical_stats.n_assets)
        samples.append(constraints.apply(w_rand))

    X_sample = np.array(samples)

    # raw_scale=False to reflect internal pymoo metrics (inverted signs)
    F_sample = evaluator.evaluate(X_sample, historical_stats, raw_scale=False)

    # In pymoo we minimize, so "worst" is maximum along each axis
    f_min = np.min(F_sample, axis=0)
    f_max = np.max(F_sample, axis=0)
    delta = np.maximum(f_max - f_min, 1e-4)

    # Reference point with safety offset
    ref_point = f_max + margin * delta
    return ref_point
