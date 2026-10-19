import numpy as np
from numba import njit


@njit(fastmath=True)
def _project_row(
    w: np.ndarray,
    min_allocation: float,
    upper_arr: np.ndarray,
    min_assets: int,
    max_assets: int,
) -> None:
    n = w.shape[0]

    # 1. Filter eligible assets (upper >= min_allocation)
    scores = np.full(n, -1e30)
    n_elig = 0
    for i in range(n):
        if upper_arr[i] >= min_allocation:
            scores[i] = w[i]
            n_elig += 1
        else:
            w[i] = 0.0

    if n_elig < min_assets:
        raise ValueError("Eligible assets less than min_assets")

    # 2. Sort descending based on initial weight
    idx_sorted = np.argsort(-scores)

    # 3. Determine active cardinality k (between min_assets and max_active)
    k_target = min_assets
    max_active = int(1.0 / min_allocation) if min_allocation > 0.0 else n_elig
    max_active = min(max_active, n_elig, max_assets)

    if k_target > max_active:
        raise ValueError("min_assets cannot exceed max_active")

    n_natural = 0
    for j in range(n_elig):
        idx = idx_sorted[j]
        if scores[idx] >= min_allocation:
            n_natural += 1
        else:
            break

    k = max(min(n_natural, max_active), k_target)

    # Initial capacity calculation
    sum_upper = 0.0
    for j in range(k):
        sum_upper += upper_arr[idx_sorted[j]]

    # Phase A: Expansion if sum upper < 1.0 and we can still add assets
    while sum_upper < 1.0 - 1e-12 and k < max_active:
        sum_upper += upper_arr[idx_sorted[k]]
        k += 1

    # Phase B: Feasibility swap if k reached max_active but sum upper < 1.0
    # Replaces active assets with smaller upper with inactive assets with larger upper
    while sum_upper < 1.0 - 1e-12:
        best_active_pos = -1
        min_active_upper = 1e30
        min_active_score = 1e30
        for j in range(k):
            idx_a = idx_sorted[j]
            u = upper_arr[idx_a]
            sc = scores[idx_a]
            if (u < min_active_upper - 1e-12) or (abs(u - min_active_upper) <= 1e-12 and sc < min_active_score):
                min_active_upper = u
                min_active_score = sc
                best_active_pos = j

        best_inactive_pos = -1
        max_inactive_upper = -1.0
        max_inactive_score = -1e30
        for j in range(k, n_elig):
            idx_i = idx_sorted[j]
            u = upper_arr[idx_i]
            sc = scores[idx_i]
            if (u > max_inactive_upper + 1e-12) or (abs(u - max_inactive_upper) <= 1e-12 and sc > max_inactive_score):
                max_inactive_upper = u
                max_inactive_score = sc
                best_inactive_pos = j

        # If no further inactive assets with larger upper exist, stop
        if best_inactive_pos == -1 or max_inactive_upper <= min_active_upper + 1e-12:
            break

        idx_act = idx_sorted[best_active_pos]
        idx_inact = idx_sorted[best_inactive_pos]
        idx_sorted[best_active_pos] = idx_inact
        idx_sorted[best_inactive_pos] = idx_act

        sum_upper += (max_inactive_upper - min_active_upper)

    # 4. Active mask: zero out excluded assets beyond k
    is_active = np.zeros(n, dtype=np.bool_)
    for j in range(k):
        is_active[idx_sorted[j]] = True

    for i in range(n):
        if not is_active[i]:
            w[i] = 0.0

    # Preliminary feasibility check
    sum_min = k * min_allocation
    sum_max = 0.0
    for j in range(k):
        sum_max += upper_arr[idx_sorted[j]]

    if sum_max < 1.0 - 1e-12 or sum_min > 1.0 + 1e-12:
        raise ValueError("Cannot satisfy constraints with sum 1.0 within asset limits")

    # Initial clamp with exact min_allocation limit
    for j in range(k):
        idx = idx_sorted[j]
        if w[idx] < min_allocation:
            w[idx] = min_allocation
        elif w[idx] > upper_arr[idx]:
            w[idx] = upper_arr[idx]

    # 5. Simplex bisection: redistributes weight across active assets
    low = -1.0
    high = 1.0
    for _ in range(40):
        mid = (low + high) * 0.5
        s = 0.0
        for j in range(k):
            idx = idx_sorted[j]
            v = w[idx] + mid
            if v < min_allocation:
                v = min_allocation
            elif v > upper_arr[idx]:
                v = upper_arr[idx]
            s += v

        if s < 1.0:
            low = mid
        else:
            high = mid

    # 6. Definitive weight assignment
    mid = (low + high) * 0.5
    s_final = 0.0
    for j in range(k):
        idx = idx_sorted[j]
        v = w[idx] + mid
        if v < min_allocation:
            v = min_allocation
        elif v > upper_arr[idx]:
            v = upper_arr[idx]
        w[idx] = v
        s_final += v

    # 7. Numerical residual absorption
    residual = 1.0 - s_final
    if abs(residual) > 1e-12:
        if residual > 0.0:
            for j in range(k):
                idx = idx_sorted[j]
                room = upper_arr[idx] - w[idx]
                if room > 1e-12:
                    delta = min(residual, room)
                    w[idx] += delta
                    residual -= delta
                    if residual <= 1e-12:
                        break
            if residual > 1e-12:
                for j in range(k):
                    idx = idx_sorted[j]
                    w[idx] += residual / k
        else:
            for j in range(k):
                idx = idx_sorted[j]
                room = w[idx] - min_allocation
                if room > 1e-12:
                    delta = min(-residual, room)
                    w[idx] -= delta
                    residual += delta
                    if abs(residual) <= 1e-12:
                        break


@njit(fastmath=True)
def _apply_constraints_jit(
    weights: np.ndarray,
    min_allocation: float,
    upper: np.ndarray,
    min_assets: int,
    max_assets: int,
) -> np.ndarray:
    if weights.ndim == 1:
        _project_row(weights, min_allocation, upper, min_assets, max_assets)
    else:
        for r in range(weights.shape[0]):
            _project_row(weights[r], min_allocation, upper, min_assets, max_assets)
    return weights


def apply_constraints(
    weights: np.ndarray,
    min_allocation: float = 0.05,
    upper: np.ndarray | None = None,
    min_assets: int = 3,
    max_assets: int | None = None,
) -> np.ndarray:
    """
    Validates constraints and projects arbitrary portfolio weights onto the feasible domain.

    Enforces total weight summing to 1, semi-continuous asset allocations (either 0
    or >= min_allocation), individual maximum caps, and cardinality constraints
    (between min_assets and max_assets) using bisection on the simplex.

    :param weights: 1D or 2D array of unconstrained portfolio weights.
    :param min_allocation: Minimum allocation threshold for active assets.
    :param upper: Optional upper bound array per asset (defaults to 1.0).
    :param min_assets: Minimum number of active assets required.
    :param max_assets: Maximum number of active assets allowed.
    :return: Projected feasible weights array of same shape.
    """
    if not isinstance(weights, np.ndarray):
        weights = np.asarray(weights, dtype=np.float64)
    elif not np.issubdtype(weights.dtype, np.floating):
        weights = weights.astype(np.float64)

    if weights.ndim not in (1, 2):
        raise ValueError(f"'weights' must have dimension 1 or 2, received ndim={weights.ndim}")

    n_cols = weights.shape[-1]

    if not isinstance(min_assets, (int, np.integer)) or min_assets < 1:
        raise ValueError(f"'min_assets' must be an integer >= 1, received {min_assets}")
    if min_assets > n_cols:
        raise ValueError(f"'min_assets' ({min_assets}) cannot exceed number of assets ({n_cols})")

    if max_assets is None:
        max_assets = n_cols
    else:
        if not isinstance(max_assets, (int, np.integer)) or max_assets < 1:
            raise ValueError(f"'max_assets' must be an integer >= 1, received {max_assets}")
        if max_assets < min_assets:
            raise ValueError(f"'max_assets' ({max_assets}) cannot be less than 'min_assets' ({min_assets})")
        max_assets = min(int(max_assets), n_cols)

    if not isinstance(min_allocation, (int, float, np.floating)):
        raise TypeError("'min_allocation' must be a numeric value")
    if not (0.0 <= min_allocation <= 1.0):
        raise ValueError(f"'min_allocation' must be in [0, 1], received {min_allocation}")

    if upper is None:
        upper = np.ones(n_cols, dtype=np.float64)
    else:
        upper = np.asarray(upper, dtype=np.float64)
        if upper.ndim != 1 or upper.shape[0] != n_cols:
            raise ValueError(f"'upper' must be a 1D array of length {n_cols}, received shape={upper.shape}")
        if np.any(upper < 0.0) or np.any(upper > 1.0):
            raise ValueError("All elements of 'upper' must be in interval [0, 1]")

    if min_assets * min_allocation > 1.0 + 1e-12:
        raise ValueError(
            f"Problem infeasible: min_assets ({min_assets}) * min_allocation ({min_allocation}) = "
            f"{min_assets * min_allocation:.4f} > 1.0"
        )

    eligible = upper >= min_allocation
    n_elig = np.count_nonzero(eligible)
    if n_elig < min_assets:
        raise ValueError(
            f"Problem infeasible: assets with upper >= min_allocation ({n_elig}) less than min_assets ({min_assets})"
        )

    effective_max = min(max_assets, int(1.0 / min_allocation)) if min_allocation > 0.0 else max_assets
    effective_max = min(effective_max, n_elig)

    if effective_max < min_assets:
        raise ValueError(
            f"Problem infeasible: effective maximum cardinality ({effective_max}) less than min_assets ({min_assets})"
        )

    upper_elig_sorted = np.sort(upper[eligible])[::-1]
    max_possible_sum = np.sum(upper_elig_sorted[:effective_max])
    if max_possible_sum < 1.0 - 1e-12:
        raise ValueError(
            f"Problem infeasible: sum of maximum {effective_max} eligible upper bounds "
            f"({max_possible_sum:.4f}) is less than 1.0"
        )

    if not weights.flags.c_contiguous:
        weights = np.ascontiguousarray(weights)
    if not upper.flags.c_contiguous:
        upper = np.ascontiguousarray(upper)

    return _apply_constraints_jit(weights, float(min_allocation), upper, int(min_assets), int(max_assets))


class PortfolioConstraints:
    """
    Encapsulates structural portfolio constraints and repair operators.

    Maintains settings for minimum allocation per active asset, upper bounds,
    and cardinality bounds, providing methods to project weights and filter out
    infeasible solutions.
    """

    def __init__(
        self,
        min_allocation: float = 0.05,
        upper: np.ndarray | None = None,
        min_assets: int = 3,
        max_assets: int | None = None,
    ) -> None:
        self.min_allocation = min_allocation
        self.upper = np.asarray(upper, dtype=np.float64) if upper is not None else None
        self.min_assets = min_assets
        self.max_assets = max_assets

    def apply(self, portfolios: np.ndarray) -> np.ndarray:
        """Projects portfolio weights onto the constrained feasible space."""
        return apply_constraints(
            portfolios,
            min_allocation=self.min_allocation,
            upper=self.upper,
            min_assets=self.min_assets,
            max_assets=self.max_assets,
        )

    def remove_infeasible(
        self,
        X: np.ndarray,
        F: np.ndarray,
        sum_tol: float = 1e-8,
        zero_tol: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filters out solutions that violate budget, bound, or cardinality constraints.

        :param X: Decision variables array (portfolio weights) of shape (P, N).
        :param F: Objective function values array of shape (P, M).
        :param sum_tol: Tolerance for budget constraint (|sum(w) - 1| <= sum_tol).
        :param zero_tol: Tolerance for non-negativity and boundary checks.
        :return: Tuple of filtered (F, X) arrays containing only valid solutions.
        """
        n = X.shape[1]
        upper = self.upper if self.upper is not None else np.ones(n)
        pct_min = self.min_allocation
        min_assets = self.min_assets
        max_assets = self.max_assets if self.max_assets is not None else n

        invalid = []
        for i in range(X.shape[0]):
            row = X[i]

            # 1. Significant negative values
            if np.any(row < -zero_tol):
                invalid.append(i)
                continue

            # 2. Above upper bound
            if np.any(row - upper > zero_tol):
                invalid.append(i)
                continue

            # 3. Sum of weights = 1
            if abs(row.sum() - 1.0) > sum_tol:
                invalid.append(i)
                continue

            # 4. Semi-continuous weights: allowed only ~0 or >= (pct_min - tol)
            mask_invalid = (row > zero_tol) & (row < pct_min - zero_tol)
            if np.any(mask_invalid):
                invalid.append(i)
                continue

            # 5. Minimum and maximum cardinality
            n_active = int(np.sum(row >= pct_min - zero_tol))
            if n_active < min_assets or n_active > max_assets:
                invalid.append(i)
                continue

        print(f"Discarded {len(invalid)} invalid solutions out of {X.shape[0]} total.")
        if len(invalid) > 0:
            invalid = np.unique(np.array(invalid, dtype=np.int64))
            X = np.delete(X, invalid, axis=0)
            F = np.delete(F, invalid, axis=0)

        return F, X
