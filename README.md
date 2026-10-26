# 📈 QuantEvo

> **Note**: This is an educational personal project. It does not provide financial or investment advice. Backtests omit trading fees, slippage, and market impact.

---

## 🎯 1. Overview and Motivation

Classic mean-variance optimization (Markowitz, 1952) assumes Gaussian returns, perfectly known parameters, and continuous convex constraints. In practice, financial returns have heavy tails and volatility clustering, while realistic allocations require cardinality limits ($k_{\min} \le \|w\| \le k_{max}  $) and semi-continuous thresholds ($w_i \ge w_{\min}$ or $w_i = 0$) that make exact solving NP-hard.

QuantEvo combines genetic optimization with non-parametric risk modeling:
* **Heuristic repair (bisection water-filling)**: Projects candidate weights onto the constrained simplex in $\mathcal{O}(k \log(1/\epsilon))$ iterations without soft penalty tuning.
* **Moving Block Bootstrap**: Resamples overlapping return blocks to keep empirical autocorrelation and volatility clustering intact.
* **Meucci's Effective Number of Bets (ENB)**: Decomposes portfolio variance via PCA into orthogonal factors and tracks entropy to detect true factor concentration.
* **Ledoit-Wolf shrinkage**: Conditions bootstrap covariance matrices against numerical singularity.
* **NSGA-II multi-objective GA**: Explores Pareto trade-offs (e.g. CVaR vs. Drawdown or CAGR) with Hypervolume early stopping and knee-point selection.
* **Numba parallel JIT**: Compiles bootstrap evaluations and constraint projections down to multi-threaded machine code.
* **Rolling walk-forward framework**

---

## 📊 2. Walk-Forward Backtest Examples

To evaluate how the algorithm handles real market regimes, QuantEvo executes a rolling walk-forward backtest over 25 years (July 2001 to July 2026, 11 windows of 3-year in-sample / 2-year out-of-sample).

To avoid synthetic splicing across 25 years, the backtest uses historical Vanguard index mutual funds (`VTSMX`, `VTMGX`, `VEIEX`, `VBMFX`, `VUSTX`, `VGSIX`, `VFISX`).

### 2.1 Example B: 7-Asset Defensive ("No-Fall") Strategy
The optimizer is configured to minimize daily CVaR (95%) and Maximum Drawdown, subject to:
* $\text{CAGR} \ge 4.0\%$
* $\text{ENB} \ge 3.5$
* $\text{ENC} \ge 3.8$
* Position size $\ge 5\%$, cardinality between 3 and 6 assets.

#### Out-of-Sample Performance Comparison (Feasible Windows 1 to 9, 2004 to 2022)

| Portfolio Strategy | Mean CAGR | Mean Daily CVaR (95%) | Mean Max Drawdown | Mean ENB | Mean ENC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QuantEvo (Generated)** | **4.31%** | **0.0060** | **5.80%** | **3.60** | **3.80** |
| World Equity 20% Conservative | 4.49% | 0.0059 | 6.36% | 4.39 | 1.52 |
| Balanced 40/60 (Long Treasury) | 7.07% | 0.0116 | 10.28% | 3.30 | 2.17 |
| Static Risk Parity Proxy | 6.45% | 0.0131 | 11.36% | 2.17 | 4.35 |
| Naive Equal Weight | 6.39% | 0.0147 | 13.72% | 1.47 | 7.00 |
| Equal Weight Core | 6.62% | 0.0155 | 13.48% | 1.65 | 5.00 |

*\*Note on benchmark comparison*: All strategies are evaluated over the identical 9 feasible windows (July 2004 to July 2022). In Windows 10 and 11, the joint optimization constraints (CAGR $\ge$ 4.0%, ENB $\ge$ 3.5, ENC $\ge$ 3.8) yielded an empty feasible set during the 2022 rate shock. Consequently, both QuantEvo and all reference portfolios are evaluated exclusively across the 9 feasible windows to ensure an identical evaluation period. Detailed window evaluations and benchmark definitions are illustrated in [EXPERIMENTS.md](./EXPERIMENTS.md).

#### Main Observations:
1. **Risk profile and trade-offs vs. World Equity 20%**: QuantEvo matches the defensive profile of World Equity 20% Conservative (5.80% vs 6.36% drawdown, 0.0060 vs 0.0059 CVaR), but delivers slightly lower return (4.31% vs 4.49%) and lower factor diversification ($\text{ENB} = 3.60$ vs $4.39$). This gap stems from enforcing multi-asset nominal dispersion ($\text{ENC} \ge 3.8$ vs $1.52$), which bars concentrating capital in a single aggregate bond instrument.
2. **Constituent vs. factor diversification**: While nominal equal weighting maximizes constituent spread ($\text{ENC} = 7.00$), its factor diversification collapses ($\text{ENB} = 1.47$) due to dominant equity risk. QuantEvo balances nominal allocation ($\text{ENC} = 3.80$) with orthogonal bets ($\text{ENB} = 3.60$), though binding ENC floors impose an observable efficiency trade-off.
3. **2008 Financial Crisis**: During the 2008 liquidation (Window 3, July 2008 to July 2010), defensive allocations to US Treasuries contained QuantEvo's maximum drawdown to 10.34%, compared to 28.64% for Naive Equal Weight and 23.48% for the Static Risk Parity Proxy.

Detailed window-by-window analyses, crisis breakdowns (including 2008 and the 2020 COVID crash), and 2022 infeasibility mechanics for both strategy examples are documented in [EXPERIMENTS.md](./EXPERIMENTS.md).

---

## 📐 3. Methodology and Formulation

### 3.1 MINLP Portfolio Formulation
Portfolio selection is modeled as a multi-objective Mixed-Integer Non-Linear Programming problem:

$$\begin{aligned}
\min_{w \in \mathbb{R}^N, \, z \in \{0, 1\}^N} \quad & \mathbf{F}(w) = \left( f_1(w), \dots, f_M(w) \right)^T \\
\text{subject to} \quad & \sum_{i=1}^N w_i = 1 \\
& z_i \cdot w_{\min} \le w_i \le z_i \cdot u_i, \quad \forall i \in \{1, \dots, N\} \\
& k_{\min} \le \sum_{i=1}^N z_i \le k_{\max} \\
& g_j(w) \le 0, \quad \forall j \in \{1, \dots, J\} \\
& z_i \in \{0, 1\}, \quad w_i \ge 0, \quad \forall i \in \{1, \dots, N\}
\end{aligned}$$

where $w_i$ is the weight of asset $i$, $z_i$ is a binary decision variable indicating whether asset $i$ is held, $u_i$ is the individual upper cap, $w_{min}$ is the semi-continuous allocation threshold, and $k_{min}, k_{max}$ are cardinality bounds.

### 3.2 Heuristic Repair Operator (Bisection Water-Filling)
Standard crossover and mutation operators almost always generate infeasible portfolios under cardinality and semi-continuous bounds. Instead of soft penalties, QuantEvo uses a deterministic repair operator that projects candidate vectors directly onto the feasible simplex (`src/optimization/repair.py`):

1. **Asset Eligibility Filtering**: Discards any asset $i$ where individual capacity $u_i < w_{min}$.
2. **Cardinality Selection via Continuous Scores**: Assigns each eligible asset a score equal to its continuous candidate weight generated by GA variation: $s_i = w_i^{\text{raw}}$. Assets are sorted in descending order of score. The active cardinality $k$ is dynamically determined by counting assets where $s_i \ge w_{min}$ (clamped to $[k_{min}, \min(k_{max}, \lfloor 1/w_{min} \rfloor, n_{\text{elig}})]$).
3. **Feasibility Swaps**: If the cumulative upper bounds of the chosen $k$ assets cannot reach the unit budget ($\sum_{j=1}^k u_j < 1.0$), the operator iteratively swaps active assets having minimal upper caps with inactive eligible assets having larger upper caps until feasibility is guaranteed.
4. **Simplex Bisection (Water-Filling)**: Solves for the shift parameter $\lambda$ such that:
   $$\sum_{j=1}^k \text{clip}(w_j + \lambda, w_{min}, u_j) = 1.0$$
   Because this sum is strictly monotonic in $\lambda$, a 1D bisection on $[-1.0, 1.0]$ converges deterministically to numerical precision in 40 iterations.
5. **Floating-Point Residual Absorption**: Any remaining numerical residue is distributed into assets with available headroom.

The repair operator is compiled with Numba JIT to enable high-throughput constraint projection during genetic variation.

### 3.3 Conditional Value-at-Risk (CVaR)
Tail risk is quantified via daily CVaR at confidence level $1 - \alpha$ (Rockafellar & Uryasev, 2000), computed non-parametrically across bootstrap return paths:

$$\widehat{\text{CVaR}}_\alpha(w) = -\frac{1}{\lfloor \alpha T \rfloor} \sum_{t=1}^{\lfloor \alpha T \rfloor} R_{(t)}$$

where $R_{(1)} \le R_{(2)} \le \dots \le R_{(T)}$ are sorted daily portfolio returns.

### 3.4 Moving Block Bootstrap
To preserve empirical autocorrelation and volatility clustering without parametric assumptions, return paths are simulated via Moving Block Bootstrap (Künsch, 1989) with overlapping blocks of length $L = 20$ trading days sampled with replacement.

### 3.5 Effective Number of Bets (Meucci, 2009)
Nominal capital weights do not reflect risk concentration. QuantEvo implements Meucci's orthogonal factor decomposition using Principal Component Analysis (PCA):
1. Decompose the covariance matrix: $\Sigma = V \Lambda V^T$, where $V$ are orthonormal eigenvectors and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$.
2. Compute the variance contribution of each orthogonal bet:
   $$p_k = \frac{(w^T v_k)^2 \lambda_k}{\sigma_p^2}, \quad \text{where } \sum_{k=1}^N p_k = 1$$
3. Evaluate the exponential Shannon entropy:
   $$\text{ENB} = \exp\left( -\sum_{k=1}^N p_k \ln p_k \right)$$

When risk is concentrated in a single factor, $\text{ENB} = 1$; when evenly spread across all $N$ dimensions, $\text{ENB} = N$.

### 3.6 Ledoit-Wolf Covariance Shrinkage
Empirical covariance matrices can become ill-conditioned across resampled bootstrap paths. To ensure positive definiteness without parametric assumptions, Ledoit-Wolf shrinkage is calculated on each bootstrap scenario:
$$\Sigma_{\text{shrunk}} = \delta F + (1 - \delta) \Sigma_{\text{sample}}, \quad F = \frac{\text{tr}(\Sigma_{\text{sample}})}{N} I_N$$
where $\delta \in [0, 1]$ is the analytically optimal shrinkage intensity (Ledoit and Wolf, 2004).

### 3.7 Capital Dynamics: Buy-and-Hold
Out-of-sample evaluations track realistic buy-and-hold price drift rather than assuming continuous, friction-free rebalancing.

### 3.8 Multi-Objective Optimization (NSGA-II)
The optimization uses `pymoo` with customized components:
* **Hypervolume Early Stopping**: Stops the GA when hypervolume growth plateaus within a patience limit.
* **Knee-Point Selection**: Picks a balanced trade-off by finding the point furthest from the secant line (2D) or plane (3D) of extreme Pareto solutions, with Euclidean distance to the utopia point as fallback for degenerate geometry.
* **Infeasibility Diagnostics**: When constraints cannot be met, the engine identifies the unsatisfied metrics and logs the closest value found in the population.
* **Multi-Seed Aggregation**: Runs multiple seeds and extracts the global non-dominated front.

---

## ⚙️ 4. Architecture and Performance

The codebase separates optimization, metrics, and data loaders, compiling bottleneck loops with Numba.

```
QuantEvo/
├── main.py                   # Single-window portfolio optimization script
├── benchmark.py              # Rolling walk-forward backtest runner
├── requirements.txt          # Python dependencies
├── examples/                 # Configuration profiles (Standard USA, No-Fall, Italian)
│   ├── example_a.py
│   ├── example_a_mf.py
│   ├── example_b.py
│   ├── example_b_mf.py
│   └── example_c.py
├── other/
│   ├── utils.py              # Plotting and CSV export helpers
│   ├── compare_exp_a.ipynb   # Visual comparison and benchmark generation for Exp A
│   ├── compare_exp_b.ipynb   # Visual comparison and benchmark generation for Exp B
│   └── output/               # Backtest CSV results and performance plots
│       ├── exp_a/
│       └── exp_b/
└── src/
    ├── constants.py          # Trading calendar constants
    ├── data/
    │   ├── download.py       # Yahoo Finance data fetcher with local disk cache
    │   ├── statistics.py     # Block Bootstrap and Ledoit-Wolf estimation
    │   └── historical_data_exception.py # Data integrity exceptions
    ├── metrics/
    │   ├── metric.py         # Abstract base class Metric
    │   ├── cagr.py           # Compound Annual Growth Rate
    │   ├── cvar.py           # Conditional Value at Risk
    │   ├── omega.py          # Omega Ratio
    │   ├── enb.py            # Meucci Effective Number of Bets (JIT)
    │   ├── enc.py            # Effective Number of Constituents (JIT)
    │   ├── drawdown.py       # Maximum Drawdown
    │   └── volatility.py     # Annualized Volatility
    └── optimization/
        ├── pymoo.py          # Problem and callback adapters for pymoo
        ├── repair.py         # JIT heuristic repair with exact bisection projection
        └── utils.py          # Knee-point detection and HV estimation
```

### Parallel Acceleration with Numba
Evaluating 250 candidate portfolios across 100 bootstrap paths over several hundred generations is computationally intensive.

To make the search practical, core loops in QuantEvo use `@njit(fastmath=True, parallel=True)`. Multi-threading via Numba's `prange` distributes bootstrap evaluations across available CPU cores, minimizing execution overhead during multi-seed runs.

### Metric Interface (Strategy Pattern)
Every risk and return function inherits from the base class `Metric` (`src/metrics/metric.py`):
* Any metric can serve as an **objective function** (minimized or maximized).
* Any metric can serve as a **non-linear constraint** by specifying a threshold and aggregation measure across bootstrap paths (`cvar`, `percentile`, or `downside_penalty` with semi-deviation), automatically normalized as $g(w) \le 0$ for `pymoo`.

---

## 🚀 5. Getting Started

### 5.1 Installation
Clone the repository and install the dependencies inside a virtual environment:

```bash
git clone https://github.com/SmartsvXD/QuantEvo.git
cd QuantEvo

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 5.2 Running an Optimization
Run a single-window optimization on the default configuration:

```bash
python main.py
```

To run a specific configuration profile from the `examples/` directory:

```bash
python main.py --config examples/example_b_mf.py
```

### 5.3 Running the Walk-Forward Benchmark
To run the full rolling backtest with in-sample and out-of-sample evaluation and export CSV results:

```bash
python benchmark.py --config examples/example_b_mf.py --output output/my_exp_b
```

This command generates:
* `metrics.csv`: Metric values for each window across in-sample and out-of-sample periods.
* `weights.csv`: Optimal weights selected for each window, with mean and standard deviation.
* `window_periods.csv`: Start and end dates for every historical window.

To reproduce the benchmark comparison tables and plot images, open and run the notebook `other/compare_exp_b.ipynb`.

### 5.4 Creating a Custom Configuration
Configurations are standalone Python modules loaded dynamically at runtime via `importlib`, allowing external strategies to be tested without modifying the package source code.

To test your own asset universe and investment strategy, create a configuration file (for example, `my_strategy.py`):

```python
from src.constants import TRADING_DAYS_PER_YEAR
from src.metrics.cagr import CAGR
from src.metrics.cvar import CVaR
from src.metrics.drawdown import Drawdown
from src.metrics.enb import ENB
from src.metrics.enc import ENC
from src.metrics.omega import Omega

random_state = 42

# Window configuration (3 years in-sample, 2 years out-of-sample)
window_size = TRADING_DAYS_PER_YEAR * 3
window_size_out_of_sample = TRADING_DAYS_PER_YEAR * 2

# Bootstrap configuration
stats_config = {
    "bootstrap_size": 100,
    "bootstrap_block_size": 20,
}

# Genetic algorithm configuration
optimization_config = {
    "pop_size": 250,
    "n_gen": 400,
    "n_seeds": 5,
    "do_early_stopping": True,
    "early_stopping_patience": 10,
    "estimate_ref_point": True,
    "hv_ref_point": None,
}

# Asset universe and individual maximum caps
assets = [
    {"name": "SPY", "upper": 0.60},
    {"name": "QQQ", "upper": 0.40},
    {"name": "EFA", "upper": 0.30},
    {"name": "TLT", "upper": 0.50},
    {"name": "GLD", "upper": 0.20},
]

# MINLP structural constraints
structural_constraints = {
    "minimum_percentage_per_asset": 0.05,
    "minimum_number_of_assets": 3,
    "maximum_number_of_assets": 5,
}

# Objectives to optimize
objective_functions = [
    CVaR(alpha=0.05),
    Drawdown(),
]

# Non-linear constraints
constraints = [
    CAGR(threshold=0.06, risk_measure="percentile", alpha=0.5),
    ENB(threshold=2.5, risk_measure="percentile", alpha=0.5),
    Omega(threshold=1.2, annual_hurdle_rate=0.02, risk_measure="percentile", alpha=0.5),
]
```

Run your new configuration:

```bash
python main.py --config my_strategy.py
```

---

## 📚 6. References

1. **Markowitz, H.** (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
2. **Meucci, A.** (2009). Managing Diversification. *Risk*, 22(5), 74-79.
3. **Ledoit, O., & Wolf, M.** (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
4. **Künsch, H. R.** (1989). The Jackknife and the Bootstrap for General Stationary Observations. *The Annals of Statistics*, 17(3), 1217-1241.
5. **Artzner, P., Delbaen, F., Eber, J. M., & Heath, D.** (1999). Coherent Measures of Risk. *Mathematical Finance*, 9(3), 203-228.
6. **Rockafellar, R. T., & Uryasev, S.** (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21-41.
7. **Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T.** (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
8. **Keating, C., & Shadwick, W. F.** (2002). A universal performance measure. *Journal of Performance Measurement*, 6(3), 59-86.

## 📄 License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later) - see the [LICENSE](LICENSE) file for details.