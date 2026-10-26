# 📈 QuantEvo

> **Disclaimer**: This is a personal project created for educational and self-study purposes. It is not financial or investment advice. The backtests do not model transaction fees, slippage, or liquidity constraints. If you use this code for real trading, you do so at your own risk.

---

## 🎯 1. Overview and Motivation

Mean-variance optimization (Markowitz, 1952) is the classical foundation of portfolio theory. However, its core assumptions fail on empirical financial time series:
1. **Non-Gaussian returns**: Asset returns exhibit fat tails, negative skewness, and volatility clustering, which variance penalizes symmetrically.
2. **Factor concentration vs. nominal diversification**: A portfolio with uniform capital allocation ($w_i = 1/N$) can remain heavily concentrated in a single systematic risk factor.
3. **Operational MINLP constraints**: Cardinality limits ($k_{\min} \le |w|_0 \le k_{\max}$) and semi-continuous bounds ($w_i \ge w_{\min}$ or $w_i = 0$) break convexity and make exact solving NP-hard.

QuantEvo addresses these limitations through an integrated framework:
* Heuristic repair with continuous projection (bisection water-filling) to enforce MINLP constraints without soft penalties.
* Moving Block Bootstrap to preserve empirical autocorrelation and volatility clustering.
* Meucci's Effective Number of Bets (ENB) via spectral PCA decomposition and Shannon entropy.
* Ledoit-Wolf shrinkage re-estimated on every individual bootstrap path for covariance conditioning.
* Multi-objective genetic optimization (NSGA-II) with Hypervolume early stopping and knee-point selection.
* Parallel Numba JIT acceleration for multi-core execution.
* Rolling walk-forward validation (3 years in-sample, 2 years out-of-sample) spanning 25 years (2001 to 2026).

---

## 📊 2. Walk-Forward Backtest Results

QuantEvo is evaluated using an 11-window rolling walk-forward backtest over 25 years (July 2001 to July 2026). Each window uses 3 years of in-sample data for optimization and 2 years of out-of-sample data for testing, without lookahead bias.

To obtain 25 years of daily data, the backtest uses historical Vanguard mutual funds tracking core asset classes (`VTSMX`, `VTMGX`, `VEIEX`, `VBMFX`, `VUSTX`, `VGSIX`, `VFISX`).

### 2.1 Experiment B: 7-Asset Defensive ("No-Fall") Strategy
The optimizer was configured to minimize daily CVaR (95%) and Maximum Drawdown, subject to:
* Annualized return: $\text{CAGR} \ge 5.0\%$
* Factor diversification: $\text{ENB} \ge 3.5$
* Asset diversification: $\text{ENC} \ge 3.9$
* Minimum asset allocation: 5% (cardinality: between 3 and 6 assets).

#### Out-of-Sample Performance Comparison (Feasible Windows 1 to 9, 2004 to 2022)

| Portfolio Strategy | Mean CAGR | Mean Daily CVaR (95%) | Mean Max Drawdown | Mean ENB | Mean ENC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QuantEvo (Generated)** | **4.34%** | **0.0065** | **6.22%** | **3.43** | **3.90** |
| World Equity 20% Conservative | 4.49% | 0.0059 | 6.36% | 4.39 | 1.52 |
| Balanced 40/60 (Long Treasury) | 7.07% | 0.0116 | 10.28% | 3.30 | 2.17 |
| Static Risk Parity Proxy | 6.45% | 0.0131 | 11.36% | 2.17 | 4.35 |
| Naive Equal Weight | 6.40% | 0.0147 | 13.67% | 1.47 | 6.99 |
| Equal Weight Core | 6.62% | 0.0155 | 13.48% | 1.65 | 5.00 |

*\*Note on benchmark comparison*: All strategies are evaluated over the identical 9 feasible windows (July 2004 to July 2022). In Windows 10 and 11, the joint optimization constraints (CAGR $\ge$ 5.0%, ENB $\ge$ 3.5, ENC $\ge$ 3.9) yielded an empty feasible set during the 2022 rate shock. Consequently, both QuantEvo and all reference portfolios are evaluated exclusively across the 9 feasible windows to ensure an identical evaluation period. Detailed window evaluations and benchmark definitions are in [EXPERIMENTS.md](./EXPERIMENTS.md).

#### Main Observations:
1. **Risk profile and asset allocation**: QuantEvo exhibits a defensive risk profile (average drawdown of 6.22% and CVaR of 0.0065), driven by substantial allocations to fixed income and short-term Treasuries (~78% across feasible windows). This risk-return profile closely aligns with the World Equity 20% Conservative benchmark (6.36% drawdown, 0.0059 CVaR, 4.49% CAGR).
2. **Constituent vs. factor diversification**: While World Equity 20% achieves its defensive profile by concentrating primarily in a single aggregate bond fund ($\text{ENC} = 1.52$), QuantEvo distributes capital across multiple asset classes ($\text{ENC} = 3.90$, $\text{ENB} = 3.43$). In contrast, while Naive Equal Weight allocates equally across all 7 assets ($\text{ENC} \approx 7.0$), its Meucci ENB drops to 1.47 due to shared equity and credit factor risk.
3. **2008 Financial Crisis**: During the 2008 liquidation (Window 3, July 2008 to July 2010), defensive allocations to US Treasuries contained QuantEvo's maximum drawdown to 11.23%, compared to 28.50% for Naive Equal Weight and 23.48% for the Static Risk Parity Proxy.

Detailed window-by-window analyses, crisis breakdowns (including 2008 and the 2020 COVID crash), and 2022 infeasibility mechanics are documented in [EXPERIMENTS.md](./EXPERIMENTS.md).

---

## 📐 3. Methodology and Mathematical Formulation

### 3.1 Formal MINLP Portfolio Formulation
QuantEvo models portfolio selection as a multi-objective Mixed-Integer Non-Linear Programming problem:

$$\begin{aligned}
\min_{w \in \mathbb{R}^N, \, z \in \{0, 1\}^N} \quad & \mathbf{F}(w) = \left( f_1(w), \dots, f_M(w) \right)^T \\
\text{subject to} \quad & \sum_{i=1}^N w_i = 1 \\
& z_i \cdot w_{\min} \le w_i \le z_i \cdot u_i, \quad \forall i \in \{1, \dots, N\} \\
& k_{\min} \le \sum_{i=1}^N z_i \le k_{\max} \\
& g_j(w) \le 0, \quad \forall j \in \{1, \dots, J\} \\
& z_i \in \{0, 1\}, \quad w_i \ge 0, \quad \forall i \in \{1, \dots, N\}
\end{aligned}$$

where $w_i$ is the weight of asset $i$, $z_i$ is a binary decision variable indicating whether asset $i$ is held, $u_i$ is the individual upper cap, $w_{min}$ is the semi-continuous allocation threshold, and $k_{min}, k_{max}$ are cardinality bounds.

### 3.2 Heuristic Repair with Continuous Projection (Bisection Water-Filling)
Genetic algorithms struggle with MINLP constraints because continuous crossover and mutation operators generate infeasible offspring with probability close to 1. QuantEvo eliminates soft penalty terms via a deterministic repair operator that combines greedy cardinality selection with continuous projection on the simplex (`src/optimization/repair.py`):

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

### 3.5 Effective Number of Bets (Attilio Meucci, 2009)
Nominal weights do not reflect true risk concentration. QuantEvo implements Meucci's orthogonal factor decomposition using Principal Component Analysis (PCA) across bootstrapped covariance matrices:
1. Decompose the covariance matrix: $\Sigma = V \Lambda V^T$, where $V$ contains the orthonormal eigenvectors (orthogonal risk factors) and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$.
2. Calculate the variance attributed to each orthogonal bet:
   $$p_k = \frac{(w^T v_k)^2 \lambda_k}{\sigma_p^2}, \quad \text{where } \sum_{k=1}^N p_k = 1$$
3. Compute the exponential Shannon entropy:
   $$\text{ENB} = \exp\left( -\sum_{k=1}^N p_k \ln p_k \right)$$

When total portfolio risk is concentrated in a single factor, $\text{ENB} = 1$. When risk is evenly distributed across all $N$ orthogonal dimensions, $\text{ENB} = N$. While PCA identifies statistical rather than macroeconomic factors, combining PCA with bootstrap scenario generation captures how factor exposures shift across resampled empirical regimes.

### 3.6 Ledoit-Wolf Optimal Covariance Shrinkage Across Bootstrap Paths
Sample covariance matrices calculated from empirical observations are prone to estimation error and potential ill-conditioning across resampled bootstrap paths.

QuantEvo applies the Ledoit-Wolf shrinkage estimator not just to static in-sample data, but **individually to each of the $B$ bootstrap paths**. For each bootstrap sample path, it shrinks the empirical covariance toward a scaled identity matrix target:
$$\Sigma_{\text{shrunk}} = \delta F + (1 - \delta) \Sigma_{\text{sample}}, \quad F = \frac{\text{tr}(\Sigma_{\text{sample}})}{N} I_N$$

where $\delta \in [0, 1]$ is the analytically optimal shrinkage intensity derived by Ledoit and Wolf (2004) to minimize expected quadratic loss under the Frobenius norm. This guarantees well-conditioned, strictly positive definite covariance matrices $\Sigma^{(b)}$ for every bootstrap scenario.

### 3.7 Capital Dynamics: Buy-and-Hold with Weight Drift
Rather than assuming continuous rebalancing at zero transaction cost, out-of-sample portfolio evaluations model discrete buy-and-hold price drift, tracking real capital allocation drift between rebalancing periods.

### 3.8 Multi-Objective Optimization (NSGA-II)
QuantEvo adapts NSGA-II (`pymoo` backend):
* **Hypervolume Early Stopping**: Tracks the Hypervolume (HV) indicator of the Pareto front across generations. When HV growth stops improving within a specified patience limit, the search terminates early.
* **Knee-Point Selection**: To choose a balanced allocation from the Pareto front, QuantEvo finds the solution maximizing perpendicular distance to the line (in 2D) or plane (in 3D) connecting extreme points of the normalized front. In higher dimensions (or as fallback), it selects the solution minimizing Euclidean distance to the ideal utopia point.
* **Multi-Seed Aggregation**: Runs optimizations across multiple random seeds, combines candidate solutions, and extracts the global non-dominated front.

---

## ⚙️ 4. Software Architecture and Performance

The project is organized into modular components with performance-critical paths accelerated via Numba.

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
* Any metric can serve as a **non-linear constraint** by specifying a threshold and risk measure (percentile or CVaR), automatically normalized as $g(w) \le 0$ for `pymoo`.

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
To test your own asset universe and investment strategy, create a new configuration file (for example, `my_strategy.py`):

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
