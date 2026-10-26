# 🔬 QuantEvo: Example Strategies and Walk-Forward Benchmark Analysis

Walk-forward backtest results across 25 years (2001 to 2026) for two sample strategies (Example A and Example B), evaluating out-of-sample window performance, crisis behavior, and portfolio weight stability across market regimes.

---

## 📅 1. Experimental Setup and Walk-Forward Protocol

### 1.1 Walk-Forward Mechanics
Following the rolling walk-forward protocol described in [README.md](./README.md#2-walk-forward-backtest-examples), portfolios are optimized and evaluated without lookahead bias:
* **In-Sample (IS) Window**: 3 trading years (756 trading days) for statistics estimation, Moving Block Bootstrap, and NSGA-II optimization.
* **Out-of-Sample (OOS) Window**: 2 trading years (504 trading days) evaluated in buy-and-hold execution without parameter updating.
* **Rolling Step**: 2 trading years, yielding disjoint consecutive out-of-sample periods.

| Window | In-Sample Period | Out-of-Sample Period |
| :--- | :--- | :--- |
| **Window 1** | July 2001 to July 2004 | July 2004 to July 2006 |
| **Window 2** | July 2003 to July 2006 | July 2006 to July 2008 |
| **Window 3** | July 2005 to July 2008 | July 2008 to July 2010 |
| **Window 4** | July 2007 to July 2010 | July 2010 to July 2012 |
| **Window 5** | July 2009 to July 2012 | July 2012 to July 2014 |
| **Window 6** | July 2011 to July 2014 | July 2014 to July 2016 |
| **Window 7** | July 2013 to July 2016 | July 2016 to July 2018 |
| **Window 8** | July 2015 to July 2018 | July 2018 to July 2020 |
| **Window 9** | July 2017 to July 2020 | July 2020 to July 2022 |
| **Window 10** | July 2019 to July 2022 | July 2022 to July 2024 |
| **Window 11** | July 2021 to July 2024 | July 2024 to July 2026 |

### 1.2 Data History and Asset Proxies
To ensure 25 years of continuous daily total returns without synthetic splicing, the backtests use historical Vanguard index mutual funds (`src/data/download.py`), which predate modern ETFs and track identical market-cap universes:
* `VTSMX`: Vanguard Total Stock Market Index (VTI equivalent).
* `VTMGX`: Vanguard Developed Markets Index (VEA equivalent).
* `VEIEX`: Vanguard Emerging Markets Stock Index (VWO equivalent).
* `VBMFX`: Vanguard Total Bond Market Index (BND equivalent).
* `VUSTX`: Vanguard Long-Term Treasury Fund (VGLT equivalent).
* `VGSIX`: Vanguard Real Estate Index (VNQ equivalent).
* `VFISX`: Vanguard Short-Term Treasury Index (VGSH equivalent).

### 1.3 Benchmark Portfolios
All tests compare QuantEvo against standard static asset allocation baselines (evaluated as fixed buy-and-hold strategies):
* **Naive Equal Weight (EW)**: Equal capital weighting across all candidate assets ($w_i = 1/N$).
* **Equal Weight Core**: Equal capital weighting across core equity and bond assets (`VTSMX`, `VTMGX`, `VUSTX`, `VGSIX`, `VFISX`).
* **Balanced 40/60 (Long Treasury)**: 40% global equities (30% US, 10% Developed) and 60% long-term US Treasuries (`VUSTX`).
* **Classic 40/60 US-Centric**: 40% US equities (`VTSMX`), 60% broad US bonds (`VBMFX`).
* **Static Risk Parity Proxy / Static ERC Proxy**: Fixed multi-asset strategic allocations reflecting risk parity asset-class diversification heuristics rather than dynamic variance scaling (in Exp B: 20% US equities, 5% broad bonds, 25% long treasuries, 25% REITs, 25% short treasuries; in Exp A: 12% US equity, 10% developed equity, 8% emerging equity, 70% total bonds).
* **World Equity Conservative**: 20% global equities, 80% fixed income.

---

## 🛡️ 2. Example B: 7-Asset Defensive ("No-Fall") Strategy

### 2.1 Problem Setup
* **Candidate Assets**: `VTSMX`, `VTMGX`, `VEIEX`, `VBMFX`, `VUSTX`, `VGSIX`, `VFISX`.
* **Objective Functions**: Minimize daily CVaR (95%) and minimize Maximum Drawdown.
* **Optimization Constraints**:
  * $\text{CAGR} \ge 4.0\%$ (annualized return floor).
  * $\text{ENB} \ge 3.5$ (Meucci factor diversification floor).
  * $\text{ENC} \ge 3.8$ (asset concentration floor).
  * Minimum position size: 5% (semi-continuous condition $w_i \in \{0\} \cup [0.05, u_i]$).
  * Cardinality: between 3 and 6 assets.
  * Individual caps: US Equity 70%, Dev Equity 40%, Emerging 15%, Broad Bonds 80%, Long Treasuries 25%, REITs 15%, Short Treasuries 40%.

### 2.2 Comprehensive Out-of-Sample Performance (Feasible Windows 1 to 9, 2004 to 2022)

| Portfolio Strategy | Mean CAGR | Mean Daily CVaR (95%) | Mean Max Drawdown | Mean ENB | Mean ENC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QuantEvo (Generated)** | **4.31%** | **0.0060** | **5.80%** | **3.60** | **3.80** |
| World Equity 20% Conservative | 4.49% | 0.0059 | 6.36% | 4.39 | 1.52 |
| Balanced 40/60 (Long Treasury) | 7.07% | 0.0116 | 10.28% | 3.30 | 2.17 |
| Static Risk Parity Proxy | 6.45% | 0.0131 | 11.36% | 2.17 | 4.35 |
| Naive Equal Weight | 6.39% | 0.0147 | 13.72% | 1.47 | 7.00 |
| Equal Weight Core | 6.62% | 0.0155 | 13.48% | 1.65 | 5.00 |

*\*Sampling and Benchmark Note*: 
All portfolio metrics represent the out-of-sample average over the identical 9 feasible windows (July 2004 to July 2022). In Windows 10 and 11, the joint optimization constraints (CAGR $\ge$ 4.0%, ENB $\ge$ 3.5, ENC $\ge$ 3.8) yielded an empty feasible set during the 2022 bond-equity correlation breakdown. Consequently, both QuantEvo and all reference portfolios are evaluated exclusively across the 9 feasible windows to ensure an identical evaluation period.
* **Benchmark Context**: The primary risk baseline is **World Equity 20% Conservative**. While World Equity achieves slightly higher CAGR (4.49% vs 4.31%) and Meucci ENB (4.39 vs 3.60) by allocating 80% to aggregate bonds, QuantEvo trades off return and factor entropy to satisfy the multi-asset nominal constraint ($\text{ENC} = 3.80$ vs $1.52$), containing drawdown to 5.80% vs 6.36%.

### 2.3 Window-by-Window Performance Breakdown

The table below reports QuantEvo's out-of-sample metrics for each walk-forward window:

| Window | OOS Period | CAGR | Daily CVaR (95%) | Max Drawdown | Meucci ENB | Inverse Herfindahl (ENC) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Window 1** | 2004 to 2006 | 5.25% | 0.0046 | 2.50% | 3.86 | 3.80 |
| **Window 2** | 2006 to 2008 | 6.44% | 0.0050 | 2.95% | 2.42 | 3.80 |
| **Window 3** | 2008 to 2010 | 2.89% | 0.0093 | 10.34% | 3.33 | 3.80 |
| **Window 4** | 2010 to 2012 | 6.43% | 0.0051 | 3.21% | 3.34 | 3.80 |
| **Window 5** | 2012 to 2014 | 5.06% | 0.0047 | 3.79% | 3.55 | 3.80 |
| **Window 6** | 2014 to 2016 | 4.08% | 0.0049 | 3.10% | 4.00 | 3.80 |
| **Window 7** | 2016 to 2018 | 2.34% | 0.0047 | 3.49% | 4.33 | 3.80 |
| **Window 8** | 2018 to 2020 | 9.74% | 0.0084 | 9.17% | 3.94 | 3.81 |
| **Window 9** | 2020 to 2022 | -3.42% | 0.0079 | 13.62% | 3.60 | 3.80 |
| **Window 10** | 2022 to 2024 | Infeasible | Infeasible | Infeasible | Infeasible | Infeasible |
| **Window 11** | 2024 to 2026 | Infeasible | Infeasible | Infeasible | Infeasible | Infeasible |

---

### 2.4 Visual Analysis: Key Historical Regimes (Example B)

Each plot below displays two panels:
* **Top Panel**: Cumulative buy-and-hold portfolio return across the 3-year In-Sample training block and the 2-year Out-of-Sample testing block, divided by the vertical split line.
* **Bottom Panel**: Evolution of the Rolling Daily CVaR (95%) computed on a 63-day trading window (one financial quarter).

#### Window 1 (2004 to 2006): Economic Expansion and Rate Normalization
The optimizer anchored risk by concentrating capital in short-duration paper (`VFISX`) and broad bonds (`VBMFX`), dampening tail risk while capturing baseline returns during policy rate normalization.

![Window 1 Performance](./other/docs_data/exp_b/images/window_1_performance.png)

#### Window 3 (2008 to 2010): 2008 Financial Crisis
During the 2008 systemic liquidation, flight-to-quality allocations to US Treasuries insulated the portfolio against equity market losses:
* Naive Equal Weight fell **28.64%** and the Static Risk Parity Proxy fell **23.48%**.
* QuantEvo contained maximum drawdown to **10.34%**, with daily CVaR peaking at 0.0093 (vs. 0.0282 for Naive Equal Weight).

![Window 3 Performance](./other/docs_data/exp_b/images/window_3_performance.png)

#### Window 5 (2012 to 2014): Post-Crisis Recovery
During the post-crisis expansion, balanced allocations across equities and fixed income captured market upside while rolling CVaR remained compressed throughout the out-of-sample period.

![Window 5 Performance](./other/docs_data/exp_b/images/window_5_performance.png)

#### Window 8 (2018 to 2020): March 2020 Liquidity Shock
During the March 2020 market crash, convex duration gains in long-term Treasuries (`VUSTX`) effectively counterbalanced equity losses, limiting maximum drawdown to 9.17% (compared to 18.63% for Naive Equal Weight).

![Window 8 Performance](./other/docs_data/exp_b/images/window_8_performance.png)

#### Window 9 (2020 to 2022): Inflation Shock and Rate Repricing
In anticipation of policy rate tightening, the optimizer reduced portfolio duration by shifting into short-term Treasuries and broad bonds, maintaining factor dispersion under shifting yield curve dynamics.

![Window 9 Performance](./other/docs_data/exp_b/images/window_9_performance.png)

### 2.5 Infeasibility in Windows 10 and 11: Feasible Space Analysis
In Windows 10 (IS 2019 to 2022) and 11 (IS 2021 to 2024), Example B identified no candidate solution satisfying the joint constraint set:

$$
\mathcal{F} = \left\{ w \in \Delta^{N-1} \;\middle|\; \text{CAGR}(w) \ge 4.0\%, \; \text{ENB}(w) \ge 3.5, \; \text{ENC}(w) \ge 3.8, \; k_{\min} \le \|w\|_0 \le k_{\max} \right\} = \emptyset
$$

The 2022 global monetary tightening caused simultaneous contractions in equity and fixed-income assets, temporarily eliminating the negative stock-bond correlation. Under these empirical returns, achieving $\text{CAGR} \ge 4.0\%$ required concentrating capital in select outperforming equities, which mutually excluded the minimum factor diversification ($\text{ENB} \ge 3.5$) and constituent count ($\text{ENC} \ge 3.8$) bounds. Consequently, the hard constraints over-constrained the problem, yielding an empty feasible set $\mathcal{F} = \emptyset$. The diagnostic engine explicitly isolated the binding bottlenecks, logging the closest attainable values across the population.

By comparison, **Example A** maintained feasible solutions across both windows (OOS CAGRs of 6.90% in Window 10 and 12.53% in Window 11) because its diversification floors ($\text{ENB} \ge 1.35$, $\text{ENC} \ge 2.0$) were compatible with the empirical joint distribution of that cycle.

---

## 📈 3. Example A: 4-Asset Balanced Core Strategy

### 3.1 Problem Setup
* **Candidate Assets**: `VTSMX` (US Stock), `VTMGX` (Dev Equity), `VEIEX` (Emerging Equity), `VBMFX` (Total Bond).
* **Objective Functions**: Maximize CAGR and minimize daily CVaR (95%).
* **Optimization Constraints**:
  * Omega Ratio $\ge 1.10$ (hurdle rate = 0%).
  * Effective Number of Constituents ($\text{ENC} \ge 2.0$).
  * Effective Number of Bets ($\text{ENB} \ge 1.35$).
  * Minimum position size: 5%.
  * Cardinality: between 3 and 4 assets.
  * Individual caps: US Stock 70%, Developed 50%, Emerging 30%, Total Bond 40%.

### 3.2 Summary of Out-of-Sample Performance (Feasible Windows, 2004 to 2026)

| Portfolio Strategy | Mean CAGR | Mean Daily CVaR (95%) | Mean Omega | Mean ENB | Mean ENC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QuantEvo (Generated)** | **7.13%** | **0.0149** | **2.22** | **1.55** | **2.71** |
| Naive Equal Weight | 8.11% | 0.0189 | 2.06 | 1.07 | 4.00 |
| Classic 40/60 US-Centric | 6.30% | 0.0109 | 2.27 | 2.33 | 1.92 |
| Static ERC Proxy | 5.09% | 0.0082 | 2.08 | 2.12 | 1.92 |
| World Equity 40% | 5.94% | 0.0104 | 2.17 | 1.87 | 2.31 |

*\*Note*: Metrics represent the out-of-sample average over the 10 feasible windows for both QuantEvo and the reference portfolios (Window 4, 2010 to 2012, was infeasible under joint Omega and ENB constraints).

### 3.3 Window-by-Window Performance Breakdown (Example A)

| Window | OOS Period | CAGR | Daily CVaR (95%) | Omega Ratio | Meucci ENB | Inverse Herfindahl (ENC) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Window 1** | 2004 to 2006 | 11.38% | 0.0102 | 2.75 | 1.52 | 3.34 |
| **Window 2** | 2006 to 2008 | 6.24% | 0.0148 | 1.55 | 1.48 | 2.88 |
| **Window 3** | 2008 to 2010 | -1.61% | 0.0276 | 0.98 | 1.31 | 2.64 |
| **Window 4** | 2010 to 2012 | Infeasible | Infeasible | Infeasible | Infeasible | Infeasible |
| **Window 5** | 2012 to 2014 | 14.08% | 0.0102 | 3.76 | 1.84 | 2.16 |
| **Window 6** | 2014 to 2016 | 4.21% | 0.0122 | 1.58 | 1.80 | 2.30 |
| **Window 7** | 2016 to 2018 | 9.31% | 0.0105 | 3.51 | 1.67 | 2.63 |
| **Window 8** | 2018 to 2020 | 7.00% | 0.0222 | 1.55 | 1.31 | 2.74 |
| **Window 9** | 2020 to 2022 | 1.31% | 0.0155 | 1.16 | 1.60 | 2.39 |
| **Window 10** | 2022 to 2024 | 6.90% | 0.0118 | 1.64 | 1.53 | 3.24 |
| **Window 11** | 2024 to 2026 | 12.53% | 0.0143 | 3.77 | 1.45 | 2.83 |

---

### 3.4 Visual Analysis: Key Historical Regimes (Example A)

#### Window 2 (2006 to 2008): Pre-Crisis Equity Cycle
The optimizer rotated equity exposure toward Developed Markets (`VTMGX`) to capture international outperformance, while fully saturating the 40.0% cap on Total Bonds (`VBMFX`).

![Window 2 Performance](./other/docs_data/exp_a/images/window_2_performance.png)

#### Window 3 (2008 to 2010): 2008 Credit Contraction
Because bond exposure was capped at 40%, mandatory equity allocations exposed the portfolio to the systemic 2008 downturn, limiting downside protection compared to the less-constrained Example B setup.

![Window 3 Performance](./other/docs_data/exp_a/images/window_3_performance.png)

#### Window 7 (2016 to 2018): Low-Volatility Expansion
During synchronized global economic growth, equity allocations generated high risk-adjusted efficiency (Omega = 3.51) under muted volatility.

![Window 7 Performance](./other/docs_data/exp_a/images/window_7_performance.png)

#### Window 11 (2024 to 2026): Large-Cap Equity Cycle
Equity exposure tilted toward US large caps captured market momentum while the saturated bond anchor stabilized rolling portfolio volatility.

![Window 11 Performance](./other/docs_data/exp_a/images/window_11_performance.png)

---

## ⚖️ 4. Portfolio Weight Stability Across Windows

A critical test for any quantitative allocation engine is weight stability across market regimes. Stable portfolio allocations ensure that the strategy is not excessively sensitive to small changes in estimation windows.

### Example B Weight Distribution (7 Assets, Feasible Windows 1 to 9)

| Asset Ticker | Description | Mean Weight | Standard Deviation | Min Weight | Max Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `VFISX` | Short-Term US Treasury | **34.82%** | 2.96% | 28.37% | 37.93% |
| `VBMFX` | Total US Bond Market | **29.42%** | 1.57% | 25.51% | 31.11% |
| `VUSTX` | Long-Term US Treasury | **16.02%** | 2.05% | 12.34% | 18.52% |
| `VTSMX` | Total US Stock Market | **11.28%** | 6.96% | 0.00% | 24.30% |
| `VTMGX` | Developed World Equity | **6.11%** | 7.08% | 0.00% | 24.33% |
| `VEIEX` | Emerging Market Equity | **1.61%** | 3.20% | 0.00% | 9.51% |
| `VGSIX` | Real Estate Investment Trusts | **0.74%** | 2.08% | 0.00% | 6.63% |

### Example A Weight Distribution (4 Assets, Feasible Windows)

The table below reports the normalized weight distribution across the 10 feasible windows (Windows 1 to 3 and 5 to 11), summing to 100.00%:

| Asset Ticker | Description | Mean Weight (Active) | Standard Deviation | Min Weight (Overall / Active) | Max Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `VBMFX` | Total US Bond Market | **40.00%** | 0.00% | 0.00% / 40.00% | 40.00% |
| `VTSMX` | Total US Stock Market | **31.86%** | 16.16% | 0.00% / 6.79% | 54.89% |
| `VTMGX` | Developed World Equity | **21.04%** | 16.44% | 0.00% / 0.00% | 49.74% |
| `VEIEX` | Emerging Market Equity | **7.10%** | 10.47% | 0.00% / 0.00% | 30.00% |

*\*Note on Example A Weights*: Metrics reflect the 10 feasible windows where allocations sum to 100.00%. In Window 4 (2010 to 2012), no portfolio satisfied joint Omega and ENB constraints; computing unconditioned averages across all 11 windows scales active mean weights by a factor of 10/11 (yielding an unallocated 0.00% minimum across all assets).

#### Observations on Weight Stability:
1. **Defensive Anchors**: In Example B, short-term treasuries (`VFISX`) and broad bonds (`VBMFX`) consistently absorb roughly 64% of total portfolio capital with very small standard deviations (2.96% and 1.57%). This shows that the optimizer does not radically flip asset classes between cycles.
2. **Active Binding Constraints**: In Example A, the zero standard deviation of `VBMFX` is explained directly by its active upper cap of 40.00% (`examples/example_a.py`). The optimizer repeatedly saturates this cap to harvest fixed-income stability, using the remaining 60% budget to rotate dynamically among equity regions.

---

## 💡 5. Synthesis and Practical Takeaways

The walk-forward backtest examples highlight three quantitative properties of the framework:

1. **Decoupling of Nominal and Factor Diversification**: Equal capital allocation ($ENC \approx 7.0$) repeatedly failed to provide downside risk mitigation during market shocks due to shared factor exposure ($ENB \approx 1.1\text{–}1.4$). Incorporating Meucci's entropy metric explicitly forced allocations into uncorrelated components, reducing maximum drawdowns to 5.80% (Example B).
2. **Allocation Regularization**: Applying Ledoit-Wolf shrinkage individually to each bootstrap path stabilized the empirical covariance estimation, preventing corner solutions and producing stable asset class weights across rolling windows without explicit turnover penalties.
3. **Sensitivity to Correlation Breakdown**: Hard multi-objective constraints bounded tail risk effectively during historical equity crashes, but produced an empty feasible set during the simultaneous stock-bond decline of 2022. Maintaining feasibility across structural regime breaks requires adaptive or regime-aware constraint thresholds, as shown by the continuous feasibility of Example A.
