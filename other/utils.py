import exchange_calendars as xcals
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick




def compute_buy_and_hold(port, window_idx, out_of_sample, in_sample=None):
    if in_sample is not None:
        R_is = in_sample[window_idx].get(bootstrap=False)["returns"][0]
        R_oos = out_of_sample[window_idx].get(bootstrap=False)["returns"][0]
        R = np.concatenate([R_is, R_oos], axis=1)
    else:
        R = out_of_sample[window_idx].get(bootstrap=False)["returns"][0]
    asset_wealth = np.cumprod(1.0 + R, axis=1)
    portfolio_value = port @ asset_wealth
    return portfolio_value - 1.0



def compute_rolling_cvar(port, cvar_metric, window_idx, out_of_sample, in_sample=None, rolling_window=63):
    if in_sample is not None:
        R_is = in_sample[window_idx].get(bootstrap=False)["returns"][0]
        R_oos = out_of_sample[window_idx].get(bootstrap=False)["returns"][0]
        R = np.concatenate([R_is, R_oos], axis=1)
    else:
        R = out_of_sample[window_idx].get(bootstrap=False)["returns"][0]
    cvar_values = []
    for t in range(rolling_window, R.shape[1] + 1):
        r_slice = np.expand_dims(R[:, t - rolling_window : t], axis=0)
        val = cvar_metric._calculate(port, tensors={"returns": r_slice})[0, 0]
        cvar_values.append(val)
    return np.array(cvar_values)


def display_and_save(
    n_windows,
    periods,
    weights,
    out_of_sample,
    reference_portfolios,
    reference_colors,
    output_dir,
    compute_metric,
    alpha,
    metric,
    in_sample=None,
    rolling_window=63,
    calendar_name="XNYS"
):
    calendar = xcals.get_calendar(calendar_name, start="1990-01-01")


    for w in range(n_windows):
        split_date = periods[w]["OutOfSampleStartDate"]
        end_date = periods[w]["OutOfSampleEndDate"]

        if in_sample is not None and "InSampleStartDate" in periods[w]:
            start_date = periods[w]["InSampleStartDate"]
            x = calendar.sessions_in_range(start_date, end_date).tz_localize(None)
            split_dt = pd.to_datetime(split_date)
        else:
            start_date = split_date
            x = calendar.sessions_in_range(start_date, end_date).tz_localize(None)
            split_dt = None

        port = np.array([weights[w][k] for k in weights[w] if k != "Window"])
        total_weight = port.sum()

        fig, (ax_ret, ax_cvar) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        if total_weight > 0:
            y_gen = compute_buy_and_hold(port.round(3), out_of_sample=out_of_sample, in_sample=in_sample, window_idx=w)
            min_len = min(len(x), len(y_gen))
            x_plot = x[-min_len:]
            y_gen_plot = y_gen[-min_len:]
            ax_ret.plot(x_plot, y_gen_plot, label="Generated Portfolio", color="blue", linewidth=2)

            cvar_gen = compute_metric(
                port.round(3), metric, window_idx=w, rolling_window=rolling_window, out_of_sample=out_of_sample, in_sample=in_sample
            )
            cvar_len = min(len(x_plot) - rolling_window + 1, len(cvar_gen))
            x_cvar = x_plot[rolling_window - 1 :][:cvar_len]
            ax_cvar.plot(x_cvar, cvar_gen[:cvar_len], label="Generated Portfolio", color="blue", linewidth=2)
        else:
            min_len = len(x)
            x_plot = x

        for name, bench_weights in reference_portfolios.items():
            y_bench = compute_buy_and_hold(bench_weights, out_of_sample=out_of_sample, in_sample=in_sample, window_idx=w)
            curr_len = min(len(x_plot), len(y_bench))
            ax_ret.plot(
                x_plot[-curr_len:], y_bench[-curr_len:], label=name, color=reference_colors[name], linestyle="--"
            )

            cvar_bench = compute_metric(
                bench_weights, metric, window_idx=w, rolling_window=rolling_window, out_of_sample=out_of_sample, in_sample=in_sample
            )
            curr_cvar_len = min(len(x_plot) - rolling_window + 1, len(cvar_bench))
            x_bench_cvar = x_plot[rolling_window - 1 :][:curr_cvar_len]
            ax_cvar.plot(
                x_bench_cvar, cvar_bench[:curr_cvar_len], label=name, color=reference_colors[name], linestyle="--"
            )

        if split_dt is not None:
            for ax in (ax_ret, ax_cvar):
                ax.axvline(x=split_dt, color="black", linestyle=":", linewidth=1.5, alpha=0.8, label="IS / OOS Split" if ax == ax_ret else None)
                ax.axvspan(x_plot[0], split_dt, color="gray", alpha=0.08)
                ax.axvspan(split_dt, x_plot[-1], color="blue", alpha=0.03)

        ax_ret.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        if split_dt is not None:
            title_text = f"Window {w + 1} Performance: In-Sample ({start_date} to {periods[w]['InSampleEndDate']}) vs Out-of-Sample ({split_date} to {end_date})"
        else:
            title_text = f"Window {w + 1} Out-of-Sample Performance ({start_date} to {end_date})"
        ax_ret.set_title(title_text, fontsize=12, fontweight="bold")
        ax_ret.set_ylabel("Cumulative Return")
        ax_ret.grid(True, linestyle="--", alpha=0.5)
        ax_ret.legend(loc="best")

        ax_cvar.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
        ax_cvar.set_ylabel(f"Rolling {metric.__class__.__name__} ({int(alpha * 100)}%)")
        ax_cvar.set_xlabel("Date")
        ax_cvar.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

        fig.savefig(f"{output_dir}/images/window_{w + 1}_performance.png", dpi=300)

def save_metrics(metrics, n_windows, config, reference_portfolios, in_sample, out_of_sample, output_dir):
    metrics_rp = []

    for i in range(len(metrics)):
        metrics_rp.append(
            {
                "Window": metrics[i]["Window"],
                "Portfolio": "Generated Portfolio",
                "Metric": metrics[i]["Metric"],
                "InSample": metrics[i]["InSample"],
                "OutOfSample": metrics[i]["OutOfSample"],
                "Difference": (metrics[i]["OutOfSample"] - metrics[i]["InSample"] if metrics[i]["InSample"] is not None or metrics[i]["OutOfSample"] is not None else None),
            }
        )

    for w in range(n_windows):
        for reference_name, reference_perc in reference_portfolios.items():
            for func in config.objective_functions + config.constraints:
                is_value = func.calculate(reference_perc, in_sample[w], raw_scale=True, do_bootstrap=False)
                oos_value = func.calculate(reference_perc, out_of_sample[w], raw_scale=True, do_bootstrap=False)

                metrics_rp.append(
                    {
                        "Window": w + 1,
                        "Portfolio": reference_name,
                        "Metric": func.__class__.__name__,
                        "InSample": is_value,
                        "OutOfSample": oos_value,
                        "Difference": oos_value - is_value,
                    }
                )

        print(f"Metrics for Window {w + 1}:")
        print(pd.DataFrame(m for m in metrics_rp if m["Window"] == w + 1))
        print("\n")

    pd.DataFrame(metrics_rp).to_csv(f"{output_dir}/metrics_comparison.csv", index=False)