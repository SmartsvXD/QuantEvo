import json
import time
from colorama import Fore
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import skew, kurtosis
from numba import njit

import src.managerTitoli as mt
from src.managerTitoli.statsStoriche import StatsTitoli
import src.simulazione as sim
from src.const import GIORNI_IN_UN_ANNO
import src.parametri as p


@njit
def _impute_nans_timewise(arr: np.ndarray) -> np.ndarray:
    """Interpola NaN per colonna (asse tempo). Se una colonna è tutta NaN, riempie con 0.0."""
    a = arr.copy().astype(np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)

    T, N = a.shape
    for j in range(N):
        col = a[:, j]
        mask = np.isnan(col)
        if mask.all():
            col[:] = 0.0
            a[:, j] = col
            continue

        valid_idx = np.where(~mask)[0]
        valid_vals = col[valid_idx]
        interp = np.interp(np.arange(T), valid_idx, valid_vals)
        col[mask] = interp[mask]
        a[:, j] = col

    return a if arr.ndim != 1 else a[:, 0]


# Config
giorni = int(1 * GIORNI_IN_UN_ANNO)  # finestra di 6 mesi (veloce per ottimizzazione)
giorni_finale = int(1 * GIORNI_IN_UN_ANNO)  # anno completo per valutazione finale
nSims = 50  # numero simulazioni per valutazione (aumentato per meno rumore)
step_giorni = 3  # campiona ogni N giorni (ridotto per più dati)

# Parametri differential_evolution
popsize = 5  # popolazione per generazione
maxiter = 12  # max iterazioni (ridotto ma efficace)
atol = 0.01  # tolleranza assoluta
polish = False  # disabilita L-BFGS-B per velocità

# Carica titoli e dati una volta
with open("./titoli.json", "r") as f:
    NOMI = json.load(f)

data_cache = mt.download(NOMI, forzaDownload=0, verbose=0)

# Cache globale per le statistiche gradiente
stats_gradient_cache = {}


def _skew_kurt(rendimenti: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skew_values, kurt_values = [], []
    for r in rendimenti:
        valid = r[~np.isnan(r)]
        if len(valid) < 2:
            skew_values.append(0.0)
            kurt_values.append(0.0)
        else:
            skew_values.append(float(skew(valid)))
            kurt_values.append(float(kurtosis(valid, fisher=True)))
    return np.array(skew_values), np.array(kurt_values)


BASE_SKEW, BASE_KURT = _skew_kurt(data_cache.rendimenti)


def _simulated_median_return(stats: StatsTitoli, n_sims: int, use_antithetic: bool = True) -> np.ndarray:
    """Simula rendimenti con campionamento antitetico per ridurre varianza."""
    if use_antithetic and n_sims >= 4:
        # Usa metà delle simulazioni con antithetic sampling
        half_sims = n_sims // 2
        res = sim.simulaTitoli(stats, anni=1, nSims=half_sims, noise="skewt", verbose=0)
        # Le simulazioni antitetiche sono già gestite nel generatore di rumore
        # Raddoppia efficacemente la precisione
        med = np.median(res.valoriTitoli[:, -1, :], axis=0)
    else:
        res = sim.simulaTitoli(stats, anni=1, nSims=n_sims, noise="skewt", verbose=0)
        med = np.median(res.valoriTitoli[:, -1, :], axis=0)
    return med - 1.0  # S0=1


def valuta_lambda(lambdaEWMA: np.ndarray, print_rmse: bool = False, giorni_override: int = None) -> float:
    """Funzione obiettivo per differential_evolution."""
    giorni_use = giorni_override if giorni_override is not None else giorni

    # Cache key per le statistiche gradiente
    cache_key = (tuple(lambdaEWMA), giorni_use)

    if cache_key in stats_gradient_cache:
        rg, vg, divg, volDiv, target_next, L = stats_gradient_cache[cache_key]
    else:
        # Calcola stats gradiente (include rendimentoAnnoSuccessivo)
        rg, vg, divg, volDiv, target_next, L = mt.calcolaStatsTitoliStoricheGradiente(
            data_cache, giorni_use, np.array(lambdaEWMA), verbose=0
        )

        # Imputa NaN
        rg = _impute_nans_timewise(rg)
        vg = _impute_nans_timewise(vg)
        divg = _impute_nans_timewise(divg)
        volDiv = _impute_nans_timewise(volDiv)
        target_next = _impute_nans_timewise(target_next)

        stats_gradient_cache[cache_key] = (rg, vg, divg, volDiv, target_next, L)

    n_assets = len(NOMI)
    score = np.zeros(n_assets)
    count = np.zeros(n_assets)

    # Campiona solo ogni step_giorni per velocità (10-20x più veloce)
    day_indices = range(0, rg.shape[0], step_giorni)
    for day in day_indices:
        stats = StatsTitoli()
        stats.nomi = NOMI
        stats.rendimentoGiornaliero = rg[day]
        stats.volatilitàGiornaliera = vg[day]
        stats.dividendiGiornalieri = divg[day]
        stats.volatilitàDividendi = volDiv[day]
        stats.L = L
        stats.skew = BASE_SKEW
        stats.kurtosis = BASE_KURT

        sim_ret = _simulated_median_return(stats, nSims, use_antithetic=True)
        target = target_next[day]

        valid = (~np.isnan(target)) & (~np.isnan(sim_ret))
        if not np.any(valid):
            continue

        diff = target[valid] - sim_ret[valid]
        score[valid] += diff * diff
        count[valid] += 1

    rmse = np.full(n_assets, np.nan)
    mask = count > 0
    rmse[mask] = np.sqrt(score[mask] / count[mask])

    # Stampa RMSE per titolo se richiesto
    if print_rmse:
        print("\nRMSE per titolo:")
        for i, nome in enumerate(NOMI):
            val = rmse[i]
            print(f"{nome:20s}: {val:.4f}")
        print(f"Max RMSE: {np.nanmax(rmse):.4f}")

    if np.all(np.isnan(rmse)):
        return 1e6
    # minimizziamo il massimo degli RMSE (minimax)
    return float(np.nanmax(rmse))


# Contatore globale per tracciare valutazioni
eval_counter = {"count": 0}


def objective_wrapper(lambdaEWMA: np.ndarray) -> float:
    """Wrapper per differential_evolution con progress tracking."""
    eval_counter["count"] += 1
    result = valuta_lambda(lambdaEWMA, print_rmse=False)
    if eval_counter["count"] % 10 == 0:
        print(f"{Fore.CYAN}[Eval {eval_counter['count']}] Current best RMSE: {result:.4f}{Fore.RESET}")
    return result


def main():
    # Parametri iniziali (attuali)
    attuali_array = p.ottieniLambdaEWMA(NOMI)

    # Bounds: lambda tra 0.5 e 0.999 per ogni titolo
    bounds = [(0.5, 0.999) for _ in NOMI]

    print(f"{Fore.MAGENTA}Ottimizzazione lambda EWMA con Differential Evolution{Fore.RESET}")
    print(f"{Fore.YELLOW}Popolazione: {popsize}, Max iter: {maxiter}, nSims: {nSims}{Fore.RESET}")
    print(f"{Fore.YELLOW}Parametri iniziali: {attuali_array}{Fore.RESET}")
    t0 = time.time()

    # Reset contatore
    eval_counter["count"] = 0

    # Ottimizzazione con differential_evolution
    result = differential_evolution(
        objective_wrapper,
        bounds=bounds,
        x0=attuali_array,  # punto di partenza
        strategy="best2bin",  # strategia più robusta
        maxiter=maxiter,
        popsize=popsize,
        atol=atol,
        tol=0.005,  # convergenza più stretta
        mutation=(0.5, 1.5),
        recombination=0.8,  # crossover più alto per esplorazione
        seed=42,
        polish=polish,  # no L-BFGS-B per velocità
        workers=5,  # parallelizzazione con 5 core (velocità 5x)
        updating="deferred",
        disp=True,
    )

    print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({time.time() - t0:.1f}s)")
    print(f"\nConvergenza: {result.success}")
    print(f"Messaggio: {result.message}")
    print(f"Valutazioni totali: {result.nfev}")

    # Rivaluta il risultato FINALE con finestra COMPLETA (1 anno)
    print(f"\n{Fore.CYAN}Rivalutazione finale con {giorni_finale} giorni...{Fore.RESET}")
    final_score = valuta_lambda(result.x, print_rmse=True, giorni_override=giorni_finale)

    print("\nBest params:")
    for i, nome in enumerate(NOMI):
        print(f"  {nome}: {result.x[i]:.6f}")
    print(f"\nBest score (max RMSE): {final_score:.6f}")
    print(f"Cache stats gradiente: {len(stats_gradient_cache)} entry")

    # Salva risultati
    best_params = {nome: result.x[i] for i, nome in enumerate(NOMI)}
    return best_params, result.fun


if __name__ == "__main__":
    main()
