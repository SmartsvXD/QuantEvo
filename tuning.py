import src.dataManager as dm
import src.monteCarlo as mt
from src.const import GIORNI_IN_UN_ANNO
import optuna
import numpy as np
from colorama import Fore
import time

giorni = int(5 * GIORNI_IN_UN_ANNO)
nSims = 500

titoli = dm.Titoli()
sim = mt.MonteCarlo()


def valutaLambdaEWMA(lambdaEWMA):
    global titoli
    # Ensure lambdaEWMA is passed to the gradient stats calculation
    (
        rendimentoGiornaliero,
        volatilitàGiornaliera,
        rendimentoAnnuo,
        volatilitàAnnua,
        rendimentoAnnuoDividendi,
        rendimentoGiornalieroDividendi,
        volatilitàDividendi,
        rendimentiAnnoSuccessivo,
        L,
    ) = titoli.calcolaStatsStoricheGradiente(giorni, forzaDownload=0, lambdaEWMA=lambdaEWMA)

    stats = dm.Stats()
    stats.nomi = titoli.stats.nomi

    n_assets = len(titoli.stats.nomi)
    score = np.zeros(n_assets, dtype=np.float64)
    counts = np.zeros(n_assets, dtype=np.int64)

    print()
    print(f"{Fore.MAGENTA}Elaborazione della simulazione dei titoli...{Fore.RESET}")

    t0 = time.time()

    print(f"0/{giorni}", end="")
    for giorno in range(giorni):
        # Use the slice for the specific day
        stats.rendimentoGiornaliero = rendimentoGiornaliero[giorno]
        stats.volatilitàGiornaliera = volatilitàGiornaliera[giorno]
        stats.rendimentoAnnuo = rendimentoAnnuo[giorno]
        stats.volatilitàAnnua = volatilitàAnnua[giorno]
        stats.rendimentoAnnuoDividendi = rendimentoAnnuoDividendi[giorno]
        stats.rendimentoGiornalieroDividendi = rendimentoGiornalieroDividendi[giorno]
        stats.volatilitàDividendi = volatilitàDividendi[giorno]
        stats.L = L

        risultati = sim.simulaTitoli(stats, nSims=500, verbose=0)
        target = rendimentiAnnoSuccessivo[giorno]

        # Mask NaN values: only accumulate errors where both target and results are finite
        valid = (~np.isnan(target)) & (~np.isnan(risultati))

        if not np.any(valid):
            # niente da accumulare per questo giorno
            print(f"Giorno {giorno}: nessun dato valido (tutti NaN), salto")
            continue

        diff = target - risultati
        # accumulate only on valid positions
        score[valid] += (diff[valid]) ** 2
        counts[valid] += 1

        print(f"\r{giorno}/{giorni}", end="")

    simTime = time.time() - t0
    print()
    print(f"{Fore.MAGENTA}Fatto!{Fore.RESET} ({simTime:.2f}s)")

    # Calcola RMSE per asset usando i conteggi effettivi
    rmse = np.full(n_assets, np.nan, dtype=np.float64)
    nonzero = counts > 0
    rmse[nonzero] = np.sqrt(score[nonzero] / counts[nonzero])

    # Stampa diagnostica compatta
    print("RMSE per asset:", rmse)

    # Aggrega in uno score scalare per l'ottimizzazione (mean ignorando NaN)
    if np.all(~nonzero):
        # nessun dato valido in tutto il run: ritorna valore grande
        return 1e6
    aggregated = float(np.nanmean(rmse))
    print("Score aggregato (mean RMSE):", aggregated)
    return aggregated


def objective(trial):
    lambdaEWMA = [trial.suggest_float(i, 0.5, 0.999) for i in titoli.stats.nomi]

    score = valutaLambdaEWMA(lambdaEWMA)

    trial.report(score, step=0)
    # possibilità di pruning future (semplice)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return score


study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=40, timeout=3600)
print("Best params:", study.best_params)
print("Best score:", study.best_value)
