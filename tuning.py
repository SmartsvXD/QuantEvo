import src.dataManager as dm
import src.monteCarlo as mt
from src.const import GIORNI_IN_UN_ANNO
import optuna
import numpy as np
from colorama import Fore
import time
import src.parametri as p


def _impute_nans_timewise(arr):
    """Imputa NaN in una matrice temporale (shape: T x N) per colonna.
    Usa interpolazione lineare lungo l'asse temporale; se una colonna è
    completamente NaN riempie con 0.0 (o con la mediana globale se preferisci).

    Accetta anche vettori 1D.
    """
    a = np.array(arr, dtype=float, copy=True)
    if a.ndim == 1:
        a = a[:, None]

    T, N = a.shape
    for j in range(N):
        col = a[:, j]
        mask = np.isnan(col)
        if mask.all():
            # colonna completamente vuota: non abbiamo info, mettere 0.0
            col[:] = 0.0
            a[:, j] = col
            continue

        valid_idx = np.where(~mask)[0]
        valid_vals = col[valid_idx]
        # interpola su tutti gli indici; fuori range prende first/last valid
        interp = np.interp(np.arange(T), valid_idx, valid_vals)
        col[mask] = interp[mask]
        a[:, j] = col

    if arr is None:
        return None
    return a if arr.ndim != 1 else a[:, 0]


giorni = int(1 * GIORNI_IN_UN_ANNO)
nSims = 5

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

    # Imputa NaN nelle matrici temporali prima di usarle nella simulazione
    # (evitiamo perdita di informazione: interpolazione per colonna/asset)
    rendimentoGiornaliero = _impute_nans_timewise(rendimentoGiornaliero)
    volatilitàGiornaliera = _impute_nans_timewise(volatilitàGiornaliera)
    rendimentoAnnuo = _impute_nans_timewise(rendimentoAnnuo)
    volatilitàAnnua = _impute_nans_timewise(volatilitàAnnua)
    rendimentoAnnuoDividendi = _impute_nans_timewise(rendimentoAnnuoDividendi)
    rendimentoGiornalieroDividendi = _impute_nans_timewise(rendimentoGiornalieroDividendi)
    volatilitàDividendi = _impute_nans_timewise(volatilitàDividendi)
    rendimentiAnnoSuccessivo = _impute_nans_timewise(rendimentiAnnoSuccessivo)

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

        risultati = sim.simulaTitoli(stats, nSims=nSims, verbose=0)
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

    return rmse


def objective(trial):
    # Rolling-window cross-validation + minimax aggregation
    lambdaEWMA = [trial.suggest_float(i, 0.5, 0.999) for i in titoli.stats.nomi]

    # Calcola statistiche una volta per l'intero periodo considerato
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

    # Imputa NaN nelle matrici temporali
    rendimentoGiornaliero = _impute_nans_timewise(rendimentoGiornaliero)
    volatilitàGiornaliera = _impute_nans_timewise(volatilitàGiornaliera)
    rendimentoAnnuo = _impute_nans_timewise(rendimentoAnnuo)
    volatilitàAnnua = _impute_nans_timewise(volatilitàAnnua)
    rendimentoAnnuoDividendi = _impute_nans_timewise(rendimentoAnnuoDividendi)
    rendimentoGiornalieroDividendi = _impute_nans_timewise(rendimentoGiornalieroDividendi)
    volatilitàDividendi = _impute_nans_timewise(volatilitàDividendi)
    rendimentiAnnoSuccessivo = _impute_nans_timewise(rendimentiAnnoSuccessivo)

    n_assets = len(titoli.stats.nomi)

    # Parametri rolling (configurabili)
    train_len = max(int(0.6 * giorni), 5)
    test_len = max(int(0.2 * giorni), 1)
    step = max(int(0.1 * giorni), 1)

    if train_len + test_len > giorni:
        # fallback: non abbastanza giorni per folding, valuta sull'intero periodo
        rmse = valutaLambdaEWMA(lambdaEWMA)
        if np.all(np.isnan(rmse)):
            return 1e6
        return float(np.nanmax(rmse))

    fold_starts = list(range(0, giorni - (train_len + test_len) + 1, step))
    if len(fold_starts) == 0:
        fold_starts = [0]

    fold_max_rmse = []

    for start in fold_starts:
        score = np.zeros(n_assets, dtype=np.float64)
        counts = np.zeros(n_assets, dtype=np.int64)

        # Per ogni giorno nella finestra train+test, simuliamo; ma accumuliamo
        # l'errore solo per i giorni di test (i >= train_len)
        window_len = train_len + test_len
        print(f"\nFold start={start} (train={train_len}, test={test_len})")
        for offset in range(window_len):
            day_idx = start + offset

            stats = dm.Stats()
            stats.nomi = titoli.stats.nomi
            stats.rendimentoGiornaliero = rendimentoGiornaliero[day_idx]
            stats.volatilitàGiornaliera = volatilitàGiornaliera[day_idx]
            stats.rendimentoAnnuo = rendimentoAnnuo[day_idx]
            stats.volatilitàAnnua = volatilitàAnnua[day_idx]
            stats.rendimentoAnnuoDividendi = rendimentoAnnuoDividendi[day_idx]
            stats.rendimentoGiornalieroDividendi = rendimentoGiornalieroDividendi[day_idx]
            stats.volatilitàDividendi = volatilitàDividendi[day_idx]
            stats.L = L

            risultati = sim.simulaTitoli(stats, nSims=nSims, verbose=0)
            target = rendimentiAnnoSuccessivo[day_idx]

            # consideriamo solo i giorni del test per il calcolo dell'errore
            if offset >= train_len:
                valid = (~np.isnan(target)) & (~np.isnan(risultati))
                if np.any(valid):
                    diff = target - risultati
                    score[valid] += (diff[valid]) ** 2
                    counts[valid] += 1

        # calcolo RMSE per asset sul periodo di test
        rmse = np.full(n_assets, np.nan, dtype=np.float64)
        nz = counts > 0
        if np.any(nz):
            rmse[nz] = np.sqrt(score[nz] / counts[nz])
            fold_max_rmse.append(float(np.nanmax(rmse)))
        else:
            # se la fold non ha dati di test validi, penalizza pesantemente
            fold_max_rmse.append(1e6)

    # Aggregazione: media dei max(RMSE) sui fold (puoi cambiare con max per un criterio più prudente)
    if len(fold_max_rmse) == 0:
        return 1e6
    return float(np.mean(fold_max_rmse))


# crea lo study single-objective che minimizza il massimo RMSE (minimax)
study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner()
)

attualiArray = p.ottieniLambdaEWMA(titoli.stats.nomi)

attualiDict = {}
for i, nome in enumerate(titoli.stats.nomi):
    attualiDict[nome] = attualiArray[i]

study.enqueue_trial(attualiDict)
study.optimize(objective, n_trials=40, timeout=3600)

# Risultato single-objective: stampare i migliori parametri trovati
print("\n\nBest params:", study.best_params)
print("Best (minimax) score:", study.best_value)
