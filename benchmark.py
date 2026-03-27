# ANCORA NON FUNZIONA

import src.managerTitoli as titoli
import json
from colorama import Fore
from src.genetica import run
from src.const import GIORNI_IN_UN_ANNO
from src.managerTitoli.downloadStorico import DownloadData
from src.valutazione import sharpe, rischioTotale, diversificazione
import numpy as np

N_ITERAZIONI = 3
LASSO = GIORNI_IN_UN_ANNO
TOTALE = 19  # Lassi

# NUOVI PARAMETRI ANTI-OVERFITTING DA BLA.md
MIN_ASSETS = 3  # Vincolo cardinality: minimo 3 asset attivi
LAMBDA_CARDINALITY = 0.15  # Peso penalità cardinality (raccomandato: 0.1-0.2)
N_BOOTSTRAP = 50
LAMBDA_TURNOVER = 0.10

# Profilo prestazioni benchmark
USE_BOOTSTRAP_INNER = False
N_BOOTSTRAP_INNER = 0
USE_BOOTSTRAP_OUTER = True
N_BOOTSTRAP_OUTER = 20
PAUSE_PER_FOLD = False

if __name__ == "__main__":
    with open("./titoli.json", "r") as file:
        nomiTitoliCap = json.load(file)

    nomiTitoli = []
    upper = []
    for nome, cap in nomiTitoliCap:
        nomiTitoli.append(nome)
        upper.append(cap)

    upper = np.array(upper)

    # Tracking anti-overfitting: memorizzo metriche per ogni fold
    fold_results = []
    prev_best_portfolio = None

    storico = titoli.download(nomiTitoli, forzaDownload=0)

    # Calcola il numero massimo di 'lassi' (anni) disponibili coerente per tutti i titoli
    max_lassi = None
    for arr in storico.close:
        # trovo l'indice del primo elemento non NaN
        first_valid_index = np.argmax(~np.isnan(arr))
        arr_cleaned = arr[first_valid_index:]
        lassi_possibili = arr_cleaned.shape[0] // LASSO
        if max_lassi is None or lassi_possibili < max_lassi:
            max_lassi = lassi_possibili

    if max_lassi is None or max_lassi < 2:
        print("Dati insufficienti per il walk-forward standard. Interrompo.")
        exit()

    if TOTALE > max_lassi:
        TOTALE = max_lassi
        print(f"Numero lassi troppo grande, correzione a: {TOTALE}")

    # Walk-forward rigido: test window = 1 LASSO (anno). number_of_folds = TOTALE - 1
    number_of_folds = TOTALE - 1

    # Nested validation grid (ampliata per ridurre overfitting)
    # Griglia più ampia: penalty, mean_shrink E lambda_ewma
    penalty_grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]  # Più valori
    mean_shrink_grid = [0.92, 0.94, 0.96, 0.98, 0.99]  # Più granularità
    lambda_ewma_grid = [0.92, 0.94, 0.96, 0.97, 0.98]  # Ottimizza anche EWMA decay!

    for f in reversed(range(number_of_folds)):
        # train termina prima dei "number_of_folds - f" anni finali
        train_end = -((number_of_folds - f) * LASSO)
        test_start = train_end
        test_end = train_end + LASSO

        # slices
        if train_end == 0:
            train_slice = slice(None, None)
        else:
            train_slice = slice(None, train_end)

        if test_end == 0:
            test_slice = slice(test_start, None)
        else:
            test_slice = slice(test_start, test_end)

        lasso = DownloadData()
        lasso.nomi = storico.nomi
        lasso.close = storico.close[:, train_slice]
        lasso.dividendi = storico.dividendi[:, train_slice]
        lasso.rendimenti = storico.rendimenti[:, train_slice]

        test = DownloadData()
        test.nomi = storico.nomi
        test.close = storico.close[:, test_slice]
        test.dividendi = storico.dividendi[:, test_slice]
        test.rendimenti = storico.rendimenti[:, test_slice]

        # NESTED VALIDATION ROBUSTA: cerchiamo iperparametri robusti su griglia estesa
        best_score = None
        best_params = (None, None, None)  # (penalty, mean_shrink, lambda_ewma)
        best_solution = None

        for penalty in penalty_grid:
            for mean_shrink in mean_shrink_grid:
                for lambda_ewma in lambda_ewma_grid:
                    # Non facciamo monkey-patch globale: passiamo hhi a `run`
                    hhi = penalty

                    # Costruisco inner-train / inner-val (80/20 temporale) sul train attuale (lasso)
                    n_days = lasso.rendimenti.shape[1] if hasattr(lasso, "rendimenti") else 0
                    # Se la finestra è troppo corta penalizziamo fortemente (ma non con inf)
                    min_inner_val_days = 30
                    if n_days <= 1:
                        score = 1e6
                    else:
                        split_idx = int(n_days * 0.8)

                        inner_train = DownloadData()
                        inner_train.nomi = lasso.nomi
                        inner_train.close = lasso.close[:, :split_idx]
                        inner_train.dividendi = lasso.dividendi[:, :split_idx]
                        inner_train.rendimenti = lasso.rendimenti[:, :split_idx]

                        inner_val = DownloadData()
                        inner_val.nomi = lasso.nomi
                        inner_val.close = lasso.close[:, split_idx:]
                        inner_val.dividendi = lasso.dividendi[:, split_idx:]
                        inner_val.rendimenti = lasso.rendimenti[:, split_idx:]

                        # calcolo statistiche su inner-train e inner-val CON lambda_ewma ottimizzato
                        stats_inner_train = titoli.calcolaStatsStoriche(
                            inner_train,
                            [lambda_ewma for _ in nomiTitoli],
                            mean_shrink=mean_shrink,
                            verbose=0,  # Silenzioso per non intasare output
                        )

                        stats_inner_val = titoli.calcolaStatsStoriche(
                            inner_val, [lambda_ewma for _ in nomiTitoli], mean_shrink=mean_shrink, verbose=0
                        )

                        # esegui GA veloce sull'inner-train (più seeds per robustezza)
                        print("Selezione...")
                        ris_val, port_val = run(
                            stats_inner_train,
                            upper=upper,
                            percentualeMinima=0.05,
                            pop_size=150,  # Ridotto per velocità inner loop
                            n_gen=100,  # Aumentato ma con early stopping
                            seeds=[1, 2, 3],  # Almeno 3 seeds per robustezza
                            hhi=hhi,
                            enable_early_stopping=True,
                            early_stopping_patience=20,  # Più aggressivo per inner loop
                            min_assets=MIN_ASSETS,  # Vincolo cardinality
                            lambda_cardinality=LAMBDA_CARDINALITY,
                            use_bootstrap=USE_BOOTSTRAP_INNER,
                            n_bootstrap=N_BOOTSTRAP_INNER,
                            lambdaEWMA=[lambda_ewma for _ in nomiTitoli],
                            mean_shrink=mean_shrink,
                            prev_weights=prev_best_portfolio,
                            lambda_turnover=LAMBDA_TURNOVER,
                        )
                        print("Fatto!\n")

                        # Se l'inner validation ha pochissimi giorni, penalizziamo
                        inner_val_days = n_days - split_idx
                        if inner_val_days < min_inner_val_days:
                            score = 1e6
                        else:
                            # valuta i top-K sull'inner-val usando ENSEMBLE AVERAGING (più robusto)
                            if len(port_val) == 0:
                                score = 1e6
                            else:
                                K = min(5, len(port_val))  # Aumentato da 3 a 5 per ensemble più robusto
                                arr = np.array([sharpe(port_val[k], stats_inner_val) for k in range(K)])

                                # Se tutti i valori sono positivi usiamo la media armonica
                                # altrimenti penalizziamo fortemente la presenza di valori <= 0
                                if np.all(arr > 0):
                                    sharpe_mean = len(arr) / np.sum(1.0 / arr)
                                else:
                                    # penalità legata al più grande valore negativo osservato
                                    neg_max = np.abs(arr[arr <= 0]).max() if np.any(arr <= 0) else 0.0
                                    sharpe_mean = arr.mean() - 2.0 * neg_max

                                sharpe_std = np.std(arr)
                                # Aumentiamo il peso della penalità di stabilità
                                stability_penalty = 0.5
                                score = -sharpe_mean + stability_penalty * sharpe_std

                    if best_score is None or score < best_score:
                        best_score = score
                        best_params = (penalty, mean_shrink, lambda_ewma)

        # Applica i parametri scelti sul train completo
        hhi = best_params[0]
        mean_shrink_best = best_params[1]
        lambda_ewma_best = best_params[2]

        print(f"\n{Fore.CYAN}Fold {f + 1}/{number_of_folds}: Parametri ottimali trovati:{Fore.RESET}")
        print(f"\t{Fore.CYAN}Penalty (HHI): {hhi:.4f}{Fore.RESET}")
        print(f"\t{Fore.CYAN}Mean Shrinkage: {mean_shrink_best:.4f}{Fore.RESET}")
        print(f"\t{Fore.CYAN}Lambda EWMA: {lambda_ewma_best:.4f}{Fore.RESET}")
        print(f"\t{Fore.CYAN}Inner Val Score: {best_score:.4f}{Fore.RESET}\n")

        statsStoriche = titoli.calcolaStatsStoriche(
            lasso, [lambda_ewma_best for _ in nomiTitoli], mean_shrink=mean_shrink_best, verbose=0
        )
        statsTest = titoli.calcolaStatsStoriche(
            test, [lambda_ewma_best for _ in nomiTitoli], mean_shrink=mean_shrink_best, verbose=0
        )

        # Esegui l'ottimizzazione su PIÙ seed per robustezza (8 seeds invece di 5)
        seeds_list = list(range(1, 9))  # Aumentato da 5 a 8
        ris, port = run(
            statsStoriche,
            n_gen=300,  # Aumentato ma con early stopping riduce overfitting
            upper=upper,
            seeds=seeds_list,
            hhi=hhi,
            enable_early_stopping=True,
            early_stopping_patience=50,  # Più paziente per outer loop
            min_assets=MIN_ASSETS,  # Vincolo cardinality
            lambda_cardinality=LAMBDA_CARDINALITY,
            use_bootstrap=USE_BOOTSTRAP_OUTER,
            n_bootstrap=N_BOOTSTRAP_OUTER,
            lambdaEWMA=[lambda_ewma_best for _ in nomiTitoli],
            mean_shrink=mean_shrink_best,
            prev_weights=prev_best_portfolio,
            lambda_turnover=LAMBDA_TURNOVER,
        )

        # Stampa top 5 soluzioni con metriche
        print(f"\n{Fore.MAGENTA}Top 5 Portafogli (Fold {f + 1}){Fore.RESET}:")
        for i in reversed(range(min(5, len(port)))):
            print(f"\t{Fore.GREEN}N.{i + 1}{Fore.RESET}")

            for j in range(len(statsStoriche.nomi)):
                if port[i][j] != 0:
                    print(f"\t\t{Fore.YELLOW}{statsStoriche.nomi[j]}{Fore.RESET}: {port[i][j]:.2%}")

            print(
                f"\t\tMETRICHE PREVISTE:\n\t\t\tVarianza Totale: {ris[i][0]:.2%}²\n\t\t\tSharpe: {-ris[i][1]:.2f}\n\t\t\tDiversificazione: {ris[i][2]:.2%}²"
            )

            sharpeReale = sharpe(port[i], statsTest)
            rischioReale = rischioTotale(port[i], statsTest)
            diverReale = diversificazione(port[i], statsTest)

            print(
                f"\t\tMETRICHE REALI:\n\t\t\tVarianza Totale: {rischioReale:.2%}²\n\t\t\tSharpe: {sharpeReale:.2f}\n\t\t\tDiversificazione: {diverReale:.2%}²"
            )

            # Calcola overfitting gap per il miglior portafoglio (i=0 in reverse order = top)
            if i == 0:  # Solo per il miglior portafoglio
                overfitting_gap_sharpe = abs(-ris[i][1] - sharpeReale)
                overfitting_gap_risk = abs(ris[i][0] - rischioReale)

                fold_results.append(
                    {
                        "fold": f + 1,
                        "params": best_params,
                        "train_sharpe": -ris[i][1],
                        "test_sharpe": sharpeReale,
                        "train_risk": ris[i][0],
                        "test_risk": rischioReale,
                        "overfitting_gap_sharpe": overfitting_gap_sharpe,
                        "overfitting_gap_risk": overfitting_gap_risk,
                        "turnover_vs_prev": float(np.sum(np.abs(port[i] - prev_best_portfolio))) if prev_best_portfolio is not None else 0.0,
                        "portfolio": port[i].copy(),
                    }
                )

                print(f"\n\t{Fore.RED}OVERFITTING GAP:{Fore.RESET}")
                print(f"\t\tSharpe Gap: {overfitting_gap_sharpe:.4f}")
                print(f"\t\tRisk Gap: {overfitting_gap_risk:.4%}²")

                prev_best_portfolio = port[i].copy()

        if PAUSE_PER_FOLD:
            input()

    # Summary finale: analisi anti-overfitting
    print(f"\n\n{Fore.MAGENTA}{'=' * 80}{Fore.RESET}")
    print(f"{Fore.MAGENTA}SUMMARY ANTI-OVERFITTING - Walk-Forward Validation{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'=' * 80}{Fore.RESET}\n")

    if len(fold_results) > 0:
        avg_gap_sharpe = np.mean([r["overfitting_gap_sharpe"] for r in fold_results])
        avg_gap_risk = np.mean([r["overfitting_gap_risk"] for r in fold_results])
        avg_test_sharpe = np.mean([r["test_sharpe"] for r in fold_results])
        std_test_sharpe = np.std([r["test_sharpe"] for r in fold_results])
        avg_turnover = np.mean([r["turnover_vs_prev"] for r in fold_results])

        print(f"{Fore.CYAN}Media Overfitting Gap Sharpe:{Fore.RESET} {avg_gap_sharpe:.4f}")
        print(f"{Fore.CYAN}Media Overfitting Gap Risk:{Fore.RESET} {avg_gap_risk:.4%}²")
        print(
            f"{Fore.CYAN}Media Test Sharpe (Out-of-Sample):{Fore.RESET} {avg_test_sharpe:.4f} ± {std_test_sharpe:.4f}"
        )
        print(f"{Fore.CYAN}Turnover Medio vs Fold precedente:{Fore.RESET} {avg_turnover:.4f}")

        # Analisi stabilità parametri
        penalties = [r["params"][0] for r in fold_results]
        shrinkages = [r["params"][1] for r in fold_results]
        lambdas = [r["params"][2] for r in fold_results]

        print(f"\n{Fore.CYAN}Stabilità Parametri:{Fore.RESET}")
        print(f"\tPenalty (HHI): μ={np.mean(penalties):.4f}, σ={np.std(penalties):.4f}")
        print(f"\tMean Shrinkage: μ={np.mean(shrinkages):.4f}, σ={np.std(shrinkages):.4f}")
        print(f"\tLambda EWMA: μ={np.mean(lambdas):.4f}, σ={np.std(lambdas):.4f}")

        # Interpretazione
        print(f"\n{Fore.GREEN}INTERPRETAZIONE:{Fore.RESET}")
        if avg_gap_sharpe < 0.3:
            print("\u2713 Overfitting minimo: gap Sharpe < 0.3")
        elif avg_gap_sharpe < 0.5:
            print("\u26a0 Overfitting moderato: 0.3 < gap Sharpe < 0.5")
        else:
            print("\u2717 Overfitting significativo: gap Sharpe > 0.5")

        if std_test_sharpe / max(abs(avg_test_sharpe), 0.01) < 0.3:
            print("\u2713 Stabilit\u00e0 alta: CV < 0.3")
        else:
            print("\u26a0 Variabilit\u00e0 significativa nelle performance")

        # Salva risultati
        import os

        os.makedirs("./data", exist_ok=True)
        with open("./data/benchmark_results.json", "w") as f:
            import json

            # Converti numpy arrays in liste per serializzazione
            results_serializable = []
            for r in fold_results:
                r_copy = r.copy()
                r_copy["portfolio"] = r_copy["portfolio"].tolist()
                results_serializable.append(r_copy)
            json.dump(results_serializable, f, indent=2)
        print(f"\n{Fore.GREEN}Risultati salvati in ./data/benchmark_results.json{Fore.RESET}")

    print(f"\n{Fore.MAGENTA}{'=' * 80}{Fore.RESET}\n")
