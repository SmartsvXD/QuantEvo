import argparse
import importlib
import time
from pathlib import Path

import pandas as pd

import config
from src.genetica import run
from src.genetica.repair import PortfolioConstraints
from src.managerTitoli import StatsTitoli
from src.valutazione import ValutatoreMultiplo


def benchmark(config_module=config, output="output"):
    titoli = [x.get("name", "") for x in config_module.assets]
    upper = [x.get("upper", 1.0) for x in config_module.assets]

    constraints = PortfolioConstraints(
        percentualeMinima=config_module.structural_constraints.get("minimum_percentage_per_asset", 0.05),
        upper=upper,
        minTitoli=config_module.structural_constraints.get("minimum_number_of_assets", 3),
        maxTitoli=config_module.structural_constraints.get("maximum_number_of_assets", 10),
    )

    stats = StatsTitoli(
        titoli,
        W=config_module.window_size,
        wOOS=config_module.window_size_out_of_sample,
        forzaDownload=False,
        verbose=True,
    )
    ott, val, anni, n = stats.preparaRollingWindow()

    objFunc = ValutatoreMultiplo(config_module.objective_functions)
    vincoli = ValutatoreMultiplo(config_module.constraints)

    metriche = []
    pesi = []

    for w in range(n):
        ott[w].preparaBootstrap(
            randomState=config_module.random_state,
            nBootstrap=config_module.stats_config.get("bootstrap_size", 100),
            lunghezzaBlocco=config_module.stats_config.get("bootstrap_block_size", 20),
        )
        val[w].preparaBootstrap(
            randomState=config_module.random_state,
            nBootstrap=config_module.stats_config.get("bootstrap_size", 100),
            lunghezzaBlocco=config_module.stats_config.get("bootstrap_block_size", 20),
        )

        port, _ = run(
            ott[w],
            objFunc,
            constraints,
            vincoli=vincoli,
            seeds=list(
                range(
                    config_module.random_state,
                    config_module.random_state + config_module.optimization_config.get("n_seeds", 10),
                )
            ),
            popSize=config_module.optimization_config.get("pop_size", 200),
            nGen=config_module.optimization_config.get("n_gen", 500),
            faiEarlyStopping=config_module.optimization_config.get("do_early_stopping", True),
            pazienzaEarlyStopping=config_module.optimization_config.get("early_stopping_patience", 10),
            stima_ref_point=config_module.optimization_config.get("esteem_ref_point", True),
            hv_ref_point=config_module.optimization_config.get("hv_ref_point", None),
        )

        if len(port) == 0:
            print("Nessun portafoglio valido trovato.")

            diz = {"Finestra": w + 1}
            for i in range(len(titoli)):
                diz[titoli[i]] = 0.0
            pesi.append(diz)

            for i in objFunc.indici + vincoli.indici:
                metriche.append(
                    {
                        "Finestra": w + 1,
                        "Metrica": i.__class__.__name__,
                        "Ottimizzazione": None,
                        "Validazione": None,
                        "Differenza": None,
                    }
                )
        else:
            print(
                "\nFinestra OOS:",
                w + 1,
                f"(dal {anni[w]['DataInizioOutOfSample']} al {anni[w]['DataFineOutOfSample']})",
            )
            print()

            diz = {}
            diz["Finestra"] = w + 1
            for i in range(len(titoli)):
                print(f"{titoli[i]}: {port[i]:.1%}")
                diz[titoli[i]] = port[i]
            pesi.append(diz)

            print()

            for i in objFunc.indici + vincoli.indici:
                ottM = i.calcola(port, ott[w], valoreOriginale=True, faiBootstrap=False)
                valM = i.calcola(port, val[w], valoreOriginale=True, faiBootstrap=False)

                metriche.append(
                    {
                        "Finestra": w + 1,
                        "Metrica": i.__class__.__name__,
                        "Ottimizzazione": ottM,
                        "Validazione": valM,
                        "Differenza": valM - ottM,
                    }
                )

                print(f"{i.__class__.__name__} (Ottimizzazione): {ottM:.4f}")
                print(f"{i.__class__.__name__} (Validazione):    {valM:.4f}")
                print(f"{i.__class__.__name__} (Differenza):     {valM - ottM:.4f}")
                print()

            time.sleep(3)

    pesi.append(
        {"Finestra": "Media", **{titoli[i]: sum(pesi[w][titoli[i]] for w in range(n)) / n for i in range(len(titoli))}}
    )

    pesi.append(
        {
            "Finestra": "Deviazione Standard",
            **{
                titoli[i]: (
                    sum((pesi[w][titoli[i]] - sum(pesi[w][titoli[i]] for w in range(n)) / n) ** 2 for w in range(n)) / n
                )
                ** 0.5
                for i in range(len(titoli))
            },
        }
    )

    Path(f"./{output}").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metriche)
    df.to_csv(f"./{output}/metriche.csv", index=False)

    df = pd.DataFrame(anni)
    df.to_csv(f"./{output}/anni_finestre.csv", index=False)

    df = pd.DataFrame(pesi)
    df.to_csv(f"./{output}/pesi.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the benchmark.")
    parser.add_argument(
        "--config", type=str, default="config", help="Name of the configuration module (without .py extension)."
    )
    parser.add_argument("--output", type=str, default="output", help="Name of the output folder.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.suffix == ".py" or "/" in args.config:
        spec = importlib.util.spec_from_file_location("dynamic_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
    else:
        config_module = importlib.import_module(args.config)

    benchmark(config_module=config_module, output=args.output)
