import time

import pandas as pd

import config
from src.genetica import run
from src.genetica.repair import PortfolioConstraints
from src.managerTitoli import StatsTitoli
from src.valutazione import ValutatoreMultiplo

if __name__ == "__main__":
    titoli = [x["name"] for x in config.assets]
    upper = [x["upper"] for x in config.assets]

    constraints = PortfolioConstraints(
        percentualeMinima=config.structural_constraints["minimum_percentage_per_asset"],
        upper=upper,
        minTitoli=config.structural_constraints["minimum_number_of_assets"],
        maxTitoli=config.structural_constraints["maximum_number_of_assets"],
    )

    stats = StatsTitoli(titoli, forzaDownload=0, verbose=1)
    ott, val, anni, n = stats.preparaRollingWindow()

    objFunc = ValutatoreMultiplo(config.objective_functions)
    vincoli = ValutatoreMultiplo(config.constraints)

    metriche = []
    pesi = []

    for w in range(n):
        ott[w].preparaBootstrap(randomState=config.random_state)
        val[w].preparaBootstrap(randomState=config.random_state)

        port, ris = run(
            ott[w],
            objFunc,
            constraints,
            vincoli=vincoli,
            seeds=list(range(config.random_state, config.random_state + config.optimization_config["n_seeds"])),
            popSize=config.optimization_config["pop_size"],
            nGen=config.optimization_config["n_gen"],
        )

        if len(port) == 0:
            print("Nessun portafoglio valido trovato.")
        else:
            print("\nFinestra OOS:", w + 1, f"(dal {anni[w]['DataInizioOutOfSample']} al {anni[w]['DataFineOutOfSample']})")
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

    df = pd.DataFrame(metriche)
    df.to_csv("metriche.csv", index=False)

    df = pd.DataFrame(anni)
    df.to_csv("anni_finestre.csv", index=False)

    df = pd.DataFrame(pesi)
    df.to_csv("pesi.csv", index=False)
