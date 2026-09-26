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
    stats = stats.preparaSingolo()
    stats.preparaBootstrap(randomState=config.random_state)

    objFunc = ValutatoreMultiplo(config.objective_functions)
    vincoli = ValutatoreMultiplo(config.contraints)

    port, ris = run(
        stats,
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
        print("\n")
        for i in range(len(titoli)):
            print(f"{titoli[i]}: {port[i]:.1%}")

        print()
        for i in objFunc.indici:
            print(f"{i.__class__.__name__}: {i.calcola(port, stats, valoreOriginale=True, faiBootstrap=False):.4f}")

        print()
        for i in vincoli.indici:
            print(f"{i.__class__.__name__}: {i.calcola(port, stats, valoreOriginale=True, faiBootstrap=False):.4f}")
