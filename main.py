import argparse
import importlib
from pathlib import Path

import config
from src.genetica import run
from src.genetica.repair import PortfolioConstraints
from src.managerTitoli import StatsTitoli
from src.valutazione import ValutatoreMultiplo


def main(config_module=config):
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
    stats = stats.preparaSingolo()
    stats.preparaBootstrap(
        randomState=config_module.random_state,
        nBootstrap=config_module.stats_config.get("bootstrap_size", 100),
        lunghezzaBlocco=config_module.stats_config.get("bootstrap_block_size", 20),
    )

    objFunc = ValutatoreMultiplo(config_module.objective_functions)
    vincoli = ValutatoreMultiplo(config_module.constraints)

    port, _ = run(
        stats,
        objFunc,
        constraints,
        vincoli=vincoli,
        seeds=list(
            range(
                config_module.random_state,
                config_module.random_state + config_module.optimization_config.get("n_seeds", 1),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the optimization.")
    parser.add_argument(
        "--config", type=str, default="config", help="Name of the configuration module (without .py extension)."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.suffix == ".py" or "/" in args.config:
        spec = importlib.util.spec_from_file_location("dynamic_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
    else:
        config_module = importlib.import_module(args.config)

    main(config_module=config_module)
