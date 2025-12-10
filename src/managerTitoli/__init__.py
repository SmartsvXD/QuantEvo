from src.managerTitoli.statsStoriche import calcolaStatsStoriche, calcolaStatsTitoliStoricheGradiente
from src.managerTitoli.downloadStorico import download

def _test():
    data = download(["SPY", ])

    calcolaStatsStoriche(data, [0.5, ])
    calcolaStatsTitoliStoricheGradiente(data, 251, [0.5, ])