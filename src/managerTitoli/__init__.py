from src.managerTitoli.downloadStorico import download
from src.managerTitoli.statsTitoliSemplice import StatsTitoliSemplice


class StatsTitoli(StatsTitoliSemplice):
    def __init__(self, nomi: tuple, forzaDownload=True, verbose=1):
        self.nomi = nomi

        rendimenti = download(nomi, fonte="yfinance", anni=50, forzaDownload=forzaDownload, verbose=verbose)

        super().__init__(rendimenti)
