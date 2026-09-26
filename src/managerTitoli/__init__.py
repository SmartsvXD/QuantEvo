from src.managerTitoli.downloadStorico import download
from src.managerTitoli.statsTitoliSemplice import StatsTitoliSemplice
from src.const import GIORNI_IN_UN_ANNO


class StatsTitoli:
    def __init__(
        self, nomi: tuple, W: int = 3 * GIORNI_IN_UN_ANNO, wOOS: int = GIORNI_IN_UN_ANNO, forzaDownload=True, verbose=1
    ):
        self.nomi = nomi

        rendimenti = download(nomi, fonte="yfinance", anni=50, forzaDownload=forzaDownload, verbose=verbose)

        self.rendimenti = rendimenti
        self.W = W
        self.wOOS = wOOS

    def preparaSingolo(self):
        return StatsTitoliSemplice(self.rendimenti[-self.W :])

    def preparaRollingWindow(self):
        ottimizzazione = []
        validazione = []

        for i in range(len(self.rendimenti) - self.W - self.wOOS + 1):
            rendimenti_ottimizzazione = self.rendimenti[i : i + self.W]
            rendimenti_validazione = self.rendimenti[i + self.W : i + self.W + self.wOOS]

            ottimizzazione.append(StatsTitoliSemplice(rendimenti_ottimizzazione))
            validazione.append(StatsTitoliSemplice(rendimenti_validazione))

        return ottimizzazione, validazione
