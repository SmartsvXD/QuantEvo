from src.const import GIORNI_IN_UN_ANNO
from src.managerTitoli.downloadStorico import download
from src.managerTitoli.statsTitoliSemplice import StatsTitoliSemplice


class StatsTitoli:
    def __init__(
        self,
        nomi: tuple,
        W: int = 3 * GIORNI_IN_UN_ANNO,
        wOOS: int = 2 *GIORNI_IN_UN_ANNO,
        forzaDownload=True,
        verbose=1,
    ):
        self.nomi = nomi
        self.rendimenti, self.date = download(
            nomi,
            fonte="yfinance",
            anni=50,
            forzaDownload=forzaDownload,
            verbose=verbose,
        )

        self.W = W
        self.wOOS = wOOS

    def preparaSingolo(self):
        return StatsTitoliSemplice(self.rendimenti[:, -self.W :])

    def preparaRollingWindow(self):
        ottimizzazione = []
        validazione = []
        intervalli = []

        T = self.rendimenti.shape[1]
        dimensione_blocco = self.W + self.wOOS

        for fine_val in range(T, dimensione_blocco - 1, -self.wOOS):
            inizio_val = fine_val - self.wOOS
            fine_opt = inizio_val
            inizio_opt = fine_opt - self.W

            rend_opt = self.rendimenti[:, inizio_opt:fine_opt]
            rend_val = self.rendimenti[:, inizio_val:fine_val]

            ottimizzazione.append(StatsTitoliSemplice(rend_opt))
            validazione.append(StatsTitoliSemplice(rend_val))

            intervalli.append(
                {
                    "DataInizioInSample": str(self.date[inizio_opt].date()),
                    "DataFineInSample": str(self.date[fine_opt - 1].date()),
                    "DataInizioOutOfSample": str(self.date[inizio_val].date()),
                    "DataFineOutOfSample": str(self.date[fine_val - 1].date()),
                }
            )

        ottimizzazione.reverse()
        validazione.reverse()
        intervalli.reverse()

        return ottimizzazione, validazione, intervalli, len(ottimizzazione)
