import numpy as np
from src.managerTitoli.statsStoriche import StatsTitoli
from src.valutazione.rischioTotale import rischioTotale
from src.const import GIORNI_IN_UN_ANNO


def sharpe(portafoglio, stats: StatsTitoli):
    """
    Calcola lo Sharpe Ratio di un portafoglio.

    Args:
        portafoglio (np.ndarray): Array dei pesi, uno per ogni titolo in stats.
        stats (StatsTitoli): Statistiche storiche dei titoli.

    Returns:
        float: Sharpe Ratio del portafoglio.
    """
    pesi = np.array(portafoglio)
    # Calcola rendimento atteso come somma di crescita del titolo (rendimentoGiornaliero) e dividendi
    rendimento_titolo = stats.rendimentoGiornaliero * GIORNI_IN_UN_ANNO
    dividendi = stats.dividendiGiornalieri * GIORNI_IN_UN_ANNO
    rendimento_atteso = np.dot(pesi, rendimento_titolo + dividendi)
    R_f = 0.02  # Rendimento privo di rischio fisso (es. 2%)
    varianza_portafoglio = rischioTotale(portafoglio, stats)
    rischio_portafoglio = np.sqrt(varianza_portafoglio)
    sharpe_ratio = (rendimento_atteso - R_f) / rischio_portafoglio
    return sharpe_ratio


def calcolaRendimentoPrivoDiRischio(rendimenti_titoli, pesi_portafoglio):
    # Da implementare, utilizzeremo FRED
    pass
