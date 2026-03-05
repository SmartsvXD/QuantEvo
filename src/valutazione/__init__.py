from src.valutazione.sharpe import sharpe
from src.valutazione.rischioTotale import rischioTotale
from src.valutazione.diversificazione import diversificazione


def valuta(portafoglio, statsStoriche):
    return (
        rischioTotale(portafoglio, statsStoriche),
        -sharpe(portafoglio, statsStoriche),
        diversificazione(portafoglio, statsStoriche),
    )


def _test():
    import src.managerTitoli as titoli
    import json

    portafoglio = [0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    with open("./titoli.json", "r") as file:
        nomiTitoli = json.load(file)

    storico = titoli.download(nomiTitoli, forzaDownload=0)

    statsStoriche = titoli.calcolaStatsStoriche(storico, [0.9 for _ in nomiTitoli])

    print(rischioTotale(portafoglio, statsStoriche))
    print(-sharpe(portafoglio, statsStoriche))
    print(diversificazione(portafoglio, statsStoriche))
