import multiprocessing as mp
import src.acceleratedFunctions as a


def __calcolaStatsTitoli(nomi, rendimenti, dividendi, close, q):
    q.put(a._calcolaStatsTitoli(nomi, rendimenti, dividendi, close))


def _calcolaStatsTitoli(nomi, rendimenti, dividendi, close):
    q = mp.Queue()
    schiavo = mp.Process(
        target=__calcolaStatsTitoli,
        name="Schiavo Calcola Stats Titoli",
        args=(nomi, rendimenti, dividendi, close, q),
    )

    schiavo.start()
    data = q.get()
    schiavo.join()

    return data


def __calcolaL(rendimenti, q):
    q.put(a._calcolaL(rendimenti))


def _calcolaL(rendimenti):
    q = mp.Queue()
    schiavo = mp.Process(
        target=__calcolaL,
        name="Schiavo Calcola Cholesky",
        args=(rendimenti, q),
    )

    schiavo.start()
    data = q.get()
    schiavo.join()

    return data


def __simulaTitoli(Z, anni, nSims, drift, coeffDiffusione, S0, mediaDividendi, q):
    q.put(a._simulaTitoli(Z, anni, nSims, drift, coeffDiffusione, S0, mediaDividendi))


def _simulaTitoli(
    Z,
    anni,
    nSims,
    drift,
    coeffDiffusione,
    S0,
    mediaDividendi,
):
    q = mp.Queue()
    schiavo = mp.Process(
        target=__simulaTitoli,
        name="Schiavo Simula Titoli",
        args=(Z, anni, nSims, drift, coeffDiffusione, S0, mediaDividendi, q),
    )

    schiavo.start()
    data = q.get()
    schiavo.join()

    return data
