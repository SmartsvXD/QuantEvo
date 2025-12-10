import numpy as np


# Skewed-t PDF e campionamento
def rskewt(n: int, df: np.ndarray, skew: float, loc: float = 0, scale: float = 1, random_state: int = 49) -> np.ndarray:
    """
    Genera n campioni da una skewed-t univariata.

    :param df: gradi di libertà (code pesanti)
    :type df: `np.ndarray`
    :param skew: parametro di skewness (0=simmetrica)
    :type skew: `float`
    :param loc: posizione
    :type loc: `float`
    :param scale: scala
    :type scale: `float`
    """

    if random_state is not None:
        np.random.seed(random_state)

    u0 = np.random.standard_t(df, size=n)
    u1 = np.random.standard_normal(size=n)

    delta = skew / np.sqrt(1 + skew**2)
    v = delta * np.abs(u0) + np.sqrt(1 - delta**2) * u1
    out = loc + scale * v

    return out


def rskewt_multivariate(n, df, skew_vec, loc_vec, scale_vec, corr, random_state=None) -> np.array:
    """
    Genera campioni heavy-tail (t) correlati e li distorce con uno skew per margine.
    I dati vengono poi ri-standardizzati a media 0 e varianza 1 per evitare drift indesiderato.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_assets = len(skew_vec)
    L = np.linalg.cholesky(corr)

    # base t indipendenti
    t_samples = np.random.standard_t(df, size=(n, n_assets))
    # applica correlazione
    Z_corr = t_samples @ L.T

    # applica skew per margine usando la stessa base correlata
    delta = skew_vec / np.sqrt(1 + skew_vec**2)
    U = np.random.standard_normal(size=(n, n_assets))
    out = delta * np.abs(Z_corr) + np.sqrt(1 - delta**2) * U

    # standardizza per evitare media/spread errati
    out_mean = out.mean(axis=0)
    out_std = out.std(axis=0)
    out_std[out_std == 0] = 1.0
    out = (out - out_mean) / out_std

    # scala e shift finali
    out = loc_vec + scale_vec * out
    return out
