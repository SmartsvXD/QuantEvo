import numpy as np
from sklearn.covariance import LedoitWolf


def _calcolaMatriceCorrelazione(rendimenti: np.ndarray) -> np.ndarray:
    if rendimenti.ndim != 2:
        n = rendimenti.shape[0] if rendimenti.ndim == 1 else 0
        return np.eye(n)

    n, T = rendimenti.shape
    data_mat = np.empty((T, n), dtype=np.float64)

    for j in range(n):
        col = rendimenti[j]
        mask = ~np.isnan(col)
        if np.any(mask):
            # Vincolo al -100% sui dati validi lineari
            valid = np.maximum(col[mask], -1.0)
            mean = np.mean(valid)

            filled = np.empty_like(col)
            filled[mask] = valid
            filled[~mask] = mean
        else:
            filled = np.zeros_like(col)

        data_mat[:, j] = filled

    # Stima shrinkage della covarianza
    lw = LedoitWolf().fit(data_mat)
    cov = lw.covariance_

    # Normalizzazione in matrice di correlazione
    std_empirica = np.sqrt(np.diag(cov))
    std_empirica = np.maximum(std_empirica, 1e-10)

    corr_matrix = cov / (std_empirica[:, None] * std_empirica[None, :])

    # Simmetrizzazione, clipping numerico e diagonale unitaria
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix
