import numpy as np
from sklearn.covariance import LedoitWolf


def _calcolaMatriceCorrelazione(rendimenti: np.ndarray) -> np.array:
    n = rendimenti.shape[0]

    if rendimenti.ndim == 2:
        T = rendimenti.shape[1]
    else:
        return np.eye(n)

    data_mat = np.empty((T, n), dtype=np.float64)

    for j in range(n):
        col = rendimenti[j]
        mask = ~np.isnan(col)
        if np.any(mask):
            mean = np.nanmean(col)
            filled = np.where(np.isnan(col), mean, col)
        else:
            filled = np.zeros_like(col)
        data_mat[:, j] = filled

    lw = LedoitWolf().fit(data_mat)
    cov = lw.covariance_

    std_empirica = np.sqrt(np.diag(cov))
    std_empirica[std_empirica < 1e-10] = 1e-10

    corr_matrix = cov / (std_empirica[:, None] * std_empirica[None, :])

    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix
