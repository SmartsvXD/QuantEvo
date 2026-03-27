import numpy as np
from src.const import DEFAULT_PESI_VALUTAZIONE


# Funzione per ordinare valutazioni e pesi rispetto alla media delle valutazioni
def ordina(F, X, upper=None, percentualeMinima=0.05, minTitoli=3, sum_tol=1e-2):
    """
    Ordina F (valutazioni) e X (pesi) rispetto alla media delle valutazioni.
    Scarta automaticamente tutte le righe di X che non rispettano i vincoli:
      - pesi >= 0
      - pesi <= upper
      - pesi = 0 oppure >= percentualeMinima
      - almeno `min_assets` asset con peso > percentualeMinima
      - somma dei pesi = 1 (entro `sum_tol`)

    Ritorna: F_ordinato, X_ordinato
    """

    if upper is None:
        n = X.shape[1]
        upper = np.ones(n) * 0.3

    invalid = []
    for i in range(X.shape[0]):
        row = X[i]

        # negativi significativi
        if np.any(row < -1e-12):
            invalid.append(i)
            continue

        # sopra upper
        if np.any(row - upper > 1e-12):
            invalid.append(i)
            continue

        # somma deve essere ~1
        s = row.sum()
        if abs(s - 1.0) > sum_tol:
            invalid.append(i)
            continue

        # pesi non nulli sotto la soglia minima
        mask_min = (row > 0) & (row < percentualeMinima)
        if np.any(mask_min):
            invalid.append(i)
            continue

        # cardinality
        n_active = int(np.sum(row > percentualeMinima))
        if n_active < minTitoli:
            invalid.append(i)
            continue

    if len(invalid) > 0:
        invalid = np.unique(np.array(invalid, dtype=np.int64))
        X = np.delete(X, invalid, axis=0)
        F = np.delete(F, invalid, axis=0)

    w = np.array(DEFAULT_PESI_VALUTAZIONE)
    if F.size == 0:
        return F, X
    indici = np.argsort(np.sum(F * w, axis=1))
    return F[indici], X[indici]
