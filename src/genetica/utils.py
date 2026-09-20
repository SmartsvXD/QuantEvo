import numpy as np

from src.const import DEFAULT_PESI_VALUTAZIONE
from src.utils.vincoli import Vincoli


# Funzione per ordinare valutazioni e pesi rispetto alla media delle valutazioni
def ordina(F, X, vincoli: Vincoli, sum_tol=1e-8, zero_tol=1e-12):
    if vincoli.upper is None:
        n = X.shape[1]
        upper = np.ones(n)
    else:
        upper = vincoli.upper

    pct_min = vincoli.percentualeMinima
    min_titoli = vincoli.minTitoli

    invalid = []
    for i in range(X.shape[0]):
        row = X[i]

        # 1. Valori negativi significativi
        if np.any(row < -zero_tol):
            invalid.append(i)
            continue

        # 2. Sopra l'upper bound
        if np.any(row - upper > zero_tol):
            invalid.append(i)
            continue

        # 3. Somma pesi = 1
        if abs(row.sum() - 1.0) > sum_tol:
            invalid.append(i)
            continue

        # 4. Pesi semi-continui: ammessi solo ~0 oppure >= (pct_min - tol)
        # Se un peso è > zero_tol ma < pct_min - zero_tol, è invalido
        mask_invalido = (row > zero_tol) & (row < pct_min - zero_tol)
        if np.any(mask_invalido):
            invalid.append(i)
            continue

        # 5. Cardinalità minima: asset con peso >= pct_min (con tolleranza)
        n_active = int(np.sum(row >= pct_min - zero_tol))
        if n_active < min_titoli:
            invalid.append(i)
            continue

    print(f"Scartate {len(invalid)} soluzioni non valide su {X.shape[0]} totali.")
    if len(invalid) > 0:
        invalid = np.unique(np.array(invalid, dtype=np.int64))
        X = np.delete(X, invalid, axis=0)
        F = np.delete(F, invalid, axis=0)

    w = np.array(DEFAULT_PESI_VALUTAZIONE, dtype=np.float64)
    if F.size == 0:
        return F, X

    n_obj = F.shape[1]
    if w.size != n_obj:
        w = np.ones(n_obj, dtype=np.float64) / n_obj

    indici = np.argsort(np.sum(F * w, axis=1))
    return F[indici], X[indici]
