import numpy as np
from src.utils.vincoli import Vincoli


def trova_knee_point(F: np.ndarray) -> int:
    n_points, n_obj = F.shape
    if n_points <= n_obj:
        return 0

    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    delta = f_max - f_min
    delta[delta == 0] = 1.0
    F_norm = (F - f_min) / delta

    if n_obj == 3:
        # Trova i 3 vertici estremi (migliori per ciascun singolo asse)
        p1 = F_norm[np.argmin(F_norm[:, 0])]
        p2 = F_norm[np.argmin(F_norm[:, 1])]
        p3 = F_norm[np.argmin(F_norm[:, 2])]

        # Vettore normale al piano passante per p1, p2, p3
        v1 = p2 - p1
        v2 = p3 - p1
        normale = np.cross(v1, v2)
        norm_len = np.linalg.norm(normale)

        if norm_len > 1e-8:
            normale = normale / norm_len
            # Distanza perpendicolare di ogni punto dal piano
            distanze = np.abs(np.dot(F_norm - p1, normale))
            return int(np.argmax(distanze))

    # Fallback TOPSIS (distanza minima dall'ideale [0, 0, ...])
    return int(np.argmin(np.linalg.norm(F_norm, axis=1)))


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

    if F.size == 0:
        return None, None

    knee_idx = trova_knee_point(F)

    return F[knee_idx], X[knee_idx]
