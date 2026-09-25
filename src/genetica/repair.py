import numpy as np
from numba import njit


@njit(fastmath=True)
def _proietta_riga(w, pct_min, upper_arr, min_titoli, max_titoli):
    n = w.shape[0]

    # 1. Filtro asset eleggibili (upper >= pct_min)
    scores = np.full(n, -1e30)
    n_elig = 0
    for i in range(n):
        if upper_arr[i] >= pct_min:
            scores[i] = w[i]
            n_elig += 1
        else:
            w[i] = 0.0

    if n_elig < min_titoli:
        raise ValueError("Asset eleggibili inferiori a minTitoli")

    # 2. Ordinamento decrescente in base al peso iniziale
    idx_sorted = np.argsort(-scores)

    # 3. Determinazione cardinalità attiva k (tra min_titoli e max_active)
    k_target = min_titoli
    max_active = int(1.0 / pct_min) if pct_min > 0.0 else n_elig
    max_active = min(max_active, n_elig, max_titoli)

    if k_target > max_active:
        raise ValueError("min_titoli non può essere superiore a max_active")

    n_natural = 0
    for j in range(n_elig):
        idx = idx_sorted[j]
        if scores[idx] >= pct_min:
            n_natural += 1
        else:
            break

    k = max(min(n_natural, max_active), k_target)

    # Calcolo capienza iniziale
    sum_upper = 0.0
    for j in range(k):
        sum_upper += upper_arr[idx_sorted[j]]

    # Fase A: Espansione se somma upper < 1.0 e possiamo ancora aggiungere titoli
    while sum_upper < 1.0 - 1e-12 and k < max_active:
        sum_upper += upper_arr[idx_sorted[k]]
        k += 1

    # Fase B: Swap di fattibilità se k ha raggiunto max_active ma somma upper < 1.0
    # Sostituisce i titoli attivi con upper minore con titoli inattivi con upper maggiore
    while sum_upper < 1.0 - 1e-12:
        best_active_pos = -1
        min_active_upper = 1e30
        min_active_score = 1e30
        for j in range(k):
            idx_a = idx_sorted[j]
            u = upper_arr[idx_a]
            sc = scores[idx_a]
            if (u < min_active_upper - 1e-12) or (abs(u - min_active_upper) <= 1e-12 and sc < min_active_score):
                min_active_upper = u
                min_active_score = sc
                best_active_pos = j

        best_inactive_pos = -1
        max_inactive_upper = -1.0
        max_inactive_score = -1e30
        for j in range(k, n_elig):
            idx_i = idx_sorted[j]
            u = upper_arr[idx_i]
            sc = scores[idx_i]
            if (u > max_inactive_upper + 1e-12) or (abs(u - max_inactive_upper) <= 1e-12 and sc > max_inactive_score):
                max_inactive_upper = u
                max_inactive_score = sc
                best_inactive_pos = j

        # Se non esistono ulteriori titoli inattivi con upper maggiore, stop
        if best_inactive_pos == -1 or max_inactive_upper <= min_active_upper + 1e-12:
            break

        idx_act = idx_sorted[best_active_pos]
        idx_inact = idx_sorted[best_inactive_pos]
        idx_sorted[best_active_pos] = idx_inact
        idx_sorted[best_inactive_pos] = idx_act

        sum_upper += (max_inactive_upper - min_active_upper)

    # 4. Maschera attivi: azzera i titoli esclusi oltre k
    is_active = np.zeros(n, dtype=np.bool_)
    for j in range(k):
        is_active[idx_sorted[j]] = True

    for i in range(n):
        if not is_active[i]:
            w[i] = 0.0

    # Verifica preliminare di fattibilità
    sum_min = k * pct_min
    sum_max = 0.0
    for j in range(k):
        sum_max += upper_arr[idx_sorted[j]]

    if sum_max < 1.0 - 1e-12 or sum_min > 1.0 + 1e-12:
        raise ValueError("Impossibile soddisfare i vincoli con somma 1.0 entro i limiti di titoli")

    # Clamp iniziale con limite esatto pct_min
    for j in range(k):
        idx = idx_sorted[j]
        if w[idx] < pct_min:
            w[idx] = pct_min
        elif w[idx] > upper_arr[idx]:
            w[idx] = upper_arr[idx]

    # 5. Bisezione sul simplesso: redistribuisce il peso sui titoli attivi
    low = -1.0
    high = 1.0
    for _ in range(40):
        mid = (low + high) * 0.5
        s = 0.0
        for j in range(k):
            idx = idx_sorted[j]
            v = w[idx] + mid
            if v < pct_min:
                v = pct_min
            elif v > upper_arr[idx]:
                v = upper_arr[idx]
            s += v

        if s < 1.0:
            low = mid
        else:
            high = mid

    # 6. Assegnazione definitiva pesi
    mid = (low + high) * 0.5
    s_final = 0.0
    for j in range(k):
        idx = idx_sorted[j]
        v = w[idx] + mid
        if v < pct_min:
            v = pct_min
        elif v > upper_arr[idx]:
            v = upper_arr[idx]
        w[idx] = v
        s_final += v

    # 7. Assorbimento residuo numerico
    residuo = 1.0 - s_final
    if abs(residuo) > 1e-12:
        if residuo > 0.0:
            for j in range(k):
                idx = idx_sorted[j]
                spazio = upper_arr[idx] - w[idx]
                if spazio > 1e-12:
                    delta = min(residuo, spazio)
                    w[idx] += delta
                    residuo -= delta
                    if residuo <= 1e-12:
                        break
            if residuo > 1e-12:
                for j in range(k):
                    idx = idx_sorted[j]
                    w[idx] += residuo / k
        else:
            for j in range(k):
                idx = idx_sorted[j]
                spazio = w[idx] - pct_min
                if spazio > 1e-12:
                    delta = min(-residuo, spazio)
                    w[idx] -= delta
                    residuo += delta
                    if abs(residuo) <= 1e-12:
                        break


@njit(fastmath=True)
def _apply_constraints_jit(pesi, percentualeMinima, upper, minTitoli, maxTitoli):
    if pesi.ndim == 1:
        _proietta_riga(pesi, percentualeMinima, upper, minTitoli, maxTitoli)
    else:
        for r in range(pesi.shape[0]):
            _proietta_riga(pesi[r], percentualeMinima, upper, minTitoli, maxTitoli)
    return pesi


def applyConstraints(pesi, percentualeMinima=0.05, upper=None, minTitoli=3, maxTitoli=None):
    """Valida i vincoli di portafoglio e delega la proiezione al kernel compilato."""
    if not isinstance(pesi, np.ndarray):
        pesi = np.asarray(pesi, dtype=np.float64)
    elif not np.issubdtype(pesi.dtype, np.floating):
        pesi = pesi.astype(np.float64)

    if pesi.ndim not in (1, 2):
        raise ValueError(f"'pesi' deve avere dimensionalità 1 o 2, ricevuto ndim={pesi.ndim}")

    n_cols = pesi.shape[-1]

    if not isinstance(minTitoli, (int, np.integer)) or minTitoli < 1:
        raise ValueError(f"'minTitoli' deve essere un intero >= 1, ricevuto {minTitoli}")
    if minTitoli > n_cols:
        raise ValueError(f"'minTitoli' ({minTitoli}) non può superare il numero di asset ({n_cols})")

    if maxTitoli is None:
        maxTitoli = n_cols
    else:
        if not isinstance(maxTitoli, (int, np.integer)) or maxTitoli < 1:
            raise ValueError(f"'maxTitoli' deve essere un intero >= 1, ricevuto {maxTitoli}")
        if maxTitoli < minTitoli:
            raise ValueError(f"'maxTitoli' ({maxTitoli}) non può essere inferiore a 'minTitoli' ({minTitoli})")
        maxTitoli = min(int(maxTitoli), n_cols)

    if not isinstance(percentualeMinima, (int, float, np.floating)):
        raise TypeError("'percentualeMinima' deve essere un valore numerico")
    if not (0.0 <= percentualeMinima <= 1.0):
        raise ValueError(f"'percentualeMinima' deve essere in [0, 1], ricevuto {percentualeMinima}")

    if upper is None:
        upper = np.ones(n_cols, dtype=np.float64)
    else:
        upper = np.asarray(upper, dtype=np.float64)
        if upper.ndim != 1 or upper.shape[0] != n_cols:
            raise ValueError(f"'upper' deve essere un array 1D di lunghezza {n_cols}, ricevuto shape={upper.shape}")
        if np.any(upper < 0.0) or np.any(upper > 1.0):
            raise ValueError("Tutti gli elementi di 'upper' devono essere compresi nell'intervallo [0, 1]")

    if minTitoli * percentualeMinima > 1.0 + 1e-12:
        raise ValueError(
            f"Problema non fattibile: minTitoli ({minTitoli}) * percentualeMinima ({percentualeMinima}) = "
            f"{minTitoli * percentualeMinima:.4f} > 1.0"
        )

    eleggibili = upper >= percentualeMinima
    n_elig = np.count_nonzero(eleggibili)
    if n_elig < minTitoli:
        raise ValueError(
            f"Problema non fattibile: asset con upper >= percentualeMinima ({n_elig}) inferiori a minTitoli ({minTitoli})"
        )

    effective_max = min(maxTitoli, int(1.0 / percentualeMinima)) if percentualeMinima > 0.0 else maxTitoli
    effective_max = min(effective_max, n_elig)

    if effective_max < minTitoli:
        raise ValueError(
            f"Problema non fattibile: cardinalità massima effettiva ({effective_max}) inferiore a minTitoli ({minTitoli})"
        )

    upper_elig_sorted = np.sort(upper[eleggibili])[::-1]
    somma_max_possibile = np.sum(upper_elig_sorted[:effective_max])
    if somma_max_possibile < 1.0 - 1e-12:
        raise ValueError(
            f"Problema non fattibile: la somma dei massimi {effective_max} upper bound eleggibili "
            f"({somma_max_possibile:.4f}) è inferiore a 1.0"
        )

    if not pesi.flags.c_contiguous:
        pesi = np.ascontiguousarray(pesi)
    if not upper.flags.c_contiguous:
        upper = np.ascontiguousarray(upper)

    return _apply_constraints_jit(pesi, float(percentualeMinima), upper, int(minTitoli), int(maxTitoli))


class PortfolioConstraints:
    def __init__(self, percentualeMinima=0.05, upper=None, minTitoli=3, maxTitoli=None):
        self.percentualeMinima = percentualeMinima
        self.upper = np.asarray(upper, dtype=np.float64) if upper is not None else None
        self.minTitoli = minTitoli
        self.maxTitoli = maxTitoli

    def applica(self, portafogli):
        return applyConstraints(
            portafogli,
            percentualeMinima=self.percentualeMinima,
            upper=self.upper,
            minTitoli=self.minTitoli,
            maxTitoli=self.maxTitoli,
        )

    def rimuoviInvalidi(self, X, F, sum_tol=1e-8, zero_tol=1e-12):
        n = X.shape[1]
        upper = self.upper if self.upper is not None else np.ones(n)
        pct_min = self.percentualeMinima
        min_titoli = self.minTitoli
        max_titoli = self.maxTitoli if self.maxTitoli is not None else n

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

            # 5. Cardinalità minima e massima
            n_active = int(np.sum(row >= pct_min - zero_tol))
            if n_active < min_titoli or n_active > max_titoli:
                invalid.append(i)
                continue

        print(f"Scartate {len(invalid)} soluzioni non valide su {X.shape[0]} totali.")
        if len(invalid) > 0:
            invalid = np.unique(np.array(invalid, dtype=np.int64))
            X = np.delete(X, invalid, axis=0)
            F = np.delete(F, invalid, axis=0)

        return F, X
