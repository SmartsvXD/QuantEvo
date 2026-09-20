import numpy as np
from numba import njit


@njit(fastmath=True)
def _proietta_riga(w, pct_min, upper_arr, min_titoli):
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

    # 2. Ordinamento decrescente
    idx_sorted = np.argsort(-scores)

    # 3. Determinazione cardinalità attiva k
    k_target = min_titoli
    max_active = int(1.0 / pct_min) if pct_min > 0.0 else n_elig
    if max_active > n_elig:
        max_active = n_elig

    n_natural = 0
    for j in range(n_elig):
        idx = idx_sorted[j]
        if scores[idx] >= pct_min:
            n_natural += 1
        else:
            break

    k = max(min(max(n_natural, k_target), max_active), k_target)

    # Espansione se somma upper < 1.0
    sum_upper = 0.0
    for j in range(k):
        sum_upper += upper_arr[idx_sorted[j]]

    while sum_upper < 1.0 and k < max_active:
        sum_upper += upper_arr[idx_sorted[k]]
        k += 1

    # 4. Maschera attivi
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
        raise ValueError("Impossibile soddisfare i vincoli con somma 1.0")

    # Clamp iniziale con limite esatto pct_min
    for j in range(k):
        idx = idx_sorted[j]
        if w[idx] < pct_min:
            w[idx] = pct_min
        elif w[idx] > upper_arr[idx]:
            w[idx] = upper_arr[idx]

    # 5. Bisezione sul simplesso
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
def _applica_vincoli_jit(pesi, percentualeMinima, upper, minTitoli):
    if pesi.ndim == 1:
        _proietta_riga(pesi, percentualeMinima, upper, minTitoli)
    else:
        for r in range(pesi.shape[0]):
            _proietta_riga(pesi[r], percentualeMinima, upper, minTitoli)
    return pesi


def applicaVincoli(pesi, percentualeMinima=0.05, upper=None, minTitoli=3):
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

    somma_upper_elig = np.sum(upper[eleggibili])
    if somma_upper_elig < 1.0 - 1e-12:
        raise ValueError(
            f"Problema non fattibile: somma degli upper bound eleggibili ({somma_upper_elig:.4f}) < 1.0"
        )

    if not pesi.flags.c_contiguous:
        pesi = np.ascontiguousarray(pesi)
    if not upper.flags.c_contiguous:
        upper = np.ascontiguousarray(upper)

    return _applica_vincoli_jit(pesi, float(percentualeMinima), upper, int(minTitoli))


class Vincoli:
    def __init__(self, percentualeMinima=0.05, upper=None, minTitoli=3):
        self.percentualeMinima = percentualeMinima
        self.upper = np.asarray(upper, dtype=np.float64) if upper is not None else None
        self.minTitoli = minTitoli

    def applica(self, portafogli):
        return applicaVincoli(
            portafogli,
            percentualeMinima=self.percentualeMinima,
            upper=self.upper,
            minTitoli=self.minTitoli,
        )
