import numpy as np
from numba import njit


@njit(fastmath=True)
def _proietta_riga(w, pct_min, upper_arr, min_titoli):
    n = w.shape[0]

    # 1. Filtro iniziale e clamp superiore
    for i in range(n):
        if w[i] < pct_min:
            w[i] = 0.0
        elif w[i] > upper_arr[i]:
            w[i] = upper_arr[i]

    # 2. Controllo e correzione Cardinalità
    k_target = min(min_titoli, n)
    n_active = 0
    for i in range(n):
        if w[i] >= pct_min:
            n_active += 1

    if n_active < k_target:
        # Seleziona i k_target asset con peso originale maggiore
        idx_sorted = np.argsort(-w)
        for k in range(k_target):
            idx = idx_sorted[k]
            if w[idx] < pct_min:
                w[idx] = pct_min
        n_active = k_target

    # 3. Verifica preliminare di fattibilità
    sum_min = 0.0
    sum_max = 0.0
    for i in range(n):
        if w[i] >= pct_min:
            sum_min += pct_min
            sum_max += upper_arr[i]

    if sum_max < 1.0:
        # Impossibile fare 1.0: saturiamo al massimo consentito
        for i in range(n):
            if w[i] >= pct_min:
                w[i] = upper_arr[i]
        return
    elif sum_min > 1.0:
        # Impossibile rispettare pct_min: fallback equiponderato
        for i in range(n):
            if w[i] >= pct_min:
                w[i] = 1.0 / n_active
            else:
                w[i] = 0.0
        return

    # 4. Bisezione sul Simplesso Vincolato (Trova lambda)
    low = -1.0
    high = 1.0
    for _ in range(35):
        mid = (low + high) * 0.5
        s = 0.0
        for i in range(n):
            if w[i] >= pct_min:
                v = w[i] + mid
                if v < pct_min:
                    v = pct_min
                elif v > upper_arr[i]:
                    v = upper_arr[i]
                s += v

        if s < 1.0:
            low = mid
        else:
            high = mid

    # 5. Assegnazione definitiva pesi
    mid = (low + high) * 0.5
    s_final = 0.0
    for i in range(n):
        if w[i] >= pct_min:
            v = w[i] + mid
            if v < pct_min:
                v = pct_min
            elif v > upper_arr[i]:
                v = upper_arr[i]
            w[i] = v
            s_final += v
        else:
            w[i] = 0.0

    # 6. Pulizia micro-residuo numerico float (< 1e-12)
    residuo = 1.0 - s_final
    if abs(residuo) > 1e-12:
        for i in range(n):
            if w[i] >= pct_min:
                nuovo = w[i] + residuo
                if pct_min <= nuovo <= upper_arr[i]:
                    w[i] = nuovo
                    break


@njit(fastmath=True)
def applicaVincoli(pesi, percentualeMinima=0.05, upper=None, minTitoli=3):
    n_cols = pesi.shape[-1]

    # Prepara upper_arr una volta sola (fuori dai loop)
    if upper is None:
        upper_arr = np.full(n_cols, 0.3)
    else:
        upper_arr = upper

    # Dispatch pulito tra 1D e 2D senza duplicazione interna
    if pesi.ndim == 1:
        _proietta_riga(pesi, percentualeMinima, upper_arr, minTitoli)
    else:
        for r in range(pesi.shape[0]):
            _proietta_riga(pesi[r], percentualeMinima, upper_arr, minTitoli)

    return pesi
