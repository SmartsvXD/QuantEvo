import numpy as np
from numba import njit


@njit(fastmath=True)
def _proietta_riga(w, pct_min, upper_arr, min_titoli):
    n = w.shape[0]

    # Tolleranza per garantire che l'asset attivo sia > pct_min (evita lo scarto in utils.py:38)
    eps_min = 1e-7 if min_titoli * (pct_min + 1e-7) <= 1.0 else 0.0
    pct_min_eff = pct_min + eps_min

    # 1. Filtro asset eleggibili e penalizzazione dei non eleggibili
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

    # 2. Ordinamento degli asset eleggibili per peso originale decrescente
    idx_sorted = np.argsort(-scores)

    # 3. Determinazione della cardinalità attiva k
    k_target = min_titoli
    max_active = int(1.0 / pct_min)
    if max_active > n_elig:
        max_active = n_elig

    n_natural = 0
    for j in range(n_elig):
        idx = idx_sorted[j]
        if scores[idx] >= pct_min:
            n_natural += 1
        else:
            break

    k = n_natural
    if k < k_target:
        k = k_target
    if k > max_active:
        k = max_active

    # Se la somma dei massimi consentiti è < 1.0, espande il set attivo
    sum_upper = 0.0
    for j in range(k):
        sum_upper += upper_arr[idx_sorted[j]]

    while sum_upper < 1.0 and k < max_active:
        sum_upper += upper_arr[idx_sorted[k]]
        k += 1

    # 4. Maschera attivi e azzeramento esclusi
    is_active = np.zeros(n, dtype=np.bool_)
    for j in range(k):
        is_active[idx_sorted[j]] = True

    for i in range(n):
        if not is_active[i]:
            w[i] = 0.0

    # Verifica preliminare di fattibilità sul sottoinsieme selezionato
    sum_min = 0.0
    sum_max = 0.0
    for j in range(k):
        idx = idx_sorted[j]
        eff_lower = pct_min_eff if upper_arr[idx] >= pct_min_eff else pct_min
        sum_min += eff_lower
        sum_max += upper_arr[idx]

    if sum_max < 1.0 - 1e-12 or sum_min > 1.0 + 1e-12:
        raise ValueError("Impossibile soddisfare i vincoli con somma 1.0")

    # Clamp iniziale dei soli titoli attivi
    for j in range(k):
        idx = idx_sorted[j]
        eff_lower = pct_min_eff if upper_arr[idx] >= pct_min_eff else pct_min
        if w[idx] < eff_lower:
            w[idx] = eff_lower
        elif w[idx] > upper_arr[idx]:
            w[idx] = upper_arr[idx]

    # 5. Bisezione sul Simplesso Vincolato
    low = -1.0
    high = 1.0
    for _ in range(40):
        mid = (low + high) * 0.5
        s = 0.0
        for j in range(k):
            idx = idx_sorted[j]
            eff_lower = pct_min_eff if upper_arr[idx] >= pct_min_eff else pct_min
            v = w[idx] + mid
            if v < eff_lower:
                v = eff_lower
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
        eff_lower = pct_min_eff if upper_arr[idx] >= pct_min_eff else pct_min
        v = w[idx] + mid
        if v < eff_lower:
            v = eff_lower
        elif v > upper_arr[idx]:
            v = upper_arr[idx]
        w[idx] = v
        s_final += v

    # 7. Assorbimento micro-residuo numerico float
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
                eff_lower = pct_min_eff if upper_arr[idx] >= pct_min_eff else pct_min
                spazio = w[idx] - eff_lower
                if spazio > 1e-12:
                    delta = min(-residuo, spazio)
                    w[idx] -= delta
                    residuo += delta
                    if abs(residuo) <= 1e-12:
                        break


@njit(fastmath=True)
def applicaVincoli(pesi, percentualeMinima=0.05, upper=None, minTitoli=3):
    n_cols = pesi.shape[-1]

    # Controllo esplicito di fattibilità globale
    n_elig = 0
    sum_upper_elig = 0.0
    for i in range(n_cols):
        if upper[i] >= percentualeMinima:
            n_elig += 1
            sum_upper_elig += upper[i]

    if n_elig < minTitoli:
        raise ValueError("Problema non fattibile: asset eleggibili inferiori a minTitoli")
    if minTitoli * percentualeMinima > 1.0 + 1e-12:
        raise ValueError("Problema non fattibile: minTitoli * percentualeMinima > 1.0")
    if sum_upper_elig < 1.0 - 1e-12:
        raise ValueError("Problema non fattibile: somma upper bound eleggibili < 1.0")

    # Dispatch vettorializzato 1D o 2D
    if pesi.ndim == 1:
        _proietta_riga(pesi, percentualeMinima, upper, minTitoli)
    else:
        for r in range(pesi.shape[0]):
            _proietta_riga(pesi[r], percentualeMinima, upper, minTitoli)

    return pesi

class Vincoli:
    def __init__(self, percentualeMinima=0.05, upper=None, minTitoli=3):
        self.percentualeMinima = percentualeMinima
        # Conversione esplicita in ndarray float64 prima di Numba
        self.upper = np.asarray(upper, dtype=np.float64) if upper is not None else None
        self.minTitoli = minTitoli

    def applica(self, portafogli):
        upper = self.upper
        if upper is None:
            n_cols = portafogli.shape[-1]
            upper = np.ones(n_cols, dtype=np.float64)

        return applicaVincoli(
            portafogli, 
            percentualeMinima=self.percentualeMinima, 
            upper=upper, 
            minTitoli=self.minTitoli
        )
