import numpy as np
from numba import njit


@njit(fastmath=True)
def applicaVincoli(pesi, percentualeMinima=0.05, upper=None, minTitoli=3):
    # Unificare gestione: trattiamo sia vettori 1D che matrici 2D
    is_1d = pesi.ndim == 1
    if is_1d:
        n_rows = 1
        n_cols = pesi.size
    else:
        n_rows = pesi.shape[0]
        n_cols = pesi.shape[1]

    for r in range(n_rows):
        # prepariamo upper_row (array di limiti per colonna)
        if upper is None:
            upper_row = np.empty(n_cols)
            for i in range(n_cols):
                upper_row[i] = 0.3
        else:
            upper_row = upper

        # 1 clamp tra 0 e max
        for i in range(n_cols):
            if is_1d:
                v = pesi[i]
            else:
                v = pesi[r, i]

            if v < 0.0:
                v = 0.0
            elif v > upper_row[i]:
                v = upper_row[i]

            if is_1d:
                pesi[i] = v
            else:
                pesi[r, i] = v

        # 2 rimuove pesi sotto soglia minima
        for i in range(n_cols):
            if is_1d:
                v = pesi[i]
            else:
                v = pesi[r, i]
            if (v > 0.0) and (v < percentualeMinima):
                if is_1d:
                    pesi[i] = 0.0
                else:
                    pesi[r, i] = 0.0

        # 3 normalizzazione
        s = 0.0
        for i in range(n_cols):
            if is_1d:
                s += pesi[i]
            else:
                s += pesi[r, i]

        if s == 0.0:
            mt = minTitoli
            if mt > n_cols:
                mt = n_cols
            for i in range(n_cols):
                if is_1d:
                    pesi[i] = 0.0
                else:
                    pesi[r, i] = 0.0
            for i in range(mt):
                if is_1d:
                    pesi[i] = 1.0 / mt
                else:
                    pesi[r, i] = 1.0 / mt
            continue

        for i in range(n_cols):
            if is_1d:
                pesi[i] = pesi[i] / s
            else:
                pesi[r, i] = pesi[r, i] / s

        # 4 CARDINALITY
        n_active = 0
        for i in range(n_cols):
            val = pesi[i] if is_1d else pesi[r, i]
            if val > percentualeMinima:
                n_active += 1

        if n_active < minTitoli and minTitoli <= n_cols:
            neg = np.empty(n_cols)
            for i in range(n_cols):
                neg[i] = -(pesi[i] if is_1d else pesi[r, i])
            sorted_indices = np.argsort(neg)

            needed = minTitoli - n_active
            total_to_distribute = needed * percentualeMinima

            any_active = False
            for i in range(n_cols):
                if (pesi[i] if is_1d else pesi[r, i]) > percentualeMinima:
                    any_active = True
                    break

            if any_active and n_active > 0:
                reduction_per_asset = total_to_distribute / n_active
                for i in range(n_cols):
                    if (pesi[i] if is_1d else pesi[r, i]) > percentualeMinima:
                        if is_1d:
                            pesi[i] -= reduction_per_asset
                        else:
                            pesi[r, i] -= reduction_per_asset

                activated = 0
                for k in range(n_cols):
                    idx = sorted_indices[k]
                    val = pesi[idx] if is_1d else pesi[r, idx]
                    if val <= percentualeMinima and activated < needed:
                        if is_1d:
                            pesi[idx] = percentualeMinima
                        else:
                            pesi[r, idx] = percentualeMinima
                        activated += 1

            # rinormalizza
            s2 = 0.0
            for i in range(n_cols):
                s2 += pesi[i] if is_1d else pesi[r, i]
            if s2 != 0.0:
                for i in range(n_cols):
                    if is_1d:
                        pesi[i] = pesi[i] / s2
                    else:
                        pesi[r, i] = pesi[r, i] / s2

        # 5 fix limiti dopo normalizzazione
        any_over = False
        any_under = False
        for i in range(n_cols):
            val = pesi[i] if is_1d else pesi[r, i]
            if val > upper_row[i]:
                any_over = True
            if (val > 0.0) and (val < percentualeMinima):
                any_under = True

        if any_over or any_under:
            for i in range(n_cols):
                val = pesi[i] if is_1d else pesi[r, i]
                if (val > 0.0) and (val < percentualeMinima):
                    if is_1d:
                        pesi[i] = 0.0
                    else:
                        pesi[r, i] = 0.0

            tmp = np.empty(n_cols)
            for i in range(n_cols):
                v = pesi[i] if is_1d else pesi[r, i]
                if v < 0.0:
                    tmp[i] = 0.0
                elif v > upper_row[i]:
                    tmp[i] = upper_row[i]
                else:
                    tmp[i] = v
            for i in range(n_cols):
                if is_1d:
                    pesi[i] = tmp[i]
                else:
                    pesi[r, i] = tmp[i]

            residuo = 1.0
            for i in range(n_cols):
                residuo -= pesi[i] if is_1d else pesi[r, i]

            if abs(residuo) > 1e-12:
                liberi_count = 0
                for i in range(n_cols):
                    val = pesi[i] if is_1d else pesi[r, i]
                    if (val > percentualeMinima) and (val < upper_row[i]):
                        liberi_count += 1

                if liberi_count > 0:
                    add = residuo / liberi_count
                    for i in range(n_cols):
                        val = pesi[i] if is_1d else pesi[r, i]
                        if (val > percentualeMinima) and (val < upper_row[i]):
                            if is_1d:
                                pesi[i] += add
                            else:
                                pesi[r, i] += add
                else:
                    non_zero_count = 0
                    for i in range(n_cols):
                        if (pesi[i] if is_1d else pesi[r, i]) > 0.0:
                            non_zero_count += 1
                    if non_zero_count > 0:
                        add = residuo / non_zero_count
                        for i in range(n_cols):
                            if (pesi[i] if is_1d else pesi[r, i]) > 0.0:
                                if is_1d:
                                    pesi[i] += add
                                else:
                                    pesi[r, i] += add
            # garantiamo infine che nessun peso superi upper_row e che la somma sia 1.0
            tmp2 = np.empty(n_cols)
            for i in range(n_cols):
                v = pesi[i] if is_1d else pesi[r, i]
                if v < 0.0:
                    tmp2[i] = 0.0
                elif v > upper_row[i]:
                    tmp2[i] = upper_row[i]
                else:
                    tmp2[i] = v

            s_tmp = 0.0
            for i in range(n_cols):
                s_tmp += tmp2[i]

            if s_tmp == 0.0:
                mt = minTitoli
                if mt > n_cols:
                    mt = n_cols
                for i in range(n_cols):
                    if is_1d:
                        pesi[i] = 0.0
                    else:
                        pesi[r, i] = 0.0
                for i in range(mt):
                    if is_1d:
                        pesi[i] = 1.0 / mt
                    else:
                        pesi[r, i] = 1.0 / mt
            else:
                # distribuisco l'eventuale residuo solo verso asset che non sono al limite superiore
                resid = 1.0 - s_tmp
                it = 0
                # iteriamo al massimo n_cols volte per raggiungere stabilità
                while (abs(resid) > 1e-12) and (it < n_cols):
                    free_count = 0
                    for i in range(n_cols):
                        if tmp2[i] < (upper_row[i] - 1e-12):
                            free_count += 1

                    if free_count == 0:
                        # nessuno può ricevere senza superare upper: esco dal ciclo
                        break

                    add = resid / free_count
                    for i in range(n_cols):
                        if tmp2[i] < (upper_row[i] - 1e-12):
                            tmp2[i] += add
                            if tmp2[i] > upper_row[i]:
                                tmp2[i] = upper_row[i]

                    s_tmp = 0.0
                    for i in range(n_cols):
                        s_tmp += tmp2[i]
                    resid = 1.0 - s_tmp
                    it += 1

                # se residuo piccolo ma non zero, distribuisco come fallback tra non-zero (potrebbe ignorare upper)
                if abs(resid) > 1e-12:
                    non_zero_count = 0
                    for i in range(n_cols):
                        if tmp2[i] > 0.0:
                            non_zero_count += 1
                    if non_zero_count > 0:
                        add = resid / non_zero_count
                        for i in range(n_cols):
                            if tmp2[i] > 0.0:
                                tmp2[i] += add

                # assegno il risultato finale (garantito tmp2[i] <= upper_row[i] salvo fallback)
                for i in range(n_cols):
                    if is_1d:
                        pesi[i] = tmp2[i]
                    else:
                        pesi[r, i] = tmp2[i]
    return pesi
