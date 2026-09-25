import numpy as np


def trova_knee_point(F: np.ndarray, X: np.ndarray) -> int:
    n_points, n_obj = F.shape
    if n_points <= n_obj:
        raise ValueError(f"Numero di punti ({n_points}) insufficiente per {n_obj} obiettivi")

    # 1. Normalizzazione min-max (spazio [0, 1])
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    delta = f_max - f_min
    delta[delta == 0.0] = 1.0
    F_norm = (F - f_min) / delta
    
    knee_idx = None

    # Caso 2 OBIETTIVI: Distanza perpendicolare dalla retta congiungente i due estremi
    if n_obj == 2:
        idx_p1 = int(np.argmin(F_norm[:, 0]))  # Migliore su asse 0
        idx_p2 = int(np.argmin(F_norm[:, 1]))  # Migliore su asse 1

        p1 = F_norm[idx_p1]
        p2 = F_norm[idx_p2]

        v = p2 - p1
        v_norm = np.linalg.norm(v)

        # Se gli estremi non coincidono (fronte non degenere)
        if v_norm > 1e-8:
            # Distanza ortogonale punto-retta in 2D tramite prodotto vettoriale bidimensionale:
            # |(x2 - x1)(y1 - y0) - (x1 - x0)(y2 - y1)| / ||v||
            numeratore = np.abs(v[0] * (p1[1] - F_norm[:, 1]) - (p1[0] - F_norm[:, 0]) * v[1])
            distanze = numeratore / v_norm
            knee_idx = int(np.argmax(distanze))

    # Caso 3 OBIETTIVI: Distanza perpendicolare dal piano passante per i 3 estremi
    elif n_obj == 3:
        idx_p1 = int(np.argmin(F_norm[:, 0]))
        idx_p2 = int(np.argmin(F_norm[:, 1]))
        idx_p3 = int(np.argmin(F_norm[:, 2]))

        p1 = F_norm[idx_p1]
        p2 = F_norm[idx_p2]
        p3 = F_norm[idx_p3]

        v1 = p2 - p1
        v2 = p3 - p1
        normale = np.cross(v1, v2)
        norm_len = np.linalg.norm(normale)

        # Se i tre punti definiscono un piano non degenere
        if norm_len > 1e-8:
            normale_unitaria = normale / norm_len
            distanze = np.abs(np.dot(F_norm - p1, normale_unitaria))
            knee_idx = int(np.argmax(distanze))

    # Fallback TOPSIS (minima distanza euclidea dall'origine ideale [0, 0, ...])
    if knee_idx is None:
        knee_idx = int(np.argmin(np.linalg.norm(F_norm, axis=1)))

    return F[knee_idx], X[knee_idx]


def stima_hv_ref_point(statsStoriche, valutatore, constraints, n_campioni=50, margine=0.2):
    """
    Calcola a priori un ref_point sicuro valutando un insieme di portafogli campione.

    :param margine: Percentuale di margine di peggioramento oltre il peggior valore trovato (es. +20%).
    """
    n_asset = len(statsStoriche.nomi)
    campioni = []

    # 1. Portafogli 100% su singolo asset (se ammessi dai constraints, o riparati)
    for i in range(n_asset):
        w = np.zeros(n_asset)
        w[i] = 1.0
        campioni.append(constraints.applica(w))

    # 2. Portafoglio equiponderato
    campioni.append(constraints.applica(np.ones(n_asset) / n_asset))

    # 3. Portafogli casuali conformi ai vincoli
    for _ in range(n_campioni):
        w_rand = np.random.rand(n_asset)
        campioni.append(constraints.applica(w_rand))

    X_sample = np.array(campioni)

    # IMPORTANTE: valoreOriginale=False per riflettere le metriche interne di pymoo (segni invertiti)
    F_sample = valutatore.calcola(X_sample, statsStoriche, valoreOriginale=False)

    # In pymoo minimizziamo, quindi il "peggiore" è il massimo lungo ogni asse
    f_min = np.min(F_sample, axis=0)
    f_max = np.max(F_sample, axis=0)
    delta = np.maximum(f_max - f_min, 1e-4)

    # Punto peggiorativo con offset di sicurezza
    ref_point = f_max + margine * delta
    return ref_point
