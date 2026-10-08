import numpy as np


def _maschera_pareto(F: np.ndarray) -> np.ndarray:
    """Restituisce una maschera booleana con True per i punti non dominati (minimizzazione)."""
    n_points = F.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)

    for i, f in enumerate(F):
        if is_efficient[i]:
            # Un punto j è dominato da i se F[j] >= F[i] su tutti gli obiettivi
            # e F[j] > F[i] su almeno uno
            dominati = np.all(F >= f, axis=1) & np.any(F > f, axis=1)
            is_efficient[dominati] = False

    return is_efficient


def trova_knee_point(F: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identifica il knee point sul fronte di Pareto non dominato."""
    if len(F) != len(X):
        raise ValueError("F e X devono avere lo stesso numero di righe")

    # 1. Filtra solo le soluzioni non dominate (fronte di Pareto globale)
    mask_pareto = _maschera_pareto(F)
    F_pareto = F[mask_pareto]
    X_pareto = X[mask_pareto]

    n_points, n_obj = F_pareto.shape
    if n_points <= n_obj:
        # Se i punti non dominati sono insufficienti, fallback sul primo punto o TOPSIS diretto
        if n_points == 0:
            return [], []
        return F_pareto[0], X_pareto[0]

    # 2. Normalizzazione min-max solo sui punti non dominati
    f_min = np.min(F_pareto, axis=0)
    f_max = np.max(F_pareto, axis=0)
    delta = f_max - f_min
    delta[delta == 0.0] = 1.0
    F_norm = (F_pareto - f_min) / delta

    knee_idx = None

    # Caso 2 OBIETTIVI: Distanza perpendicolare dalla retta congiungente i due estremi
    if n_obj == 2:
        idx_p1 = int(np.argmin(F_norm[:, 0]))
        idx_p2 = int(np.argmin(F_norm[:, 1]))

        p1 = F_norm[idx_p1]
        p2 = F_norm[idx_p2]

        v = p2 - p1
        v_norm = np.linalg.norm(v)

        if v_norm > 1e-8:
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

        if norm_len > 1e-8:
            normale_unitaria = normale / norm_len
            distanze = np.abs(np.dot(F_norm - p1, normale_unitaria))
            knee_idx = int(np.argmax(distanze))

    # Fallback TOPSIS (minima distanza euclidea dall'origine ideale [0, 0, ...])
    if knee_idx is None:
        knee_idx = int(np.argmin(np.linalg.norm(F_norm, axis=1)))

    return F_pareto[knee_idx], X_pareto[knee_idx]


def stima_hv_ref_point(statsStoriche, valutatore, constraints, n_campioni=50, margine=0.2):
    """
    Calcola a priori un ref_point sicuro valutando un insieme di portafogli campione.

    :param margine: Percentuale di margine di peggioramento oltre il peggior valore trovato (es. +20%).
    """
    campioni = []

    # 1. Portafogli 100% su singolo asset (se ammessi dai constraints, o riparati)
    for i in range(statsStoriche.nTitoli):
        w = np.zeros(statsStoriche.nTitoli)
        w[i] = 1.0
        campioni.append(constraints.applica(w))

    # 2. Portafoglio equiponderato
    campioni.append(constraints.applica(np.ones(statsStoriche.nTitoli) / statsStoriche.nTitoli))

    # 3. Portafogli casuali conformi ai vincoli
    for _ in range(n_campioni):
        w_rand = np.random.rand(statsStoriche.nTitoli)
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
