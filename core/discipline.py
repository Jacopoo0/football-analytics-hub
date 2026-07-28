"""Calcolo del Discipline Score per squadra per partita.

La metrica usa falli (Fls), cartellini gialli (CrdY) e
cartellini rossi (CrdR) per partita, normalizzati rispetto
alla distribuzione dell'intera lega.

Piu' alto e' il punteggio (0-100), migliori sono la
disciplina e il fair play della squadra in quella partita.
"""

import pandas as pd

# Pesi per ciascun indicatore disciplinare
WEIGHTS = {"CrdR": 3.0, "CrdY": 1.5, "Fls": 1.0}


def compute_discipline(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Calcola il Discipline Score per ogni partita e squadra.

    La logica:
    1. Per ogni partita, calcola un indice disciplinare grezzo:
       indicator = CrdR * 3 + CrdY * 1.5 + Fls * 1.0
    2. Normalizza l'indice attraverso tutta la lega usando
       una scala min-max invertita (min = massima disciplina).
    3. Il punteggio finale 0-100 dove 100 = nessun fallo/cartellino.

    Parameters
    ----------
    team_stats_df : pd.DataFrame
        DataFrame con almeno le colonne:
        - game_id
        - team_id
        - Fls : numero di falli commessi
        - CrdY : numero di cartellini gialli
        - CrdR : numero di cartellini rossi

    Returns
    -------
    pd.DataFrame
        DataFrame con colonne:
        - game_id
        - team_id
        - discipline_score (0-100)
        - fouls (Fls)
        - yellows (CrdY)
        - reds (CrdR)
        - raw_index : indice disciplinare grezzo
    """
    df = team_stats_df.copy()

    # Assicura che le colonne necessarie esistano
    for col in ["Fls", "CrdY", "CrdR"]:
        if col not in df.columns:
            df[col] = 0
    df[["Fls", "CrdY", "CrdR"]] = df[["Fls", "CrdY", "CrdR"]].fillna(0)

    # Indice disciplinare grezzo (piu' alto = peggiore)
    df["raw_index"] = (
        df["CrdR"] * WEIGHTS["CrdR"]
        + df["CrdY"] * WEIGHTS["CrdY"]
        + df["Fls"] * WEIGHTS["Fls"]
    )

    # Normalizzazione min-max su tutta la lega
    idx_min = df["raw_index"].min()
    idx_max = df["raw_index"].max()
    if idx_max > idx_min:
        df["discipline_score"] = (
            100 * (1 - (df["raw_index"] - idx_min) / (idx_max - idx_min))
        )
    else:
        df["discipline_score"] = 100.0

    df["discipline_score"] = df["discipline_score"].round(2)

    return pd.DataFrame(
        {
            "game_id": df["game_id"],
            "team_id": df["team_id"],
            "discipline_score": df["discipline_score"],
            "fouls": df["Fls"].astype(int),
            "yellows": df["CrdY"].astype(int),
            "reds": df["CrdR"].astype(int),
            "raw_index": df["raw_index"].round(2),
        }
    )
