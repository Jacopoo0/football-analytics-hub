"""Calcolo del Network (Coesione) Score per squadra per partita.

La metrica valuta la capacita' di una squadra di mantenere
il possesso e far circolare la palla, usando il possesso palla
come indicatore principale (dato che FBref non fornisce
dati passaggio a livello di partita nel dataset schedule/misc).

Piu' alto e' il punteggio (0-100), maggiore e' la coesione
e la circolazione di palla della squadra.
"""

import pandas as pd


def compute_network(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Calcola il Network Score per ogni partita e squadra.

    La logica:
    1. Usa il possesso palla (Poss) come metrica principale
       di coesione e circolazione palla.
    2. Aggiunge un bonus per partite con molti gol (GF + GA >= 3)
       come indicatore di transizioni offensive riuscite.
    3. Normalizza il possesso 0-100:
       - Poss 0-100% -> punteggio 0-100
       - Bonus goal: +5 se match con 3+ gol totali

    Parameters
    ----------
    team_stats_df : pd.DataFrame
        DataFrame con almeno le colonne:
        - game_id
        - team_id
        - Poss : possesso palla percentuale
        - GF : gol fatti
        - GA : gol subiti

    Returns
    -------
    pd.DataFrame
        DataFrame con colonne:
        - game_id
        - team_id
        - network_score (0-100)
        - possession (Poss %)
        - goal_bonus : punti extra per gol totali
    """
    df = team_stats_df.copy()

    if "Poss" not in df.columns:
        df["Poss"] = 50.0
    df["Poss"] = pd.to_numeric(df["Poss"], errors="coerce").fillna(50.0)

    if "GF" not in df.columns:
        df["GF"] = 0
    if "GA" not in df.columns:
        df["GA"] = 0
    df[["GF", "GA"]] = df[["GF", "GA"]].fillna(0)

    # Punteggio base = possesso palla
    df["network_score"] = df["Poss"]

    # Bonus per partite con molti gol totali (transizioni rapide)
    df["total_goals"] = df["GF"] + df["GA"]
    df["goal_bonus"] = df["total_goals"].apply(
        lambda g: min(5, g * 2) if g >= 3 else 0
    )

    df["network_score"] = (df["network_score"] + df["goal_bonus"]).clip(0, 100).round(2)

    return pd.DataFrame(
        {
            "game_id": df["game_id"],
            "team_id": df["team_id"],
            "network_score": df["network_score"],
            "possession": df["Poss"],
            "goal_bonus": df["goal_bonus"],
        }
    )
