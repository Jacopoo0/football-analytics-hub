"""Calcolo del Pressure Score per squadra per partita.

La metrica misura la pericolosita' offensiva di una squadra
usando i percentile rank di tiri totali (Sh), tiri in porta (SoT)
e gol (Gls) rispetto a tutta la lega.

Piu' alto e' il punteggio (0-100), piu' la squadra esercita
pressione offensiva nella partita.
"""

import pandas as pd


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Calcola il percentile rank (0-100) di una serie."""
    return series.rank(pct=True) * 100


def compute_pressure(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Calcola il Pressure Score per ogni partita e squadra.

    La logica:
    1. Per ogni partita-squadra, calcola il percentile rank di:
       - Sh (tiri totali)
       - SoT (tiri in porta)
       - Gls (gol)
       rispetto a tutte le altre partite della lega.
    2. Combina i 3 percentile rank in una media semplice.
    3. Il risultato e' un punteggio 0-100 dove valori alti indicano
       alta pressione offensiva.

    Parameters
    ----------
    team_stats_df : pd.DataFrame
        DataFrame con colonne:
        - game_id, team_id
        - Sh : tiri totali
        - SoT : tiri in porta
        - Gls : gol

    Returns
    -------
    pd.DataFrame
        DataFrame con colonne:
        - game_id
        - team_id
        - pressure_score (0-100)
        - shots_pctl : percentile rank tiri totali
        - sot_pctl : percentile rank tiri in porta
        - goals_pctl : percentile rank gol
        - Sh : tiri nella partita
        - SoT : tiri in porta nella partita
        - Gls : gol nella partita
    """
    df = team_stats_df.copy()

    # Assicura colonne numeriche
    for col in ["Sh", "SoT", "Gls"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Percentile rank per ogni metrica
    df["shots_pctl"] = _percentile_rank(df["Sh"])
    df["sot_pctl"] = _percentile_rank(df["SoT"])
    df["goals_pctl"] = _percentile_rank(df["Gls"])

    # Media dei 3 percentile rank
    df["pressure_score"] = (
        (df["shots_pctl"] + df["sot_pctl"] + df["goals_pctl"]) / 3
    ).round(2)

    return pd.DataFrame(
        {
            "game_id": df["game_id"],
            "team_id": df["team_id"],
            "pressure_score": df["pressure_score"],
            "shots_pctl": df["shots_pctl"].round(1),
            "sot_pctl": df["sot_pctl"].round(1),
            "goals_pctl": df["goals_pctl"].round(1),
            "Sh": df["Sh"].astype(int),
            "SoT": df["SoT"].astype(int),
            "Gls": df["Gls"].astype(int),
        }
    )
