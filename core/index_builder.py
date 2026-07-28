"""Index Builder del TacticalPulse Index.

Combina le 3 componenti (Pressure, Discipline, Network)
in un punteggio composito 0-100 con pesi configurabili.
Aggiunge una colonna 'rank' basata sul TacticalPulse Score.
"""

import pandas as pd


def build_index(
    pressure_df: pd.DataFrame,
    discipline_df: pd.DataFrame,
    network_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Combina le 3 componenti nel TacticalPulse Score finale (0-100).

    Parameters
    ----------
    pressure_df : pd.DataFrame
        Output di compute_pressure().
        Colonne: [game_id, team_id, pressure_score].
    discipline_df : pd.DataFrame
        Output di compute_discipline().
        Colonne: [game_id, team_id, discipline_score].
    network_df : pd.DataFrame
        Output di compute_network().
        Colonne: [game_id, team_id, network_score].
    weights : dict, optional
        Pesi personalizzati, es. {"pressure": 0.4, "discipline": 0.3, "network": 0.3}.
        Default: equal-weight (1/3 ciascuno).

    Returns
    -------
    pd.DataFrame
        Colonne: [game_id, team_id, pressure_score, discipline_score,
                  network_score, tactical_pulse_score, rank].
    """
    if weights is None:
        weights = {"pressure": 1 / 3, "discipline": 1 / 3, "network": 1 / 3}

    w_p = weights.get("pressure", 1 / 3)
    w_d = weights.get("discipline", 1 / 3)
    w_n = weights.get("network", 1 / 3)
    total = w_p + w_d + w_n
    w_p, w_d, w_n = w_p / total, w_d / total, w_n / total

    merged = pressure_df.merge(
        discipline_df, on=["game_id", "team_id"], how="outer"
    ).merge(network_df, on=["game_id", "team_id"], how="outer")

    merged = merged.fillna(0)

    merged["tactical_pulse_score"] = (
        merged["pressure_score"] * w_p
        + merged["discipline_score"] * w_d
        + merged["network_score"] * w_n
    ).round(1)

    # Ranking 1..N basato sul punteggio composito
    merged["rank"] = (
        merged.groupby("game_id")["tactical_pulse_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    cols = [
        "game_id",
        "team_id",
        "pressure_score",
        "discipline_score",
        "network_score",
        "tactical_pulse_score",
        "rank",
    ]
    return merged[cols].sort_values(["game_id", "rank"]).reset_index(drop=True)
