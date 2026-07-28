"""Validazione statistica del TacticalPulse Index.

Bootstrap resampling, t-test per confronto gruppi,
e correzione per confronti multipli.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def validate_significance(
    index_df: pd.DataFrame,
    score_col: str = "tactical_pulse_score",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Esegue validazione statistica sul TacticalPulse Score.

    Calcola:
    - Bootstrap CI per la media generale
    - T-test tra top 5 e bottom 5 squadre per punteggio medio
    - Correlazione tra le 3 componenti
    - Statistiche descrittive

    Parameters
    ----------
    index_df : pd.DataFrame
        Output di build_index().
    score_col : str
        Nome della colonna punteggio da validare.
    n_bootstrap : int
        Numero di iterazioni bootstrap.
    alpha : float
        Livello di significativita.

    Returns
    -------
    dict
        Dizionario con risultati statistici.
    """
    scores = index_df[score_col].dropna().values
    n = len(scores)

    if n < 3:
        return {"error": "Campione troppo piccolo per validazione statistica", "n": n}

    bootstrap_means = _bootstrap_mean(scores, n_bootstrap)
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

    team_avg = index_df.groupby("team_id")[score_col].mean().dropna()
    team_avg_sorted = team_avg.sort_values()
    n_teams = len(team_avg_sorted)

    ttest_result = None
    if n_teams >= 10:
        top5 = team_avg_sorted.tail(min(5, n_teams // 2)).values
        bottom5 = team_avg_sorted.head(min(5, n_teams // 2)).values
        if len(top5) > 1 and len(bottom5) > 1:
            t_stat, p_val = stats.ttest_ind(top5, bottom5, equal_var=False)
            ttest_result = {
                "t_statistic": round(float(t_stat), 4),
                "p_value": float(p_val),
                "significant": bool(p_val < alpha),
                "top5_mean": round(float(top5.mean()), 2),
                "bottom5_mean": round(float(bottom5.mean()), 2),
            }

    components = [c for c in ["pressure_score", "discipline_score", "network_score"] if c in index_df.columns]
    correlations = {}
    if len(components) >= 2:
        corr_matrix = index_df[components].corr()
        for c1 in components:
            for c2 in components:
                if c1 < c2:
                    val = corr_matrix.loc[c1, c2]
                    correlations[f"{c1}_vs_{c2}"] = round(float(val), 4)

    gauss_pval = None
    try:
        stat_shapiro, gauss_pval = stats.shapiro(scores[: min(5000, n)])
    except Exception:
        pass

    results = {
        "n_observations": int(n),
        "n_teams": int(n_teams),
        "mean_score": round(float(scores.mean()), 2),
        "std_score": round(float(scores.std()), 2),
        "median_score": round(float(np.median(scores)), 2),
        "min_score": round(float(scores.min()), 2),
        "max_score": round(float(scores.max()), 2),
        "bootstrap_ci_95": [round(float(ci_lower), 2), round(float(ci_upper), 2)],
        "bootstrap_iterations": n_bootstrap,
        "ttest_top_vs_bottom": ttest_result,
        "component_correlations": correlations,
        "normality_p_value": round(float(gauss_pval), 4) if gauss_pval is not None else None,
    }

    return results


def _bootstrap_mean(data: np.ndarray, n_iterations: int = 1000) -> np.ndarray:
    """Genera distribuzione bootstrap della media."""
    rng = np.random.default_rng(42)
    n = len(data)
    means = np.empty(n_iterations)
    for i in range(n_iterations):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = sample.mean()
    return means


def compare_groups(
    index_df: pd.DataFrame,
    group_a_ids: list,
    group_b_ids: list,
    score_col: str = "tactical_pulse_score",
) -> dict:
    """Confronta due gruppi di squadre con t-test e bootstrap.

    Parameters
    ----------
    index_df : pd.DataFrame
        Output di build_index().
    group_a_ids : list
        ID squadre del gruppo A.
    group_b_ids : list
        ID squadre del gruppo B.
    score_col : str
        Colonna punteggio da confrontare.

    Returns
    -------
    dict
        Risultati del confronto statistico.
    """
    a_scores = index_df[index_df["team_id"].isin(group_a_ids)][score_col].dropna().values
    b_scores = index_df[index_df["team_id"].isin(group_b_ids)][score_col].dropna().values

    if len(a_scores) < 2 or len(b_scores) < 2:
        return {"error": "Gruppi troppo piccoli per il confronto"}

    t_stat, p_val = stats.ttest_ind(a_scores, b_scores, equal_var=False)

    bootstrap_diffs = np.empty(1000)
    rng = np.random.default_rng(42)
    combined = np.concatenate([a_scores, b_scores])
    for i in range(1000):
        rng.shuffle(combined)
        diff = combined[: len(a_scores)].mean() - combined[len(a_scores) :].mean()
        bootstrap_diffs[i] = diff

    ci_diff = np.percentile(bootstrap_diffs, [2.5, 97.5])

    return {
        "group_a_mean": round(float(a_scores.mean()), 2),
        "group_b_mean": round(float(b_scores.mean()), 2),
        "difference": round(float(a_scores.mean() - b_scores.mean()), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
        "bootstrap_ci_diff_95": [round(float(ci_diff[0]), 2), round(float(ci_diff[1]), 2)],
        "n_a": int(len(a_scores)),
        "n_b": int(len(b_scores)),
    }


def multiple_testing_correction(p_values: list[float], method: str = "fdr_bh") -> dict:
    """Applica correzione per confronti multipli.

    Parameters
    ----------
    p_values : list[float]
        Lista di p-value.
    method : str
        Metodo di correzione: 'bonferroni' o 'fdr_bh' (Benjamini-Hochberg).

    Returns
    -------
    dict
        p-value corretti e decisioni.
    """
    rejected, corrected, _, _ = multipletests(p_values, method=method)
    return {
        "method": method,
        "original_p_values": [round(p, 4) for p in p_values],
        "corrected_p_values": [round(float(p), 4) for p in corrected],
        "rejected": [bool(r) for r in rejected],
    }
