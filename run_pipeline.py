#!/usr/bin/env python3
"""Script entry-point per il TacticalPulse Index."""

import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.config import load_environment
from core.data_loader import load_events, clear_cache

# Carica GROQ_API_KEY da .env / secrets
load_environment()
from core.discipline import compute_discipline
from core.index_builder import build_index
from core.network import compute_network
from core.pressure import compute_pressure
from stats.significance import validate_significance


def _resolve_season(league: str, preferred: str) -> str:
    """Prova la stagione preferita, altrimenti restituisce quella preferita."""
    from core.data_loader import _create_fbref
    fb = _create_fbref(league, preferred)
    try:
        sched = fb.read_schedule()
        if sched is not None and not sched.empty:
            return preferred
    except Exception:
        pass
    return preferred


def main():
    parser = argparse.ArgumentParser(
        description="TacticalPulse Index - Pipeline End-to-End"
    )
    parser.add_argument(
        "--league",
        default="ENG-Premier League",
        help="Lega (es. ENG-Premier League, ITA-Serie A)",
    )
    parser.add_argument(
        "--season",
        default="2024-2025",
        help="Stagione (es. 2024-2025, 2023-2024)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=[1 / 3, 1 / 3, 1 / 3],
        help="Pesi per Pressure Discipline Network (default: equal)",
    )
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Numero massimo di partite (ogni partita ha 2 righe, una per squadra)")
    parser.add_argument("--no-cache", action="store_true", help="Ignora cache e riscarica")
    parser.add_argument("--clear-cache", action="store_true", help="Pulisce la cache prima di eseguire")
    args = parser.parse_args()

    weights = {
        "pressure": args.weights[0],
        "discipline": args.weights[1],
        "network": args.weights[2],
    }

    if args.clear_cache:
        print("Pulizia cache...")
        clear_cache()
        print("Cache pulita.")

    league = args.league
    season = _resolve_season(league, args.season)

    print(f"\n{'='*60}")
    print(f"  TacticalPulse Index")
    print(f"  Lega: {league}")
    print(f"  Stagione: {season}")
    print(f"  Pesi: P={weights['pressure']:.0%} D={weights['discipline']:.0%} N={weights['network']:.0%}")
    print(f"{'='*60}\n")

    print("[1/5] Caricamento statistiche partite da FBref...")
    stats_df = load_events(league, season, force_download=args.no_cache, max_matches=args.max_matches)
    n_matches = stats_df["game_id"].nunique() if "game_id" in stats_df.columns else 0
    print(f"       Scaricate {len(stats_df)} righe per {n_matches} partite.")

    print("[2/5] Calcolo Pressure Component...")
    pressure_df = compute_pressure(stats_df)
    print(f"       {len(pressure_df)} record, media score={pressure_df['pressure_score'].mean():.1f}")

    print("[3/5] Calcolo Discipline Component...")
    discipline_df = compute_discipline(stats_df)
    print(f"       {len(discipline_df)} record, media score={discipline_df['discipline_score'].mean():.1f}")

    print("[4/5] Calcolo Network Component...")
    network_df = compute_network(stats_df)
    print(f"       {len(network_df)} record, media score={network_df['network_score'].mean():.1f}")

    print("[5/5] Costruzione TacticalPulse Index...")
    index_df = build_index(pressure_df, discipline_df, network_df, weights=weights)
    print(f"       {len(index_df)} record.")

    print("\nValidazione statistica...")
    stats = validate_significance(index_df)
    print(f"       Media: {stats.get('mean_score', 'N/A')}")
    print(f"       Mediana: {stats.get('median_score', 'N/A')}")
    print(f"       Min: {stats.get('min_score', 'N/A')}  Max: {stats.get('max_score', 'N/A')}")
    print(f"       Bootstrap CI 95%: {stats.get('bootstrap_ci_95', 'N/A')}")
    ttest = stats.get('ttest_top_vs_bottom')
    if ttest:
        print(f"       T-test top vs bottom: p={ttest['p_value']:.4f} {'significativo' if ttest['significant'] else 'non significativo'}")

    print("\n--- Top 10 Squadre per TacticalPulse Score ---")
    team_avg = (
        index_df.groupby("team_id")[
            ["pressure_score", "discipline_score", "network_score", "tactical_pulse_score"]
        ]
        .mean()
        .round(1)
        .sort_values("tactical_pulse_score", ascending=False)
    )
    print(f"{'#':<4} {'Squadra':<20} {'Press':<8} {'Disc':<8} {'Net':<8} {'Totale':<8}")
    print("-" * 60)
    for rank, (team, row) in enumerate(team_avg.head(10).iterrows(), 1):
        print(f"{rank:<4} {team:<20} {row['pressure_score']:<8} {row['discipline_score']:<8} {row['network_score']:<8} {row['tactical_pulse_score']:<8}")

    # Genera report
    from agents.orchestrator import _get_llm, _generate_data_only_report

    llm = _get_llm()
    if llm is None:
        print("\nGROQ_API_KEY non configurata -> report data-only.")
        report = _generate_data_only_report(index_df, stats, league, season, weights)
    else:
        print("\nGROQ_API_KEY configurata -> generazione report narrativo via CrewAI...")
        try:
            from agents.orchestrator import _generate_llm_report
            report = _generate_llm_report(index_df, stats, league, season, weights, llm)
        except Exception as e:
            print(f"  Errore: {e}")
            print("  Fallback a report data-only.")
            report = _generate_data_only_report(index_df, stats, league, season, weights)

    reports_dir = _HERE / "reports" / "output"
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_name = league.replace(" ", "_").replace("-", "_")
    fname = f"tacticalpulse_{safe_name}_{season}.md"
    path = reports_dir / fname
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport salvato in: {path}")
    print(f"\n{'='*60}")
    print("  Pipeline completata con successo!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
