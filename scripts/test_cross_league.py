"""Cross-league end-to-end test: verify ALL league/season combos produce data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ["SOCCERDATA_LOGLEVEL"] = "ERROR"
import pandas as pd

from core.data_loader import load_events
from core.pressure import compute_pressure
from core.discipline import compute_discipline
from core.network import compute_network
from core.index_builder import build_index
from stats.significance import validate_significance

LEAGUES = ["ENG-Premier League", "ITA-Serie A", "ESP-La Liga", "GER-Bundesliga", "FRA-Ligue 1"]
SEASONS = ["2024-2025", "2023-2024", "2022-2023"]

required_cols = ["Sh", "SoT", "Gls", "Fls", "CrdY", "CrdR", "Poss"]

results = []
for league in LEAGUES:
    for season in SEASONS:
        print(f"\n{'='*60}")
        print(f"  {league} / {season}")
        print(f"{'='*60}")
        row = {"league": league, "season": season}

        try:
            stats_df = load_events(league, season, max_matches=10)
        except Exception as e:
            row["status"] = f"ERROR load_events: {type(e).__name__}: {str(e)[:100]}"
            results.append(row)
            continue

        if stats_df.empty:
            row["status"] = "EMPTY: load_events returned empty DataFrame"
            results.append(row)
            continue

        row["n_records"] = len(stats_df)
        row["n_matches"] = stats_df["game_id"].nunique() if "game_id" in stats_df.columns else 0
        row["n_teams"] = stats_df["team_id"].nunique() if "team_id" in stats_df.columns else 0
        row["columns"] = list(stats_df.columns)

        # Check required columns
        missing = [c for c in required_cols if c not in stats_df.columns]
        row["missing_cols"] = missing

        if missing:
            row["status"] = f"COLUMNS_MISSING: {missing}"
            results.append(row)
            continue

        # Check numeric values
        for col in required_cols:
            vals = pd.to_numeric(stats_df[col], errors="coerce")
            if vals.isna().all():
                row[f"{col}_all_na"] = True

        row["all_numeric_valid"] = True

        # Run full pipeline
        try:
            pressure_df = compute_pressure(stats_df)
            discipline_df = compute_discipline(stats_df)
            network_df = compute_network(stats_df)
            index_df = build_index(pressure_df, discipline_df, network_df)
            stats_results = validate_significance(index_df)

            row["n_index"] = len(index_df)
            row["mean_score"] = round(index_df["tactical_pulse_score"].mean(), 1)
            row["pressure_mean"] = round(pressure_df["pressure_score"].mean(), 1)
            row["discipline_mean"] = round(discipline_df["discipline_score"].mean(), 1)
            row["network_mean"] = round(network_df["network_score"].mean(), 1)
            row["status"] = "OK"
        except Exception as e:
            row["status"] = f"ERROR pipeline: {type(e).__name__}: {str(e)[:100]}"

        results.append(row)

# Summary
print(f"\n\n{'='*70}")
print(f"  SUMMARY: {len(results)} combinations tested")
print(f"{'='*70}")

df_results = pd.DataFrame(results)
ok = df_results[df_results["status"] == "OK"]
fail = df_results[df_results["status"] != "OK"]
print(f"\n  ✅ OK: {len(ok)}")
print(f"  ❌ FAIL: {len(fail)}")

if len(fail) > 0:
    print(f"\n  FAILURES:")
    for _, r in fail.iterrows():
        print(f"    - {r['league']:20s} {r['season']:10s}: {r['status']}")

if len(ok) > 0:
    print(f"\n  SUCCESSES:")
    for _, r in ok.iterrows():
        print(f"    ✅ {r['league']:20s} {r['season']:10s} | "
              f"{r['n_records']:4d} records | "
              f"{r['n_matches']:2d} matches | "
              f"{r['n_teams']:2d} teams | "
              f"TP={r['mean_score']:.1f}")

print(f"\n{'='*70}")
