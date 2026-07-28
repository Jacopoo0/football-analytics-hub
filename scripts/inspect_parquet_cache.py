"""Inspect actual Parquet cache data for all leagues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ["SOCCERDATA_LOGLEVEL"] = "ERROR"
import pandas as pd

from core.data_loader import load_events

for league in ["ENG-Premier League", "ITA-Serie A", "GER-Bundesliga"]:
    season = "2024-2025"
    print(f"\n{'='*60}")
    print(f"  {league} {season}")
    print(f"{'='*60}")
    try:
        df = load_events(league, season)
        teams = sorted(df["team_id"].unique().tolist())
        n_matches = df["game_id"].nunique() if "game_id" in df.columns else 0
        print(f"  Records: {len(df)}, Matches: {n_matches}")
        print(f"  Teams ({len(teams)}): {teams}")
        for c in ["Sh", "SoT", "Gls", "Fls", "CrdY", "CrdR", "Poss"]:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce")
                print(f"  {c}: present, mean={vals.mean():.1f}")
            else:
                print(f"  {c}: MISSING")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
