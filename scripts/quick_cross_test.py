"""Quick cross-league test: all 5 leagues, 2024-2025 season."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ["SOCCERDATA_LOGLEVEL"] = "ERROR"
import pandas as pd

from core.data_loader import load_events

LEAGUES = ["ENG-Premier League", "ITA-Serie A", "ESP-La Liga", "GER-Bundesliga", "FRA-Ligue 1"]
required = ["Sh", "SoT", "Gls", "Fls", "CrdY", "CrdR", "Poss"]

for league in LEAGUES:
    print(f"\n--- {league} 2024-2025 ---")
    try:
        df = load_events(league, "2024-2025", max_matches=5, force_download=True)
        n_records = len(df)
        n_matches = df["game_id"].nunique() if "game_id" in df.columns else 0
        n_teams = df["team_id"].nunique() if "team_id" in df.columns else 0
        print(f"  OK: {n_records} records, {n_matches} matches, {n_teams} teams")
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  COLONNE MANCANTI: {missing}")
        else:
            for c in required:
                vals = pd.to_numeric(df[c], errors="coerce")
                print(f"  {c}: media={vals.mean():.1f}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:150]}")
