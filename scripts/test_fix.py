"""Verify FBref download works without monkey-patch for a non-English league."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ["SOCCERDATA_LOGLEVEL"] = "ERROR"
import pandas as pd

from core.data_loader import _create_fbref, _flatten_columns

league = "ITA-Serie A"
season = "2024-2025"

print(f"{'='*60}")
print(f"  TEST: {league} / {season}")
print(f"{'='*60}")

# Step 1: Create FBref instance
print("\n[STEP 1] Creazione FBref instance...")
try:
    fb = _create_fbref(league, season)
    print(f"  OK: leagues={fb.leagues}, seasons={fb.seasons}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

# Step 2: Read seasons (1 request)
print("\n[STEP 2] read_seasons()...")
try:
    seasons = fb.read_seasons()
    print(f"  OK: {type(seasons).__name__}, shape={seasons.shape}")
    print(seasons.head())
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

# Step 3: Read team season stats (1 request)
print("\n[STEP 3] read_team_season_stats('standard')...")
try:
    tss = fb.read_team_season_stats()
    print(f"  OK: {len(tss)} rows, columns={list(tss.columns[:10])}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

# Step 4: Read schedule match logs (1 request per team, ~20)
# We limit to first 2 teams to save time
print("\n[STEP 4] read_team_match_stats('schedule') per team (first 2 teams)...")
teams = tss.index.get_level_values('team').unique().tolist() if 'team' in tss.index.names else []
if not teams:
    teams = list(tss.index.unique(level=2)) if tss.index.nlevels >= 3 else []
print(f"  Squadre trovate: {len(teams)}")
for team in teams[:2]:
    print(f"\n  --- {team} ---")
    try:
        df = fb.read_team_match_stats(stat_type="schedule", team=team)
        if df is not None and not df.empty:
            flat = _flatten_columns(df)
            print(f"  OK: {len(flat)} rows, cols={list(flat.columns[:10])}")
        else:
            print(f"  EMPTY")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:120]}")
    time.sleep(7)  # Rate limit

print(f"\n{'='*60}")
print("  TEST COMPLETATO")
print(f"{'='*60}")
