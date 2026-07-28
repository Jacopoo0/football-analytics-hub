"""Test: does Selenium-based FBref download work without monkey-patch?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ["SOCCERDATA_LOGLEVEL"] = "ERROR"

from soccerdata import FBref

for league in ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A"]:
    for season in ["2024-2025", "2023-2024"]:
        print(f"\n{'='*60}")
        print(f"  {league} / {season}")
        print(f"{'='*60}")
        try:
            fb = FBref(leagues=league, seasons=season)
            sched = fb.read_team_match_stats(stat_type="schedule")
            if sched is None or sched.empty:
                print(f"  SCHEDULE: EMPTY")
            else:
                n = len(sched)
                cols = list(sched.columns[:8])
                print(f"  SCHEDULE: {n} rows, cols={cols}")
            misc = fb.read_team_match_stats(stat_type="misc")
            if misc is not None and not misc.empty:
                print(f"  MISC: {len(misc)} rows")
            else:
                print(f"  MISC: empty")
            shoot = fb.read_team_match_stats(stat_type="shooting")
            if shoot is not None and not shoot.empty:
                print(f"  SHOOTING: {len(shoot)} rows")
            else:
                print(f"  SHOOTING: empty")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
