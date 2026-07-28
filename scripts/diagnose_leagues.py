"""Diagnostic script: test FBref data fetch for all leagues/seasons."""
import sys
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
os.environ["SOCCERDATA_LOGLEVEL"] = "ERROR"

from core.data_loader import _create_fbref, _flatten_columns, _merge_stat_types

LEAGUES = ["ENG-Premier League", "ITA-Serie A", "ESP-La Liga", "GER-Bundesliga", "FRA-Ligue 1"]
SEASONS = ["2024-2025", "2023-2024", "2022-2023"]


def test_fbref_init():
    for league in LEAGUES:
        for season in SEASONS:
            try:
                fb = _create_fbref(league, season)
                print(f"  INIT OK  {league:25s} {season}")
            except Exception as e:
                print(f"  INIT ERR {league:25s} {season}: {e}")


def test_fbref_download():
    for league in LEAGUES:
        for season in SEASONS:
            for stat_type in ["schedule", "misc", "shooting"]:
                try:
                    fb = _create_fbref(league, season)
                    df = fb.read_team_match_stats(stat_type=stat_type)
                    if df is None:
                        print(f"  NULL     {league:25s} {season}  {stat_type}")
                    elif df.empty:
                        print(f"  EMPTY    {league:25s} {season}  {stat_type}")
                    else:
                        n = len(df)
                        cols = list(df.columns[:6]) if hasattr(df, 'columns') else []
                        print(f"  OK ({n:4d}r) {league:25s} {season}  {stat_type:10s} cols={cols}")
                except Exception as e:
                    print(f"  ERR      {league:25s} {season}  {stat_type:10s}: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "init"
    if mode == "init":
        print("=== TEST FBref INIT ===")
        test_fbref_init()
    elif mode == "download":
        print("=== TEST FBref DOWNLOAD (cache exists) ===")
        test_fbref_download()
    elif mode == "download_full":
        print("=== TEST FBref DOWNLOAD (fresh) ===")
        # Force no_cache via env - but recall the monkey-patch issue
        # We'll skip cache for one league
        from core.data_loader import _download_team_match_stats
        for league in LEAGUES:
            for season in SEASONS:
                print(f"\n--- {league} {season} ---")
                try:
                    sched, misc, shoot = _download_team_match_stats(league, season)
                    print(f"  schedule: {type(sched).__name__} {'empty' if sched is None or sched.empty else f'{len(sched)} rows'}")
                    if sched is not None and not sched.empty:
                        print(f"  schedule cols: {list(sched.columns[:12])}")
                    print(f"  misc: {type(misc).__name__} {'empty' if misc is None or misc.empty else f'{len(misc)} rows'}")
                    print(f"  shoot: {type(shoot).__name__} {'empty' if shoot is None or shoot.empty else f'{len(shoot)} rows'}")
                except Exception as e:
                    print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
    else:
        print(f"Unknown mode: {mode}")
