"""Trace the 'Comp' column through the soccerdata pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ["SOCCERDATA_LOGLEVEL"] = "ERROR"

from core.data_loader import _create_fbref, _flatten_columns

# Create FBref for Serie A and get raw schedule data
fb = _create_fbref("ITA-Serie A", "2024-2025")

# Read schedule - DON'T flatten, check raw columns
sched = fb.read_team_match_stats(stat_type="schedule")
print("=== RAW schedule columns ===")
print(sched.columns.tolist()[:20])
print()

# Check if 'Comp' exists at any level
if isinstance(sched.columns, pd.MultiIndex):
    for i, col in enumerate(sched.columns):
        if 'omp' in str(col[1]) or 'omp' in str(col[0]):
            print(f"Found Comp at [{i}]: {col}")
        if 'omp' in str(col[1]).lower() or 'omp' in str(col[0]).lower():
            print(f"Found comp (lower) at [{i}]: {col}")
else:
    for i, col in enumerate(sched.columns):
        if 'omp' in str(col).lower():
            print(f"Found comp at [{i}]: {col}")

# Check 'round' column  
if isinstance(sched.columns, pd.MultiIndex):
    for i, col in enumerate(sched.columns):
        if 'round' in str(col[1]).lower() or 'round' in str(col[0]).lower():
            print(f"Found round at [{i}]: {col}")

# Read misc and shooting too
for stat_type in ["misc", "shooting"]:
    df = fb.read_team_match_stats(stat_type=stat_type)
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns[:20]:
            if 'omp' in str(col).lower():
                print(f"{stat_type}: Found comp at {col}")
    else:
        for col in df.columns[:20]:
            if 'omp' in str(col).lower():
                print(f"{stat_type}: Found comp at {col}")
