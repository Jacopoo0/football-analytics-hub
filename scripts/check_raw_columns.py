"""Check raw column structure from FBref HTML files."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

from soccerdata.fbref import _parse_table
from lxml import html as lh

BASE = Path.home() / "soccerdata" / "data" / "FBref"

def parse_match_log(fp):
    with open(fp, "rb") as f:
        tree = lh.parse(f)
    for table in tree.xpath("//table"):
        caption = table.xpath(".//caption")
        cap_text = caption[0].text_content().strip().lower() if caption else ""
        if "match logs" in cap_text or "scores & fixtures" in cap_text:
            df = _parse_table(table)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[1] if c[1] and c[1] != "" else c[0] for c in df.columns]
            df.columns = [str(c).strip() for c in df.columns]
            return df
    return None

# Check schedule + misc + shooting for Inter
for stat_type in ["schedule", "misc", "shooting"]:
    fp = BASE / f"matchlogs_Inter_2425_{stat_type}.html"
    if not fp.exists():
        print(f"\n{stat_type}: NOT FOUND")
        continue
    df = parse_match_log(fp)
    if df is None:
        print(f"\n{stat_type}: No table found")
        continue
    print(f"\n=== {stat_type} ===")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    if "Comp" in df.columns:
        print(f"Comp values: {sorted(df['Comp'].dropna().unique().tolist())}")
    else:
        print("WARNING: NO 'Comp' column!")
    if "Opponent" in df.columns:
        print(f"Opponents: {sorted(df['Opponent'].dropna().unique().tolist())}")
