"""Inspect raw FBref HTML to determine competition info structure."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from lxml import html

# Check Arsenal schedule file
html_path = Path.home() / "soccerdata" / "data" / "FBref" / "matchlogs_Arsenal_2425_schedule.html"
with open(html_path, "rb") as f:
    tree = html.parse(f)

# Find all Comp cells
comp_cells = tree.xpath("//td[@data-stat='comp']")
comps = set()
for cell in comp_cells:
    t = cell.text_content().strip() if cell.text_content() else ""
    if t:
        comps.add(t)
print(f"Arsenal competitions: {sorted(comps)}")

# Find all Round cells  
round_cells = tree.xpath("//td[@data-stat='round']")
rounds = set()
for cell in round_cells:
    t = cell.text_content().strip() if cell.text_content() else ""
    if t:
        rounds.add(t)
print(f"Arsenal rounds: {sorted(rounds)}")

# Now check a non-PL team
for team_file in [
    "matchlogs_Atalanta_2425_schedule.html",
    "matchlogs_Inter_2425_schedule.html",
]:
    fp = Path.home() / "soccerdata" / "data" / "FBref" / team_file
    if not fp.exists():
        print(f"{team_file}: NOT FOUND")
        continue
    with open(fp, "rb") as f:
        tree = html.parse(f)
    comp_cells = tree.xpath("//td[@data-stat='comp']")
    comps = set()
    for cell in comp_cells:
        t = cell.text_content().strip() if cell.text_content() else ""
        if t:
            comps.add(t)
    print(f"{team_file}: competitions = {sorted(comps)}")
