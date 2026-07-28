"""Check misc HTML table structure."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
from lxml import html as lh

fp = Path.home() / "soccerdata" / "data" / "FBref" / "matchlogs_Inter_2425_misc.html"
with open(fp, "rb") as f:
    tree = lh.parse(f)

print(f"Tables: {len(tree.xpath('//table'))}")
for i, table in enumerate(tree.xpath("//table")):
    caption = table.xpath(".//caption")
    cap_text = caption[0].text_content().strip().lower()[:80] if caption else "(no caption)"
    print(f"  [{i}] {cap_text}")
    ths = table.xpath(".//th")
    headers = [th.text_content().strip() if th.text_content() else "" for th in ths[:8]]
    if headers:
        print(f"       headers: {headers}")
