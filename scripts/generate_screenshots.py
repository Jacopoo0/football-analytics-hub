"""Genera screenshot demo della dashboard TacticalPulse Index.

Utilizza dati sintetici (senza FBref) e plotly + PIL per creare
immagini representative di ogni pagina. Salvate in docs/screenshots/.
"""

import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = _HERE / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def _synthetic_data():
    rng = np.random.default_rng(42)
    teams = [f"Squadra{i}" for i in range(1, 7)]
    rows = []
    for team in teams:
        for m in range(1, 6):
            rows.append({
                "team_id": team,
                "game_id": f"{team}_G{m}",
                "Sh": int(rng.integers(5, 20)),
                "SoT": int(rng.integers(2, 10)),
                "Gls": int(rng.integers(0, 5)),
                "Fls": int(rng.integers(5, 18)),
                "CrdY": int(rng.integers(0, 5)),
                "CrdR": int(rng.integers(0, 2)),
                "Poss": round(rng.uniform(35, 65), 1),
            })
    stats_df = pd.DataFrame(rows)

    from core.pressure import compute_pressure
    from core.discipline import compute_discipline
    from core.network import compute_network
    from core.index_builder import build_index
    pressure_df = compute_pressure(stats_df)
    discipline_df = compute_discipline(stats_df)
    network_df = compute_network(stats_df)
    index_df = build_index(pressure_df, discipline_df, network_df)

    avg = (
        index_df.groupby("team_id")[
            ["pressure_score", "discipline_score", "network_score", "tactical_pulse_score"]
        ]
        .mean()
        .round(1)
    )
    avg["rank"] = avg["tactical_pulse_score"].rank(ascending=False).astype(int)
    avg = avg.sort_values("rank")

    return stats_df, index_df, avg


def _render_text_png(filename: str, title: str, body_lines: list[str],
                     chart_path: str | None = None):
    """Crea un'immagine PNG con titolo, corpo testo e optional chart."""
    W, H = 1200, 800
    img = Image.new("RGB", (W, H), (14, 17, 23))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    y = 30
    draw.text((40, y), title, fill=(155, 89, 182), font=title_font)
    y += 50
    for line in body_lines:
        draw.text((40, y), line, fill=(200, 200, 200), font=body_font)
        y += 28

    if chart_path:
        try:
            chart_img = Image.open(chart_path)
            chart_img = chart_img.resize((1000, 450), Image.LANCZOS)
            img.paste(chart_img, (100, y + 20))
        except Exception:
            pass

    path = SCREENSHOTS_DIR / filename
    img.save(path)
    print(f"  Salvato {path}")
    return path


def _save_plotly_fig(fig, filename: str, width=1000, height=500):
    """Salva una figura plotly come PNG."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    path = str(SCREENSHOTS_DIR / filename)
    fig.write_image(path, format="png", width=width, height=height)
    return path


def generate_overview():
    print("[1/5] Overview screenshot...")
    _, _, avg = _synthetic_data()

    fig = px.bar(
        avg.reset_index().sort_values("tactical_pulse_score", ascending=True),
        x="tactical_pulse_score",
        y="team_id",
        orientation="h",
        title="TacticalPulse Score per Squadra",
        color="tactical_pulse_score",
        color_continuous_scale="Purples",
        labels={"team_id": "", "tactical_pulse_score": "Score"},
    )
    fig.update_layout(height=400)
    chart_path = _save_plotly_fig(fig, "_chart_overview.png", height=400)

    _render_text_png("overview.png", "Overview — TacticalPulse Index", [
        "La pagina principale mostra una panoramica dell'indice tattico per la lega selezionata.",
        "",
        f"KPI: {len(avg)} squadre analizzate · Media punteggio: {avg['tactical_pulse_score'].mean():.1f}",
        f"Top squadra: {avg.index[0]} ({avg.iloc[0]['tactical_pulse_score']:.1f})",
        "",
        "La sezione 'Classifica completa' mostra il ranking con Pressure, Discipline, Network e Totale.",
        "I pesi delle componenti sono regolabili dal sidebar.",
    ], chart_path=chart_path)


def generate_comparison():
    print("[2/5] Comparison screenshot...")
    _, index_df, avg = _synthetic_data()
    teams = avg.index.tolist()
    a_name, b_name = teams[0], teams[1]

    a_data = index_df[index_df["team_id"] == a_name][["pressure_score", "discipline_score", "network_score"]].mean()
    b_data = index_df[index_df["team_id"] == b_name][["pressure_score", "discipline_score", "network_score"]].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=a_data.values.tolist() + [a_data.values[0]],
        theta=["Pressure", "Discipline", "Network", "Pressure"],
        fill="toself", name=a_name, line_color="#E74C3C",
        fillcolor="rgba(231, 76, 60, 0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=b_data.values.tolist() + [b_data.values[0]],
        theta=["Pressure", "Discipline", "Network", "Pressure"],
        fill="toself", name=b_name, line_color="#3498DB",
        fillcolor="rgba(52, 152, 219, 0.15)"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
        title=f"Confronto {a_name} vs {b_name}",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
    )
    chart_path = _save_plotly_fig(fig, "_chart_comparison.png", height=450)

    _render_text_png("comparison.png", "Team Comparison — Confronto Tattico", [
        f"Confronto diretto tra {a_name} e {b_name}.",
        "",
        "Radar chart: mostra il profilo delle due squadre nelle 3 componenti.",
        "La tabella comparativa elenca i punteggi con vincitore per ogni metrica.",
        "",
        "Include insight automatici: 'TeamA e' superiore in pressione offensiva'.",
        "Sezione AI: genera un'analisi comparativa narrativa via CrewAI.",
    ], chart_path=chart_path)


def generate_deep_dive():
    print("[3/5] Deep Dive screenshot...")
    _, index_df, avg = _synthetic_data()
    team = avg.index[0]
    team_data = index_df[index_df["team_id"] == team]

    avg_press = team_data["pressure_score"].mean()
    avg_disc = team_data["discipline_score"].mean()
    avg_net = team_data["network_score"].mean()
    avg_total = team_data["tactical_pulse_score"].mean()

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=avg_total,
        title={"text": "TacticalPulse", "font": {"size": 14, "color": "#ccc"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#9B59B6"}}
    ))
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=avg_press,
        title={"text": "Pressure", "font": {"size": 14, "color": "#ccc"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E74C3C"}}
    ))
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=avg_disc,
        title={"text": "Discipline", "font": {"size": 14, "color": "#ccc"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#F1C40F"}}
    ))
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=avg_net,
        title={"text": "Network", "font": {"size": 14, "color": "#ccc"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#3498DB"}}
    ))
    fig.update_layout(grid={"rows": 2, "columns": 2, "pattern": "independent"}, height=400)
    chart_path = _save_plotly_fig(fig, "_chart_deepdive.png", height=400, width=1000)

    _render_text_png("deep_dive.png", f"Single Team Deep Dive — {team}", [
        f"Analisi approfondita di {team} con tutti i KPI disponibili.",
        "",
        "Gauge charts: TacticalPulse, Pressure, Discipline, Network (0-100).",
        "Breakdown donut: contributo percentuale di ogni componente.",
        f"Punti di forza e debolezza generati automaticamente per {team}.",
        "",
        "Momentum: In crescita / Stabile / In calo (basato su media mobile 3 partite).",
        "Forma recente: media ultime 5 partite vs media generale.",
        "Evoluzione temporale del punteggio con rolling average.",
    ], chart_path=chart_path)


def generate_ai_report():
    print("[4/5] AI Report screenshot...")
    _, _, avg = _synthetic_data()
    data_only_lines = [
        f"# TacticalPulse Index Report (Data-Only)",
        "",
        f"**Lega:** ENG-Premier League | **Stagione:** 2024-2025",
        f"**Generato il:** 2026-07-26 15:30",
        "",
        "---",
        "## Ranking Squadre per TacticalPulse Score",
        "",
        "|   | Squadra | Pressure | Discipline | Network | Totale |",
        "|---|---------|----------|------------|---------|--------|",
    ]
    for rank, (team, row) in enumerate(avg.iterrows(), 1):
        data_only_lines.append(
            f"| {rank} | {team} | {row['pressure_score']} | {row['discipline_score']} | "
            f"{row['network_score']} | {row['tactical_pulse_score']} |"
        )
    data_only_lines += [
        "",
        "## Statistiche Descrittive",
        f"- Media: {avg['tactical_pulse_score'].mean():.1f}",
        f"- Dev Std: {avg['tactical_pulse_score'].std():.1f}",
        f"- Min: {avg['tactical_pulse_score'].min():.1f} | Max: {avg['tactical_pulse_score'].max():.1f}",
    ]

    _render_text_png("ai_report.png", "AI Report — Report Tattico Automatico", [
        "Report narrativo generato da 4 agenti CrewAI (Analyst → Statistician → Writer → Critic).",
        "Con Groq Llama 3.3 70B per analisi in linguaggio naturale.",
        "",
        "Senza chiave AI: report data-only con tabelle ranking, statistiche descrittive,",
        "intervallo di confidenza bootstrap, t-test top vs bottom, correlazioni.",
        "",
        "Con chiave AI: report narrativo professionale in italiano con interpretazioni tattiche.",
        "Il report puo' essere scaricato in formato .md.",
    ])


def generate_validation():
    print("[5/5] Statistical Validation screenshot...")
    _, index_df, avg = _synthetic_data()
    scores = index_df["tactical_pulse_score"].dropna().values

    fig = px.box(
        index_df, y="tactical_pulse_score",
        title="Distribuzione TacticalPulse Score",
        points="all",
        labels={"tactical_pulse_score": "Score"},
        color_discrete_sequence=["#9B59B6"],
    )
    fig.update_layout(height=350)
    chart_path = _save_plotly_fig(fig, "_chart_validation.png", height=350)

    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    _render_text_png("validation.png", "Statistical Validation — Validazione Statistica", [
        "Verifica la robustezza statistica dei risultati del TacticalPulse Index.",
        "",
        "Bootstrap CI 95%: stima l'intervallo di confidenza del punteggio medio.",
        f"CI 95%: [{ci_low:.1f}, {ci_high:.1f}] — ampiezza: {ci_high - ci_low:.1f} punti.",
        "",
        "T-test Top 5 vs Bottom 5: confronta le squadre migliori con le peggiori.",
        "Distribuzione dei punteggi: box plot con tutti i punti individuali.",
        "Correlazioni tra componenti: Pressure-Discipline-Network.",
    ], chart_path=chart_path)


def main():
    print("Generazione screenshot demo TacticalPulse...")
    print(f"Output: {SCREENSHOTS_DIR}")

    generate_overview()
    generate_comparison()
    generate_deep_dive()
    generate_ai_report()
    generate_validation()

    # Pulizia chart intermedi
    for f in SCREENSHOTS_DIR.glob("_chart_*.png"):
        f.unlink()

    print(f"\nFatto! {len(list(SCREENSHOTS_DIR.glob('*.png')))} screenshot generati in {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
