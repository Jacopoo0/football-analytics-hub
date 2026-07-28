"""Dashboard Streamlit interattiva per il TacticalPulse Index.

5 pagine: Overview, Team Comparison, Single Team Deep Dive, AI Report, Statistical Validation.
"""

import os
import sys
import traceback
import io
from pathlib import Path
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.config import load_environment, get_ai_config, get_groq_api_key
from core.data_loader import load_events, is_demo_mode, get_demo_cache_combinations
from core.pressure import compute_pressure
from core.discipline import compute_discipline
from core.network import compute_network
from core.index_builder import build_index
from stats.significance import validate_significance

_ = load_environment()

st.set_page_config(page_title="Football Analytics Hub", page_icon="⚽", layout="wide")

# ── Costanti ───────────────────────────────────────────────────────

COLORS = {
    "pressure": "#E74C3C",
    "discipline": "#F1C40F",
    "network": "#3498DB",
    "tactical_pulse": "#9B59B6",
}
COLORS_LIGHT = {
    "pressure": "rgba(231, 76, 60, 0.25)",
    "discipline": "rgba(241, 196, 15, 0.25)",
    "network": "rgba(52, 152, 219, 0.25)",
    "tactical_pulse": "rgba(155, 89, 182, 0.25)",
}
COLORS_BG = {
    "pressure": "rgba(231, 76, 60, 0.08)",
    "discipline": "rgba(241, 196, 15, 0.08)",
    "network": "rgba(52, 152, 219, 0.08)",
    "tactical_pulse": "rgba(155, 89, 182, 0.08)",
}

LEAGUES = ["ENG-Premier League", "ITA-Serie A", "ESP-La Liga", "GER-Bundesliga", "FRA-Ligue 1"]
SEASONS = ["2024-2025", "2023-2024", "2022-2023"]
PAGES = ["Overview", "Team Comparison", "Single Team Deep Dive", "AI Report", "Statistical Validation"]

SCORE_LEVELS = [(67, "Alto", "🟢"), (34, "Medio", "🟡"), (0, "Basso", "🔴")]

GLOSSARY = {
    "Pressure Score": "Misura la capacità di creare pericolo offensivo (0–100). Combina tiri totali, tiri in porta e gol segnati usando il percentile rank rispetto alla lega. **Più alto = squadra più offensiva e pericolosa.**",
    "Discipline Score": "Misura il controllo di falli e cartellini (0–100). Più alto significa meno falli, meno gialli, meno rossi. **Più alto = squadra più disciplinata e meno sanzionata.**",
    "Network Score": "Misura la qualità del possesso palla e della costruzione di gioco (0–100). Basato sul possesso medio con un bonus per i gol segnati. **Più alto = squadra che tiene palla e costruisce meglio.**",
    "TacticalPulse Score": "Punteggio composito (0–100) che combina Pressure, Discipline e Network con pesi regolabili. **Più alto = profilo tattico complessivamente più forte ed equilibrato.**",
    "Bootstrap CI 95%": "Intervallo di confidenza calcolato ricampionando i dati migliaia di volte. Dice con sicurezza al 95% dove cade il vero punteggio medio della lega. Più stretto = stima più affidabile.",
    "p-value": "Probabilità che la differenza osservata tra due gruppi sia dovuta al caso. **p < 0.05** = differenza statisticamente significativa. **p ≥ 0.05** = differenza non dimostrabile con i dati disponibili.",
    "Ranking": "Classifica delle squadre per TacticalPulse Score. La prima squadra ha il profilo tattico migliore secondo il modello.",
    "Top vs Bottom": "Confronto tra le 5 squadre con punteggio più alto e le 5 con punteggio più basso. Se la differenza è significativa, l'indice distingue bene le squadre forti da quelle deboli.",
}

# ── Helper functions ───────────────────────────────────────────────

def _interpret_score(score: float) -> tuple[str, str]:
    for threshold, label, icon in SCORE_LEVELS:
        if score >= threshold:
            return label, icon
    return "Basso", "🔴"


def _classify_profile(row: pd.Series) -> str:
    pressure = row.get("pressure_score", 0)
    discipline = row.get("discipline_score", 0)
    network = row.get("network_score", 0)
    scores = {"pressure": pressure, "discipline": discipline, "network": network}
    max_comp = max(scores, key=scores.get)
    spread = max(scores.values()) - min(scores.values())
    total = row.get("tactical_pulse_score", 0)
    if total < 25:
        return "Instabile"
    if total < 40:
        return "Poco incisiva"
    if spread < 12:
        return "Equilibrata"
    if max_comp == "pressure" and pressure >= 58:
        return "Aggressiva" if discipline >= 40 else "Aggressiva ma rischiosa"
    if max_comp == "discipline" and discipline >= 58:
        return "Disciplinata"
    if max_comp == "network" and network >= 58:
        return "Tecnica"
    low_comp = min(scores, key=scores.get)
    if low_comp == "pressure" and pressure < 35:
        return "Poco incisiva"
    return "Equilibrata"


def _get_strengths_weaknesses(pressure: float, discipline: float, network: float) -> tuple[list[str], list[str]]:
    strengths = []
    weaknesses = []
    if pressure >= 60:
        strengths.append("Alta capacità di creare occasioni")
    elif pressure <= 30:
        weaknesses.append("Bassa incisività offensiva")
    elif pressure < 40:
        weaknesses.append("Pressione offensiva sotto la media")
    if discipline >= 60:
        strengths.append("Eccellente disciplina difensiva")
    elif discipline <= 30:
        weaknesses.append("Disciplina difensiva fragile")
    elif discipline < 40:
        weaknesses.append("Troppi falli e cartellini")
    if network >= 60:
        strengths.append("Buona qualità del possesso")
    elif network <= 30:
        weaknesses.append("Bassa qualità del possesso")
    elif network < 40:
        weaknesses.append("Network di costruzione debole")
    if not strengths:
        strengths = ["Nessun punto di forza dominante"]
    if not weaknesses:
        weaknesses = ["Nessuna debolezza critica evidente"]
    return strengths[:3], weaknesses[:3]


def _compute_momentum(team_data: pd.DataFrame, window: int = 3) -> str:
    sorted_data = team_data.sort_values("game_id").reset_index(drop=True)
    if len(sorted_data) < window + 1:
        return "Stabile"
    recent = sorted_data["tactical_pulse_score"].tail(window).mean()
    previous = sorted_data["tactical_pulse_score"].iloc[-(window + 1):-1].mean()
    diff = recent - previous
    if diff > 3:
        return "In crescita"
    if diff < -3:
        return "In calo"
    return "Stabile"


def _compute_recent_form(team_data: pd.DataFrame, window: int = 5) -> tuple[float | None, float | None]:
    sorted_data = team_data.sort_values("game_id").reset_index(drop=True)
    overall = sorted_data["tactical_pulse_score"].mean()
    if len(sorted_data) >= window:
        recent = sorted_data["tactical_pulse_score"].tail(window).mean()
    else:
        recent = overall if len(sorted_data) > 0 else None
    return recent, overall


def _generate_overview_insights(avg: pd.DataFrame) -> list[str]:
    if avg.empty:
        return []
    top = avg.head(1).iloc[0]
    top_name = top.name
    comps = {"pressione offensiva": top["pressure_score"], "disciplina": top["discipline_score"], "qualità del network": top["network_score"]}
    best_comp = max(comps, key=comps.get)
    worst_comp = min(comps, key=comps.get)
    insights = []
    spread = max(comps.values()) - min(comps.values())
    if spread < 10:
        insights.append(f"**{top_name}** domina grazie a un profilo bilanciato tra tutte le componenti.")
    else:
        insights.append(f"**{top_name}** eccelle in **{best_comp}** (punto di forza principale).")
        if comps[worst_comp] < 40:
            insights.append(f"**{top_name}** mostra fragilità in **{worst_comp}**, area di possibile miglioramento.")
    if len(avg) > 1:
        insights.append(f"TacticalPulse medio della lega: **{avg['tactical_pulse_score'].mean():.1f}**")
        bot = avg.tail(1).iloc[0]
        diff = top["tactical_pulse_score"] - bot["tactical_pulse_score"]
        insights.append(f"Divario tra prima e ultima: **{diff:.1f} punti**")
    return insights


def _generate_comparison_insights(team_a: str, team_b: str, a_avg: pd.Series, b_avg: pd.Series) -> list[str]:
    insights = []
    for col, label in [
        ("pressure_score", "pressione offensiva"),
        ("discipline_score", "disciplina"),
        ("network_score", "qualità del network"),
    ]:
        diff = a_avg[col] - b_avg[col]
        if abs(diff) > 5:
            winner = team_a if diff > 0 else team_b
            verb = "superiore" if abs(diff) > 10 else "leggermente superiore"
            insights.append(f"**{winner}** è {verb} in **{label}**")
        else:
            insights.append(f"Le due squadre sono simili in **{label}**")
    return insights


def _generate_weight_insight(base_avg: pd.DataFrame, new_avg: pd.DataFrame, w: dict) -> list[str]:
    if base_avg.empty or new_avg.empty:
        return []
    base = base_avg["tactical_pulse_score"]
    new = new_avg["tactical_pulse_score"]
    moves = (new - base).sort_values(ascending=False)
    insights = []
    top_gain = moves.head(1)
    top_loss = moves.tail(1)
    if top_gain.iloc[0] > 1:
        insights.append(f"Aumentando il peso di {_dominant_weight_component(w)}, **{top_gain.index[0]}** guadagna {top_gain.iloc[0]:.1f} punti.")
    if top_loss.iloc[0] < -1:
        insights.append(f"Con questi pesi **{top_loss.index[0]}** perde {abs(top_loss.iloc[0]):.1f} punti.")
    return insights


def _dominant_weight_component(w: dict) -> str:
    comp_map = {"pressure": "Pressione offensiva", "discipline": "Disciplina", "network": "Network"}
    return comp_map.get(max(w, key=w.get), "?")


def _format_pct(diff: float, total: float) -> str:
    if total == 0:
        return "—"
    return f"{abs(diff) / total * 100:.1f}%"


def _interpretation_box(text: str, icon: str = "💡"):
    st.markdown(
        f"""<div style="padding:0.75rem 1rem;border-radius:8px;background:#1a1a2e;border-left:4px solid #9B59B6;margin:0.75rem 0;">
        <span style="font-size:1.1rem;">{icon}</span> {text}</div>""",
        unsafe_allow_html=True,
    )


def _metric_with_help(label: str, value, help_text: str):
    st.metric(label, value, help=help_text)


def _csv_download_button(df: pd.DataFrame, filename: str, button_label: str):
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(label=button_label, data=csv, file_name=filename, mime="text/csv", use_container_width=True)


def _render_glossary():
    with st.sidebar.expander("📖 Glossario", expanded=False):
        for term, defn in GLOSSARY.items():
            st.markdown(f"**{term}**")
            st.markdown(f"<small>{defn}</small>", unsafe_allow_html=True)
            st.markdown("---")


# ── Pipeline dati ─────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Caricamento dati in corso...")
def _load_and_compute_cached(league: str, season: str, max_matches: int) -> dict:
    return _load_and_compute(league, season, max_matches)


def _load_and_compute(league: str, season: str, max_matches: int) -> dict:
    stats_df = load_events(league, season, max_matches=max_matches)
    if stats_df.empty:
        st.error("Nessun dato trovato per la selezione corrente.")
        return {}
    pressure_df = compute_pressure(stats_df)
    discipline_df = compute_discipline(stats_df)
    network_df = compute_network(stats_df)
    index_df = build_index(pressure_df, discipline_df, network_df)
    stats_results = validate_significance(index_df)
    teams = sorted(index_df["team_id"].unique().tolist())
    return {
        "stats_df": stats_df,
        "pressure_df": pressure_df,
        "discipline_df": discipline_df,
        "network_df": network_df,
        "index_df": index_df,
        "stats_results": stats_results,
        "teams": teams,
        "n_matches": stats_df["game_id"].nunique(),
        "n_records": len(stats_df),
    }


def _compute_with_weights(base: dict, w_pressure: float, w_discipline: float, w_network: float) -> dict:
    weights = {"pressure": w_pressure, "discipline": w_discipline, "network": w_network}
    index_df = build_index(base["pressure_df"], base["discipline_df"], base["network_df"], weights=weights)
    stats_results = validate_significance(index_df)
    return {"index_df": index_df, "stats_results": stats_results}


def _team_avg_table(index_df: pd.DataFrame) -> pd.DataFrame:
    avg = (
        index_df.groupby("team_id")[
            ["pressure_score", "discipline_score", "network_score", "tactical_pulse_score"]
        ]
        .mean()
        .round(1)
    )
    avg["rank"] = avg["tactical_pulse_score"].rank(ascending=False).astype(int)
    avg["profile"] = avg.apply(_classify_profile, axis=1)
    return avg.sort_values("rank")


def _safe_style_table(display_df: pd.DataFrame):
    try:
        styled = display_df.style.format(
            {c: "{:.1f}" for c in ["Pressure", "Discipline", "Network", "Totale"] if c in display_df.columns}
        )
        for col_name, color_key in [("Pressure", "pressure"), ("Discipline", "discipline"), ("Network", "network"), ("Totale", "tactical_pulse")]:
            if col_name not in display_df.columns:
                continue
            top_val = display_df[col_name].max()
            styled = styled.map(
                lambda v, t=top_val, ck=color_key: (
                    f"background-color: {COLORS_LIGHT.get(ck, 'rgba(255,255,255,0.05)')}; font-weight: bold;"
                    if v == t else ""
                ),
                subset=[col_name],
            )
        return styled
    except Exception:
        return display_df


def _render_ai_debug():
    try:
        is_cloud = st.runtime.exists() and hasattr(st, "secrets")
    except Exception:
        is_cloud = False
    if is_cloud:
        return
    cfg = get_ai_config()
    env = load_environment()
    with st.sidebar.expander("🔧 Debug AI", expanded=False):
        st.markdown(f"**AI enabled:** {'si' if cfg['enabled'] else 'no'}")
        st.markdown(f"**Source:** `{cfg['key_source']}`")
        st.markdown(f"**Env path:** `{env.get('env_path', 'non trovato')}`")
        if cfg["masked_key"]:
            st.markdown(f"**Key:** `{cfg['masked_key']}`")
        st.caption("La chiave non viene mai mostrata per intero.")


# ── Sidebar ───────────────────────────────────────────────────────

st.sidebar.title("⚽ Football Analytics Hub")
st.sidebar.markdown("---")

# ── Demo mode detection ──────────────────────────────────────────
_demo_mode = is_demo_mode()
_demo_combos: list[tuple[str, str]] | None = None
if _demo_mode:
    _demo_combos = get_demo_cache_combinations(LEAGUES, SEASONS)
    if _demo_combos:
        st.sidebar.info(
            "**Demo mode**: this public deployment uses preloaded match data "
            "for reliable cloud access. Live FBref downloads remain available "
            "in local environments with Chrome installed."
        )
    else:
        st.sidebar.warning(
            "**Demo mode**: nessun dato pre caricato in data/demo_cache/. "
            "Installa Google Chrome per il download live da FBref."
        )

# ── Selectors ────────────────────────────────────────────────────
if _demo_mode and _demo_combos:
    _demo_leagues = sorted(set(l for l, s in _demo_combos))
    sel_league = st.sidebar.selectbox("Lega", _demo_leagues, key="sel_league")
    _demo_seasons_for_league = sorted(
        set(s for l, s in _demo_combos if l == sel_league)
    )
    sel_season = st.sidebar.selectbox(
        "Stagione", _demo_seasons_for_league, key="sel_season"
    )
else:
    sel_league = st.sidebar.selectbox("Lega", LEAGUES, key="sel_league")
    sel_season = st.sidebar.selectbox("Stagione", SEASONS, key="sel_season")
sel_max = st.sidebar.slider("Numero partite", 10, 380, 50, key="sel_max")

if st.sidebar.button("🚀 Carica dati", type="primary", use_container_width=True):
    # Pre-check: in demo mode, validate the combo before attempting load
    if _demo_mode and _demo_combos and (sel_league, sel_season) not in _demo_combos:
        _available = ", ".join(f"{l} {s}" for l, s in sorted(_demo_combos))
        st.info(
            f"Dati demo non disponibili per **{sel_league} {sel_season}**. "
            f"Combinazioni disponibili: {_available}."
        )
    else:
        with st.spinner("Caricamento dati e calcolo indici in corso..."):
            try:
                result = _load_and_compute_cached(sel_league, sel_season, sel_max)
                if result:
                    st.session_state["base_data"] = result
                    st.session_state["data"] = result
                    st.session_state["data_loaded"] = True
                    st.session_state["current_league"] = sel_league
                    st.session_state["current_season"] = sel_season
                    st.session_state["weights"] = {"pressure": 1.0, "discipline": 1.0, "network": 1.0}
                    # Resetta squadre selezionate se non presenti nella nuova lega/stagione
                    teams = result["teams"]
                    for key in ["team_a", "team_b"]:
                        val = st.session_state.get(key)
                        if val is not None and val not in teams:
                            st.session_state.pop(key, None)
                    st.success(f"Dati caricati: {result['n_matches']} partite, {result['n_records']} record, {len(teams)} squadre")
                    st.rerun()
            except RuntimeError as e:
                st.info(
                    "**Demo mode**: this public deployment uses preloaded match data "
                    "for reliable cloud access. Live FBref downloads remain available "
                    "in local environments with Chrome installed."
                )
            except ValueError as e:
                st.error("📭 Dati non disponibili")
                with st.expander("Dettaglio errore"):
                    st.markdown(
                        f"La combinazione **{sel_league} {sel_season}** non e' disponibile su FBref.\n\n"
                        "Possibili cause:\n"
                        "- La lega non esiste per questa stagione\n"
                        "- Il nome della lega non corrisponde ai codici FBref\n"
                        "- La stagione non e' ancora iniziata o e' troppo vecchia"
                    )
                    st.code(str(e))
            except ConnectionError as e:
                st.error("🌐 Connessione fallita")
                with st.expander("Dettaglio errore"):
                    st.markdown(
                        "Impossibile raggiungere FBref. Verifica la connessione internet.\n\n"
                        "Possibili cause:\n"
                        "- Blocco firewall o proxy\n"
                        "- FBref temporaneamente irraggiungibile\n"
                        "- Rate limiting di FBref (troppe richieste)"
                    )
                    st.code(str(e))
            except Exception as e:
                st.error(f"Errore durante il caricamento: {type(e).__name__}")
                with st.expander("Dettaglio errore"):
                    st.code(str(e))

data_loaded = st.session_state.get("data_loaded", False)

if data_loaded:
    d = st.session_state["data"]
    st.sidebar.info(f"📊 {d['n_matches']} partite, {d['n_records']} record")

    with st.sidebar.expander("⚙️ Pesi del modello", expanded=False):
        w_p = st.slider("Pressure", 0.0, 2.0, st.session_state.get("weights", {}).get("pressure", 1.0), 0.05, key="w_pressure")
        w_d = st.slider("Discipline", 0.0, 2.0, st.session_state.get("weights", {}).get("discipline", 1.0), 0.05, key="w_discipline")
        w_n = st.slider("Network", 0.0, 2.0, st.session_state.get("weights", {}).get("network", 1.0), 0.05, key="w_network")
        old_w = st.session_state.get("weights", {})
        if old_w and (old_w.get("pressure") != w_p or old_w.get("discipline") != w_d or old_w.get("network") != w_n):
            base = st.session_state["base_data"]
            base_avg = _team_avg_table(base["index_df"])
            new_data = _compute_with_weights(base, w_p, w_d, w_n)
            new_data["stats_df"] = base["stats_df"]
            new_data["pressure_df"] = base["pressure_df"]
            new_data["discipline_df"] = base["discipline_df"]
            new_data["network_df"] = base["network_df"]
            new_data["teams"] = base["teams"]
            new_data["n_matches"] = base["n_matches"]
            new_data["n_records"] = base["n_records"]
            new_avg = _team_avg_table(new_data["index_df"])
            st.session_state["weight_insights"] = _generate_weight_insight(base_avg, new_avg, old_w)
            st.session_state["data"] = new_data
            st.session_state["weights"] = {"pressure": w_p, "discipline": w_d, "network": w_n}
            st.rerun()
        weight_insights = st.session_state.get("weight_insights", [])
        if weight_insights:
            for ins in weight_insights:
                st.markdown(f"<small style='color:#9B59B6;'>📌 {ins}</small>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    team_a = st.sidebar.selectbox("Squadra 1", d["teams"], key="team_a")
    team_b = st.sidebar.selectbox("Squadra 2", d["teams"], key="team_b", index=min(1, len(d["teams"]) - 1))

    ai_cfg = get_ai_config()
    use_ai = st.sidebar.checkbox("🤖 Usa AI per report", value=False, disabled=not ai_cfg["enabled"], help=ai_cfg["message"], key="use_ai")
    if not ai_cfg["enabled"] and st.session_state.get("use_ai", False):
        st.sidebar.warning("AI non disponibile senza GROQ_API_KEY")
    elif ai_cfg["enabled"]:
        st.sidebar.success(ai_cfg["message"])
    _render_ai_debug()

    _render_glossary()

page = st.sidebar.radio("Pagina", PAGES, index=0)
st.sidebar.markdown("---")
st.sidebar.caption("TacticalPulse Index © 2025")

# ── Config summary ────────────────────────────────────────────────

def _render_config_summary():
    if not data_loaded:
        return
    w = st.session_state.get("weights", {"pressure": 1.0, "discipline": 1.0, "network": 1.0})
    ai_cfg = get_ai_config()
    ai_status = "🤖 Attiva" if ai_cfg["enabled"] and st.session_state.get("use_ai", False) else "📊 Data-only"
    st.caption(
        f"⚙️ {st.session_state['current_league']} · {st.session_state['current_season']} · "
        f"Pesi: P={w['pressure']:.2f} D={w['discipline']:.2f} N={w['network']:.2f} · "
        f"AI: {ai_status}"
    )


# ── Pagine ────────────────────────────────────────────────────────


def _page_overview():
    d = st.session_state["data"]
    index_df = d["index_df"]
    stats = d["stats_results"]
    avg = _team_avg_table(index_df)
    le = st.session_state["current_league"]
    se = st.session_state["current_season"]
    w = st.session_state.get("weights", {"pressure": 1.0, "discipline": 1.0, "network": 1.0})

    st.header(f"📊 TacticalPulse Index — {le} {se}")
    _render_config_summary()

    weight_insights = st.session_state.get("weight_insights", [])
    if weight_insights:
        for ins in weight_insights:
            _interpretation_box(ins, "⚙️")

    with st.expander("ℹ️ Cos'è TacticalPulse Index?", expanded=False):
        st.markdown("""
**TacticalPulse Index** è uno strumento di analisi tattica calcistica basato su dati reali **FBref**.
Supporta **5 leghe** (Premier League, Serie A, La Liga, Bundesliga, Ligue 1) e **3 stagioni** recenti.
Calcola un punteggio da **0 a 100** per ogni squadra, combinando 3 dimensioni:

- 🔴 **Pressure** — capacità di creare pericolo offensivo (tiri, tiri in porta, gol)
- 🟡 **Discipline** — controllo di falli e cartellini (più alto = più disciplinato)
- 🔵 **Network** — qualità e coesione del possesso/passaggio (possesso + bonus gol)

*Legenda punteggi:* 🔴 0–33 = Basso · 🟡 34–66 = Medio · 🟢 67–100 = Alto

**Pesi attuali:** Pressure **{w['pressure']:.2f}** · Discipline **{w['discipline']:.2f}** · Network **{w['network']:.2f}**
""")

    with st.expander("📖 Come leggere i risultati", expanded=False):
        st.markdown("""
- **Pressure alto** (≥67) → squadra offensivamente pericolosa, crea molte occasioni
- **Discipline alto** (≥67) → squadra controllata, commette pochi falli e prende pochi cartellini
- **Network alto** (≥67) → squadra che tiene possesso e costruisce bene il gioco
- **TacticalPulse alto** (≥67) → profilo complessivamente forte o bilanciato

*Una squadra può eccellere in una componente e avere carenze in un'altra: il profilo tattico completo dà un quadro più sfumato.*
""")

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Partite analizzate", d["n_matches"], help="Numero di partite analizzate per la lega selezionata")
        col2.metric("Media TacticalPulse", f"{stats.get('mean_score', 'N/A')}", help="Punteggio medio della lega. Confronta con la mediana per capire la distribuzione.")
        top_team = avg.head(1).index[0] if len(avg) > 0 else "N/A"
        top_score = avg.head(1)["tactical_pulse_score"].values[0] if len(avg) > 0 else "N/A"
        col3.metric("🥇 Prima", f"{top_team} ({top_score})", help="Squadra con il miglior TacticalPulse Score attuale")
        ttest = stats.get("ttest_top_vs_bottom")
        pval = ttest["p_value"] if ttest else "N/A"
        col4.metric("p-value (top vs bottom)", f"{pval:.4f}" if isinstance(pval, float) else pval,
                     help="Se p < 0.05, la differenza tra le migliori e peggiori squadre è statisticamente significativa")

    st.divider()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("🥇 Top 3")
        t3 = avg.head(3).reset_index()
        medals = ["🥇", "🥈", "🥉"]
        for i, (_, row) in enumerate(t3.iterrows()):
            level, icon = _interpret_score(row["tactical_pulse_score"])
            st.markdown(
                f"""<div style="padding:0.5rem 0.75rem;margin:0.25rem 0;border-radius:6px;
                background:{COLORS_BG['tactical_pulse']};border-left:3px solid {COLORS['tactical_pulse']};">
                <strong>{medals[i]} #{row['rank']} {row['team_id']}</strong> — 
                <span style="font-size:1.2rem;font-weight:700;color:{COLORS['tactical_pulse']};">{row['tactical_pulse_score']}</span>
                <span style="font-size:0.85rem;margin-left:0.5rem;">{icon} {level}</span>
                <span style="font-size:0.8rem;color:#888;display:block;margin-top:0.2rem;">Profilo: {row['profile']}</span></div>""",
                unsafe_allow_html=True,
            )
        st.subheader("📉 Bottom 3")
        b3 = avg.tail(3).reset_index()
        for i, (_, row) in enumerate(b3.iterrows()):
            level, icon = _interpret_score(row["tactical_pulse_score"])
            st.markdown(
                f"""<div style="padding:0.5rem 0.75rem;margin:0.25rem 0;border-radius:6px;
                background:#1c1c1c;border-left:3px solid #555;">
                <strong>#{row['rank']} {row['team_id']}</strong> — 
                <span style="font-size:1.1rem;font-weight:600;">{row['tactical_pulse_score']}</span>
                <span style="font-size:0.85rem;margin-left:0.5rem;">{icon} {level}</span>
                <span style="font-size:0.8rem;color:#888;display:block;margin-top:0.2rem;">Profilo: {row['profile']}</span></div>""",
                unsafe_allow_html=True,
            )

    with col_right:
        try:
            avg_sorted = avg.sort_values("tactical_pulse_score", ascending=True)
            fig_bar = px.bar(
                avg_sorted.reset_index(),
                x="tactical_pulse_score",
                y="team_id",
                orientation="h",
                title="TacticalPulse Score per Squadra",
                color="tactical_pulse_score",
                color_continuous_scale="Purples",
                labels={"team_id": "", "tactical_pulse_score": "Score"},
            )
            fig_bar.update_layout(height=500, yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.warning(f"Grafico non disponibile: {e}")
            st.dataframe(avg_sorted.reset_index()[["team_id", "tactical_pulse_score"]].sort_values("tactical_pulse_score", ascending=False),
                         use_container_width=True, hide_index=True)

    insights = _generate_overview_insights(avg)
    if insights:
        st.divider()
        st.subheader("💡 Insight automatici")
        for ins in insights:
            _interpretation_box(ins, "📌")

    st.divider()
    st.subheader("🏆 Classifica completa")
    show_top10 = st.toggle("Mostra solo top 10", value=True)
    display_df = avg.head(10).reset_index() if show_top10 else avg.reset_index()
    display_df = display_df.rename(columns={
        "team_id": "Squadra",
        "pressure_score": "Pressure",
        "discipline_score": "Discipline",
        "network_score": "Network",
        "tactical_pulse_score": "Totale",
        "rank": "Rank",
        "profile": "Profilo",
    })
    columns = ["Squadra", "Pressure", "Discipline", "Network", "Totale", "Rank", "Profilo"]
    display_df = display_df[columns]
    styled = _safe_style_table(display_df)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    col_exp_csv, _ = st.columns([1, 3])
    with col_exp_csv:
        _csv_download_button(display_df, f"ranking_{le}_{se}.csv", "⬇️ Scarica CSV ranking")


def _page_team_comparison():
    d = st.session_state["data"]
    index_df = d["index_df"]
    stats_df = d["stats_df"]
    teams = d["teams"]

    team_a = st.session_state.get("team_a", teams[0])
    team_b = st.session_state.get("team_b", teams[1] if len(teams) > 1 else teams[0])

    a_data = index_df[index_df["team_id"] == team_a]
    b_data = index_df[index_df["team_id"] == team_b]
    if a_data.empty or b_data.empty:
        st.warning("Seleziona due squadre valide dal sidebar.")
        return

    if len(a_data) < 2:
        st.warning(f"⚠️ {team_a} ha solo {len(a_data)} partita — i dati potrebbero non essere rappresentativi.")
    if len(b_data) < 2:
        st.warning(f"⚠️ {team_b} ha solo {len(b_data)} partita — i dati potrebbero non essere rappresentativi.")

    a_avg = a_data[["pressure_score", "discipline_score", "network_score"]].mean()
    b_avg = b_data[["pressure_score", "discipline_score", "network_score"]].mean()
    a_total = a_data["tactical_pulse_score"].mean()
    b_total = b_data["tactical_pulse_score"].mean()

    st.markdown(f"<h1 style='text-align:center;font-size:1.8rem;'>⚔️ {team_a} &nbsp;vs&nbsp; {team_b}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#aaa;margin-top:-0.5rem;'>Confronto diretto tra profili tattici</p>", unsafe_allow_html=True)
    _render_config_summary()
    st.divider()

    total_diff = a_total - b_total
    winner = team_a if total_diff > 0 else team_b
    abs_diff = abs(total_diff)
    pct_diff = _format_pct(total_diff, max(a_total, b_total))

    cols = st.columns([1, 2, 1])
    with cols[0]:
        level_a, icon_a = _interpret_score(a_total)
        st.markdown(f"<div style='text-align:center;padding:1rem;background:{COLORS_BG['tactical_pulse']};border-radius:10px;border:2px solid {COLORS['tactical_pulse']};'><div style='font-size:0.85rem;color:#aaa;'>{team_a}</div><div style='font-size:2.5rem;font-weight:800;color:{COLORS['tactical_pulse']};'>{a_total:.1f}</div><div style='font-size:0.8rem;'>{icon_a} {level_a}</div></div>", unsafe_allow_html=True)
    with cols[1]:
        direction = "▲" if total_diff > 0 else "▼"
        badge = f"<span style='background:{COLORS['tactical_pulse']};color:#111;padding:0.2rem 0.6rem;border-radius:4px;font-weight:700;font-size:0.9rem;'>Vince {winner}</span>" if abs_diff > 1 else "<span style='color:#aaa;'>Sostanziale pareggio</span>"
        diff_display = f"{abs_diff:.1f}" if abs_diff > 1 else "—"
        st.markdown(f"<div style='text-align:center;padding:1rem;'><div style='font-size:0.8rem;color:#888;'>Differenza</div><div style='font-size:2rem;font-weight:700;'>{direction} {diff_display}</div><div style='font-size:0.85rem;color:#aaa;margin-top:0.25rem;'>({pct_diff})</div><div style='margin-top:0.5rem;'>{badge}</div></div>", unsafe_allow_html=True)
    with cols[2]:
        level_b, icon_b = _interpret_score(b_total)
        st.markdown(f"<div style='text-align:center;padding:1rem;background:{COLORS_BG['tactical_pulse']};border-radius:10px;border:2px solid {COLORS['tactical_pulse']};'><div style='font-size:0.85rem;color:#aaa;'>{team_b}</div><div style='font-size:2.5rem;font-weight:800;color:{COLORS['tactical_pulse']};'>{b_total:.1f}</div><div style='font-size:0.8rem;'>{icon_b} {level_b}</div></div>", unsafe_allow_html=True)

    st.divider()
    cols_radar = st.columns([3, 2])
    with cols_radar[0]:
        try:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=a_avg.values.tolist() + [a_avg.values[0]], theta=["Pressure", "Discipline", "Network", "Pressure"], fill="toself", name=team_a, line_color=COLORS["pressure"], fillcolor="rgba(231, 76, 60, 0.15)"))
            fig_radar.add_trace(go.Scatterpolar(r=b_avg.values.tolist() + [b_avg.values[0]], theta=["Pressure", "Discipline", "Network", "Pressure"], fill="toself", name=team_b, line_color=COLORS["network"], fillcolor="rgba(52, 152, 219, 0.15)"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100], visible=True), bgcolor="rgba(0,0,0,0)"), title="Confronto profilo tattico", height=450, margin=dict(l=60, r=60, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.warning(f"Radar chart non disponibile: {e}")
            comp_df = pd.DataFrame({"Componente": ["Pressure", "Discipline", "Network"], team_a: a_avg.values, team_b: b_avg.values})
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with cols_radar[1]:
        st.subheader("📊 Componenti")
        for label, va, vb, color_key in [("Pressione offensiva", a_avg["pressure_score"], b_avg["pressure_score"], "pressure"), ("Disciplina", a_avg["discipline_score"], b_avg["discipline_score"], "discipline"), ("Network", a_avg["network_score"], b_avg["network_score"], "network")]:
            diff_v = va - vb
            w_name = team_a if diff_v > 0 else (team_b if diff_v < 0 else "—")
            w_icon = "▲" if diff_v > 0 else ("▼" if diff_v < 0 else "—")
            w_color = "white" if diff_v == 0 else (COLORS["network"] if diff_v > 0 else COLORS["pressure"])
            st.markdown(f"""<div style="padding:0.6rem 0.75rem;margin:0.35rem 0;border-radius:6px;background:{COLORS_BG[color_key]};border-left:3px solid {COLORS[color_key]};"><div style="font-size:0.8rem;color:#aaa;">{label}</div><div style="display:flex;justify-content:space-between;align-items:center;"><span style="font-weight:600;">{team_a}: <strong>{va:.1f}</strong></span><span style="font-size:0.9rem;color:{w_color};font-weight:700;">{w_icon} {w_name}</span><span style="font-weight:600;">{team_b}: <strong>{vb:.1f}</strong></span></div></div>""", unsafe_allow_html=True)
        try:
            bar_df = pd.DataFrame({"Componente": ["Pressure", "Discipline", "Network"], team_a: a_avg.values, team_b: b_avg.values})
            fig_side = px.bar(bar_df.melt(id_vars="Componente", var_name="Squadra", value_name="Score"), x="Componente", y="Score", color="Squadra", barmode="group", title="Confronto side-by-side", color_discrete_map={team_a: COLORS["pressure"], team_b: COLORS["network"]})
            fig_side.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig_side, use_container_width=True)
        except Exception as e:
            st.warning(f"Grafico a barre non disponibile: {e}")

    st.divider()
    st.subheader("💡 Analisi comparativa")
    ins = _generate_comparison_insights(team_a, team_b, a_avg, b_avg)
    for i_text in ins:
        _interpretation_box(i_text, "📌")

    st.divider()
    st.subheader("📋 Tabella comparativa")
    comp_rows = []
    for metric, col in [("Pressure Score", "pressure_score"), ("Discipline Score", "discipline_score"), ("Network Score", "network_score"), ("TacticalPulse Score", "tactical_pulse_score")]:
        va = round(a_data[col].mean(), 1)
        vb = round(b_data[col].mean(), 1)
        comp_rows.append({"Metrica": metric, team_a: va, team_b: vb, "Vince": team_a if va > vb else (team_b if vb > va else "Pareggio")})
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    _csv_download_button(comp_df, f"confronto_{team_a}_vs_{team_b}_{st.session_state['current_league']}.csv", "⬇️ Scarica CSV confronto")

    st.divider()
    st.subheader("📈 Statistiche dettagliate")
    tab_a, tab_b = st.tabs([team_a, team_b])
    for tab, raw, name in [(tab_a, stats_df[stats_df["team_id"] == team_a], team_a), (tab_b, stats_df[stats_df["team_id"] == team_b], team_b)]:
        with tab:
            if raw.empty:
                st.warning(f"Nessun dato per {name}")
                continue
            c1, c2, c3 = st.columns(3)
            c1.metric("Tiri medi (Sh)", f"{raw['Sh'].mean():.1f}" if "Sh" in raw.columns else "N/A", help="Media di tiri totali per partita")
            c2.metric("Tiri in porta (SoT)", f"{raw['SoT'].mean():.1f}" if "SoT" in raw.columns else "N/A", help="Media di tiri in porta per partita")
            c3.metric("Gol medi (Gls)", f"{raw['Gls'].mean():.1f}" if "Gls" in raw.columns else "N/A", help="Media di gol segnati per partita")
            c1.metric("Falli medi", f"{raw['Fls'].mean():.1f}" if "Fls" in raw.columns else "N/A", help="Media falli commessi per partita")
            c2.metric("Cart. gialli medi", f"{raw['CrdY'].mean():.1f}" if "CrdY" in raw.columns else "N/A", help="Media cartellini gialli per partita")
            c3.metric("Cart. rossi totali", f"{raw['CrdR'].sum()}" if "CrdR" in raw.columns else "N/A", help="Totale cartellini rossi nella stagione")
            c1.metric("Possesso medio %", f"{raw['Poss'].mean():.1f}" if "Poss" in raw.columns else "N/A", help="Possesso palla medio percentuale")

    st.divider()
    st.subheader("🤖 AI Insight")
    ai_cfg = get_ai_config()
    use_ai = st.session_state.get("use_ai", False)
    if not ai_cfg["enabled"]:
        st.warning("GROQ_API_KEY non configurata. Imposta GROQ_API_KEY nel file .env per l'analisi AI.")
    elif use_ai:
        if st.button("Genera analisi comparativa AI", type="primary", use_container_width=True):
            with st.spinner("Generazione analisi con CrewAI..."):
                text = _generate_comparison_ai_text(team_a, team_b, index_df, d["stats_results"])
                if text:
                    st.markdown(text)
                else:
                    st.warning("Impossibile generare analisi AI.")
    else:
        st.info("Attiva '🤖 Usa AI per report' nel sidebar per l'analisi comparativa con CrewAI.")


def _generate_comparison_ai_text(team_a: str, team_b: str, index_df: pd.DataFrame, stats_results: dict) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        return ""
    try:
        from crewai import Agent, Task, Crew, Process, LLM
        llm = LLM(model="openai/llama-3.3-70b-versatile", api_key=api_key, base_url="https://api.groq.com/openai/v1", temperature=0.3)
    except Exception:
        return ""
    subset = index_df[index_df["team_id"].isin([team_a, team_b])]
    avg = subset.groupby("team_id")[["pressure_score", "discipline_score", "network_score", "tactical_pulse_score"]].mean().round(1)
    context = f"Confronto tattico tra {team_a} e {team_b}:\n\n{avg.to_string()}\n\nStatistiche globali: media={stats_results.get('mean_score')}, CI 95%={stats_results.get('bootstrap_ci_95')}"
    writer = Agent(role="Tactical Analyst", goal="Scrivere analisi comparativa tra due squadre", backstory="Esperto di analisi tattica calcistica con focus su dati avanzati.", llm=llm, verbose=False)
    task = Task(description=f"Confronta {team_a} e {team_b} basandoti sui dati del TacticalPulse Index. Spiega le differenze nelle 3 componenti (Pressure, Discipline, Network) e quale squadra ha il profilo tattico migliore e perche'.\n\nDati:\n{context}", expected_output="Un paragrafo di 4-6 righe in italiano con l'analisi comparativa.", agent=writer)
    crew = Crew(agents=[writer], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)


def _page_single_team():
    d = st.session_state["data"]
    index_df = d["index_df"]
    teams = d["teams"]

    team = st.session_state.get("team_a", teams[0])
    team_data = index_df[index_df["team_id"] == team]
    if team_data.empty:
        st.warning("Nessun dato per la squadra selezionata.")
        return

    if len(team_data) < 2:
        st.warning(f"⚠️ {team} ha solo {len(team_data)} partita — i dati potrebbero non essere rappresentativi.")

    avg_score = team_data["tactical_pulse_score"].mean()
    avg_press = team_data["pressure_score"].mean()
    avg_disc = team_data["discipline_score"].mean()
    avg_net = team_data["network_score"].mean()

    st.header(f"🔍 Analisi approfondita: {team}")
    _render_config_summary()

    profile_label = _classify_profile(pd.Series({"pressure_score": avg_press, "discipline_score": avg_disc, "network_score": avg_net, "tactical_pulse_score": avg_score}))
    level, icon = _interpret_score(avg_score)
    strengths, weaknesses = _get_strengths_weaknesses(avg_press, avg_disc, avg_net)
    momentum = _compute_momentum(team_data)
    recent_avg, overall_avg = _compute_recent_form(team_data)

    momentum_icon = {"In crescita": "📈", "In calo": "📉", "Stabile": "➡️"}.get(momentum, "➡️")
    st.markdown(f"<p style='color:#aaa;'>Profilo tattico: <strong>{profile_label}</strong> &nbsp;·&nbsp; TacticalPulse Score: <strong>{avg_score:.1f}</strong> {icon} {level} &nbsp;·&nbsp; Momentum: {momentum_icon} <strong>{momentum}</strong></p>", unsafe_allow_html=True)

    col_str, col_weak = st.columns(2)
    with col_str:
        s_text = '<br>'.join(f'• {s}' for s in strengths)
        st.markdown(f"""<div style="padding:0.6rem 0.75rem;border-radius:8px;background:#0a2e1a;border-left:4px solid #2ecc71;"><strong style="color:#2ecc71;">✅ Punti di forza</strong><br>{s_text}</div>""", unsafe_allow_html=True)
    with col_weak:
        w_text = '<br>'.join(f'• {w}' for w in weaknesses)
        st.markdown(f"""<div style="padding:0.6rem 0.75rem;border-radius:8px;background:#2e0a0a;border-left:4px solid #e74c3c;"><strong style="color:#e74c3c;">⚠️ Punti deboli</strong><br>{w_text}</div>""", unsafe_allow_html=True)

    if recent_avg is not None and overall_avg is not None:
        form_diff = recent_avg - overall_avg
        form_text = f"📊 **Media ultime 5:** {recent_avg:.1f} vs **Media generale:** {overall_avg:.1f} ({'+' if form_diff > 0 else ''}{form_diff:.1f})"
        _interpretation_box(form_text, "📅")

    st.divider()

    def _gauge(value, title, color):
        return go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": title, "font": {"size": 16, "color": "#ccc"}}, gauge={"axis": {"range": [0, 100], "tickfont": {"color": "#aaa"}}, "bar": {"color": color}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0, "steps": [{"range": [0, 33], "color": "#2a2a2a"}, {"range": [33, 66], "color": "#3a3a3a"}, {"range": [66, 100], "color": "#4a4a4a"}], "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": value}}))

    col1, col2, col3, col4 = st.columns(4)
    try:
        col1.plotly_chart(_gauge(avg_score, "TacticalPulse", COLORS["tactical_pulse"]), use_container_width=True)
        col2.plotly_chart(_gauge(avg_press, "Pressure", COLORS["pressure"]), use_container_width=True)
        col3.plotly_chart(_gauge(avg_disc, "Discipline", COLORS["discipline"]), use_container_width=True)
        col4.plotly_chart(_gauge(avg_net, "Network", COLORS["network"]), use_container_width=True)
    except Exception as e:
        st.warning(f"Gauge chart non disponibile: {e}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TacticalPulse", f"{avg_score:.1f}")
        c2.metric("Pressure", f"{avg_press:.1f}")
        c3.metric("Discipline", f"{avg_disc:.1f}")
        c4.metric("Network", f"{avg_net:.1f}")

    st.markdown("<p style='text-align:center;color:#888;font-size:0.85rem;'>🔴 0–33 = Basso · 🟡 34–66 = Medio · 🟢 67–100 = Alto</p>", unsafe_allow_html=True)

    st.divider()
    col_explain, col_ranking = st.columns([1, 1])
    with col_explain:
        st.subheader("🧩 Breakdown TacticalPulse Score")
        try:
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Pressure", "Discipline", "Network"],
                values=[avg_press, avg_disc, avg_net],
                marker=dict(colors=[COLORS["pressure"], COLORS["discipline"], COLORS["network"]]),
                hole=0.5,
                textinfo="label+value",
                textposition="outside",
            )])
            fig_donut.update_layout(title=f"Contributo componenti — Totale {avg_score:.1f}", height=320, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)
        except Exception as e:
            st.warning(f"Donut chart non disponibile: {e}")
            breakdown = pd.DataFrame({"Componente": ["Pressure", "Discipline", "Network"], "Punteggio": [avg_press, avg_disc, avg_net]})
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

    with col_ranking:
        avg_all = _team_avg_table(index_df)
        if team in avg_all.index:
            pos = int(avg_all.loc[team, "rank"])
            st.markdown(f"**Posizione in classifica:** #{pos} / {len(avg_all)} squadre")
            st.markdown("**Dettaglio componenti rispetto alla lega:**")
            for label, val, color_key in [("Pressione offensiva", avg_press, "pressure"), ("Disciplina", avg_disc, "discipline"), ("Network", avg_net, "network")]:
                all_vals = avg_all[f"{color_key}_score"]
                pct = (all_vals < val).sum() / len(all_vals) * 100
                st.markdown(f"""<div style="margin:0.3rem 0;"><span style="color:{COLORS[color_key]};">■</span> {label}: <strong>{val:.1f}</strong> <span style="color:#888;font-size:0.85rem;">— migliore del {pct:.0f}% delle squadre</span></div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("📈 Evoluzione TacticalPulse Score")

    try:
        team_sorted = team_data.sort_values("game_id").reset_index(drop=True)
        team_sorted["match_num"] = range(1, len(team_sorted) + 1)
        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(x=team_sorted["match_num"], y=team_sorted["tactical_pulse_score"], mode="lines+markers", name="TacticalPulse", line=dict(color=COLORS["tactical_pulse"], width=2), marker=dict(size=6)))
        if len(team_sorted) >= 3:
            team_sorted["rolling_avg"] = team_sorted["tactical_pulse_score"].rolling(window=3, min_periods=1).mean()
            fig_evol.add_trace(go.Scatter(x=team_sorted["match_num"], y=team_sorted["rolling_avg"], mode="lines", name="Media mobile (3)", line=dict(color="#e8e8e8", width=2, dash="dot")))
        fig_evol.add_hline(y=avg_score, line_dash="dot", line_color="#888", annotation_text=f"Media {avg_score:.1f}")
        fig_evol.update_layout(title=f"Evoluzione {team}", height=350, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5))
        st.plotly_chart(fig_evol, use_container_width=True)
    except Exception as e:
        st.warning(f"Grafico evoluzione non disponibile: {e}")
        st.dataframe(team_data[["game_id", "tactical_pulse_score"]].head(20), use_container_width=True, hide_index=True)

    if len(team_data) >= 3:
        st.subheader("📊 Evoluzione componenti")
        try:
            team_sorted_c = team_data.sort_values("game_id").reset_index(drop=True)
            team_sorted_c["match_num"] = range(1, len(team_sorted_c) + 1)
            fig_comps = go.Figure()
            fig_comps.add_trace(go.Scatter(x=team_sorted_c["match_num"], y=team_sorted_c["pressure_score"], mode="lines+markers", name="Pressure", line=dict(color=COLORS["pressure"])))
            fig_comps.add_trace(go.Scatter(x=team_sorted_c["match_num"], y=team_sorted_c["discipline_score"], mode="lines+markers", name="Discipline", line=dict(color=COLORS["discipline"])))
            fig_comps.add_trace(go.Scatter(x=team_sorted_c["match_num"], y=team_sorted_c["network_score"], mode="lines+markers", name="Network", line=dict(color=COLORS["network"])))
            fig_comps.update_layout(title="Andamento Pressure, Discipline, Network", height=300, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5))
            st.plotly_chart(fig_comps, use_container_width=True)
        except Exception as e:
            st.warning(f"Grafico componenti non disponibile: {e}")

    st.divider()
    st.subheader("📊 Distribuzione punteggi")
    try:
        fig_box = px.box(team_data, y="tactical_pulse_score", title="Variabilità del TacticalPulse Score", points="all", labels={"tactical_pulse_score": "Score"}, color_discrete_sequence=[COLORS["tactical_pulse"]])
        fig_box.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig_box, use_container_width=True)
    except Exception as e:
        st.warning(f"Box chart non disponibile: {e}")

    csv_data = team_data[["game_id", "tactical_pulse_score", "pressure_score", "discipline_score", "network_score"]].head(50)
    _csv_download_button(csv_data, f"dettaglio_{team}.csv", "⬇️ Scarica CSV dettaglio")


def _page_ai_report():
    d = st.session_state["data"]
    index_df = d["index_df"]
    stats = d["stats_results"]
    le = st.session_state["current_league"]
    se = st.session_state["current_season"]

    st.header("🤖 Report Tattico Automatico")
    st.markdown("Report narrativo generato da **4 agenti CrewAI** (Analyst → Statistician → Writer → Critic) con **Llama 3.3 70B** su Groq. Se l'AI non è configurata, viene mostrato un report data-only con tabelle e statistiche descrittive.")
    _render_config_summary()
    st.divider()

    ai_cfg = get_ai_config()
    use_ai_key = st.session_state.get("use_ai", False)
    ai_available = ai_cfg["enabled"] and use_ai_key

    col_badge, col_info = st.columns([1, 3])
    with col_badge:
        if ai_cfg["enabled"]:
            st.markdown(f"""<div style="padding:0.5rem 1rem;border-radius:8px;background:#0a2e1a;border:1px solid #2ecc71;text-align:center;"><span style="font-size:1.5rem;">🤖</span><br><span style="color:#2ecc71;font-weight:700;">AI attiva</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="padding:0.5rem 1rem;border-radius:8px;background:#2e1a0a;border:1px solid #e67e22;text-align:center;"><span style="font-size:1.5rem;">📊</span><br><span style="color:#e67e22;font-weight:700;">Data-only</span></div>""", unsafe_allow_html=True)
    with col_info:
        if ai_cfg["enabled"]:
            st.info(f"🤖 {ai_cfg['message']}")
        else:
            st.warning(f"📊 {ai_cfg['message']}")
            st.markdown("Per abilitare l'AI:\n1. Crea un file **`.env`** con `GROQ_API_KEY=tua_chiave`\n2. Oppure **`.streamlit/secrets.toml`**\n3. **Riavvia** l'app")

    st.divider()

    report_path = _HERE / "reports" / "output" / f"tacticalpulse_{le.replace(' ', '_').replace('-', '_')}_{se}.md"
    if report_path.exists():
        mtime = datetime.fromtimestamp(report_path.stat().st_mtime)
        st.caption(f"📁 Ultimo report salvato su disco: {mtime.strftime('%d/%m/%Y %H:%M')}")

    scope = st.radio("Ambito del report", ["Tutta la lega", f"Solo {st.session_state.get('team_a', 'Squadra1')} vs {st.session_state.get('team_b', 'Squadra2')}"], horizontal=True)
    report_status = st.session_state.get("report_status", None)

    col_gen, col_load, col_dl = st.columns([1, 1, 1])
    with col_gen:
        label = "🚀 Genera Report AI" if ai_available else "📊 Genera Report"
        gen_btn = st.button(label, type="primary", use_container_width=True)
    with col_load:
        if report_path.exists() and st.button("📂 Ricarica ultimo report", use_container_width=True):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    st.session_state["last_report"] = f.read()
                st.session_state["report_status"] = "loaded_from_disk"
                st.success("Report ricaricato da disco.")
                st.rerun()
            except Exception as e:
                st.error(f"Errore caricamento report: {e}")
    with col_dl:
        if "last_report" in st.session_state and st.session_state["last_report"]:
            st.download_button(label="⬇️ Scarica report .md", data=st.session_state["last_report"].encode("utf-8"), file_name=f"tacticalpulse_{le}_{se}.md", mime="text/markdown", use_container_width=True)

    if report_status == "ai_success":
        st.success("✅ **Report AI generato con successo**", icon="🤖")
    elif report_status == "data_only":
        st.info("📊 **Report generato in modalità data-only** (AI non disponibile)")
    elif report_status == "loaded_from_disk":
        st.info("📂 **Report caricato da disco**")
    elif report_status == "ai_fallback":
        st.warning("⚠️ **AI non riuscita** — mostrato fallback data-only")

    if gen_btn:
        st.session_state["report_status"] = None
        if ai_available:
            with st.spinner("🧠 Generazione report con 4 agenti CrewAI... Attendi 30–60 secondi."):
                try:
                    from agents.orchestrator import _generate_llm_report, _get_llm
                    llm = _get_llm()
                    if scope.startswith("Solo"):
                        parts = scope.split(" vs ")
                        t_a = parts[0].replace("Solo ", "")
                        t_b = parts[1]
                        sub_idx = index_df[index_df["team_id"].isin([t_a, t_b])]
                        try:
                            sub_s = validate_significance(sub_idx)
                        except Exception:
                            sub_s = stats
                        report = _generate_llm_report(sub_idx, sub_s, le, se, None, llm)
                    else:
                        report = _generate_llm_report(index_df, stats, le, se, None, llm)
                    st.session_state["last_report"] = report
                    st.session_state["report_status"] = "ai_success"
                    st.rerun()
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error(f"Errore generazione report AI: {e}")
                    with st.expander("🔍 Dettaglio errore tecnico"):
                        st.code(tb, language="python")
                    from agents.orchestrator import _generate_data_only_report
                    report = _generate_data_only_report(index_df, stats, le, se)
                    st.session_state["last_report"] = report
                    st.session_state["report_status"] = "ai_fallback"
                    st.rerun()
        else:
            with st.spinner("Generazione report data-only..."):
                from agents.orchestrator import _generate_data_only_report
                report = _generate_data_only_report(index_df, stats, le, se)
                st.session_state["last_report"] = report
                st.session_state["report_status"] = "data_only"
                st.rerun()

    if "last_report" in st.session_state and st.session_state["last_report"]:
        st.divider()
        show_raw = st.checkbox("📄 Mostra report raw markdown", value=True)
        if show_raw:
            st.markdown(st.session_state["last_report"])
    elif not gen_btn and not report_status:
        st.info("Clicca **Genera Report** per creare il report tattico.")

    st.divider()
    with st.expander("💡 Come funziona l'AI"):
        st.markdown("Il report è generato da **4 agenti CrewAI** che lavorano in sequenza:\n\n1. **Analyst** — Analizza i dati crudi, identifica pattern e squadre anomale.\n2. **Statistician** — Valida statisticamente i pattern trovati (bootstrap, t-test).\n3. **Writer** — Scrive il report markdown professionale in italiano.\n4. **Critic** — Verifica che ogni affermazione sia supportata dai dati.\n\nIl LLM utilizzato è **Llama 3.3 70B** tramite **Groq API**, con prompt in italiano.")


def _page_statistical_validation():
    d = st.session_state["data"]
    index_df = d["index_df"]
    stats = d["stats_results"]

    st.header("📐 Validazione Statistica")
    st.markdown("Questa pagina verifica se i risultati del TacticalPulse Index sono **statisticamente robusti** — cioè se le differenze tra squadre sono reali e non dovute al caso. Usiamo **bootstrap** (ricampionamento casuale) e **t-test** per misurare l'affidabilità dei punteggi. **Tooltip:** passa il mouse sulle icone 🎲 e 📊 per capire ogni concetto.")
    _render_config_summary()
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""<div style="padding:0.75rem;border-radius:8px;background:{COLORS_BG['tactical_pulse']};border-left:4px solid {COLORS['tactical_pulse']};height:100%;" title="Il bootstrap ricampiona casualmente i dati per simulare la variabilità della stima."><strong>🎲 Cos'è il Bootstrap CI 95%?</strong><br><span style="color:#ccc;font-size:0.9rem;">Il bootstrap ricampiona casualmente i dati migliaia di volte per simulare "cosa succederebbe se rifacessimo l'analisi su dati simili". L'intervallo di confidenza al 95% indica dove cade il vero punteggio medio con una sicurezza del 95%. Più stretto è l'intervallo, più stabile è la stima.</span></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style="padding:0.75rem;border-radius:8px;background:{COLORS_BG['tactical_pulse']};border-left:4px solid {COLORS['tactical_pulse']};height:100%;" title="Il p-value misura la probabilità che una differenza sia dovuta al caso."><strong>📊 Cos'è il p-value?</strong><br><span style="color:#ccc;font-size:0.9rem;">Il p-value dice se la differenza tra due gruppi (es. top 5 vs bottom 5) è statisticamente significativa o potrebbe essere dovuta al caso.<br>• <strong style="color:#2ecc71;">p &lt; 0.05</strong> → differenza reale (significativa)<br>• <strong style="color:#e67e22;">p ≥ 0.05</strong> → differenza non dimostrabile</span></div>""", unsafe_allow_html=True)

    st.divider()

    ci = stats.get("bootstrap_ci_95")
    if ci:
        st.subheader("🎲 Bootstrap CI 95%")
        ci_low, ci_high = ci
        try:
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(x=["Media campionaria"], y=[stats.get("mean_score", 0)], mode="markers", marker=dict(size=14, color=COLORS["tactical_pulse"]), name="Media"))
            fig_ci.add_trace(go.Scatter(x=["Media campionaria", "Media campionaria"], y=[ci_low, ci_high], mode="lines", line=dict(width=3, color=COLORS["network"]), name=f"CI 95% [{ci_low:.1f}, {ci_high:.1f}]"))
            fig_ci.update_layout(title=f"Intervallo di confidenza al 95%: [{ci_low:.1f}, {ci_high:.1f}]", yaxis=dict(range=[max(0, ci_low - 10), min(100, ci_high + 10)]), height=300, showlegend=True, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig_ci, use_container_width=True)
        except Exception as e:
            st.warning(f"Grafico CI non disponibile: {e}")
            st.metric("CI 95%", f"[{ci_low:.2f}, {ci_high:.2f}]")
    else:
        st.info("Bootstrap CI 95% non disponibile.")

    if ci:
        ci_width = ci_high - ci_low
        stability = "molto stabile" if ci_width < 5 else ("moderatamente stabile" if ci_width < 10 else "variabile")
        stability_icon = "🟢" if ci_width < 5 else ("🟡" if ci_width < 10 else "🔴")
        _interpretation_box(f"L'intervallo di confidenza ha ampiezza **{ci_width:.1f} punti**: la stima è **{stability}**. {stability_icon}", "📌")

    st.divider()
    st.subheader("📊 Distribuzione TacticalPulse Score")
    if not index_df.empty:
        try:
            fig_box = px.box(index_df, y="tactical_pulse_score", title="Distribuzione dei punteggi (tutte le squadre, tutte le partite)", points="all", labels={"tactical_pulse_score": "Score"}, color_discrete_sequence=[COLORS["tactical_pulse"]])
            fig_box.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.warning(f"Box chart non disponibile: {e}")
    else:
        st.info("Nessun dato per la distribuzione.")

    st.divider()
    ttest = stats.get("ttest_top_vs_bottom")
    if ttest:
        st.subheader("⚔️ Confronto Top 5 vs Bottom 5")
        st.markdown("Abbiamo raggruppato le 5 squadre con il punteggio più alto e le 5 con il più basso, poi abbiamo confrontato le loro medie con un **t-test** (Welch).")
        col1, col2, col3 = st.columns(3)
        col1.metric("🏆 Top 5 media", f"{ttest['top5_mean']:.2f}")
        col2.metric("📉 Bottom 5 media", f"{ttest['bottom5_mean']:.2f}")
        p_val = ttest["p_value"]
        is_sig = ttest["significant"]
        p_color = "#2ecc71" if is_sig else "#e67e22"
        p_icon = "🟢 Significativo" if is_sig else "🟡 Non significativo"
        col3.markdown(f"""<div style="padding:0.5rem;border-radius:6px;background:#1a1a2e;text-align:center;"><div style="font-size:0.85rem;color:#aaa;">p-value</div><div style="font-size:1.5rem;font-weight:700;color:{p_color};">{p_val:.4f}</div><div style="font-size:0.85rem;color:{p_color};">{p_icon}</div></div>""", unsafe_allow_html=True)
        try:
            fig_ttest = px.bar(x=["Top 5", "Bottom 5"], y=[ttest["top5_mean"], ttest["bottom5_mean"]], color=["Top 5", "Bottom 5"], color_discrete_map={"Top 5": COLORS["network"], "Bottom 5": COLORS["pressure"]}, title=f"Differenza Top 5 vs Bottom 5 (p = {p_val:.4f})", labels={"x": "", "y": "Media TacticalPulse Score"}, text=[f"{ttest['top5_mean']:.1f}", f"{ttest['bottom5_mean']:.1f}"])
            fig_ttest.update_traces(textposition="outside", texttemplate="%{text}")
            fig_ttest.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig_ttest, use_container_width=True)
        except Exception as e:
            st.warning(f"Grafico t-test non disponibile: {e}")

        if is_sig:
            _interpretation_box("Il test è **statisticamente significativo** (p < 0.05): la differenza tra top 5 e bottom 5 è reale, non casuale. Il TacticalPulse Index distingue efficacemente squadre forti da squadre deboli.", "✅")
        else:
            _interpretation_box("Il test **non** è statisticamente significativo (p ≥ 0.05): la differenza osservata potrebbe essere dovuta al caso. Servono più dati o la lega potrebbe essere molto equilibrata.", "⚠️")
    else:
        st.info("Confronto Top 5 vs Bottom 5 non disponibile (servono almeno 10 squadre).")

    st.divider()
    st.subheader("📋 Statistiche descrittive complete")
    desc_rows = [
        ("Media", stats.get("mean_score", "N/A")), ("Mediana", stats.get("median_score", "N/A")),
        ("Deviazione Standard", stats.get("std_score", "N/A")), ("Minimo", stats.get("min_score", "N/A")),
        ("Massimo", stats.get("max_score", "N/A")), ("N. Osservazioni (partite×squadre)", stats.get("n_observations", "N/A")),
        ("N. Squadre", stats.get("n_teams", "N/A")),
    ]
    corrs = stats.get("component_correlations", {})
    for k, v in corrs.items():
        desc_rows.append((f"Correlazione {k.replace('_vs_', ' vs ').replace('_score', '').replace('_', ' ').title()}", f"{v:.4f}"))
    desc_df = pd.DataFrame(desc_rows, columns=["Metrica", "Valore"])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)
    _csv_download_button(desc_df, f"statistiche_{st.session_state['current_league']}.csv", "⬇️ Scarica CSV statistiche")

    if corrs:
        st.divider()
        st.subheader("🔗 Correlazioni tra componenti")
        for k, v in corrs.items():
            name_parts = k.replace("_vs_", " vs ").replace("_score", "").replace("_", " ").title()
            strength = "correlazione debole" if abs(v) < 0.3 else ("correlazione moderata" if abs(v) < 0.7 else "correlazione forte")
            direction = "positiva" if v > 0 else "negativa"
            extra = "Le due componenti tendono a muoversi insieme." if v > 0 else "Le due componenti tendono a muoversi in direzione opposta."
            _interpretation_box(f"{name_parts}: **{v:+.4f}** ({strength}, {direction}). {extra}", "📌")


# ── Rendering ─────────────────────────────────────────────────────

if not data_loaded:
    st.title("⚽ Football Analytics Hub")
    ai_cfg = get_ai_config()
    ai_badge = "🤖 AI attivo" if ai_cfg["enabled"] else "📊 Modalità data-only"
    st.markdown(f"**Stato AI:** {ai_badge} — {ai_cfg['message']}")

    with st.expander("ℹ️ Cos'è TacticalPulse Index?", expanded=True):
        st.markdown("""
**TacticalPulse Index** è uno strumento di analisi tattica calcistica basato su dati reali **FBref**.

Supporta **5 leghe** (Premier League, Serie A, La Liga, Bundesliga, Ligue 1) e **3 stagioni** (2022-2023, 2023-2024, 2024-2025).

Calcola un punteggio da **0 a 100** per ogni squadra, combinando 3 dimensioni:
- 🔴 **Pressure** — capacità di creare pericolo offensivo (tiri, tiri in porta, gol)
- 🟡 **Discipline** — controllo di falli e cartellini (più alto = più disciplinato)
- 🔵 **Network** — qualità e coesione del possesso/passaggio (possesso + bonus gol)

Il punteggio finale (**TacticalPulse Score**) serve per confrontare squadre, identificare profili tattici e punti di forza/debolezza.

*Legenda punteggi:* 🔴 0–33 = Basso · 🟡 34–66 = Medio · 🟢 67–100 = Alto

**Come usare la dashboard:**
1. Scegli **lega, stagione e numero partite** nel sidebar a sinistra
2. Clicca **"Carica dati"** per scaricare e analizzare
3. Sfoglia le pagine: **Overview**, **Team Comparison**, **Single Team Deep Dive**, **AI Report**, **Statistical Validation**
4. Regola i **pesi del modello** nel sidebar per esplorare scenari tattici alternativi
""")

    _render_glossary()

    st.info("Seleziona lega, stagione e numero di partite nel sidebar, poi clicca **\"🚀 Carica dati\"** per iniziare.")
else:
    if page == "Overview":
        _page_overview()
    elif page == "Team Comparison":
        _page_team_comparison()
    elif page == "Single Team Deep Dive":
        _page_single_team()
    elif page == "AI Report":
        _page_ai_report()
    elif page == "Statistical Validation":
        _page_statistical_validation()
