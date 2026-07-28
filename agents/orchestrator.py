"""Orchestrator della pipeline CrewAI per il TacticalPulse Index.

Esegue la sequenza: Analisi -> Validazione -> Report -> Revisione.
Con fallback data-only se GROQ_API_KEY non e' configurata.
"""

import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.config import get_groq_api_key, load_environment
from core.data_loader import load_events
from core.discipline import compute_discipline
from core.index_builder import build_index
from core.network import compute_network
from core.pressure import compute_pressure
from stats.significance import validate_significance

# Carica ambiente in modo robusto
load_environment()

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "output"


def _get_llm():
    """Crea un'istanza LLM CrewAI per Groq, o None se la API key manca o e' placeholder."""
    api_key = get_groq_api_key()
    if not api_key:
        return None
    try:
        from crewai import LLM

        return LLM(
            model="openai/llama-3.3-70b-versatile",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
        )
    except Exception as e:
        warnings.warn(f"Impossibile inizializzare LLM: {e}")
        return None


def _compute_index_data(
    league: str,
    season: str,
    weights: dict[str, float] | None = None,
    max_matches: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Esegue la pipeline di calcolo completa.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (index_df, stats_results)
    """
    events = load_events(league, season, max_matches=max_matches)

    pressure_df = compute_pressure(events)
    discipline_df = compute_discipline(events)
    network_df = compute_network(events)

    index_df = build_index(pressure_df, discipline_df, network_df, weights=weights)

    stats_results = validate_significance(index_df)

    return index_df, stats_results


def _generate_data_only_report(
    index_df: pd.DataFrame,
    stats_results: dict,
    league: str,
    season: str,
    weights: dict[str, float] | None = None,
) -> str:
    """Genera un report data-only (tabelle, nessuna narrazione LLM)."""
    w = weights or {"pressure": 1 / 3, "discipline": 1 / 3, "network": 1 / 3}
    lines = []
    lines.append(f"# TacticalPulse Index Report (Data-Only)")
    lines.append(f"**Lega:** {league} | **Stagione:** {season}")
    lines.append(f"**Generato il:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("> ⚠️ **Modalita data-only**: GROQ_API_KEY non configurata. ")
    lines.append("> Il report mostra solo i dati calcolati senza narrazione LLM. ")
    lines.append("> Imposta GROQ_API_KEY nel file .env per un report completo.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Pesi utilizzati")
    lines.append(f"- Pressure: {w['pressure']*100:.0f}%")
    lines.append(f"- Discipline: {w['discipline']*100:.0f}%")
    lines.append(f"- Network: {w['network']*100:.0f}%")
    lines.append("")

    lines.append("## Ranking Squadre per TacticalPulse Score")
    lines.append("")
    team_avg = (
        index_df.groupby("team_id")[
            ["pressure_score", "discipline_score", "network_score", "tactical_pulse_score"]
        ]
        .mean()
        .round(1)
        .sort_values("tactical_pulse_score", ascending=False)
        .reset_index()
    )
    team_avg.index = range(1, len(team_avg) + 1)
    lines.append(team_avg.to_markdown(index=True, floatfmt=".1f"))
    lines.append("")

    lines.append("## Statistiche Descrittive")
    lines.append("")
    stats_table = pd.DataFrame(
        {
            "Metrica": [
                "Media", "Mediana", "Dev. Std", "Min", "Max", "N. Osservazioni"
            ],
            "Valore": [
                stats_results.get("mean_score", "N/A"),
                stats_results.get("median_score", "N/A"),
                stats_results.get("std_score", "N/A"),
                stats_results.get("min_score", "N/A"),
                stats_results.get("max_score", "N/A"),
                stats_results.get("n_observations", "N/A"),
            ],
        }
    )
    lines.append(stats_table.to_markdown(index=False))
    lines.append("")

    if stats_results.get("bootstrap_ci_95"):
        ci = stats_results["bootstrap_ci_95"]
        lines.append(f"**Bootstrap CI 95%**: [{ci[0]}, {ci[1]}]")
        lines.append("")

    ttest = stats_results.get("ttest_top_vs_bottom")
    if ttest:
        lines.append("## Confronto Top 5 vs Bottom 5")
        lines.append("")
        lines.append(f"- **Top 5 media**: {ttest['top5_mean']}")
        lines.append(f"- **Bottom 5 media**: {ttest['bottom5_mean']}")
        lines.append(f"- **p-value**: {ttest['p_value']:.4f}")
        lines.append(f"- **Significativo**: {'Si' if ttest['significant'] else 'No'}")
        lines.append("")

    if stats_results.get("component_correlations"):
        lines.append("## Correlazioni tra Componenti")
        lines.append("")
        for k, v in stats_results["component_correlations"].items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("---")
    lines.append("*Report generato automaticamente da TacticalPulse Index.*")
    lines.append("")

    return "\n".join(lines)


def _generate_llm_report(
    index_df: pd.DataFrame,
    stats_results: dict,
    league: str,
    season: str,
    weights: dict[str, float] | None = None,
    llm=None,
) -> str:
    """Genera un report narrativo usando CrewAI con LLM."""
    from crewai import Crew, Process, Task

    from agents.analyst_agent import create_analyst_agent
    from agents.critic_agent import create_critic_agent
    from agents.statistician_agent import create_statistician_agent
    from agents.writer_agent import create_writer_agent

    analyst = create_analyst_agent(llm=llm)
    statistician = create_statistician_agent(llm=llm)
    writer = create_writer_agent(llm=llm)
    critic = create_critic_agent(llm=llm)

    team_ranking = (
        index_df.groupby("team_id")[
            ["pressure_score", "discipline_score", "network_score", "tactical_pulse_score"]
        ]
        .mean()
        .round(1)
        .sort_values("tactical_pulse_score", ascending=False)
        .to_csv()
    )

    context_data = f"""
Lega: {league}
Stagione: {season}
Pesi: {weights or 'equal-weight (33.3% ciascuno)'}

--- Ranking Squadre ---
{team_ranking}

--- Statistiche ---
Media: {stats_results.get('mean_score')}
Mediana: {stats_results.get('median_score')}
Dev Std: {stats_results.get('std_score')}
Min: {stats_results.get('min_score')}
Max: {stats_results.get('max_score')}
Bootstrap CI 95%: {stats_results.get('bootstrap_ci_95')}
T-test top vs bottom: {stats_results.get('ttest_top_vs_bottom')}
Correlazioni: {stats_results.get('component_correlations')}
"""

    task_analysis = Task(
        description=(
            "Analizza i dati del TacticalPulse Index per la lega e stagione specificate. "
            "Identifica le squadre con punteggi anomali, pattern interessanti nelle "
            "tre componenti (Pressure, Discipline, Network), e fornisci una sintesi "
            "tecnica dei risultati.\n\n"
            f"Dati:\n{context_data}"
        ),
        expected_output=(
            "Una sintesi tecnica in italiano con l'analisi delle squadre top/bottom, "
            "pattern rilevati per ogni componente, e possibili interpretazioni tattiche."
        ),
        agent=analyst,
    )

    task_validation = Task(
        description=(
            "Valida statisticamente i risultati dell'analisi. Verifica se i pattern "
            "identificati sono statisticamente significativi o possono essere rumore. "
            "Usa i dati di bootstrap, t-test e correlazioni forniti.\n\n"
            f"Dati statistici:\n{stats_results}"
        ),
        expected_output=(
            "Un resoconto statistico in italiano che indica quali pattern sono "
            "significativi (p<0.05), quali no, e perche'. Includi gli intervalli "
            "di confidenza e le correlazioni tra componenti."
        ),
        agent=statistician,
        context=[task_analysis],
    )

    task_report = Task(
        description=(
            "Scrivi un report markdown professionale in italiano sul TacticalPulse Index "
            "per la lega e stagione analizzate. Includi:\n"
            "1. Introduzione sul TacticalPulse Index\n"
            "2. Ranking completo delle squadre con punteggi\n"
            "3. Highlight dei pattern piu interessanti per ogni componente\n"
            "4. Validazione statistica dei risultati\n"
            "5. Conclusioni tattiche\n\n"
            "Stile leggibile e professionale, adatto a un pubblico di appassionati di calcio.\n\n"
            f"Dati:\n{context_data}"
        ),
        expected_output=(
            "Un report markdown completo in italiano, ben formattato, con sezioni "
            "chiare, tabelle e interpretazioni tattiche."
        ),
        agent=writer,
        context=[task_analysis, task_validation],
    )

    task_critique = Task(
        description=(
            "Verifica che ogni affermazione nel report sia supportata dai dati "
            "calcolati. Controlla che i numeri citati corrispondano ai dati reali, "
            "che le conclusioni siano coerenti con le statistiche, e che non ci "
            "siano esagerazioni o interpretazioni errate. Se tutto e' corretto, "
            "approva il report. Se trovi errori, elenca le correzioni necessarie.\n\n"
            f"Dati di riferimento:\n{context_data}"
        ),
        expected_output=(
            "Report finale approvato in markdown, oppure una lista di correzioni "
            "necessarie con spiegazioni dettagliate."
        ),
        agent=critic,
        context=[task_analysis, task_validation, task_report],
    )

    crew = Crew(
        agents=[analyst, statistician, writer, critic],
        tasks=[task_analysis, task_validation, task_report, task_critique],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


def run_pipeline(
    league: str = "ENG-Premier League",
    season: str = "2024-2025",
    weights: dict[str, float] | None = None,
    save_report: bool = True,
) -> str:
    """Esegue la pipeline completa del TacticalPulse Index.

    1. Calcola le 3 componenti e l'indice composito (core/)
    2. Valida statisticamente i risultati (stats/)
    3. Genera report narrativo con CrewAI o fallback data-only

    Parameters
    ----------
    league : str
        Identificativo lega, es. "ENG-Premier League".
    season : str
        Stagione, es. "2024-2025".
    weights : dict, optional
        Pesi personalizzati per le 3 componenti.
    save_report : bool
        Se True, salva il report in reports/output/.

    Returns
    -------
    str
        Report markdown generato.
    """
    print(f"[TacticalPulse] Calcolo indice per {league} {season}...")
    index_df, stats_results = _compute_index_data(league, season, weights=weights)
    print(f"[TacticalPulse] Calcolati {len(index_df)} record.")

    llm = _get_llm()
    if llm is None:
        print("[TacticalPulse] GROQ_API_KEY non trovata -> modalita data-only.")
        report = _generate_data_only_report(index_df, stats_results, league, season, weights)
    else:
        print("[TacticalPulse] Groq LLM configurato -> generazione report narrativo...")
        try:
            report = _generate_llm_report(index_df, stats_results, league, season, weights, llm)
        except Exception as e:
            print(f"[TacticalPulse] Errore generazione LLM: {e}")
            print("[TacticalPulse] Fallback a report data-only.")
            report = _generate_data_only_report(index_df, stats_results, league, season, weights)

    if save_report:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = league.replace(" ", "_").replace("-", "_")
        fname = f"tacticalpulse_{safe_name}_{season}.md"
        path = _REPORTS_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[TacticalPulse] Report salvato in {path}")

    return report
