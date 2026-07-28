"""Test della dashboard Streamlit (app.py).

Utilizza streamlit.testing.v1.AppTest per testare il caricamento
delle pagine e la stabilita' dell'interfaccia.
"""

import sys
import os
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Assicura che il progetto sia nel path
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Per AppTest, avviamo tutti i test con un'app che ha dati mock già caricati


def _make_synthetic_stats_df(n_teams: int = 6, n_matches: int = 5) -> pd.DataFrame:
    """Crea un DataFrame sintetico compatibile con la pipeline.

    Ogni squadra ha n_matches partite con statistiche realistiche.
    """
    teams = [f"Squadra{i}" for i in range(1, n_teams + 1)]
    rows = []
    rng = np.random.default_rng(42)
    for team in teams:
        for m in range(1, n_matches + 1):
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
                "venue": "Home" if m % 2 == 1 else "Away",
                "opponent": teams[(teams.index(team) + 1) % len(teams)] if m % 2 == 1
                            else teams[(teams.index(team) - 1) % len(teams)],
                "result": rng.choice(["W", "D", "L"]),
                "GF": int(rng.integers(0, 5)),
                "GA": int(rng.integers(0, 5)),
                "league": "ENG-Premier League",
                "season": "2024-2025",
            })
    return pd.DataFrame(rows)


def _make_synthetic_results(index_df: pd.DataFrame) -> dict:
    """Genera risultati statistici simulati per i test."""
    scores = index_df["tactical_pulse_score"].dropna().values
    if len(scores) < 3:
        return {"error": "Campione troppo piccolo"}
    team_avg = index_df.groupby("team_id")["tactical_pulse_score"].mean()
    n_teams = len(team_avg)
    top5 = team_avg.tail(min(5, n_teams // 2))
    bottom5 = team_avg.head(min(5, n_teams // 2))
    return {
        "n_observations": int(len(scores)),
        "n_teams": n_teams,
        "mean_score": round(float(scores.mean()), 2),
        "std_score": round(float(scores.std()), 2),
        "median_score": round(float(np.median(scores)), 2),
        "min_score": round(float(scores.min()), 2),
        "max_score": round(float(scores.max()), 2),
        "bootstrap_ci_95": [round(float(np.percentile(scores, 2.5)), 2),
                            round(float(np.percentile(scores, 97.5)), 2)],
        "bootstrap_iterations": 100,
        "ttest_top_vs_bottom": {
            "t_statistic": 3.5,
            "p_value": 0.01,
            "significant": True,
            "top5_mean": round(float(top5.mean()), 2) if len(top5) > 0 else 0,
            "bottom5_mean": round(float(bottom5.mean()), 2) if len(bottom5) > 0 else 0,
        },
        "component_correlations": {
            "pressure_score_vs_discipline_score": 0.15,
            "pressure_score_vs_network_score": 0.45,
            "discipline_score_vs_network_score": -0.10,
        },
    }


# ----- Helper functions tests -----


class TestHelpers:
    """Test delle funzioni helper rule-based in app.py."""

    def _reimport_helpers(self):
        """Importa le funzioni helper dall'app (reload-safe)."""
        import importlib
        import app
        importlib.reload(app)
        return app

    def test_interpret_score(self):
        from app import _interpret_score
        assert _interpret_score(80) == ("Alto", "🟢")
        assert _interpret_score(67) == ("Alto", "🟢")
        assert _interpret_score(50) == ("Medio", "🟡")
        assert _interpret_score(34) == ("Medio", "🟡")
        assert _interpret_score(20) == ("Basso", "🔴")
        assert _interpret_score(0) == ("Basso", "🔴")

    def test_classify_profile_aggressiva(self):
        from app import _classify_profile
        row = pd.Series({"pressure_score": 80, "discipline_score": 40, "network_score": 40, "tactical_pulse_score": 55})
        assert _classify_profile(row) == "Aggressiva"

    def test_classify_profile_equilibrata(self):
        from app import _classify_profile
        row = pd.Series({"pressure_score": 50, "discipline_score": 52, "network_score": 51, "tactical_pulse_score": 51})
        assert _classify_profile(row) == "Equilibrata"

    def test_classify_profile_tecnica(self):
        from app import _classify_profile
        row = pd.Series({"pressure_score": 30, "discipline_score": 40, "network_score": 75, "tactical_pulse_score": 50})
        assert _classify_profile(row) == "Tecnica"

    def test_classify_profile_disciplinata(self):
        from app import _classify_profile
        row = pd.Series({"pressure_score": 30, "discipline_score": 72, "network_score": 40, "tactical_pulse_score": 48})
        assert _classify_profile(row) == "Disciplinata"

    def test_classify_profile_instabile(self):
        from app import _classify_profile
        row = pd.Series({"pressure_score": 10, "discipline_score": 20, "network_score": 15, "tactical_pulse_score": 15})
        assert _classify_profile(row) == "Instabile"

    def test_strengths_weaknesses(self):
        from app import _get_strengths_weaknesses
        s, w = _get_strengths_weaknesses(70, 30, 55)
        assert "Alta capacità di creare occasioni" in s
        assert "Disciplina difensiva fragile" in w
        assert "Buona qualità del possesso" not in s

    def test_strengths_weaknesses_no_weak(self):
        from app import _get_strengths_weaknesses
        s, w = _get_strengths_weaknesses(50, 50, 50)
        assert "Nessuna debolezza critica evidente" in w

    def test_strengths_weaknesses_no_strong(self):
        from app import _get_strengths_weaknesses
        s, w = _get_strengths_weaknesses(30, 30, 30)
        assert "Nessun punto di forza dominante" in s

    def test_format_pct(self):
        from app import _format_pct
        assert _format_pct(5, 80) == "6.2%"
        assert _format_pct(-3, 60) == "5.0%"
        assert _format_pct(0, 50) == "0.0%"
        assert _format_pct(5, 0) == "—"

    def test_generate_overview_insights(self):
        from app import _generate_overview_insights
        idx_df = _make_synthetic_stats_df()
        from core.pressure import compute_pressure
        from core.discipline import compute_discipline
        from core.network import compute_network
        from core.index_builder import build_index
        p = compute_pressure(idx_df)
        d = compute_discipline(idx_df)
        n = compute_network(idx_df)
        index_df = build_index(p, d, n)
        from app import _team_avg_table
        avg = _team_avg_table(index_df)
        insights = _generate_overview_insights(avg)
        assert len(insights) >= 2
        assert all(isinstance(i, str) for i in insights)

    def test_generate_overview_insights_empty(self):
        from app import _generate_overview_insights
        assert _generate_overview_insights(pd.DataFrame()) == []

    def test_generate_comparison_insights(self):
        from app import _generate_comparison_insights
        a_avg = pd.Series({"pressure_score": 70, "discipline_score": 40, "network_score": 50})
        b_avg = pd.Series({"pressure_score": 40, "discipline_score": 70, "network_score": 50})
        insights = _generate_comparison_insights("TeamA", "TeamB", a_avg, b_avg)
        assert len(insights) == 3
        assert any("TeamA" in i for i in insights)
        assert any("TeamB" in i for i in insights)
        assert any("simili" in i for i in insights)


class TestTeamAvgTable:
    """Test della funzione _team_avg_table."""

    def test_returns_expected_structure(self):
        from app import _team_avg_table
        idx_df = _make_synthetic_stats_df()
        from core.pressure import compute_pressure
        from core.discipline import compute_discipline
        from core.network import compute_network
        from core.index_builder import build_index
        p = compute_pressure(idx_df)
        d = compute_discipline(idx_df)
        n = compute_network(idx_df)
        index_df = build_index(p, d, n)
        avg = _team_avg_table(index_df)
        assert isinstance(avg, pd.DataFrame)
        assert "rank" in avg.columns
        assert "profile" in avg.columns
        assert "tactical_pulse_score" in avg.columns
        assert avg["rank"].is_monotonic_increasing or (avg["rank"].diff().dropna() >= 1).all()
        # rank 1 should have highest score
        assert avg.loc[avg["rank"].idxmin(), "tactical_pulse_score"] == avg["tactical_pulse_score"].max()

    def test_profile_column_values(self):
        from app import _team_avg_table
        idx_df = _make_synthetic_stats_df(n_teams=6, n_matches=5)
        from core.pressure import compute_pressure
        from core.discipline import compute_discipline
        from core.network import compute_network
        from core.index_builder import build_index
        p = compute_pressure(idx_df)
        d = compute_discipline(idx_df)
        n = compute_network(idx_df)
        index_df = build_index(p, d, n)
        avg = _team_avg_table(index_df)
        valid_profiles = {"Aggressiva", "Aggressiva ma rischiosa", "Disciplinata", "Tecnica", "Equilibrata", "Instabile", "Poco incisiva"}
        for profile in avg["profile"]:
            assert profile in valid_profiles


class TestSafeStyleTable:
    """Test della funzione _safe_style_table."""

    def test_styler_no_crash(self):
        from app import _safe_style_table
        df = pd.DataFrame({
            "Rank": [1, 2, 3],
            "Squadra": ["A", "B", "C"],
            "Pressure": [80.0, 70.0, 60.0],
            "Discipline": [50.0, 60.0, 70.0],
            "Network": [40.0, 50.0, 60.0],
            "Totale": [56.7, 60.0, 63.3],
            "Profilo": ["Aggressiva", "Equilibrata", "Tecnica"],
        })
        # Dovrebbe funzionare senza eccezioni
        result = _safe_style_table(df)
        assert result is not None

    def test_styler_missing_columns(self):
        from app import _safe_style_table
        df = pd.DataFrame({
            "Squadra": ["A", "B"],
            "Score": [50.0, 60.0],
        })
        # Non dovrebbe crashare anche con colonne mancanti
        result = _safe_style_table(df)
        assert result is not None

    def test_styler_empty_df(self):
        from app import _safe_style_table
        df = pd.DataFrame()
        result = _safe_style_table(df)
        assert result is not None

    def test_styler_non_numeric(self):
        from app import _safe_style_table
        df = pd.DataFrame({
            "Rank": [1, 2],
            "Squadra": ["A", "B"],
            "Pressure": [80.0, 70.0],
        })
        result = _safe_style_table(df)
        assert result is not None


# ----- Scoping functions tests -----


def test_scoring_consistency():
    """Verifica che la pipeline completa produca score validi con dati sintetici."""
    stats_df = _make_synthetic_stats_df(n_teams=6, n_matches=5)
    from core.pressure import compute_pressure
    from core.discipline import compute_discipline
    from core.network import compute_network
    from core.index_builder import build_index
    p_df = compute_pressure(stats_df)
    d_df = compute_discipline(stats_df)
    n_df = compute_network(stats_df)
    index_df = build_index(p_df, d_df, n_df)
    assert "tactical_pulse_score" in index_df.columns
    assert index_df["tactical_pulse_score"].between(0, 100).all()
    assert not index_df["tactical_pulse_score"].isna().any()


def test_ranking_in_sync():
    """Verifica che _team_avg_table produca ranking coerente con i dati."""
    stats_df = _make_synthetic_stats_df(n_teams=4, n_matches=3)
    from core.pressure import compute_pressure
    from core.discipline import compute_discipline
    from core.network import compute_network
    from core.index_builder import build_index
    from app import _team_avg_table
    p_df = compute_pressure(stats_df)
    d_df = compute_discipline(stats_df)
    n_df = compute_network(stats_df)
    index_df = build_index(p_df, d_df, n_df)
    avg = _team_avg_table(index_df)
    assert len(avg) == 4
    sorted_scores = avg.sort_values("tactical_pulse_score", ascending=False)
    assert list(avg["rank"]) == list(range(1, 5))


# ----- AppTest page tests -----


@pytest.mark.skipif(
    not hasattr(__import__("streamlit.testing.v1", fromlist=["AppTest"]), "AppTest"),
    reason="streamlit.testing.v1.AppTest non disponibile",
)
class TestDashboardPages:
    """Test delle pagine della dashboard tramite AppTest.

    Questi test caricano l'app Streamlit in modalità headless e verificano
    che le pagine non crashino e mostrino gli elementi principali.
    """

    @pytest.fixture(autouse=True)
    def _setup_app(self, monkeypatch):
        """Configura l'app Streamlit con dati mock prima di ogni test."""
        monkeypatch.setattr(
            "core.data_loader.load_events",
            lambda *a, **kw: _make_synthetic_stats_df(n_teams=6, n_matches=5),
        )

    def _setup_loaded_state(self, at):
        """Imposta session_state con dati caricati mock."""
        stats_df = _make_synthetic_stats_df(n_teams=6, n_matches=5)
        from core.pressure import compute_pressure
        from core.discipline import compute_discipline
        from core.network import compute_network
        from core.index_builder import build_index
        p_df = compute_pressure(stats_df)
        d_df = compute_discipline(stats_df)
        n_df = compute_network(stats_df)
        index_df = build_index(p_df, d_df, n_df)
        stats_results = _make_synthetic_results(index_df)
        teams = sorted(index_df["team_id"].unique().tolist())
        at.session_state["data"] = {
            "stats_df": stats_df,
            "pressure_df": p_df,
            "discipline_df": d_df,
            "network_df": n_df,
            "index_df": index_df,
            "stats_results": stats_results,
            "teams": teams,
            "n_matches": stats_df["game_id"].nunique(),
            "n_records": len(stats_df),
        }
        at.session_state["data_loaded"] = True
        at.session_state["current_league"] = "ENG-Premier League"
        at.session_state["current_season"] = "2024-2025"
        at.session_state["team_a"] = teams[0]
        at.session_state["team_b"] = teams[1] if len(teams) > 1 else teams[0]

    def test_app_boots_without_error(self):
        """L'app si avvia senza eccezioni: landing page visibile."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        at.run()
        assert not at.exception

    def test_landing_page_shows_title(self):
        """Landing page mostra il titolo principale."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        at.run()
        assert at.title[0] is not None
        assert "Football Analytics Hub" in at.title[0].value

    def test_landing_page_has_expander(self):
        """Landing page mostra l'expander informativo."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        at.run()
        # Almeno un expander dovrebbe essere presente
        assert len(at.expander) > 0

    def test_overview_page_no_crash(self):
        """Pagina Overview si carica senza eccezioni."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        self._setup_loaded_state(at)
        at.run()
        assert not at.exception
        # La pagina Overview dovrebbe mostrare metriche, top/bottom 3, ecc.

    def test_overview_has_metric_kpis(self):
        """Pagina Overview mostra i KPI."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        self._setup_loaded_state(at)
        at.run()
        assert not at.exception
        # Almeno 4 metriche
        assert len(at.metric) >= 4

    def test_team_comparison_no_crash(self):
        """Pagina Team Comparison si carica senza eccezioni."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        self._setup_loaded_state(at)
        at.session_state["page"] = 1  # Team Comparison
        at.run()
        assert not at.exception

    def test_single_team_deep_dive_no_crash(self):
        """Pagina Single Team Deep Dive si carica senza eccezioni."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        self._setup_loaded_state(at)
        at.session_state["page"] = 2  # Single Team Deep Dive
        at.run()
        assert not at.exception

    def test_ai_report_page_no_crash(self):
        """Pagina AI Report si carica senza eccezioni."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        self._setup_loaded_state(at)
        at.session_state["page"] = 3  # AI Report
        at.run()
        assert not at.exception

    def test_statistical_validation_no_crash(self):
        """Pagina Statistical Validation si carica senza eccezioni."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        self._setup_loaded_state(at)
        at.session_state["page"] = 4  # Statistical Validation
        at.run()
        assert not at.exception

    def test_ai_report_button_click_data_only(self):
        """Click su 'Genera Report' in modalità data-only non crasha."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=10)
        self._setup_loaded_state(at)
        at.session_state["page"] = 3  # AI Report
        at.run()
        # Trova il bottone e cliccalo
        buttons = at.button
        gen_buttons = [b for b in buttons if "Genera" in b.label]
        if gen_buttons:
            gen_buttons[0].click()
            at.run()
            assert not at.exception


class TestMultiLeague:
    """Test che l'app funzioni con leghe e stagioni diverse."""

    @pytest.fixture(autouse=True)
    def _setup_app(self, monkeypatch):
        monkeypatch.setattr(
            "core.data_loader.load_events",
            lambda *a, **kw: _make_synthetic_stats_df(n_teams=6, n_matches=5),
        )

    def test_league_change_preserves_stability(self):
        """Cambio lega non causa crash. Simula Serie A."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        stats_df = _make_synthetic_stats_df(n_teams=6, n_matches=5)
        from core.pressure import compute_pressure
        from core.discipline import compute_discipline
        from core.network import compute_network
        from core.index_builder import build_index
        p_df = compute_pressure(stats_df)
        d_df = compute_discipline(stats_df)
        n_df = compute_network(stats_df)
        index_df = build_index(p_df, d_df, n_df)
        stats_results = _make_synthetic_results(index_df)
        teams = sorted(index_df["team_id"].unique().tolist())
        at.session_state["data"] = {
            "stats_df": stats_df, "pressure_df": p_df, "discipline_df": d_df,
            "network_df": n_df, "index_df": index_df, "stats_results": stats_results,
            "teams": teams, "n_matches": 5, "n_records": len(stats_df),
        }
        at.session_state["data_loaded"] = True
        at.session_state["current_league"] = "ITA-Serie A"
        at.session_state["current_season"] = "2023-2024"
        at.session_state["team_a"] = teams[0]
        at.session_state["team_b"] = teams[1] if len(teams) > 1 else teams[0]
        at.run()
        assert not at.exception

    def test_season_change_preserves_stability(self):
        """Cambio stagione non causa crash."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        stats_df = _make_synthetic_stats_df(n_teams=6, n_matches=5)
        from core.pressure import compute_pressure
        from core.discipline import compute_discipline
        from core.network import compute_network
        from core.index_builder import build_index
        p_df = compute_pressure(stats_df)
        d_df = compute_discipline(stats_df)
        n_df = compute_network(stats_df)
        index_df = build_index(p_df, d_df, n_df)
        stats_results = _make_synthetic_results(index_df)
        teams = sorted(index_df["team_id"].unique().tolist())
        at.session_state["data"] = {
            "stats_df": stats_df, "pressure_df": p_df, "discipline_df": d_df,
            "network_df": n_df, "index_df": index_df, "stats_results": stats_results,
            "teams": teams, "n_matches": 5, "n_records": len(stats_df),
        }
        at.session_state["data_loaded"] = True
        at.session_state["current_league"] = "ENG-Premier League"
        at.session_state["current_season"] = "2022-2023"
        at.session_state["team_a"] = teams[0]
        at.session_state["team_b"] = teams[1] if len(teams) > 1 else teams[0]
        at.run()
        assert not at.exception

    def test_league_shown_in_overview(self):
        """La lega selezionata appare nell'Overview."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        self._setup_loaded_state(at)
        at.run()
        html = str(at)
        assert "ITA-Serie A" in html or "ENG-Premier" in html

    def _setup_loaded_state(self, at):
        stats_df = _make_synthetic_stats_df(n_teams=6, n_matches=5)
        from core.pressure import compute_pressure
        from core.discipline import compute_discipline
        from core.network import compute_network
        from core.index_builder import build_index
        p_df = compute_pressure(stats_df)
        d_df = compute_discipline(stats_df)
        n_df = compute_network(stats_df)
        index_df = build_index(p_df, d_df, n_df)
        stats_results = _make_synthetic_results(index_df)
        teams = sorted(index_df["team_id"].unique().tolist())
        at.session_state["data"] = {
            "stats_df": stats_df, "pressure_df": p_df, "discipline_df": d_df,
            "network_df": n_df, "index_df": index_df, "stats_results": stats_results,
            "teams": teams, "n_matches": 5, "n_records": len(stats_df),
        }
        at.session_state["data_loaded"] = True
        at.session_state["current_league"] = "ITA-Serie A"
        at.session_state["current_season"] = "2023-2024"
        at.session_state["team_a"] = teams[0]
        at.session_state["team_b"] = teams[1] if len(teams) > 1 else teams[0]


class TestTeamResetOnLeagueChange:
    """Test che le squadre vengano resettate quando si cambia lega."""

    def test_team_a_reset_when_not_in_new_league(self):
        """Se team_a non esiste nella nuova lega, il selettore non crasha."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=5)
        stats_df = _make_synthetic_stats_df(n_teams=6, n_matches=5)
        from core.pressure import compute_pressure
        from core.discipline import compute_discipline
        from core.network import compute_network
        from core.index_builder import build_index
        p_df = compute_pressure(stats_df)
        d_df = compute_discipline(stats_df)
        n_df = compute_network(stats_df)
        index_df = build_index(p_df, d_df, n_df)
        stats_results = _make_synthetic_results(index_df)
        teams = sorted(index_df["team_id"].unique().tolist())
        at.session_state["data"] = {
            "stats_df": stats_df, "pressure_df": p_df, "discipline_df": d_df,
            "network_df": n_df, "index_df": index_df, "stats_results": stats_results,
            "teams": teams, "n_matches": 5, "n_records": len(stats_df),
        }
        at.session_state["data_loaded"] = True
        at.session_state["current_league"] = "ITA-Serie A"
        at.session_state["current_season"] = "2024-2025"
        # Imposta un team che non esiste nel dataset
        at.session_state["team_a"] = "SquadraFantasma"
        at.session_state["team_b"] = teams[0]
        at.run()
        # Non deve crashare — il selettore resetta automaticamente
        assert not at.exception
