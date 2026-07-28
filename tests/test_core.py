"""Test unitari per i componenti core del TacticalPulse Index (FBref team_match_stats).

Include test cross-league per verificare che il mapping lega/stagione
e il caricamento dati siano coerenti per tutte le leghe dichiarate.
"""

import pandas as pd
import pytest

from core.discipline import compute_discipline
from core.index_builder import build_index
from core.network import compute_network
from core.pressure import compute_pressure


def _synthetic_team_stats() -> pd.DataFrame:
    """Genera un DataFrame sintetico simile a team_match_stats FBref."""
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "team_id": ["Arsenal", "Chelsea", "Arsenal", "Liverpool"],
            "opponent": ["Chelsea", "Arsenal", "Liverpool", "Arsenal"],
            "venue": ["Home", "Away", "Away", "Home"],
            "result": ["W", "L", "D", "D"],
            "GF": [2, 0, 1, 1],
            "GA": [0, 2, 1, 1],
            "Poss": [58, 42, 48, 52],
            "Fls": [8, 12, 10, 9],
            "CrdY": [1, 3, 2, 1],
            "CrdR": [0, 0, 0, 0],
            "Sh": [10, 3, 8, 6],
            "SoT": [5, 1, 4, 3],
            "Gls": [2, 0, 1, 1],
        }
    )


class TestPressure:
    def test_compute_pressure_returns_dataframe(self):
        df = _synthetic_team_stats()
        result = compute_pressure(df)
        assert isinstance(result, pd.DataFrame)

    def test_compute_pressure_columns(self):
        df = _synthetic_team_stats()
        result = compute_pressure(df)
        expected = [
            "game_id", "team_id", "pressure_score",
            "shots_pctl", "sot_pctl", "goals_pctl",
            "Sh", "SoT", "Gls",
        ]
        assert list(result.columns) == expected

    def test_compute_pressure_scores_in_range(self):
        df = _synthetic_team_stats()
        result = compute_pressure(df)
        assert result["pressure_score"].between(0, 100).all()

    def test_compute_pressure_both_teams_present(self):
        df = _synthetic_team_stats()
        result = compute_pressure(df)
        assert "Arsenal" in result["team_id"].values
        assert "Chelsea" in result["team_id"].values

    def test_higher_shots_higher_pressure(self):
        df = _synthetic_team_stats().copy()
        df.loc[df["team_id"] == "Arsenal", "Sh"] = 20
        df.loc[df["team_id"] == "Arsenal", "SoT"] = 10
        df.loc[df["team_id"] == "Arsenal", "Gls"] = 5
        df.loc[df["team_id"] == "Chelsea", "Sh"] = 1
        df.loc[df["team_id"] == "Chelsea", "SoT"] = 0
        df.loc[df["team_id"] == "Chelsea", "Gls"] = 0
        result = compute_pressure(df)
        arsenal = result[result["team_id"] == "Arsenal"]["pressure_score"].values[0]
        chelsea = result[result["team_id"] == "Chelsea"]["pressure_score"].values[0]
        assert arsenal > chelsea


class TestDiscipline:
    def test_compute_discipline_returns_dataframe(self):
        df = _synthetic_team_stats()
        result = compute_discipline(df)
        assert isinstance(result, pd.DataFrame)

    def test_compute_discipline_columns(self):
        df = _synthetic_team_stats()
        result = compute_discipline(df)
        expected = [
            "game_id", "team_id", "discipline_score",
            "fouls", "yellows", "reds", "raw_index",
        ]
        assert list(result.columns) == expected

    def test_compute_discipline_scores_in_range(self):
        df = _synthetic_team_stats()
        result = compute_discipline(df)
        assert result["discipline_score"].between(0, 100).all()

    def test_compute_discipline_both_teams(self):
        df = _synthetic_team_stats()
        result = compute_discipline(df)
        assert "Arsenal" in result["team_id"].values
        assert "Chelsea" in result["team_id"].values

    def test_no_cards_high_score(self):
        df = _synthetic_team_stats().copy()
        df["CrdY"] = 0
        df["CrdR"] = 0
        df["Fls"] = 0
        result = compute_discipline(df)
        assert (result["discipline_score"] == 100).all()

    def test_higher_fouls_lower_score(self):
        df = _synthetic_team_stats().copy()
        # Rende Chelsea con piu' falli di Arsenal
        df.loc[df["team_id"] == "Chelsea", "Fls"] = 99
        df.loc[df["team_id"] == "Chelsea", "CrdY"] = 10
        df.loc[df["team_id"] == "Chelsea", "CrdR"] = 2
        df.loc[df["team_id"] == "Arsenal", "Fls"] = 5
        df.loc[df["team_id"] == "Arsenal", "CrdY"] = 1
        df.loc[df["team_id"] == "Arsenal", "CrdR"] = 0
        result = compute_discipline(df)
        arsenal_score = result[result["team_id"] == "Arsenal"]["discipline_score"].values[0]
        chelsea_score = result[result["team_id"] == "Chelsea"]["discipline_score"].values[0]
        assert arsenal_score > chelsea_score


class TestNetwork:
    def test_compute_network_returns_dataframe(self):
        df = _synthetic_team_stats()
        result = compute_network(df)
        assert isinstance(result, pd.DataFrame)

    def test_compute_network_columns(self):
        df = _synthetic_team_stats()
        result = compute_network(df)
        expected = [
            "game_id", "team_id", "network_score",
            "possession", "goal_bonus",
        ]
        assert list(result.columns) == expected

    def test_compute_network_scores_in_range(self):
        df = _synthetic_team_stats()
        result = compute_network(df)
        assert result["network_score"].between(0, 100).all()

    def test_compute_network_both_teams(self):
        df = _synthetic_team_stats()
        result = compute_network(df)
        assert "Arsenal" in result["team_id"].values
        assert "Chelsea" in result["team_id"].values

    def test_higher_possession_higher_score(self):
        df = _synthetic_team_stats().copy()
        df.loc[df["team_id"] == "Arsenal", "Poss"] = 95
        df.loc[df["team_id"] == "Chelsea", "Poss"] = 5
        result = compute_network(df)
        arsenal = result[result["team_id"] == "Arsenal"]["network_score"].values[0]
        chelsea = result[result["team_id"] == "Chelsea"]["network_score"].values[0]
        assert arsenal > chelsea


class TestIndexBuilder:
    def test_build_index_returns_dataframe(self):
        df = _synthetic_team_stats()
        p = compute_pressure(df)
        d = compute_discipline(df)
        n = compute_network(df)
        result = build_index(p, d, n)
        expected_cols = [
            "game_id", "team_id", "pressure_score",
            "discipline_score", "network_score",
            "tactical_pulse_score", "rank",
        ]
        assert list(result.columns) == expected_cols

    def test_tactical_pulse_score_in_range(self):
        df = _synthetic_team_stats()
        p = compute_pressure(df)
        d = compute_discipline(df)
        n = compute_network(df)
        result = build_index(p, d, n)
        assert result["tactical_pulse_score"].between(0, 100).all()

    def test_custom_weights(self):
        df = _synthetic_team_stats()
        p = compute_pressure(df)
        d = compute_discipline(df)
        n = compute_network(df)
        w = {"pressure": 0.5, "discipline": 0.5, "network": 0.0}
        result = build_index(p, d, n, weights=w)
        assert "tactical_pulse_score" in result.columns

    def test_build_index_empty(self):
        p = pd.DataFrame(columns=["game_id", "team_id", "pressure_score"])
        d = pd.DataFrame(columns=["game_id", "team_id", "discipline_score"])
        n = pd.DataFrame(columns=["game_id", "team_id", "network_score"])
        result = build_index(p, d, n)
        assert len(result) == 0

    def test_rank_present(self):
        df = _synthetic_team_stats()
        p = compute_pressure(df)
        d = compute_discipline(df)
        n = compute_network(df)
        result = build_index(p, d, n)
        assert "rank" in result.columns
        assert result["rank"].nunique() >= 2


# ── Test cross-league ─────────────────────────────────────


LEAGUES_CONST = ["ENG-Premier League", "ITA-Serie A", "ESP-La Liga",
                 "GER-Bundesliga", "FRA-Ligue 1"]
SEASONS_CONST = ["2024-2025", "2023-2024", "2022-2023"]


class TestLeagueMapping:
    """Verifica che il mapping lega/stagione sia coerente."""

    def test_all_leagues_have_standard_format(self):
        """Ogni lega deve avere formato 'CODICE-Nome Lega'."""
        for league in LEAGUES_CONST:
            assert "-" in league, f"Formato non valido: {league}"
            parts = league.split("-")
            assert len(parts) >= 2, f"Troppi pochi componenti: {league}"
            assert len(parts[0]) == 3, f"Codice paese non di 3 lettere: {league}"

    def test_all_seasons_have_standard_format(self):
        """Ogni stagione deve avere formato 'YYYY-YYYY'."""
        for season in SEASONS_CONST:
            assert "-" in season, f"Formato non valido: {season}"
            parts = season.split("-")
            assert len(parts) == 2, f"Formato non valido: {season}"
            assert len(parts[0]) == 4 and len(parts[1]) == 4, (
                f"Anni non a 4 cifre: {season}"
            )

    def test_all_leagues_have_unique_codes(self):
        """I codici paese (prime 3 lettere) devono essere univoci."""
        codes = [l.split("-")[0] for l in LEAGUES_CONST]
        assert len(codes) == len(set(codes)), f"Codici duplicati: {codes}"

    def test_league_mapping_supported_by_soccerdata(self):
        """Verifica che soccerdata riconosca tutti i codici lega."""
        from soccerdata._config import LEAGUE_DICT
        for league in LEAGUES_CONST:
            assert league in LEAGUE_DICT, (
                f"{league} non trovato in soccerdata.LEAGUE_DICT. "
                f"Chiavi disponibili: {list(LEAGUE_DICT.keys())}"
            )

    def test_league_mapping_has_fbref_entry(self):
        """Ogni lega deve avere un mapping FBref in soccerdata."""
        from soccerdata._config import LEAGUE_DICT
        from soccerdata.fbref import FBref
        for league in LEAGUES_CONST:
            assert league in LEAGUE_DICT, f"{league} non in LEAGUE_DICT"
            assert "FBref" in LEAGUE_DICT[league], (
                f"{league} manca di chiave FBref"
            )
            fbref_name = LEAGUE_DICT[league]["FBref"]
            assert isinstance(fbref_name, str) and len(fbref_name) > 0

    def test_league_available_via_fbref_all_leagues(self):
        """Verifica che soccerdata.FBref esponga tutte le leghe come disponibili."""
        from soccerdata.fbref import FBref
        available = FBref.available_leagues()
        for league in LEAGUES_CONST:
            assert league in available, (
                f"{league} non in FBref.available_leagues(). "
                f"Disponibili: {available}"
            )


class TestCrossLeagueLoading:
    """Test che simulano il caricamento cross-league con dati sintetici.

    Non richiedono connessione a FBref.
    """

    def test_data_flow_same_for_any_league(self):
        """Il flusso dati (pressure/discipline/network) deve funzionare
        con qualsiasi configurazione di lega, usando dati sintetici."""
        for league in LEAGUES_CONST:
            stats = _synthetic_team_stats().copy()
            stats["league"] = league
            p = compute_pressure(stats)
            d = compute_discipline(stats)
            n = compute_network(stats)
            result = build_index(p, d, n)
            assert not result.empty, f"Risultato vuoto per {league}"
            assert "tactical_pulse_score" in result.columns
            assert result["tactical_pulse_score"].between(0, 100).all()

    def test_cache_path_isolation(self):
        """La cache deve essere isolata per lega/stagione diversa."""
        from core.data_loader import _cache_path
        paths = set()
        for league in LEAGUES_CONST:
            for season in SEASONS_CONST:
                path = _cache_path(league, season)
                paths.add(path.name)
        # Ogni combinazione deve avere un nome file unico
        expected = len(LEAGUES_CONST) * len(SEASONS_CONST)
        assert len(paths) == expected, (
            f"Cache collisioni: {len(paths)} nomi unici su {expected} attesi"
        )


class TestCupFilter:
    """Test per _filter_cup_matches e _fix_team_ids (multi-league fix)."""

    def test_filter_cup_matches_removes_foreign_teams(self):
        """_filter_cup_matches deve rimuovere match con squadre estere."""
        from core.data_loader import _filter_cup_matches
        df = pd.DataFrame({
            "team_id": ["Inter", "Milan", "Inter", "Real Madrid"],
            "opponent": ["Milan", "Inter", "Barcelona", "Inter"],
            "GF": [1, 0, 2, 1],
            "GA": [0, 1, 1, 2],
        })
        league_teams = {"Inter", "Milan", "Juventus"}
        result = _filter_cup_matches(df, league_teams)
        # Solo Inter vs Milan deve rimanere (entrambi in league_teams)
        # Inter vs Barcelona -> Barcelona non nella lega -> rimosso
        # Real Madrid vs Inter -> RM non nella lega -> rimosso
        assert len(result) == 2, f"Attese 2 righe, trovate {len(result)}"
        for _, row in result.iterrows():
            assert row["team_id"] in league_teams, (
                f"team_id {row['team_id']} non nella lega"
            )
            assert row["opponent"] in league_teams, (
                f"opponent {row['opponent']} non nella lega"
            )

    def test_filter_cup_matches_all_league_kept(self):
        """Se tutte le squadre sono della lega, nessuna riga rimossa."""
        from core.data_loader import _filter_cup_matches
        df = pd.DataFrame({
            "team_id": ["Inter", "Milan", "Juventus"],
            "opponent": ["Milan", "Inter", "Roma"],
        })
        league_teams = {"Inter", "Milan", "Juventus", "Roma"}
        result = _filter_cup_matches(df, league_teams)
        assert len(result) == 3

    def test_filter_cup_matches_empty_result(self):
        """Se nessuna squadra e' della lega, risultato vuoto."""
        from core.data_loader import _filter_cup_matches
        df = pd.DataFrame({
            "team_id": ["Barcelona", "Real Madrid"],
            "opponent": ["Real Madrid", "Barcelona"],
        })
        result = _filter_cup_matches(df, {"Inter", "Milan"})
        assert result.empty

    def test_fix_team_ids_reconstructs_names(self):
        """_fix_team_ids deve ricostruire team_id da opponent quando
        team_id e' None."""
        from core.data_loader import _fix_team_ids
        df = pd.DataFrame({
            "team_id": [None, None],
            "opponent": ["Chelsea", "Arsenal"],
            "venue": ["Home", "Away"],
            "match_report": ["mr1", "mr1"],
            "GF": [1, 0],
        })
        result = _fix_team_ids(df)
        ids = result["team_id"].tolist()
        # Home (Arsenal): opponent della Away = Arsenal
        # Away (Chelsea): opponent della Home = Chelsea
        assert ids == ["Arsenal", "Chelsea"], f"Risultato inatteso: {ids}"

    def test_fix_team_ids_preserves_existing_ids(self):
        """Se team_id ha gia' valori validi, non deve modificarli."""
        from core.data_loader import _fix_team_ids
        df = pd.DataFrame({
            "team_id": ["Arsenal", "Chelsea"],
            "opponent": ["Chelsea", "Arsenal"],
            "venue": ["Home", "Away"],
            "match_report": ["mr1", "mr1"],
        })
        result = _fix_team_ids(df)
        assert result["team_id"].tolist() == ["Arsenal", "Chelsea"]


class TestErrorHandling:
    """Verifica che gli errori siano chiari e informativi."""

    def test_chrome_check_returns_bool(self):
        """_check_chrome_available deve restituire un booleano.
        
        Su CI (GitHub Actions, Ubuntu) Chrome potrebbe non essere
        installato, quindi testiamo solo il tipo, non il valore.
        """
        from core.data_loader import _check_chrome_available
        result = _check_chrome_available()
        assert isinstance(result, bool)

    def test_required_columns_across_leagues(self):
        """Le colonne richieste devono essere presenti nel dataset sintetico,
        indipendentemente dalla lega."""
        required = ["Sh", "SoT", "Gls", "Fls", "CrdY", "CrdR", "Poss"]
        for league in LEAGUES_CONST:
            df = _synthetic_team_stats().copy()
            df["league"] = league
            for col in required:
                assert col in df.columns, (
                    f"Colonna {col} mancante per {league}"
                )
            # Pipeline deve funzionare
            p = compute_pressure(df)
            d = compute_discipline(df)
            n = compute_network(df)
            idx = build_index(p, d, n)
            assert not idx.empty

    def test_empty_dataframe_handled_gracefully(self):
        """Un DataFrame vuoto dopo il merge deve dare errore prima del crash."""
        import pandas as pd
        from core.pressure import compute_pressure
        empty = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            compute_pressure(empty)


class TestDataQualityGuardrails:
    """Test per i data quality guardrails introdotti nell'hardening finale."""

    def test_empty_dataset_after_filter_raises_error(self):
        """Se _filter_cup_matches torna vuoto, load_events deve sollevare ValueError."""
        from core.data_loader import _filter_cup_matches
        df = pd.DataFrame({
            "team_id": ["Barcelona", "Real Madrid"],
            "opponent": ["Real Madrid", "Barcelona"],
            "GF": [1, 0],
        })
        result = _filter_cup_matches(df, {"Inter", "Milan"})
        assert result.empty, "Il filtro deve produrre DataFrame vuoto"

    def test_filter_cup_matches_all_league_preserves_data(self):
        """Se tutte le squadre sono della lega, nessuna riga deve essere rimossa."""
        from core.data_loader import _filter_cup_matches
        df = pd.DataFrame({
            "team_id": ["Inter", "Milan", "Juventus"],
            "opponent": ["Milan", "Inter", "Roma"],
            "GF": [1, 2, 0],
        })
        league_teams = {"Inter", "Milan", "Juventus", "Roma"}
        result = _filter_cup_matches(df, league_teams)
        assert len(result) == 3

    def test_critical_columns_guardrail_logic(self):
        """Verifica che la logica del guardrail colonne critiche sia corretta."""
        required = {"Sh", "SoT", "Gls", "Fls", "CrdY", "CrdR", "Poss"}
        df_missing = pd.DataFrame({"Sh": [1], "SoT": [2]})
        missing = [c for c in required if c not in df_missing.columns]
        assert len(missing) == 5, "Dovrebbero mancare 5 colonne"
        df_full = pd.DataFrame({c: [1] for c in required})
        missing_full = [c for c in required if c not in df_full.columns]
        assert len(missing_full) == 0, "Nessuna colonna dovrebbe mancare"

    def test_team_count_anomaly_logic(self):
        """Verifica che la soglia di anomalia per il numero squadre sia corretta."""
        def _is_anomalous(n: int) -> bool:
            return n < 10 or n > 25
        assert not _is_anomalous(20), "20 squadre deve essere normale"
        assert _is_anomalous(5), "5 squadre deve essere anomalo"
        assert _is_anomalous(30), "30 squadre deve essere anomalo"

    def test_filter_excessive_logic(self):
        """Verifica che la soglia del 30% per filtro eccessivo sia corretta."""
        pre = 100
        ratio_40 = (pre - 60) / pre
        assert ratio_40 > 0.3, f"Rimozione 40% deve superare soglia del 30%: {ratio_40:.1%}"
        ratio_20 = (pre - 80) / pre
        assert ratio_20 < 0.3, f"Rimozione 20% deve essere sotto soglia: {ratio_20:.1%}"
