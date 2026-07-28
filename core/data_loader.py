"""Caricamento dati statistiche partite da FBref con caching locale in Parquet.

Utilizza soccerdata.FBref per ottenere le statistiche complete
per squadra per partita (tiri, falli, cartellini, possesso, ecc.)
e le memorizza in data/raw/ per riutilizzo rapido senza riscaricare.
"""

import shutil
import time
import warnings
from pathlib import Path

import pandas as pd

_DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
_DEMO_CACHE = Path(__file__).resolve().parent.parent / "data" / "demo_cache"


def _create_fbref(
    league: str, season: str
) -> "FBref":
    """Crea un'istanza FBref con Selenium in modalità headless.

    Utilizza Chrome via seleniumbase (undetected-chromedriver) per bypassare
    Cloudflare. Chrome deve essere installato sul sistema.

    Parameters
    ----------
    league : str
        Codice lega standardizzato, es. 'ENG-Premier League'.
    season : str
        Stagione, es. '2024-2025'.

    Returns
    -------
    FBref
        Istanza pronta per leggere dati da FBref.
    """
    from soccerdata import FBref  # noqa: E402

    return FBref(leagues=league, seasons=season, headless=True)


def _cache_path(league: str, season: str) -> Path:
    """Restituisce il percorso del file Parquet per una lega/stagione."""
    _DATA_RAW.mkdir(parents=True, exist_ok=True)
    safe_league = league.replace(" ", "_").replace("-", "_")
    return _DATA_RAW / f"{safe_league}_{season}.parquet"


def _demo_cache_path(league: str, season: str) -> Path:
    """Restituisce il percorso del file Parquet in data/demo_cache/."""
    safe_league = league.replace(" ", "_").replace("-", "_")
    return _DEMO_CACHE / f"{safe_league}_{season}.parquet"


def is_demo_mode() -> bool:
    """True se Chrome non e' disponibile (cloud/demo deployment)."""
    return not _check_chrome_available()


def get_demo_cache_combinations(
    leagues: list[str], seasons: list[str]
) -> list[tuple[str, str]]:
    """Restituisce le combinazioni (lega, stagione) disponibili in demo_cache.

    Parametri
    ---------
    leagues : list[str]
        Lista di codici lega da verificare.
    seasons : list[str]
        Lista di stagioni da verificare.

    Returns
    -------
    list[tuple[str, str]]
        Combinazioni (lega, stagione) presenti nel demo_cache.
    """
    combos: list[tuple[str, str]] = []
    for league in leagues:
        for season in seasons:
            if _demo_cache_path(league, season).exists():
                combos.append((league, season))
    return combos


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Appiattisce colonne multi-livello di FBref in colonne semplici.

    Esempio: ('Performance', 'CrdY') -> 'CrdY'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[1] if c[1] and c[1] != "" else c[0] for c in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _get_league_teams(league: str, season: str) -> list[str]:
    """Restituisce la lista dei team di una lega da read_team_season_stats."""
    fb = _create_fbref(league, season)
    tss = fb.read_team_season_stats(stat_type="standard")
    # Indice: (league, season, team)
    return list(tss.index.get_level_values("team").unique())


def _filter_cup_matches(
    df: pd.DataFrame, league_teams: set[str]
) -> pd.DataFrame:
    """Rimuove partite di coppa da un DataFrame unificato.

    Una partita e' di campionato solo se ENTRAMBE le squadre
    (team_id e opponent) sono nella lista della lega.
    """
    before = len(df)
    mask = df["team_id"].isin(league_teams) & df["opponent"].isin(league_teams)
    df = df[mask].copy()
    after = len(df)
    excluded = before - after
    if excluded > 0:
        print(f"[INFO] Filtrate {excluded} righe di coppe "
              f"(squadre non appartenenti alla lega)")
    return df


def _merge_stat_types(
    schedule: pd.DataFrame,
    misc: pd.DataFrame | None = None,
    shooting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fonde schedule, misc e shooting in un unico DataFrame.

    Parameters
    ----------
    schedule : pd.DataFrame
        Statistiche 'schedule' (GF, GA, Poss, Venue, Opponent, ecc.).
    misc : pd.DataFrame | None
        Statistiche 'misc' (Fls, CrdY, CrdR, ecc.).
    shooting : pd.DataFrame | None
        Statistiche 'shooting' (Sh, SoT, Gls, ecc.).

    Returns
    -------
    pd.DataFrame
        DataFrame unificato con indice [league, season, team, game].
    """
    sched = _flatten_columns(schedule.copy())

    if "Poss" not in sched.columns and "possession" in sched.columns:
        sched.rename(columns={"possession": "Poss"}, inplace=True)
    if "GF" not in sched.columns and "goals_for" in sched.columns:
        sched.rename(columns={"goals_for": "GF"}, inplace=True)
    if "GA" not in sched.columns and "goals_against" in sched.columns:
        sched.rename(columns={"goals_against": "GA"}, inplace=True)

    for extra in [misc, shooting]:
        if extra is not None:
            extra_f = _flatten_columns(extra.copy())
            extra_cols = [c for c in extra_f.columns if c not in sched.columns]
            if extra_cols:
                sched = sched.join(extra_f[extra_cols], how="left")

    return sched


def _fix_team_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ricostruisce team_id quando FBref lo restituisce come None.

    Abbinamento: per ogni match_report, ci sono 2 righe (Home/Away).
    La riga Home ha opponent = away_team = team_id dell'altra riga.
    La riga Away ha opponent = home_team = team_id dell'altra riga.
    """
    df = df.copy()
    if "team_id" not in df.columns or df["team_id"].notna().any():
        # Se team_id ha gia' valori validi, non serve ricostruire
        return df

    if "match_report" not in df.columns or "opponent" not in df.columns:
        return df

    # Per ogni match_report, ricava Home e Away
    matches = df.groupby("match_report")
    new_team_ids = {}
    for mr, grp in matches:
        home_rows = grp[grp["venue"] == "Home"]
        away_rows = grp[grp["venue"] == "Away"]
        if len(home_rows) == 1 and len(away_rows) == 1:
            home_idx = home_rows.index[0]
            away_idx = away_rows.index[0]
            # Home row: opponent e' l'away team
            # Away row: opponent e' l'home team
            new_team_ids[home_idx] = away_rows["opponent"].values[0]
            new_team_ids[away_idx] = home_rows["opponent"].values[0]
        elif len(home_rows) == 1:
            home_idx = home_rows.index[0]
            new_team_ids[home_idx] = away_rows["opponent"].values[0] if len(away_rows) > 0 else df.loc[home_idx, "opponent"]
        elif len(away_rows) == 1:
            away_idx = away_rows.index[0]
            new_team_ids[away_idx] = home_rows["opponent"].values[0] if len(home_rows) > 0 else df.loc[away_idx, "opponent"]

    for idx, team_name in new_team_ids.items():
        df.loc[idx, "team_id"] = team_name

    return df


def _check_chrome_available() -> bool:
    """Verifica se Chrome e' installato e raggiungibile."""
    import shutil as _shutil
    return _shutil.which("chrome") is not None or (
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe").exists()
    )


def _download_team_match_stats(
    league: str, season: str
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Scarica le statistiche FBref (schedule + misc + shooting) da ciascuna squadra.

    Ogni stat_type viene scaricato con retry ed exponential backoff.
    Richiede Chrome installato per bypassare Cloudflare.

    Returns
    -------
    tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]
        (schedule, misc, shooting)

    Raises
    ------
    RuntimeError
        Se Chrome non e' disponibile.
    ConnectionError
        Se il download fallisce dopo tutti i retry.
    ValueError
        Se la lega/stagione non e' disponibile su FBref.
    """
    if not _check_chrome_available():
        raise RuntimeError(
            "Google Chrome non trovato. Chrome e' necessario per scaricare dati da FBref "
            "(Cloudflare bypass). Installa Chrome o usa una combinazione lega/stagione gia' in cache."
        )

    def _try_download(stat_type, max_retries=3):
        last_exc = None
        for attempt in range(max_retries):
            try:
                fb = _create_fbref(league, season)
                df = fb.read_team_match_stats(stat_type=stat_type)
                if df is None or df.empty:
                    raise ValueError(f"FBref ha restituito DataFrame vuoto per {stat_type}")
                n = len(df)
                cols = list(df.columns[:6])
                print(f"[INFO] {stat_type}: {n} righe, colonne={cols}")
                return df
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 5
                    exc_msg = str(exc)[:120]
                    print(f"[RETRY] {stat_type} (tentativo {attempt+1}/{max_retries})"
                          f" — riprovo tra {delay}s: {type(exc).__name__}: {exc_msg}")
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    sched = _try_download("schedule")
    misc = _try_download("misc")
    shoot = _try_download("shooting")

    return sched, misc, shoot


def _get_team_season_stats(
    league: str, season: str
) -> pd.DataFrame:
    """Carica le statistiche stagionali aggregate per squadra."""
    fb = _create_fbref(league, season)
    tss = fb.read_team_season_stats(stat_type="standard")
    tss = _flatten_columns(tss)
    return tss


def load_events(
    league: str,
    season: str,
    max_matches: int | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Carica le statistiche per squadra per partita da FBref.

    Parameters
    ----------
    league : str
        Identificativo lega, es. "ENG-Premier League".
    season : str
        Stagione, es. "2024-2025".
    max_matches : int, optional
        Limita il caricamento alle prime N partite (utile per test).
    force_download : bool
        Se True, ignora la cache Parquet e riscarica.

    Returns
    -------
    pd.DataFrame
        Statistiche per squadra per partita con colonne:
        team, game_id, opponent, venue, result, GF, GA, Poss,
        Fls, CrdY, CrdR, league, season.
    """
    cache = _cache_path(league, season)

    if cache.exists() and not force_download:
        df = pd.read_parquet(cache)
        n_matches = df["game_id"].nunique() if "game_id" in df.columns else 0
        n_events = len(df)
        print(f"[INFO] Caricati {n_events} record per {n_matches} partite (da cache Parquet)")
        if max_matches is not None:
            game_ids = df["game_id"].unique()[:max_matches]
            df = df[df["game_id"].isin(game_ids)].copy()
            print(f"[INFO] Limitate a {len(df)} record per {len(game_ids)} partite (max_matches={max_matches})")
        return df

    # Se Chrome non e' disponibile, prova il demo cache
    if not _check_chrome_available():
        demo = _demo_cache_path(league, season)
        if demo.exists():
            df = pd.read_parquet(demo)
            n_matches = df["game_id"].nunique() if "game_id" in df.columns else 0
            n_events = len(df)
            print(f"[INFO] Caricati {n_events} record per {n_matches} partite (da demo cache)")
            if max_matches is not None:
                game_ids = df["game_id"].unique()[:max_matches]
                df = df[df["game_id"].isin(game_ids)].copy()
                print(f"[INFO] Limitate a {len(df)} record per {len(game_ids)} partite (max_matches={max_matches})")
            return df
        raise ValueError(
            f"Dati demo non disponibili per {league} {season}. "
            "Se sei in modalita' demo (cloud), assicurati che i file Parquet "
            "siano presenti in data/demo_cache/. "
            "In locale, installa Google Chrome per il download live."
        )

    try:
        sched, misc, shoot = _download_team_match_stats(league, season)
    except RuntimeError as e:
        raise RuntimeError(str(e))
    except ConnectionError as e:
        raise ConnectionError(
            f"Impossibile scaricare dati da FBref per {league} {season}. "
            f"Verifica la connessione internet e che FBref sia raggiungibile. "
            f"Dettaglio: {e}"
        )
    except Exception as e:
        # Prova a capire se e' un problema di lega/stagione non disponibile
        exc_str = str(e)
        if "404" in exc_str or "not found" in exc_str.lower() or "no data" in exc_str.lower():
            raise ValueError(
                f"Dati FBref non disponibili per {league} {season}. "
                f"La combinazione potrebbe non esistere su FBref. "
                f"Dettaglio: {exc_str[:200]}"
            )
        raise

    # Diagnostica: verifica stato dei tre DataFrame
    for name, df in [("schedule", sched), ("misc", misc), ("shooting", shoot)]:
        if df is None:
            print(f"[DIAG] {name}: None")
        elif df.empty:
            print(f"[DIAG] {name}: DataFrame vuoto")
        else:
            print(f"[DIAG] {name}: {len(df)} righe, {len(df.columns)} colonne — {list(df.columns[:10])}")
            for col in ["Sh", "SoT", "Gls", "Fls", "CrdY", "CrdR", "Poss"]:
                if col in df.columns:
                    print(f"[DIAG]   {col}: {pd.to_numeric(df[col], errors='coerce').mean():.1f} (media)")
                else:
                    print(f"[DIAG]   {col}: COLONNA MANCANTE")

    if sched is None or sched.empty:
        raise ValueError(
            f"Nessuna statistica trovata per {league} {season}. "
            f"La combinazione potrebbe non essere disponibile su FBref."
        )

    # Ottieni lista team della lega (per filtrare coppe)
    league_teams = _get_league_teams(league, season)
    league_teams_set = set(league_teams)
    print(f"[INFO] Team della lega ({len(league_teams_set)}): {sorted(league_teams_set)}")

    # DATA QUALITY GUARDRAIL 1: anomalo numero squadre
    if len(league_teams_set) < 10 or len(league_teams_set) > 25:
        warnings.warn(
            f"Numero squadre anomalo per {league} {season}: "
            f"{len(league_teams_set)} squadre. Attese circa 18-20."
        )

    # Unisci schedule + misc + shooting
    merged = _merge_stat_types(sched, misc=misc, shooting=shoot)
    print(f"[DIAG] Dopo merge: {len(merged)} righe, {len(merged.columns)} colonne")
    # DATA QUALITY GUARDRAIL 2: colonne critiche mancanti
    critical_cols = ["Sh", "SoT", "Gls", "Fls", "CrdY", "CrdR", "Poss"]
    missing_critical = [c for c in critical_cols if c not in merged.columns]
    if missing_critical:
        raise ValueError(
            f"Colonne critiche mancanti per {league} {season}: {missing_critical}. "
            "I dati FBref potrebbero essere cambiati o la stagione non e' disponibile."
        )

    # Reset index per avere league, season, team, game come colonne
    merged = merged.reset_index()

    # Rinomina 'game' in 'game_id' se presente
    if "game" in merged.columns and "game_id" not in merged.columns:
        merged.rename(columns={"game": "game_id"}, inplace=True)

    # Rinomina 'team' in 'team_id' per consistenza
    if "team" in merged.columns and "team_id" not in merged.columns:
        merged.rename(columns={"team": "team_id"}, inplace=True)

    # Ricostruisci team_id (FBref restituisce team=None nell'indice).
    # L'abbinamento Home/Away via opponent e' corretto per ogni match_report.
    merged = _fix_team_ids(merged)

    # DATA QUALITY GUARDRAIL 3: registro dimensione pre-filtro per valutazione
    pre_filter_len = len(merged)

    # Filtra via team list: tieni solo partite dove ENTRAMBE le squadre
    # appartengono alla lega (rimuove coppe nazionali/internazionali).
    merged = _filter_cup_matches(merged, league_teams_set)

    # DATA QUALITY GUARDRAIL 4: filtro eccessivo
    filter_excluded = pre_filter_len - len(merged)
    if pre_filter_len > 0 and filter_excluded / pre_filter_len > 0.3:
        warnings.warn(
            f"Filtro cup ha rimosso il {filter_excluded / pre_filter_len:.1%} "
            f"delle righe per {league} {season} ({filter_excluded}/{pre_filter_len}). "
            "Verificare che la lega sia corretta."
        )

    # DATA QUALITY GUARDRAIL 5: dataset vuoto dopo filtraggio
    if merged.empty:
        raise ValueError(
            f"Dataset vuoto per {league} {season} dopo il filtraggio. "
            "Nessuna partita di campionato trovata per questa combinazione. "
            "Verificare che la lega e stagione siano corretti."
        )

    # Aggiungi league/season se non presenti
    if "league" not in merged.columns:
        merged["league"] = league
    if "season" not in merged.columns:
        merged["season"] = season

    # Normalizza colonne numeriche
    numeric_cols = ["GF", "GA", "Poss", "Fls", "CrdY", "CrdR", "Sh", "SoT", "Gls"]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Salva cache COMPLETO (senza limiti di max_matches)
    cache.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache, index=False)

    n_full_matches = merged["game_id"].nunique() if "game_id" in merged.columns else 0
    print(f"[INFO] Scaricati {len(merged)} record per {n_full_matches} partite (salvati in cache)")

    # Applica max_matches solo al RETURN, non al salvataggio
    if max_matches is not None and max_matches < n_full_matches and "game_id" in merged.columns:
        game_ids = merged["game_id"].dropna().unique()[:max_matches]
        merged = merged[merged["game_id"].isin(game_ids)].copy()
        n_limited = merged["game_id"].nunique()
        print(f"[INFO] Limitate a {len(merged)} record per {n_limited} partite (max_matches={max_matches})")

    return merged


def get_team_season_stats(
    league: str, season: str
) -> pd.DataFrame:
    """Restituisce le statistiche stagionali aggregate per ogni squadra."""
    tss = _get_team_season_stats(league, season)
    tss["league"] = league
    tss["season"] = season
    tss = tss.reset_index()
    if "team" in tss.columns and "team_id" not in tss.columns:
        tss.rename(columns={"team": "team_id"}, inplace=True)
    return tss


def clear_cache(league: str | None = None, season: str | None = None) -> None:
    """Elimina i file di cache."""
    if league and season:
        path = _cache_path(league, season)
        if path.exists():
            path.unlink()
    elif _DATA_RAW.exists():
        shutil.rmtree(_DATA_RAW)
        _DATA_RAW.mkdir(parents=True, exist_ok=True)
