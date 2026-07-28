# AGENTS.md — Football Analytics Hub

## Project Overview
TacticalPulse Index e' un sistema di analisi tattica per il calcio che calcola un indice composito (0-100) combinando 3 componenti: Pressure, Discipline, Network. Usa agenti CrewAI per generare report narrativi automatici. Dati da FBref via soccerdata.

## Hardening Finale (Jul 2026)
Stato: **pronto per demo pubblica** — vedi sotto per blocker residui.

### Multi-Season Verification
Tutte le combinazioni testate senza contaminazione:
- 5 leghe 2024-2025: PL, Serie A, La Liga, Bundesliga, Ligue 1
- 2 leghe 2023-2024: PL (20 team), Serie A (20 team, 94 cup match filtrati)
- 1 lega 2022-2023: PL (20 team, 0 contaminanti)
- Zero self-matches, zero contaminanti cross-league

### Data Quality Guardrails (data_loader.py)
1. **Team count anomaly** — warning se <10 o >25 squadre
2. **Critical columns** — ValueError se mancano Sh/SoT/Gls/Fls/CrdY/CrdR/Poss
3. **Pre-filter size tracking** — registra dimensione prima del filtro cup
4. **Excessive filtering** — warning se il filtro cup rimuove >30% righe
5. **Empty dataset** — ValueError se il dataset finale e' vuoto

### Session State / Selector Reset
- `team_a` / `team_b` resettati automaticamente se non presenti nel nuovo dataset (app.py:368-373)
- Selectbox con key funzionano via session_state; Streamlit gestisce valori non validi
- Pesi modificabili senza perdere selezione squadre

### Deploy Readiness
- `.streamlit/config.toml` — tema dark, upload limit 200MB
- `.streamlit/secrets.toml.example` — template per Streamlit Cloud
- `.env.example` — template per GROQ_API_KEY
- `core/config.py` — cerca key in: st.secrets > .env > os.environ > secrets.toml
- Fallback data-only funziona senza GROQ_API_KEY

### Test Coverage (97 tests)
- `test_core.py`: 38 test (pressure, discipline, network, index, league mapping, cup filter, error handling, **+5 guardrails**)
- `test_config.py`: 20 test (mask_key, load env, ai config, dotenv)
- `test_dashboard.py`: 39 test (helpers, team avg, safe style, **AppTest smoke** per tutte 5 pagine, multi-league, season change, selector reset, AI fallback)

## Tech Stack
- Python 3.12
- Streamlit (dashboard)
- Plotly (visualizzazioni)
- CrewAI (multi-agent orchestration)
- Groq API Llama 3.3 70B (LLM)
- soccerdata / FBref (data source)
- NetworkX (network analysis)
- scipy, statsmodels (statistical validation)
- pandas, numpy (data manipulation)

## Key Commands
```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run dashboard
streamlit run app.py

# Run pipeline CLI
python run_pipeline.py --max-matches 20

# Run tests
python -m pytest tests/ -v
```

## Architecture
- `core/data_loader.py`: scarica dati da FBref, caching in Parquet
- `core/pressure.py`: Pressure Component (Sh, SoT, Gls percentile rank)
- `core/discipline.py`: Discipline Component (falli, cartellini — inverso)
- `core/network.py`: Network Component (possesso, bonus gol)
- `core/index_builder.py`: combina 3 componenti in TacticalPulse Score
- `stats/significance.py`: bootstrap CI 95%, t-test, correzioni multiple
- `agents/orchestrator.py`: pipeline CrewAI (Analyst -> Statistician -> Writer -> Critic)
- `app.py`: dashboard Streamlit (5 pagine)

## Code Style
- Python PEP 8 con type hints
- Docstring in italiano su tutte le funzioni pubbliche (Google style)
- Naming: snake_case per funzioni/variabili, PascalCase per classi
- Niente commenti ovvi — solo commenti che spiegano il "perche'"

## Testing
- Framework: pytest
- Files: `tests/test_core.py` (core logic), `tests/test_config.py` (env/config), `tests/test_dashboard.py` (UI + helpers)
- Coverage minima: ogni funzione pubblica in `core/` deve avere almeno 1 test
- Dati sintetici che simulano la struttura di FBref team_match_stats
- Dashboard tests usano `streamlit.testing.v1.AppTest` con session_state mock (non serve FBref live)
- Esegui: `python -m pytest tests/ -v`

### UI Stability Rules
- Nessuna pagina della dashboard deve crashare (nessuna eccezione non catturata)
- Ogni grafico (Plotly) deve avere un fallback: tabella numerica o metric display
- Ogni feature AI deve avere fallback data-only
- Ogni bottone visibile deve produrre un effetto osservabile (report, messaggio, rerun)
- Le tabelle ranking devono usare `Styler.map` (non `Styler.applymap`, deprecato in pandas 2.1+)
- Tutto lo styling della tabella è in `_safe_style_table()` con try/except fallback al DataFrame base

### Error Handling Rules
- Ogni chiamata Plotly (`px.*`, `go.*`) deve essere in try/except con fallback testuale
- `st.error()` per errori critici, `st.warning()` per fallback
- Errori tecnici AI: mostra `st.error()` + expander con `traceback.format_exc()`
- Dati insufficienti: `st.warning()` con messaggio esplicito, mai crash
- `st.rerun()` dopo azioni asincrone (generazione report, caricamento dati)

### Testing Rules
- Ogni nuova pagina Streamlit deve avere almeno un test smoke (no-crash)
- I test dashboard devono usare dati sintetici (`_make_synthetic_stats_df()`)
- Non dipendere da FBref live per i test
- Helper rule-based (`_classify_profile`, `_interpret_score`, ecc.) devono essere testati unitariamente
- `_safe_style_table()` deve essere testato con vari input (colonne mancanti, DataFrame vuoto)
- Aggiungere test anti-regressione per bug fix (es. `Styler.applymap`)
- **Test multi-lega/multi-stagione**: almeno 1 test per cambio lega e 1 per cambio stagione
- **Test anti-hardcode**: verificare che il nome della lega/stagione corrente appaia nell'Overview
- **Test reset selettori**: verificare che team_a/team_b vengano resettati se non presenti nel nuovo dataset

### Secrets & AI Rules
- Mai mostrare la chiave API completa — sempre `_mask_key()` o `get_ai_config()["masked_key"]`
- CLI e dashboard devono condividere la stessa logica config (`core.config`)
- AI Report checkbox deve avere `key="use_ai"` per sincronizzare stato tra sidebar e pagina
- Se GROQ_API_KEY manca: bottone sempre clickabile, genera report data-only
- Se GROQ_API_KEY presente + checkbox attiva: bottone abilitato, tenta generazione AI con fallback

## Boundaries
### Always
- Mantieni il fallback senza GROQ_API_KEY (data-only report)
- Mantieni la cache locale in Parquet per i dati FBref
- Aggiorna requirements.txt quando aggiungi dipendenze
- Tutti i punteggi devono essere normalizzati 0-100
- Usa encoding utf-8 e sys.stdout.reconfigure(encoding='utf-8') su Windows
- **Nessun hardcode su lega/stagione** — ogni feature deve funzionare su tutte le leghe (`LEAGUES`) e stagioni (`SEASONS`) configurate in app.py
- I selettori team (team_a, team_b) devono resettarsi automaticamente se la lega/stagione cambia e la squadra non esiste piu' nel dataset
- Ogni report (AI o data-only) deve riflettere dinamicamente lega e stagione correnti, mai valori fissi
- Le path di cache e report devono includere lega e stagione per evitare sovrascritture tra contesti diversi

### Ask First
- Modificare la formula del TacticalPulse Score o i pesi
- Cambiare fonte dati (attualmente FBref)
- Aggiungere nuove dipendenze

### Never
- Committare .env o file con API key
- Usare dati da fonti a pagamento
- Modificare i test esistenti senza motivo valido
- Aggiungere print() di debug nel codice finale

## Critical Files
- `run_pipeline.py`: entry point CLI
- `app.py`: entry point dashboard Streamlit
- `core/data_loader.py`: punto di ingresso per tutti i dati
- `agents/orchestrator.py`: orchestrazione CrewAI
- `.env`: contiene GROQ_API_KEY (mai committare)

## Common Pitfalls
- **Sintomo**: "GROQ_API_KEY non configurata" -> **Causa**: load_dotenv() non chiamata -> **Fix**: aggiungi load_dotenv() all'inizio di run_pipeline.py
- **Sintomo**: warning 'charmap' codec -> **Causa**: encoding console Windows -> **Fix**: sys.stdout.reconfigure(encoding='utf-8') prima di qualsiasi output
- **Sintomo**: pressure score identico per tutte le squadre -> **Causa**: logica non differenziante -> **Fix**: usa percentile rank su Sh/SoT/Gls
- **Sintomo**: KeyError su colonne FBref -> **Causa**: nomi colonne cambiati -> **Fix**: print(df.columns.tolist()) per verificare nomi reali
- **Sintomo**: team_id = None -> **Causa**: bug soccerdata -> **Fix**: _fix_team_ids() ricostruisce da pairing Home/Away
- **Sintomo**: LLM non si inizializza -> **Causa**: CrewAI non riconosce 'groq/' -> **Fix**: usa 'openai/' prefisso con base_url Groq
- **Sintomo**: Chrome processes orfani dopo test -> **Causa**: soccerdata non cleanup Chrome -> **Fix**: taskkill /F /IM chrome.exe manuale
- **Sintomo**: Test con timeout lungo vengono killati -> **Causa**: troppi Chrome process -> **Fix**: chiudere Chrome tra test consecutivi
- **Sintomo**: Dataset contiene squadre estere -> **Causa**: FBref include match di coppa -> **Fix**: _filter_cup_matches() usa la team list della lega
- **Sintomo**: colonne mancanti su stagione vecchia -> **Causa**: FBref usa struttura diversa -> **Fix**: guardrail ValueError in load_events()
