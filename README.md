# ⚽ Football Analytics Hub

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)]()
[![CrewAI](https://img.shields.io/badge/CrewAI-1.15-green)]()
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()
[![CI](https://github.com/tuo-username/football-analytics-hub/actions/workflows/ci.yml/badge.svg)]()

**Multi-League Tactical Dashboard with AI / Data-Only Reporting**

**Football Analytics Hub** e' un sistema di analisi tattica calcistica che calcola il **TacticalPulse Index** — un indice composito (0–100) combinando 3 dimensioni: **Pressure** (pericolosità offensiva), **Discipline** (controllo e fair play) e **Network** (possesso e costruzione).

Include una **dashboard interattiva Streamlit** con 5 pagine, **report AI automatici** generati da 4 agenti CrewAI (Llama 3.3 70B su Groq), e una **pipeline CLI** per esecuzione batch.

**Supporto multi-lega / multi-stagione:** Premier League, Serie A, La Liga, Bundesliga, Ligue 1 — stagioni 2022–2023, 2023–2024, 2024–2025. Ottenute verificando che tutte le squadre coinvolte in ogni partita appartengano effettivamente alla lega.

---

## Funzionalità

- **5 pagine dashboard**: Overview (KPI, ranking, insight), Team Comparison (radar, metriche), Single Team Deep Dive (gauge, trend, momentum), AI Report (narrativo o data-only), Statistical Validation (bootstrap CI, t-test)
- **Multi-lega / multi-stagione**: seleziona liberamente qualsiasi combinazione tra 5 leghe e 3 stagioni
- **Pesi modificabili**: regola l'importanza di Pressure, Discipline e Network in tempo reale
- **Profilo automatico squadra**: 7 profili tattici rule-based (Aggressiva, Disciplinata, Tecnica, Equilibrata, ecc.)
- **Punti di forza/debolezza**: generati automaticamente per ogni squadra
- **Analisi temporale**: momentum, forma recente (ultime 5 partite), media mobile
- **Report AI**: generato da 4 agenti CrewAI (Analyst → Statistician → Writer → Critic) con Llama 3.3 70B su Groq
- **Fallback data-only**: funziona completamente senza chiave AI — tabelle e statistiche descrittive sempre disponibili
- **Cache intelligente**: dati salvati in Parquet per riutilizzo immediato senza riscaricare

## Stack

| Componente | Tecnologia |
|---|---|
| Linguaggio | Python 3.12 |
| Dashboard | Streamlit + Plotly |
| AI Multi-Agente | CrewAI |
| LLM | Groq API — Llama 3.3 70B |
| Dati | FBref via `soccerdata` (Chrome headless) |
| Statistica | scipy, statsmodels (bootstrap, t-test) |
| Cache | Parquet via pyarrow |
| Testing | pytest (97 test) |
| CI | GitHub Actions |

## Quick Start

```bash
# Clona
git clone https://github.com/tuo-username/football-analytics-hub.git
cd football-analytics-hub

# Crea ambiente virtuale e installa dipendenze
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Configura AI (opzionale — senza funziona in modalità data-only)
cp .env.example .env
# Modifica .env: GROQ_API_KEY=gsk_tua_chiave

# Avvia dashboard
streamlit run app.py
```

Apri il browser su `http://localhost:8501`. Seleziona lega, stagione e numero partite nel sidebar, clicca **"Carica dati"** e naviga tra le 5 pagine.

### Requisiti

- **Python 3.12**
- **Google Chrome** installato (necessario per scaricare dati da FBref via seleniumbase)
- **Connessione Internet** per il primo caricamento dati (le successive letture usano la cache locale)

## AI Setup

Il report AI richiede una chiave Groq API (gratuita, con rate limit generoso):

1. Registrati su [console.groq.com](https://console.groq.com/)
2. Genera una API key
3. Copia `.env.example` in `.env` e inserisci la chiave, oppure usa il pannello secrets di Streamlit Cloud
4. Riavvia l'app

**Senza chiave AI**, il sistema funziona comunque in **modalità data-only** con tabelle, grafici e statistiche descrittive — nessuna funzionalità è bloccata.

## CLI Pipeline

```bash
# Esegui con dati reali (20 partite)
python run_pipeline.py --max-matches 20

# Specifica lega e stagione
python run_pipeline.py --league "ITA-Serie A" --season 2024-2025 --max-matches 10

# Pesi personalizzati (Pressure, Discipline, Network)
python run_pipeline.py --weights 0.4 0.3 0.3 --max-matches 38

# Ignora cache e riscarica da FBref
python run_pipeline.py --max-matches 20 --no-cache
```

## Deploy su Streamlit Cloud

1. Fai push del repository su GitHub
2. Vai su [share.streamlit.io](https://share.streamlit.io/) e connetti il repo
3. Nel pannello **Secrets** di Streamlit Cloud, aggiungi:
   ```
   GROQ_API_KEY = "gsk_tua_chiave_qui"
   ```
4. L'app legge automaticamente `st.secrets["GROQ_API_KEY"]` (gestito da `core/config.py`)

## Tests

```bash
# Tutti i test
python -m pytest tests/ -v

# Solo test specifici
python -m pytest tests/test_core.py -v
python -m pytest tests/test_dashboard.py -v
python -m pytest tests/test_config.py -v
```

**97 test totali:** 43 backend + 20 config + 34 dashboard (inclusi AppTest smoke per tutte 5 pagine, test multi-lega/multi-stagione, test fallback AI, test selector reset, test data quality guardrails).

## Struttura del progetto

```
football-analytics-hub/
├── app.py                       # Dashboard Streamlit (5 pagine)
├── run_pipeline.py              # Entry point CLI
├── AGENTS.md                    # Convenzioni per AI coding agent
├── README.md
├── requirements.txt
├── .env.example                 # Template per GROQ_API_KEY
├── .gitignore
├── LICENSE                      # MIT
├── .github/workflows/ci.yml     # GitHub Actions CI
├── .streamlit/
│   ├── config.toml              # Tema dark Streamlit
│   └── secrets.toml.example     # Template secrets per Streamlit Cloud
├── core/
│   ├── config.py                # Config/env loader (condiviso CLI + dashboard)
│   ├── data_loader.py           # Download/cache FBref in Parquet
│   ├── pressure.py              # Pressure Component (Sh, SoT, Gls)
│   ├── discipline.py            # Discipline Component (falli, cartellini)
│   ├── network.py               # Network Component (possesso, bonus gol)
│   └── index_builder.py         # TacticalPulse Score composito
├── agents/
│   ├── orchestrator.py          # Pipeline CrewAI + fallback data-only
│   ├── analyst_agent.py
│   ├── statistician_agent.py
│   ├── writer_agent.py
│   └── critic_agent.py
├── stats/
│   └── significance.py          # Bootstrap CI 95%, t-test, correlazioni
├── tests/
│   ├── test_core.py             # 43 test: componenti, cup filter, guardrails
│   ├── test_config.py           # 20 test: mask_key, load env, ai config
│   └── test_dashboard.py        # 34 test: helpers, AppTest, multi-league
├── data/raw/                    # Cache Parquet (gitignorato)
├── reports/output/              # Report generati (gitignorato)
├── docs/screenshots/            # Screenshot demo
└── scripts/                     # Script diagnostici (gitignorato)
```

## Architettura del modello

### Componenti del TacticalPulse Index

| Componente | Input | Metodo | Range |
|---|---|---|---|
| **Pressure** | Sh (tiri), SoT (tiri in porta), Gls (gol) | Percentile rank per lega | 0–100 |
| **Discipline** | Fls (falli), CrdY (gialli), CrdR (rossi) | Min-max inversa con pesi (1×, 1.5×, 3×) | 0–100 |
| **Network** | Poss (possesso %), gol totali | Possesso medio + bonus gol (scaled) | 0–100 |
| **TacticalPulse** | 3 componenti | Media ponderata (default: 1/3 ciascuno) | 0–100 |

### Pipeline AI

Il report narrativo è generato da 4 agenti CrewAI in sequenza:

1. **Analyst** — Analizza i dati, identifica pattern e anomalie
2. **Statistician** — Valida statisticamente i pattern (bootstrap, t-test, correlazioni)
3. **Writer** — Scrive il report markdown professionale in italiano
4. **Critic** — Verifica che ogni affermazione sia supportata dai dati

### Data Quality Guardrails

Il sistema include 5 guardrail automatici durante il caricamento dati:

| Guardrail | Tipo | Soglia |
|---|---|---|
| Team count anomaly | `warnings.warn` | <10 o >25 squadre |
| Critical columns missing | `ValueError` | Sh/SoT/Gls/Fls/CrdY/CrdR/Poss |
| Excessive cup filtering | `warnings.warn` | >30% righe rimosse |
| Empty dataset | `ValueError` | 0 righe dopo filtraggio |
| Pre-filter size tracking | Log | Registra dimensione prima del filtro |

## Screenshot

| Pagina | Anteprima |
|---|---|
| **Overview** — KPI, bar chart, classifica completa con profili tattici | ![overview](docs/screenshots/overview.png) |
| **Team Comparison** — Radar chart, metriche comparative, insight automatici | ![comparison](docs/screenshots/comparison.png) |
| **Single Team Deep Dive** — Gauge charts, breakdown componenti, trend temporale | ![deep_dive](docs/screenshots/deep_dive.png) |
| **AI Report** — Report narrativo CrewAI (o data-only come fallback) | ![ai_report](docs/screenshots/ai_report.png) |
| **Statistical Validation** — Bootstrap CI, t-test, distribuzione punteggi | ![validation](docs/screenshots/validation.png) |

## Combinazioni verificate

Tutte le seguenti combinazioni sono state testate con zero contaminazioni cross-league (nessuna partita di coppa infiltrata):

| Lega | 2024–2025 | 2023–2024 | 2022–2023 |
|---|---|---|---|
| Premier League | ✅ 20 squadre | ✅ 20 squadre | ✅ 20 squadre |
| Serie A | ✅ 20 squadre | ✅ 20 squadre (94 righe coppe filtrate) | — |
| La Liga | ✅ 20 squadre | — | — |
| Bundesliga | ✅ 18 squadre | — | — |
| Ligue 1 | ✅ 18 squadre | — | — |

## Limiti noti

- **Chrome orphan processes**: `soccerdata` non fa cleanup dei processi Chrome dopo ogni `load_events()`. In sessioni di sviluppo intense, esegui `taskkill /F /IM chrome.exe` per liberare memoria.
- **Fresh download non verificato**: alcune combinazioni non prioritarie (es. Bundesliga/Ligue 1 2022–2023) non sono state testate con download fresh Chrome; richiedono la cache Parquet locale.
- **Stagioni supportate**: ufficialmente testate 2022–2023, 2023–2024, 2024–2025. Stagioni antecedenti potrebbero funzionare ma non sono verificate.
- **FBref coverage**: i dati provengono da FBref, che ha copertura non uniforme tra leghe e stagioni. Leghe minori potrebbero non essere disponibili.
- **Network proxy**: la componente Network usa solo possesso e gol — non ci sono dati di passaggi disponibili da FBref per un'analisi più fine.
- **Cache cloud**: la cache Parquet è locale. Ogni deploy su Streamlit Cloud deve riscaricare i dati al primo avvio (richiede Chrome e qualche minuto).
- **Cup filtering**: FBref non espone una colonna `Comp` per distinguere campionato da coppe. Il filtro si basa sulla team list della lega (entrambe le squadre devono appartenere alla stessa lega).

## Roadmap

- [x] Multi-lega (5 leghe supportate)
- [x] Multi-stagione (3 stagioni)
- [x] Profilo squadra automatico
- [x] Pesi modificabili nell'interfaccia
- [x] Analisi temporale e momentum
- [x] Data quality guardrails
- [x] GitHub Actions CI
- [ ] Confronto inter-stagionale diretto
- [ ] xG (Expected Goals) quando disponibile su FBref
- [ ] Download PDF dei report AI
- [ ] Docker container

## Licenza

MIT. Vedi il file [LICENSE](LICENSE).

---

*Creato con passione per il calcio e i dati.*
