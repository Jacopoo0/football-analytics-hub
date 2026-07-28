"""Configurazione centralizzata per il TacticalPulse Index.

Gestisce il caricamento di GROQ_API_KEY da .env, .streamlit/secrets.toml,
o variabili d'ambiente. Fornisce funzioni condivise tra CLI e dashboard Streamlit.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = None
    find_dotenv = None

_HERE = Path(__file__).resolve().parent.parent
_ENV_CACHE: dict | None = None


def _project_env_exists() -> bool:
    """Restituisce True se .env esiste nella root del progetto (mockabile)."""
    return _HERE.joinpath(".env").exists()


def _mask_key(key: str | None) -> str | None:
    """Offusca una chiave API mostrando solo prime 6 + ultime 4."""
    if not key or len(key) < 12:
        return None
    return key[:6] + "..." + key[-4:]


def load_environment() -> dict:
    """Carica GROQ_API_KEY da tutte le fonti possibili.

    Ordine di priorita':
    1. st.secrets["GROQ_API_KEY"] (Streamlit Cloud deploy)
    2. .env via load_dotenv(override=True)
    3. os.environ gia' presente
    4. .streamlit/secrets.toml (locale)

    Returns
    -------
    dict
        groq_api_key: str | None
        source: "process_env" | "dotenv" | "streamlit_secrets" | "missing"
        env_path: str | None
        masked_key: str | None
    """
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE

    env_path = None
    source = "missing"
    key = None

    # 1. st.secrets (Streamlit Cloud o secrets.toml locale)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            try:
                if "GROQ_API_KEY" in st.secrets:
                    key = st.secrets["GROQ_API_KEY"]
                    source = "streamlit_secrets"
            except Exception:
                pass
    except ImportError:
        pass

    # 2. .env file via python-dotenv
    if key is None and load_dotenv is not None:
        found = find_dotenv(usecwd=True)
        if found:
            env_path = found
            load_dotenv(found, override=True)
            key = os.getenv("GROQ_API_KEY")
            source = "dotenv"
        elif _project_env_exists():
            env_path = str(_HERE / ".env")
            load_dotenv(env_path, override=True)
            key = os.getenv("GROQ_API_KEY")
            source = "dotenv"

    # 3. os.environ gia' presente
    if key is None:
        key = os.getenv("GROQ_API_KEY")
        if key:
            source = "process_env"

    # 4. .streamlit/secrets.toml locale
    if key is None:
        streamlit_secrets = _HERE / ".streamlit" / "secrets.toml"
        if streamlit_secrets.exists():
            try:
                import tomllib
                with open(streamlit_secrets, "rb") as f:
                    data = tomllib.load(f)
                key = data.get("GROQ_API_KEY")
                if key:
                    source = "streamlit_secrets"
                    env_path = str(streamlit_secrets)
            except (ImportError, tomllib.TOMLDecodeError):
                pass

    # Validazione: scarta placeholder
    if key and ("tua_chiave" in key or "sk-" in key and len(key) < 20):
        key = None
        source = "missing"

    _ENV_CACHE = {
        "groq_api_key": key,
        "source": source,
        "env_path": env_path,
        "masked_key": _mask_key(key),
    }
    return _ENV_CACHE


def get_groq_api_key() -> str | None:
    """Restituisce GROQ_API_KEY o None."""
    return load_environment()["groq_api_key"]


def is_ai_enabled() -> bool:
    """True se GROQ_API_KEY e' presente e valida."""
    return get_groq_api_key() is not None


def get_ai_config() -> dict:
    """Restituisce metadati sulla configurazione AI per la UI.

    Returns
    -------
    dict
        enabled: bool
        key_present: bool
        key_source: str
        masked_key: str | None
        message: str
    """
    env = load_environment()
    key = env["groq_api_key"]
    source = env["source"]
    present = key is not None

    messages = {
        "streamlit_secrets": "Chiave Groq rilevata da Streamlit secrets",
        "dotenv": "Chiave Groq rilevata da .env",
        "process_env": "Chiave Groq rilevata da variabili d'ambiente",
        "missing": "Chiave Groq non trovata: modalita' data-only",
    }

    return {
        "enabled": present,
        "key_present": present,
        "key_source": source,
        "masked_key": env["masked_key"],
        "message": messages.get(source, "Stato sconosciuto"),
    }


def clear_env_cache():
    """Resetta la cache (utile per test)."""
    global _ENV_CACHE
    _ENV_CACHE = None
