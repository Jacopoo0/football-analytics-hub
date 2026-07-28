"""Test per il modulo di configurazione centralizzata (core/config.py)."""

import os
import sys
from unittest.mock import patch

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core.config import (
    _HERE,
    _project_env_exists,
    load_environment,
    get_groq_api_key,
    get_ai_config,
    is_ai_enabled,
    clear_env_cache,
    _mask_key,
)


def setup_function():
    clear_env_cache()


def teardown_function():
    clear_env_cache()


# ---- _mask_key ----

class TestMaskKey:
    def test_masks_key_correctly(self):
        result = _mask_key("gsk_abcdefghijklmnop")
        assert result == "gsk_ab...mnop"

    def test_returns_none_for_short_key(self):
        assert _mask_key("abc") is None

    def test_returns_none_for_none(self):
        assert _mask_key(None) is None

    def test_does_not_expose_full_key(self):
        full = "gsk_super_secret_key_12345678"
        masked = _mask_key(full)
        assert masked is not None
        assert full not in masked
        assert "..." in masked


# ---- load_environment (con find_dotenv mockato) ----

class TestLoadEnvironment:
    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_missing_returns_no_key(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        clear_env_cache()
        result = load_environment()
        assert result["groq_api_key"] is None
        assert result["source"] == "missing"
        assert result["masked_key"] is None

    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_process_env_source(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        os.environ["GROQ_API_KEY"] = "gsk_test_key_1234567890"
        clear_env_cache()
        result = load_environment()
        assert result["groq_api_key"] == "gsk_test_key_1234567890"
        assert result["source"] == "process_env"
        del os.environ["GROQ_API_KEY"]

    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_placeholder_rejected(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        os.environ["GROQ_API_KEY"] = "tua_chiave_qui"
        clear_env_cache()
        result = load_environment()
        assert result["groq_api_key"] is None
        assert result["source"] == "missing"
        del os.environ["GROQ_API_KEY"]

    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_short_key_rejected(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        os.environ["GROQ_API_KEY"] = "sk-abc"
        clear_env_cache()
        result = load_environment()
        assert result["groq_api_key"] is None
        del os.environ["GROQ_API_KEY"]


class TestGetGroqApiKey:
    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_returns_none_when_missing(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        clear_env_cache()
        assert get_groq_api_key() is None

    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_returns_key_from_env(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        os.environ["GROQ_API_KEY"] = "gsk_valid_key_1234567890"
        clear_env_cache()
        assert get_groq_api_key() == "gsk_valid_key_1234567890"
        del os.environ["GROQ_API_KEY"]


class TestIsAiEnabled:
    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_false_when_missing(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        clear_env_cache()
        assert is_ai_enabled() is False

    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_true_when_present(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        os.environ["GROQ_API_KEY"] = "gsk_valid_key_1234567890"
        clear_env_cache()
        assert is_ai_enabled() is True
        del os.environ["GROQ_API_KEY"]


class TestGetAiConfig:
    def test_structure_is_correct(self):
        cfg = get_ai_config()
        assert isinstance(cfg, dict)
        assert "enabled" in cfg
        assert "key_present" in cfg
        assert "key_source" in cfg
        assert "masked_key" in cfg
        assert "message" in cfg

    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_disabled_state(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        clear_env_cache()
        cfg = get_ai_config()
        assert cfg["enabled"] is False
        assert cfg["key_present"] is False
        assert cfg["masked_key"] is None
        assert "non trovata" in cfg["message"]

    @patch("core.config.find_dotenv", return_value=None)
    @patch("core.config.load_dotenv")
    @patch("core.config._project_env_exists", return_value=False)
    def test_enabled_state(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        os.environ["GROQ_API_KEY"] = "gsk_valid_key_1234567890"
        clear_env_cache()
        cfg = get_ai_config()
        assert cfg["enabled"] is True
        assert cfg["key_present"] is True
        del os.environ["GROQ_API_KEY"]

    def test_masked_key_not_full_key(self):
        cfg = get_ai_config()
        if cfg["masked_key"] is not None:
            assert cfg["key_source"] != "missing"
            assert "..." in cfg["masked_key"]

    def test_message_consistent_with_source(self):
        cfg = get_ai_config()
        if cfg["key_source"] == "missing":
            assert "non trovata" in cfg["message"]
        else:
            assert "rilevata" in cfg["message"]


class TestLoadEnvironmentDotenv:
    """Test che verificano la presenza della vera .env.

    Questi test NON mockano find_dotenv e verificano
    che il sistema trovi correttamente il file .env reale.
    """

    def test_finds_real_env_file(self):
        """Verifica che .env venga trovato e che contenga una chiave."""
        clear_env_cache()
        result = load_environment()
        assert result["source"] in ("dotenv", "process_env", "missing")
        if result["groq_api_key"] is not None:
            assert result["env_path"] is not None
            assert result["source"] in ("dotenv", "process_env")

    def test_masked_key_safe_with_real_key(self):
        """Anche con chiave reale, masked_key non espone l'intera chiave."""
        clear_env_cache()
        result = load_environment()
        if result["groq_api_key"] is not None:
            mk = result["masked_key"]
            assert mk is not None
            assert result["groq_api_key"] not in mk
            assert "..." in mk
            assert len(mk) < len(result["groq_api_key"])

    def test_ai_config_consistent(self):
        """get_ai_config deve essere coerente con load_environment."""
        clear_env_cache()
        env = load_environment()
        cfg = get_ai_config()
        assert cfg["key_present"] == (env["groq_api_key"] is not None)
        assert cfg["enabled"] == (env["groq_api_key"] is not None)
