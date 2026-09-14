"""Offline tests for CHURRO startup configuration.

Covers `CHURRO_PROVIDER` / `CHURRO_MODEL` startup selection end-to-end
through the real CLI entry point. No API keys, no network, no live
Ollama/OpenAI requests.
"""

import contextlib
import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from churro.core.session import load_session
from churro.main import main as run_churro_main
from churro.providers.factory import ProviderSpec
from churro.providers.provider import Provider


class FakeStartupProvider(Provider):
    def __init__(self, name="openai", model="gpt-test"):
        self.name = name
        self.model = model

    def send(self, messages):
        raise AssertionError("no provider sends expected during startup tests")


@contextmanager
def env_scope(**updates):
    touched = set(updates) | {
        "OPENAI_API_KEY",
        "CHURRO_PROVIDER",
        "CHURRO_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
    }
    saved = {key: os.environ.get(key) for key in touched}
    for key in touched:
        os.environ.pop(key, None)
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


def run_main(inputs=(), env=None, sessions_dir=None):
    """Run ``python -m churro`` offline with scripted input + env."""
    env = env or {}
    sessions = sessions_dir or Path(tempfile.mkdtemp())
    buf = io.StringIO()
    with env_scope(**env):
        with patch("builtins.input", side_effect=list(inputs)):
            with patch("churro.main.SESSIONS_DIR", sessions):
                with contextlib.redirect_stdout(buf):
                    with contextlib.redirect_stderr(buf):
                        code = run_churro_main([])
    return code, buf.getvalue(), sessions


def test_ollama_startup_without_openai_key_is_the_goal_case():
    code, text, _ = run_main(
        inputs=["Fix the login bug", "/status", EOFError()],
        env={"CHURRO_PROVIDER": "ollama", "CHURRO_MODEL": "qwen3:14b"},
    )

    assert code == 0
    assert "Model: ollama/qwen3:14b" in text
    assert "Provider:  ollama" in text
    assert "Model:     qwen3:14b" in text
    assert "OPENAI_API_KEY" not in text


def test_openai_startup_still_validates_api_key():
    code, text, _ = run_main(env={"CHURRO_PROVIDER": "openai"})

    assert code == 1
    assert "OPENAI_API_KEY" in text


def test_default_startup_behavior_remains_openai_requiring_key():
    code, text, _ = run_main()

    assert code == 1
    assert "OPENAI_API_KEY" in text


def test_new_session_records_startup_provider_and_model():
    code, _, sessions = run_main(
        inputs=["Fix the login bug", EOFError()],
        env={"CHURRO_PROVIDER": "ollama", "CHURRO_MODEL": "qwen3:14b"},
    )

    assert code == 0
    files = list(sessions.glob("*.json"))
    assert len(files) == 1
    loaded = load_session(str(files[0]))
    assert loaded.active_provider == "ollama"
    assert loaded.active_model == "qwen3:14b"


def test_factory_is_used_for_provider_construction():
    captured = {}

    class RecordingFactory:
        def create(self, spec):
            captured["spec"] = spec
            return FakeStartupProvider(
                name=spec.provider_name, model=spec.model_name or "default"
            )

    with env_scope(CHURRO_PROVIDER="ollama", CHURRO_MODEL="qwen3:14b"):
        buf = io.StringIO()
        with patch("builtins.input", side_effect=iter(["Fix it", EOFError()])):
            with patch("churro.main.SESSIONS_DIR", Path(tempfile.mkdtemp())):
                with patch("churro.main.default_provider_factory", return_value=RecordingFactory()):
                    with contextlib.redirect_stdout(buf):
                        code = run_churro_main([])

    assert code == 0
    assert captured["spec"] == ProviderSpec("ollama", "qwen3:14b")


def test_main_constructs_providers_only_through_factory():
    main_py = (Path(__file__).resolve().parents[1] / "churro" / "main.py").read_text(encoding="utf-8")

    assert "factory.create(" in main_py
    assert "startup_provider_spec()" in main_py
    assert "OpenAIProvider(" not in main_py
    assert "OllamaProvider(" not in main_py


def test_invalid_provider_handled_cleanly():
    code, text, _ = run_main(env={"CHURRO_PROVIDER": "anthropic"})

    assert code == 1
    assert "not available" in text
    assert "openai" in text
    assert "ollama" in text


def test_invalid_config_missing_model_handled_cleanly():
    code, text, _ = run_main(env={"CHURRO_PROVIDER": "ollama:"})

    assert code == 1
    assert "missing model name" in text


def test_blank_provider_config_falls_back_to_default():
    code, text, _ = run_main(env={"CHURRO_PROVIDER": "   "})

    assert code == 1
    assert "OPENAI_API_KEY" in text


def test_no_network_calls_during_startup():
    with env_scope(CHURRO_PROVIDER="ollama", CHURRO_MODEL="qwen3:14b"):
        buf = io.StringIO()
        with patch("urllib.request.urlopen") as urlopen:
            with patch("builtins.input", side_effect=["Fix it", EOFError()]):
                with patch("churro.main.SESSIONS_DIR", Path(tempfile.mkdtemp())):
                    with contextlib.redirect_stdout(buf):
                        code = run_churro_main([])

    assert code == 0
    assert urlopen.call_count == 0


def test_startup_output_leaks_no_secrets():
    _, text, _ = run_main(
        inputs=["Fix the login bug", "/status", EOFError()],
        env={"CHURRO_PROVIDER": "ollama", "CHURRO_MODEL": "qwen3:14b"},
    )

    assert "sk-" not in text
    assert "CHURRO_PROVIDER" not in text
    assert "CHURRO_MODEL" not in text


def test_core_modules_untouched_by_startup_config():
    root = Path(__file__).resolve().parents[1]
    for relative in [
        "churro/agent/runner.py",
        "churro/tools/executor.py",
        "churro/tools/registry.py",
    ]:
        src = (root / relative).read_text(encoding="utf-8")
        assert "CHURRO_PROVIDER" not in src
        assert "CHURRO_MODEL" not in src
        assert "startup_provider_spec" not in src


def test_agent_runner_surface_unchanged():
    runner_src = (
        Path(__file__).resolve().parents[1] / "churro" / "agent" / "runner.py"
    ).read_text(encoding="utf-8")

    assert "def __init__(\n        self,\n        provider: Provider,\n        registry: ToolRegistry,\n        max_iterations: int" in runner_src


TEST_FUNCTIONS = [
    test_ollama_startup_without_openai_key_is_the_goal_case,
    test_openai_startup_still_validates_api_key,
    test_default_startup_behavior_remains_openai_requiring_key,
    test_new_session_records_startup_provider_and_model,
    test_factory_is_used_for_provider_construction,
    test_main_constructs_providers_only_through_factory,
    test_invalid_provider_handled_cleanly,
    test_invalid_config_missing_model_handled_cleanly,
    test_blank_provider_config_falls_back_to_default,
    test_no_network_calls_during_startup,
    test_startup_output_leaks_no_secrets,
    test_core_modules_untouched_by_startup_config,
    test_agent_runner_surface_unchanged,
]


def main() -> None:
    failures = 0
    for fn in TEST_FUNCTIONS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(f"\n{failures} test(s) failed")
    print(f"\nAll {len(TEST_FUNCTIONS)} tests passed.")


if __name__ == "__main__":
    main()