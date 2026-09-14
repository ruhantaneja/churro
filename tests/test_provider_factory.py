"""Offline tests for the provider factory and provider specifications.

No API keys, no network, and no Ollama/OpenAI calls. The SDK-import check
runs first because later tests import the concrete provider modules (which
is exactly what the factory avoids at import time).
"""

import os
import sys
from contextlib import contextmanager

from churro.providers.factory import (
    DEFAULT_STARTUP_PROVIDER,
    ProviderFactory,
    ProviderSpec,
    create_provider,
    default_provider_factory,
    parse_provider_spec,
    startup_provider_spec,
)
from churro.providers.provider import Provider, ProviderError, ProviderSpecError


class FakeProvider(Provider):
    def __init__(self, name="fake", model="fake-v1"):
        self.name = name
        self.model = model

    def send(self, messages):
        return "ok"


@contextmanager
def env_as(**updates):
    saved = {}
    keys = set(updates) | {"OPENAI_API_KEY", "OLLAMA_MODEL", "OLLAMA_BASE_URL"}
    for key in keys:
        saved[key] = os.environ.get(key)
        os.environ.pop(key, None)
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


# ------------------------------------------------------------------ SDK
def test_factory_imports_no_provider_sdks():
    import churro.providers.factory as factory_module

    sdk_packages = {
        name
        for name in sys.modules
        if name == "openai" or name.startswith("openai.") or name.startswith("ollama")
    }
    assert sdk_packages == set()
    assert not hasattr(factory_module, "OpenAI")
    assert not hasattr(factory_module, "Ollama")


# ---------------------------------------------------------------- parse
def test_parse_openai_provider_spec():
    spec = parse_provider_spec("openai:gpt-4o")
    assert spec.provider_name == "openai"
    assert spec.model_name == "gpt-4o"
    assert str(spec) == "openai:gpt-4o"


def test_parse_ollama_provider_spec_with_colon_in_model():
    spec = parse_provider_spec("ollama:qwen3:14b")
    assert spec.provider_name == "ollama"
    assert spec.model_name == "qwen3:14b"
    assert str(spec) == "ollama:qwen3:14b"


def test_parse_bare_provider_name_has_no_model():
    spec = parse_provider_spec("ollama")
    assert spec.provider_name == "ollama"
    assert spec.model_name is None
    assert str(spec) == "ollama"


def test_parse_bare_openai_has_no_model():
    spec = parse_provider_spec("openai")
    assert spec.provider_name == "openai"
    assert spec.model_name is None


def test_parse_lowercases_provider_name():
    spec = parse_provider_spec("OpenAI:gpt-4o")
    assert spec.provider_name == "openai"
    assert spec.model_name == "gpt-4o"


def test_parse_strips_surrounding_whitespace():
    spec = parse_provider_spec("  ollama : qwen3:14b  ")
    assert spec.provider_name == "ollama"
    assert spec.model_name == "qwen3:14b"


def test_parse_empty_spec_invalid():
    with env_as():
        try:
            parse_provider_spec("")
        except ProviderSpecError as exc:
            assert "empty" in str(exc)
        else:
            raise AssertionError("expected ProviderSpecError")


def test_parse_whitespace_only_invalid():
    try:
        parse_provider_spec("   ")
    except ProviderSpecError:
        pass
    else:
        raise AssertionError("expected ProviderSpecError")


def test_parse_model_without_provider_invalid():
    try:
        parse_provider_spec(":gpt-4o")
    except ProviderSpecError as exc:
        assert "missing provider name" in str(exc)
    else:
        raise AssertionError("expected ProviderSpecError")


def test_parse_provider_without_model_invalid():
    try:
        parse_provider_spec("ollama:")
    except ProviderSpecError as exc:
        assert "missing model name" in str(exc)
    else:
        raise AssertionError("expected ProviderSpecError")


def test_parse_non_string_invalid():
    try:
        parse_provider_spec(123)  # type: ignore[arg-type]
    except ProviderSpecError as exc:
        assert "string" in str(exc)
    else:
        raise AssertionError("expected ProviderSpecError")


def test_spec_equality_and_str_roundtrip():
    assert ProviderSpec("openai", "gpt-4o") == ProviderSpec("openai", "gpt-4o")
    assert parse_provider_spec("ollama:qwen3:14b") == ProviderSpec("ollama", "qwen3:14b")
    assert str(ProviderSpec("ollama")) == "ollama"


# --------------------------------------------------------- startup spec
def test_startup_spec_defaults_to_openai():
    assert startup_provider_spec({}) == DEFAULT_STARTUP_PROVIDER
    assert startup_provider_spec() == DEFAULT_STARTUP_PROVIDER


def test_startup_spec_provider_only():
    assert startup_provider_spec({"CHURRO_PROVIDER": "ollama"}) == "ollama"


def test_startup_spec_provider_and_model():
    assert (
        startup_provider_spec({"CHURRO_PROVIDER": "ollama", "CHURRO_MODEL": "qwen3:14b"})
        == "ollama:qwen3:14b"
    )


def test_startup_spec_inline_model_wins_over_separate_model():
    assert (
        startup_provider_spec(
            {
                "CHURRO_PROVIDER": "ollama:qwen3:32b",
                "CHURRO_MODEL": "qwen3:14b",
            }
        )
        == "ollama:qwen3:32b"
    )


def test_startup_spec_model_applies_to_default_provider():
    assert startup_provider_spec({"CHURRO_MODEL": "gpt-4o-mini"}) == "openai:gpt-4o-mini"


def test_startup_spec_strips_whitespace():
    assert (
        startup_provider_spec({"CHURRO_PROVIDER": "  ollama  ", "CHURRO_MODEL": " qwen3:14b "})
        == "ollama:qwen3:14b"
    )


def test_startup_spec_blank_provider_falls_back_to_default():
    assert startup_provider_spec({"CHURRO_PROVIDER": "   "}) == DEFAULT_STARTUP_PROVIDER


def test_startup_spec_parses_back_into_provider_spec():
    parsed = parse_provider_spec(
        startup_provider_spec({"CHURRO_PROVIDER": "ollama", "CHURRO_MODEL": "qwen3:14b"})
    )
    assert parsed.provider_name == "ollama"
    assert parsed.model_name == "qwen3:14b"


def test_startup_spec_never_constructs_providers():
    spec = startup_provider_spec({"CHURRO_PROVIDER": "ollama", "CHURRO_MODEL": "qwen3:14b"})
    assert isinstance(spec, str)
    assert spec == "ollama:qwen3:14b"


# -------------------------------------------------------------- factory
def test_empty_factory_reports_no_providers():
    factory = ProviderFactory()
    try:
        factory.create("ollama")
    except ProviderError as exc:
        assert "No providers registered" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_register_and_create_custom_builder():
    factory = ProviderFactory()
    factory.register("echo", lambda model=None: FakeProvider(name="echo", model=model or "e-1"))
    provider = factory.create(ProviderSpec("echo", "e-2"))
    assert provider.name == "echo"
    assert provider.model == "e-2"


def test_register_normalizes_and_rejects_duplicates():
    factory = ProviderFactory()
    factory.register("Echo", lambda model=None: FakeProvider(name="echo"))
    try:
        factory.register("echo", lambda model=None: FakeProvider(name="echo"))
    except ProviderError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_register_rejects_empty_name():
    factory = ProviderFactory()
    try:
        factory.register("  ", lambda model=None: FakeProvider())
    except ProviderError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_unknown_provider_clean_error():
    factory = default_provider_factory()
    try:
        factory.create("anthropic:claude-opus")
    except ProviderError as exc:
        assert "anthropic" in str(exc)
        assert "not available" in str(exc)
        assert "ollama" in str(exc)
        assert "openai" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_default_factory_supports_openai_and_ollama():
    factory = default_provider_factory()
    assert factory.supported_providers() == ["ollama", "openai"]


def test_default_factory_builds_ollama_provider():
    with env_as():
        provider = create_provider("ollama:qwen3:14b")
    assert provider.name == "ollama"
    assert provider.model == "qwen3:14b"
    assert provider.base_url == "http://localhost:11434/v1"


def test_create_accepts_provider_spec_object():
    factory = default_provider_factory()
    provider = factory.create(ProviderSpec("ollama", "qwen3:14b"))
    assert provider.model == "qwen3:14b"


def test_ollama_default_model_comes_from_env():
    with env_as(OLLAMA_MODEL="qwen3:14b"):
        provider = create_provider("ollama")
    assert provider.model == "qwen3:14b"


def test_ollama_bare_without_env_has_no_hardcoded_model():
    with env_as():
        try:
            create_provider("ollama")
        except ProviderError as exc:
            assert "OLLAMA_MODEL" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_openai_construction_offline():
    with env_as(OPENAI_API_KEY="sk-dummy-for-offline-test"):
        provider = create_provider("openai:gpt-4o-mini")
    assert provider.name == "openai"
    assert provider.model == "gpt-4o-mini"


def test_openai_default_model_when_bare():
    with env_as(OPENAI_API_KEY="sk-dummy-for-offline-test"):
        provider = create_provider("openai")
    assert provider.name == "openai"
    assert provider.model == "gpt-4o"


def test_openai_missing_key_raises_clean_error():
    with env_as():
        try:
            create_provider("openai")
        except ProviderError as exc:
            assert "OPENAI_API_KEY" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_provider_spec_error_is_provider_error():
    assert issubclass(ProviderSpecError, ProviderError)


TEST_FUNCTIONS = [
    test_factory_imports_no_provider_sdks,
    test_parse_openai_provider_spec,
    test_parse_ollama_provider_spec_with_colon_in_model,
    test_parse_bare_provider_name_has_no_model,
    test_parse_bare_openai_has_no_model,
    test_parse_lowercases_provider_name,
    test_parse_strips_surrounding_whitespace,
    test_parse_empty_spec_invalid,
    test_parse_whitespace_only_invalid,
    test_parse_model_without_provider_invalid,
    test_parse_provider_without_model_invalid,
    test_parse_non_string_invalid,
    test_spec_equality_and_str_roundtrip,
    test_startup_spec_defaults_to_openai,
    test_startup_spec_provider_only,
    test_startup_spec_provider_and_model,
    test_startup_spec_inline_model_wins_over_separate_model,
    test_startup_spec_model_applies_to_default_provider,
    test_startup_spec_strips_whitespace,
    test_startup_spec_blank_provider_falls_back_to_default,
    test_startup_spec_parses_back_into_provider_spec,
    test_startup_spec_never_constructs_providers,
    test_empty_factory_reports_no_providers,
    test_register_and_create_custom_builder,
    test_register_normalizes_and_rejects_duplicates,
    test_register_rejects_empty_name,
    test_unknown_provider_clean_error,
    test_default_factory_supports_openai_and_ollama,
    test_default_factory_builds_ollama_provider,
    test_create_accepts_provider_spec_object,
    test_ollama_default_model_comes_from_env,
    test_ollama_bare_without_env_has_no_hardcoded_model,
    test_openai_construction_offline,
    test_openai_default_model_when_bare,
    test_openai_missing_key_raises_clean_error,
    test_provider_spec_error_is_provider_error,
]


def main():
    failures = 0
    for test in TEST_FUNCTIONS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - test runner reports
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(TEST_FUNCTIONS) - failures}/{len(TEST_FUNCTIONS)} provider factory tests passed.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()