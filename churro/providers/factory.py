"""Provider selection and construction.

The CLI asks :class:`ProviderFactory` for a provider instead of building
providers itself. A provider specification (``"ollama:qwen3:14b"``) is
parsed into a normalized :class:`ProviderSpec`, and the factory resolves it
to a concrete :class:`~churro.providers.provider.Provider`.

Adding another provider later means adding one provider implementation and
one registration here, not touching the CLI.
"""

from dataclasses import dataclass
import os
from typing import Callable, Mapping

from churro.providers.provider import Provider, ProviderError, ProviderSpecError

Builder = Callable[[str | None], Provider]

STARTUP_PROVIDER_ENV = "CHURRO_PROVIDER"
STARTUP_MODEL_ENV = "CHURRO_MODEL"
DEFAULT_STARTUP_PROVIDER = "openai"


def startup_provider_spec(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Build the CLI startup provider spec from the environment.

    Returns a provider specification to feed straight into
    :func:`parse_provider_spec`. Rules:

    - ``CHURRO_PROVIDER`` set       -> that provider (`"ollama"`).
    - ``CHURRO_PROVIDER`` + ``CHURRO_MODEL``
                                    -> `"ollama:qwen3:14b"`.
    - ``CHURRO_PROVIDER`` already carries a model (`"ollama:qwen3:14b"`)
                                    -> used verbatim; ``CHURRO_MODEL``
                                      is ignored.
    - neither set                   -> default provider
                                      :data:`DEFAULT_STARTUP_PROVIDER`.
    - ``CHURRO_MODEL`` set with no provider -> applied to the default
      provider (`"openai:<model>"`).

    Accepts an explicit ``environ`` mapping for tests; defaults to
    ``os.environ``. No provider is constructed here, so no network or key
    validation ever happens inside this function.
    """
    env = environ if environ is not None else os.environ
    provider = (env.get(STARTUP_PROVIDER_ENV) or "").strip()
    model = (env.get(STARTUP_MODEL_ENV) or "").strip()

    if not provider:
        provider = DEFAULT_STARTUP_PROVIDER
    if ":" in provider:
        return provider
    if model:
        return f"{provider}:{model}"
    return provider


@dataclass(frozen=True)
class ProviderSpec:
    """A normalized provider selection: provider name plus optional model.

    ``"ollama:qwen3:14b"`` parses to ``provider_name="ollama"`` and
    ``model_name="qwen3:14b"``; a bare ``"openai"`` leaves ``model_name``
    unset so the provider constructor's default (or environment
    configuration) applies.
    """

    provider_name: str
    model_name: str | None = None

    def __str__(self) -> str:
        if self.model_name:
            return f"{self.provider_name}:{self.model_name}"
        return self.provider_name


def parse_provider_spec(spec: str) -> ProviderSpec:
    """Parse a provider specification string into a :class:`ProviderSpec`.

    Valid forms: ``"provider"`` or ``"provider:model"``. Whitespace is
    stripped; the provider name is lowercased; model names are kept verbatim
    (Ollama models legitimately contain a colon, e.g. ``qwen3:14b``). Empty
    provider names, empty model names, and non-string input raise
    :class:`ProviderSpecError` with a clear message.
    """
    if not isinstance(spec, str):
        raise ProviderSpecError(
            f"Provider spec must be a string, got {type(spec).__name__}"
        )

    raw = spec.strip()
    if not raw:
        raise ProviderSpecError(
            "Provider spec is empty. Use 'openai', 'ollama', or '<provider>:<model>'."
        )

    name, separator, model = raw.partition(":")
    name = name.strip()
    model = model.strip() if separator else None

    if not name:
        raise ProviderSpecError(f"Invalid provider spec {spec!r}: missing provider name.")
    if separator and not model:
        raise ProviderSpecError(f"Invalid provider spec {spec!r}: missing model name.")

    return ProviderSpec(provider_name=name.lower(), model_name=model)


class ProviderFactory:
    """Builds :class:`Provider` instances from :class:`ProviderSpec` values.

    Providers register a name and a small builder callable
    ``(model: str | None) -> Provider``. The factory knows nothing about how
    a provider is constructed; builders hold that knowledge.
    """

    def __init__(self) -> None:
        self._builders: dict[str, Builder] = {}

    def register(self, name: str, builder: Builder) -> None:
        key = name.strip().lower()
        if not key:
            raise ProviderError("Provider name cannot be empty.")
        if key in self._builders:
            raise ProviderError(f"Provider {key!r} is already registered.")
        self._builders[key] = builder

    def supported_providers(self) -> list[str]:
        return sorted(self._builders)

    def create(self, spec: ProviderSpec | str) -> Provider:
        """Resolve a provider specification to a concrete provider.

        Unknown provider names raise :class:`ProviderError` listing the
        supported providers so callers (and users) get a clear message.
        """
        parsed = parse_provider_spec(spec) if isinstance(spec, str) else spec
        builder = self._builders.get(parsed.provider_name)
        if builder is None:
            available = ", ".join(self.supported_providers())
            available = f"Available providers: {available}" if available else "No providers registered."
            raise ProviderError(
                f"Provider {parsed.provider_name!r} is not available in CHURRO. {available}"
            )
        return builder(parsed.model_name)


def _openai_builder(model: str | None) -> Provider:
    from churro.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(model=model) if model else OpenAIProvider()


def _ollama_builder(model: str | None) -> Provider:
    from churro.providers.ollama_provider import OllamaProvider

    return OllamaProvider(model=model)


def default_provider_factory() -> ProviderFactory:
    """The standard factory backed by the ``openai`` and ``ollama`` builders.

    All builder modules are imported lazily inside their builder, so
    importing this factory never requires the openai SDK (or any SDK).
    """
    factory = ProviderFactory()
    factory.register("openai", _openai_builder)
    factory.register("ollama", _ollama_builder)
    return factory


_DEFAULT_FACTORY: ProviderFactory | None = None


def create_provider(spec: ProviderSpec | str) -> Provider:
    """Convenience: build a provider with the default factory.

    Ollama honors ``OLLAMA_MODEL`` / ``OLLAMA_BASE_URL`` from the
    environment when the spec omits the model; OpenAI honors the existing
    ``OPENAI_API_KEY`` configuration. No specific Ollama model is hardcoded.
    """
    global _DEFAULT_FACTORY
    if _DEFAULT_FACTORY is None:
        _DEFAULT_FACTORY = default_provider_factory()
    return _DEFAULT_FACTORY.create(spec)