from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError


class ToolArgs(BaseModel):
    """Required base class for every tool's argument model.

    Unknown arguments are rejected (extra="forbid") so a typo in a tool
    call fails loudly as a controlled error instead of being ignored.
    """

    model_config = ConfigDict(extra="forbid")


A = TypeVar("A", bound=ToolArgs)


class ToolError(Exception):
    """Base class for every tool error surfaced to CHURRO."""


class ToolDefinitionError(ToolError):
    """A tool class or registration is invalid."""


class UnknownToolError(ToolError):
    """A tool was looked up but is not registered."""


class DuplicateToolError(ToolError):
    """A tool with the same name is already registered."""


class ToolResult(BaseModel):
    """Structured, validated result of executing a tool."""

    success: bool
    output: str = ""
    error: str = ""


def _validation_error_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<arguments>"
        parts.append(f"{loc}: {err['msg']}")
    return "Invalid arguments: " + "; ".join(parts)


class Tool(ABC, Generic[A]):
    """Base class for every CHURRO tool.

    A tool owns its own definition (name, description, argument schema)
    and its implementation. The ToolRegistry only dispatches by name; it
    never contains tool implementations, so adding a tool never requires
    touching the registry.
    """

    name: str = ""
    description: str = ""
    args_model: type[A]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for attr in ("name", "description"):
            if not getattr(cls, attr, None):
                raise ToolDefinitionError(
                    f"Tool subclass {cls.__name__!r} must define a non-empty {attr!r}"
                )
        args_model = getattr(cls, "args_model", None)
        if args_model is None:
            raise ToolDefinitionError(
                f"Tool subclass {cls.__name__!r} must define an 'args_model'"
            )
        if not isinstance(args_model, type) or not issubclass(args_model, ToolArgs):
            raise ToolDefinitionError(
                f"Tool subclass {cls.__name__!r} args_model must be a subclass "
                "of ToolArgs"
            )

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def execute(self, arguments: Any) -> ToolResult:
        """Validate arguments and run the tool.

        This method is total: it always returns a ToolResult and never
        lets a bad argument or a failing tool crash CHURRO.
        """
        try:
            validated = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(success=False, error=_validation_error_message(exc))
        except Exception as exc:
            return ToolResult(success=False, error=f"Invalid arguments: {exc}")

        try:
            result = self._run(validated)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {type(exc).__name__}: {exc}",
            )

        if not isinstance(result, ToolResult):
            return ToolResult(
                success=False,
                error=(
                    f"Tool {self.name!r} returned {type(result).__name__}, "
                    "expected a ToolResult"
                ),
            )
        return result

    @abstractmethod
    def _run(self, args: A) -> ToolResult:
        """Run the tool against already-validated arguments."""