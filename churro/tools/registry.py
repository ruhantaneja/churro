from typing import Any

from churro.tools.tool import (
    DuplicateToolError,
    Tool,
    ToolDefinitionError,
    ToolResult,
    UnknownToolError,
)


class ToolRegistry:
    """Registry that knows HOW to dispatch a tool, never the implementation.

    Individual tool implementations live in their own Tool subclasses; the
    registry stores them by name and forwards executions. Adding a new tool
    only requires registering a new Tool subclass — no registry changes.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise ToolDefinitionError(
                f"Cannot register {type(tool).__name__}: expected a Tool instance"
            )
        if not tool.name:
            raise ToolDefinitionError("Cannot register a tool with an empty name")
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(f"Unknown tool: {name!r}") from None

    def list_tools(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda tool: tool.name)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Dispatch to a tool and always return a controlled ToolResult.

        An unknown tool name produces a success=False result instead of
        raising, so callers can safely forward any tool request.
        """
        try:
            tool = self.get(name)
        except UnknownToolError as exc:
            return ToolResult(success=False, error=str(exc))
        return tool.execute(arguments)