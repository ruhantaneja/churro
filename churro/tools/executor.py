"""Execute normalized LLM tool calls through the ToolRegistry.

THIS is the single controlled cycle that connects the provider API
boundary to the tool system:

    ProviderResponse -> ToolCall -> ToolExecutor -> ToolRegistry -> ToolResult

The coordinator is provider-neutral: it only knows churro's normalized
types (``ToolCall`` / ``ProviderResponse``) and the existing
``ToolRegistry``. It never imports an LLM SDK, never calls an LLM, never
mutates state, and never starts an autonomous loop. An agent loop on top
of this executor is a separate, later step.
"""

from pydantic import BaseModel, ConfigDict, Field

from churro.providers.responses import ProviderResponse, ToolCall
from churro.tools.registry import ToolRegistry
from churro.tools.tool import ToolResult


class ToolExecutionResult(BaseModel):
    """A single normalized record of one executed tool call.

    ``call_id`` is the identifier assigned by the original provider call,
    so the future conversation can associate this result back to it.
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    result: ToolResult

    @property
    def success(self) -> bool:
        return self.result.success


class ToolExecutionBatch(BaseModel):
    """An ordered set of results for every tool call in one response."""

    model_config = ConfigDict(extra="forbid")

    executions: list[ToolExecutionResult] = Field(default_factory=list)

    @property
    def has_executions(self) -> bool:
        return bool(self.executions)

    @property
    def all_succeeded(self) -> bool:
        return bool(self.executions) and all(
            execution.result.success for execution in self.executions
        )


class ToolExecutor:
    """Executes every ``ToolCall`` in a ``ProviderResponse`` in order.

    Unknown tools, invalid arguments, and tools that raise mid-run all
    become controlled ``ToolResult`` failures; one failing call never
    prevents the remaining calls in the same response from running.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, response: ProviderResponse) -> ToolExecutionBatch:
        """Execute the response's tool calls and return an ordered batch.

        A response with no tool calls returns an empty batch, never an
        error. The original ``ProviderResponse`` and its ``ToolCall``
        argument dicts are only read, never mutated.
        """
        batch = ToolExecutionBatch()
        for call in response.tool_calls:
            batch.executions.append(self._execute_call(call))
        return batch

    def _execute_call(self, call: ToolCall) -> ToolExecutionResult:
        try:
            result = self.registry.execute(call.name, call.arguments)
        except Exception as exc:  # defensiveness; registry is total
            result = ToolResult(
                success=False,
                error=f"Tool execution failed: {type(exc).__name__}: {exc}",
            )
        return ToolExecutionResult(call_id=call.id, tool_name=call.name, result=result)