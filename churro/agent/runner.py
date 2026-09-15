"""A bounded, provider-neutral agent loop.

``AgentRunner`` is pure orchestration. It drives one injected provider
over repeated normalized requests, executes any returned ``ToolCall``
objects through the injected ``ToolRegistry``, converts the results into
provider-neutral ``ToolResultMessage`` objects, and feeds them back to the
provider until the model stops calling tools or the iteration budget runs
out.

The runner imports no provider SDK, holds no hidden global state, and does
not touch SessionState or session history. A future integration layer owns
deciding how/when session history is updated.
"""

from pydantic import BaseModel, ConfigDict, Field

from churro.providers.provider import ChatMessage, Provider, ProviderError
from churro.providers.responses import ProviderResponse, ToolResultMessage
from churro.providers.tool_definition import ToolDefinition
from churro.tools.executor import ToolExecutionResult, ToolExecutor
from churro.tools.registry import ToolRegistry
from churro.tools.result_messages import result_messages_from_batch

DEFAULT_MAX_ITERATIONS = 10


class AgentResult(BaseModel):
    """The outcome of one bounded agent run.

    ``completed`` is True only when the provider produced a final response
    with no tool calls. A run that exhausts ``max_iterations``, hits a
    provider error, or is interrupted returns ``completed=False`` with
    everything observed so far, so a future integration layer can decide
    how to handle it. ``interrupted`` distinguishes a Ctrl+C (or other
    ``KeyboardInterrupt``) stop, where ``error`` stays ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    final_text: str = ""
    completed: bool = False
    interrupted: bool = False
    iterations: int = 0
    error: str | None = None
    tool_executions: list[ToolExecutionResult] = Field(default_factory=list)
    tool_results: list[ToolResultMessage] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)


class AgentRunner:
    """Runs one provider to completion (or its iteration budget).

    Everything is injected: ``provider`` and ``registry`` come from the
    caller, so tests can pass fake providers and fake tools. The public
    surface is intentionally small; the loop lives here and nowhere else.
    """

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise ValueError("max_iterations must be a positive integer")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        self.provider = provider
        self.registry = registry
        self.max_iterations = max_iterations
        self._executor = ToolExecutor(registry)

    def run(self, messages: list[ChatMessage]) -> AgentResult:
        """Run the bounded loop over ``messages`` and return an
        ``AgentResult``.

        The caller's list is copied, never mutated. Tool results are
        accumulated as read-only ``ToolResultMessage`` objects and carried
        on each subsequent provider request.
        """
        if not messages:
            raise ValueError("run() requires a non-empty list of messages")

        conversation = list(messages)
        tool_results: list[ToolResultMessage] = []
        tool_executions: list[ToolExecutionResult] = []
        definitions = [
            ToolDefinition.from_tool(tool) for tool in self.registry.list_tools()
        ]

        for iteration in range(1, self.max_iterations + 1):
            try:
                response = self.provider.send_normalized(
                    conversation,
                    tools=definitions,
                    tool_results=tool_results,
                )
            except KeyboardInterrupt:
                return AgentResult(
                    completed=False,
                    interrupted=True,
                    iterations=iteration,
                    tool_executions=tool_executions,
                    tool_results=tool_results,
                    messages=conversation,
                )
            except ProviderError as exc:
                return AgentResult(
                    completed=False,
                    error=str(exc),
                    iterations=iteration,
                    tool_executions=tool_executions,
                    tool_results=tool_results,
                    messages=conversation,
                )

            if not isinstance(response, ProviderResponse):
                return AgentResult(
                    completed=False,
                    error=(
                        f"Provider returned {type(response).__name__}, "
                        "expected a ProviderResponse"
                    ),
                    iterations=iteration,
                    tool_executions=tool_executions,
                    tool_results=tool_results,
                    messages=conversation,
                )

            if not response.has_tool_calls:
                return AgentResult(
                    final_text=response.text,
                    completed=True,
                    iterations=iteration,
                    tool_executions=tool_executions,
                    tool_results=tool_results,
                    messages=conversation,
                )

            try:
                batch = self._executor.execute(response)
            except KeyboardInterrupt:
                return AgentResult(
                    completed=False,
                    interrupted=True,
                    iterations=iteration,
                    tool_executions=tool_executions,
                    tool_results=tool_results,
                    messages=conversation,
                )
            tool_executions.extend(batch.executions)
            tool_results.extend(result_messages_from_batch(batch))
            conversation.append(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": list(response.tool_calls),
                }
            )

        # Budget exhausted: the provider kept calling tools. Everything
        # observed is reported, and no further tool runs or provider calls
        # are made.
        return AgentResult(
            completed=False,
            iterations=self.max_iterations,
            tool_executions=tool_executions,
            tool_results=tool_results,
            messages=conversation,
        )