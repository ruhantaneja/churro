"""Convert executed tool results into provider-neutral result messages.

This is the result-side of the provider boundary:

    ToolExecutionResult  ->  ToolResultMessage   (one execution)
    ToolExecutionBatch   ->  list[ToolResultMessage]  (ordered)

The conversion is pure: it never mutates the batch, the execution, or the
underlying ``ToolResult``. Only churro's normalized types are imported --
no OpenAI/Anthropic/Gemini/Ollama SDK -- so a provider adapter can consume
``ToolResultMessage`` and translate it into its native message format at
some later step.
"""

from churro.providers.responses import ToolResultMessage
from churro.tools.executor import ToolExecutionBatch, ToolExecutionResult
from churro.tools.tool import ToolResult

_SUCCESS_FALLBACK = "(tool returned no output)"
_FAILURE_FALLBACK = "(tool failed with no error details)"


def result_content(result: ToolResult) -> str:
    """Return the useful content for a ToolResult.

    Successful results expose their output; failed results expose their
    error. An empty string is replaced with a fixed placeholder so the
    content is always deterministic and never silently blank.
    """
    if result.success:
        return result.output or _SUCCESS_FALLBACK
    return result.error or _FAILURE_FALLBACK


def result_message_from_execution(execution: ToolExecutionResult) -> ToolResultMessage:
    """Convert one ``ToolExecutionResult`` into a ``ToolResultMessage``."""
    return ToolResultMessage(
        call_id=execution.call_id,
        tool_name=execution.tool_name,
        content=result_content(execution.result),
        success=execution.result.success,
    )


def result_messages_from_batch(
    batch: ToolExecutionBatch,
) -> list[ToolResultMessage]:
    """Convert a ``ToolExecutionBatch`` into an ordered list of messages.

    An empty batch yields an empty list. Ordering follows the batch's
    executions exactly.
    """
    return [result_message_from_execution(execution) for execution in batch.executions]