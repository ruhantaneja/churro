"""Offline tests for the normalized tool-call executor (Step 10).

The executor connects a ProviderResponse (normalized ToolCalls) to the
ToolRegistry. Everything here uses fake Tool subclasses and fake
ProviderResponse values -- no API keys, no network, no LLM.
"""

import tempfile
from pathlib import Path

import churro.tools.executor as executor_module
from churro.providers.responses import ProviderResponse, ToolCall
from churro.providers.tool_definition import ToolDefinition
from churro.tools import (
    ToolExecutor,
    ToolRegistry,
    ToolArgs,
    ToolResult,
)
from churro.tools.tool import Tool
from churro.tools.executor import ToolExecutionBatch, ToolExecutionResult


# ---------------------------------------------------------------- fakes
class AdderArgs(ToolArgs):
    a: int
    b: int


class AdderTool(Tool):
    name = "adder"
    description = "adds two integers"
    args_model = AdderArgs

    def _run(self, args: AdderArgs) -> ToolResult:
        return ToolResult(success=True, output=str(args.a + args.b))


class FailAlwaysTool(Tool):
    name = "fail_always"
    description = "always returns a failed result"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        return ToolResult(success=False, error="boom")


class CrashTool(Tool):
    name = "crash"
    description = "raises inside _run"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        raise RuntimeError("kaboom")


class NoArgTool(Tool):
    name = "no_arg"
    description = "takes no arguments"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        return ToolResult(success=True, output="done")


class SpyingTool(Tool):
    """Records the exact arguments object handed to ``execute`` so a test
    can prove the coordinator passed the arguments through unchanged."""

    name = "spy"
    description = "records the arguments object it received"
    args_model = ToolArgs
    received: list = []

    def execute(self, arguments):
        SpyingTool.received.append(arguments)
        return ToolResult(success=True, output="spied")

    def _run(self, args) -> ToolResult:
        return ToolResult(success=True, output="spied")


class RequestRaisingRegistry(ToolRegistry):
    """A registry whose execute raises, to prove exceptions are contained."""

    def execute(self, name, arguments):
        raise RuntimeError(f"registry exploded for {name}")


ROOT_DIR = Path(tempfile.mkdtemp())


def fresh_registry():
    return ToolRegistry()


def or_die(response, registry=None):
    registry = registry or fresh_registry()
    return ToolExecutor(registry).execute(response), registry


def tool_call(call_id, name, arguments=None):
    return ToolCall(id=call_id, name=name, arguments=arguments or {})


# -------------------------------------------------------------- tests
def test_no_tool_calls_returns_empty_batch():
    response = ProviderResponse(text="no tools", finish_reason="stop")
    batch, _ = or_die(response)
    assert isinstance(batch, ToolExecutionBatch)
    assert batch.executions == []
    assert not batch.has_executions
    assert not batch.all_succeeded


def test_one_successful_tool_call():
    registry = fresh_registry()
    registry.register(AdderTool())
    response = ProviderResponse(
        tool_calls=[tool_call("call_1", "adder", {"a": 2, "b": 3})],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    assert len(batch.executions) == 1
    assert batch.executions[0].result.success
    assert batch.executions[0].result.output == "5"
    assert batch.all_succeeded


def test_multiple_successful_tool_calls():
    registry = fresh_registry()
    registry.register(AdderTool())
    response = ProviderResponse(
        tool_calls=[
            tool_call("c1", "adder", {"a": 1, "b": 2}),
            tool_call("c2", "adder", {"a": 10, "b": 20}),
            tool_call("c3", "adder", {"a": 0, "b": 0}),
        ],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    assert [e.result.output for e in batch.executions] == ["3", "30", "0"]


def test_tool_call_ordering_preserved():
    registry = fresh_registry()
    registry.register(AdderTool())
    response = ProviderResponse(
        tool_calls=[
            tool_call("first", "adder", {"a": 1, "b": 1}),
            tool_call("second", "adder", {"a": 2, "b": 2}),
            tool_call("third", "adder", {"a": 3, "b": 3}),
        ],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    assert [e.result.output for e in batch.executions] == ["2", "4", "6"]


def test_tool_call_ids_preserved():
    registry = fresh_registry()
    registry.register(AdderTool())
    response = ProviderResponse(
        tool_calls=[
            tool_call("call-alpha", "adder", {"a": 1, "b": 1}),
            tool_call("call-beta", "adder", {"a": 2, "b": 2}),
        ],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    assert [e.call_id for e in batch.executions] == ["call-alpha", "call-beta"]


def test_tool_names_preserved():
    registry = fresh_registry()
    registry.register(AdderTool())
    registry.register(NoArgTool())
    response = ProviderResponse(
        tool_calls=[
            tool_call("c1", "adder", {"a": 1, "b": 1}),
            tool_call("c2", "no_arg"),
        ],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    assert [e.tool_name for e in batch.executions] == ["adder", "no_arg"]


def test_structured_arguments_passed_unchanged():
    SpyingTool.received.clear()
    registry = fresh_registry()
    registry.register(SpyingTool())
    args = {"path": "test.py"}
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "spy", args)],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    assert batch.executions[0].result.output == "spied"
    # the exact arguments object stored on the ToolCall reached
    # tool.execute unchanged -- the coordinator never copies or rewraps it
    assert SpyingTool.received[-1] is response.tool_calls[0].arguments
    assert SpyingTool.received[-1] == {"path": "test.py"}


def test_unknown_tool_handled_safely():
    registry = fresh_registry()
    registry.register(AdderTool())
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "does_not_exist", {"a": 1, "b": 1})],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    execution = batch.executions[0]
    assert not execution.result.success
    assert "does_not_exist" in execution.result.error
    assert not batch.all_succeeded


def test_invalid_arguments_handled_safely():
    registry = fresh_registry()
    registry.register(AdderTool())
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "adder", {"a": "not-a-number", "b": 1})],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    execution = batch.executions[0]
    assert not execution.result.success
    assert "Invalid arguments" in execution.result.error


def test_tool_execution_failure_handled_safely():
    registry = fresh_registry()
    registry.register(FailAlwaysTool())
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "fail_always")],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    execution = batch.executions[0]
    assert not execution.result.success
    assert execution.result.error == "boom"


def test_crashing_tool_handled_safely():
    registry = fresh_registry()
    registry.register(CrashTool())
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "crash")],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    execution = batch.executions[0]
    assert not execution.result.success
    assert "RuntimeError" in execution.result.error
    assert "kaboom" in execution.result.error


def test_failed_tool_does_not_block_later_calls():
    registry = fresh_registry()
    registry.register(AdderTool())
    registry.register(FailAlwaysTool())
    response = ProviderResponse(
        tool_calls=[
            tool_call("ok1", "adder", {"a": 1, "b": 1}),
            tool_call("bad", "fail_always"),
            tool_call("ok2", "adder", {"a": 5, "b": 5}),
        ],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    assert [e.result.success for e in batch.executions] == [True, False, True]
    assert batch.executions[2].result.output == "10"
    assert len(batch.executions) == 3


def test_toolresult_success_information_preserved():
    registry = fresh_registry()
    registry.register(AdderTool())
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "adder", {"a": 2, "b": 2})],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    execution = batch.executions[0]
    assert isinstance(execution.result, ToolResult)
    assert execution.result.success is True
    assert execution.result.output == "4"
    assert execution.result.error == ""


def test_toolresult_error_information_preserved():
    registry = fresh_registry()
    registry.register(FailAlwaysTool())
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "fail_always")],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    execution = batch.executions[0]
    assert isinstance(execution.result, ToolResult)
    assert execution.result.success is False
    assert execution.result.error == "boom"
    assert execution.result.output == ""
    assert execution.success is False


def test_original_provider_response_not_mutated():
    registry = fresh_registry()
    registry.register(AdderTool())
    response = ProviderResponse(
        tool_calls=[
            tool_call("c1", "adder", {"a": 1, "b": 2}),
            tool_call("c2", "adder", {"a": 3, "b": 4}),
        ],
        finish_reason="tool_calls",
    )
    snapshot = response.model_copy(deep=True)
    ToolExecutor(registry).execute(response)
    assert response == snapshot
    assert [tc.arguments for tc in response.tool_calls] == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_tool_call_arguments_not_mutated():
    SpyingTool.received.clear()
    registry = fresh_registry()
    registry.register(SpyingTool())
    args = {"path": "x.py"}
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "spy", args)],
        finish_reason="tool_calls",
    )
    ToolExecutor(registry).execute(response)
    assert args == {"path": "x.py"}
    assert response.tool_calls[0].arguments == {"path": "x.py"}


def test_coordinator_touches_no_session_state_or_history():
    # the coordinator never imports session/handoff modules and only reads
    # the response; a locally constructed history stays untouched
    import churro.tools.executor as mod

    assert "session" not in vars(mod)
    assert "history" not in vars(mod)

    registry = fresh_registry()
    registry.register(AdderTool())
    history = [{"role": "user", "content": "hi"}]
    snapshot = list(history)
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "adder", {"a": 1, "b": 2})],
        finish_reason="tool_calls",
    )
    ToolExecutor(registry).execute(response)
    assert history == snapshot


def test_registry_dispatch_is_actually_used():
    SpyingTool.received.clear()
    registry = fresh_registry()
    spy = SpyingTool()
    registry.register(spy)
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "spy", {"k": "v"})],
        finish_reason="tool_calls",
    )
    ToolExecutor(registry).execute(response)
    assert len(SpyingTool.received) == 1
    assert SpyingTool.received[0] == {"k": "v"}


def test_coordinator_imports_no_provider_sdks():
    assert "openai" not in vars(executor_module)
    assert not hasattr(executor_module, "OpenAI")
    assert "anthropic" not in vars(executor_module)
    assert "ollama" not in vars(executor_module)
    assert "gemini" not in vars(executor_module)
    # the module tree it pulls in stays provider-neutral
    for candidate in (
        "churro.providers.responses",
        "churro.tools.tool",
        "churro.tools.registry",
    ):
        module = __import__(candidate, fromlist=["*"])
        assert "openai" not in str(module).lower()


def test_multiple_different_tools_work_together():
    registry = fresh_registry()
    registry.register(AdderTool())
    registry.register(NoArgTool())
    registry.register(FailAlwaysTool())
    registry.register(CrashTool())
    response = ProviderResponse(
        tool_calls=[
            tool_call("a", "adder", {"a": 2, "b": 3}),
            tool_call("b", "no_arg"),
            tool_call("c", "fail_always"),
            tool_call("d", "crash"),
            tool_call("e", "adder", {"a": 1, "b": 1}),
        ],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    assert [e.tool_name for e in batch.executions] == [
        "adder",
        "no_arg",
        "fail_always",
        "crash",
        "adder",
    ]
    assert [e.result.success for e in batch.executions] == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert [e.call_id for e in batch.executions] == ["a", "b", "c", "d", "e"]


def test_executor_contains_registry_exception():
    registry = RequestRaisingRegistry()
    response = ProviderResponse(
        tool_calls=[tool_call("c1", "adder", {"a": 1, "b": 1})],
        finish_reason="tool_calls",
    )
    batch = ToolExecutor(registry).execute(response)
    execution = batch.executions[0]
    assert not execution.result.success
    assert "registry exploded" in execution.result.error


def test_tooldefinition_still_describes_registered_tools():
    from churro.tools import WriteFileTool

    tool = WriteFileTool(ROOT_DIR)
    definition = ToolDefinition.from_tool(tool)
    assert definition.name == "write_file"
    assert definition.input_schema == tool.input_schema
    assert "properties" in definition.input_schema


def test_empty_batch_supports_values():
    batch = ToolExecutionBatch()
    assert batch.executions == []
    assert not batch.has_executions
    assert not batch.all_succeeded

    result = ToolExecutionResult(
        call_id="c1", tool_name="adder", result=ToolResult(success=True, output="2")
    )
    batch2 = ToolExecutionBatch(executions=[result])
    assert batch2.has_executions
    assert batch2.all_succeeded


TEST_FUNCTIONS = [
    test_no_tool_calls_returns_empty_batch,
    test_one_successful_tool_call,
    test_multiple_successful_tool_calls,
    test_tool_call_ordering_preserved,
    test_tool_call_ids_preserved,
    test_tool_names_preserved,
    test_structured_arguments_passed_unchanged,
    test_unknown_tool_handled_safely,
    test_invalid_arguments_handled_safely,
    test_tool_execution_failure_handled_safely,
    test_crashing_tool_handled_safely,
    test_failed_tool_does_not_block_later_calls,
    test_toolresult_success_information_preserved,
    test_toolresult_error_information_preserved,
    test_original_provider_response_not_mutated,
    test_tool_call_arguments_not_mutated,
    test_coordinator_touches_no_session_state_or_history,
    test_registry_dispatch_is_actually_used,
    test_coordinator_imports_no_provider_sdks,
    test_multiple_different_tools_work_together,
    test_executor_contains_registry_exception,
    test_tooldefinition_still_describes_registered_tools,
    test_empty_batch_supports_values,
]


def main():
    failed = 0
    for func in TEST_FUNCTIONS:
        try:
            func()
            print(f"PASS {func.__name__}")
        except Exception:
            import traceback

            failed += 1
            print(f"FAIL {func.__name__}")
            traceback.print_exc()
    total = len(TEST_FUNCTIONS)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())