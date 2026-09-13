"""Offline tests for the bounded agent runner (Step 12).

Uses only fake providers and fake tools. No API keys, no network, no
provider SDK.
"""

import churro.agent.runner as runner_module
from churro.agent import AgentResult, AgentRunner
from churro.providers.provider import Provider, ProviderError
from churro.providers.responses import ProviderResponse, ToolCall, ToolResultMessage
from churro.providers.tool_definition import ToolDefinition
from churro.tools import ToolArgs, ToolExecutor, ToolRegistry, ToolResult
from churro.tools.tool import Tool


# ---------------------------------------------------------------- fakes
class FakeProviderError(ProviderError):
    pass


class ScriptedProvider(Provider):
    name = "script"
    model = "script-1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, messages):
        return "script"

    def send_normalized(self, messages, tools=None, tool_results=None):
        if not self.responses:
            raise AssertionError("provider called after its script ran out")
        self.requests.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": list(tools) if tools is not None else None,
                "tool_results": list(tool_results) if tool_results is not None else None,
            }
        )
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class AdderArgs(ToolArgs):
    a: int
    b: int


class AdderTool(Tool):
    name = "adder"
    description = "adds two integers"
    args_model = AdderArgs

    def _run(self, args: AdderArgs) -> ToolResult:
        return ToolResult(success=True, output=str(args.a + args.b))


class NoArgTool(Tool):
    name = "no_arg"
    description = "takes no arguments"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        return ToolResult(success=True, output="done")


class FailTool(Tool):
    name = "fail"
    description = "always fails"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        return ToolResult(success=False, error="boom")


def tool(response_call_id, name, arguments=None):
    return ToolCall(id=response_call_id, name=name, arguments=arguments or {})


def final(text="Hello!"):
    return ProviderResponse(text=text, finish_reason="stop")


def make_runner(responses, tools=None, max_iterations=10):
    registry = ToolRegistry()
    for tool in (tools or [AdderTool(), NoArgTool(), FailTool()]):
        registry.register(tool)
    return AgentRunner(provider=ScriptedProvider(responses), registry=registry,
                       max_iterations=max_iterations), registry


MESSAGES = [{"role": "user", "content": "hi"}]


# -------------------------------------------------------------- tests
def test_immediate_final_response():
    runner, _ = make_runner([final("immediate")])
    result = runner.run(MESSAGES)
    assert isinstance(result, AgentResult)
    assert result.completed is True
    assert result.final_text == "immediate"
    assert result.iterations == 1
    assert result.tool_executions == []
    assert result.tool_results == []
    assert result.error is None


def test_one_tool_call_then_final_response():
    runner, _ = make_runner(
        [ProviderResponse(tool_calls=[tool("c1", "adder", {"a": 2, "b": 3})],
                          finish_reason="tool_calls"), final("5!")]
    )
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert result.final_text == "5!"
    assert result.iterations == 2
    assert len(result.tool_executions) == 1
    assert result.tool_executions[0].call_id == "c1"
    assert result.tool_executions[0].result.output == "5"


def test_multiple_tool_calls_in_one_response():
    runner, _ = make_runner(
        [
            ProviderResponse(
                tool_calls=[
                    tool("c1", "adder", {"a": 1, "b": 1}),
                    tool("c2", "no_arg"),
                ],
                finish_reason="tool_calls",
            ),
            final("done"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert len(result.tool_executions) == 2
    assert [e.result.output for e in result.tool_executions] == ["2", "done"]


def test_multiple_sequential_tool_iterations():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "adder", {"a": 1, "b": 1})],
                             finish_reason="tool_calls"),
            ProviderResponse(tool_calls=[tool("c2", "adder", {"a": 5, "b": 5})],
                             finish_reason="tool_calls"),
            final("10"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert result.iterations == 3
    assert [e.result.output for e in result.tool_executions] == ["2", "10"]


def test_tool_failure_fed_back_to_provider():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "fail")], finish_reason="tool_calls"),
            final("handled"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert result.tool_executions[0].result.success is False
    assert len(result.tool_results) == 1
    assert result.tool_results[0].content == "boom"
    # the second provider request carried the failure message back
    second_request = runner.provider.requests[1]["tool_results"]
    assert second_request[0].content == "boom"
    assert second_request[0].success is False


def test_unknown_tool_fed_back_as_failure():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "not_a_real_tool")],
                             finish_reason="tool_calls"),
            final("told you"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert result.tool_executions[0].result.success is False
    assert not result.tool_results[0].success
    assert "not_a_real_tool" in result.tool_results[0].content
    assert "not_a_real_tool" in runner.provider.requests[1]["tool_results"][0].content


def test_max_iteration_limit_stops_without_extra_calls():
    responses = [
        ProviderResponse(tool_calls=[tool(f"c{i}", "adder", {"a": i, "b": i})],
                         finish_reason="tool_calls")
        for i in range(10)
    ]
    runner, _ = make_runner(responses, max_iterations=3)
    result = runner.run(MESSAGES)
    assert result.completed is False
    assert result.iterations == 3
    assert result.error is None
    assert len(result.tool_executions) == 3
    assert len(runner.provider.requests) == 3  # no 4th provider call


def test_max_iteration_exactly_reached():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "adder", {"a": 1, "b": 1})],
                             finish_reason="tool_calls"),
            ProviderResponse(tool_calls=[tool("c2", "adder", {"a": 2, "b": 2})],
                             finish_reason="tool_calls"),
            final("never seen"),
        ],
        max_iterations=2,
    )
    result = runner.run(MESSAGES)
    assert result.completed is False
    assert result.iterations == 2
    assert len(result.tool_executions) == 2
    assert result.final_text == ""


def test_invalid_max_iterations_raise():
    for bad in (0, -1, 1.5, "10", None, True):
        try:
            AgentRunner(provider=ScriptedProvider([]), registry=ToolRegistry(),
                        max_iterations=bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass


def test_provider_failure_is_surfaced():
    runner, _ = make_runner(
        [FakeProviderError("api down")],
    )
    result = runner.run(MESSAGES)
    assert result.completed is False
    assert "api down" in (result.error or "")
    assert result.iterations == 1
    assert result.tool_executions == []


def test_provider_failure_after_some_tool_runs():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "adder", {"a": 1, "b": 1})],
                             finish_reason="tool_calls"),
            FakeProviderError("boom mid-loop"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.completed is False
    assert "boom mid-loop" in (result.error or "")
    assert result.iterations == 2
    assert len(result.tool_executions) == 1


def test_empty_final_response():
    runner, _ = make_runner([final("")])
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert result.final_text == ""


def test_empty_tool_result_fed_back():
    class EmptyOkTool(Tool):
        name = "empty_ok"
        description = "succeeds with no output"
        args_model = ToolArgs

        def _run(self, args):
            return ToolResult(success=True, output="")

    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "empty_ok")],
                             finish_reason="tool_calls"),
            final("ok"),
        ],
        tools=[EmptyOkTool(), AdderTool(), NoArgTool()],
    )
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert result.tool_results[0].success is True
    assert result.tool_results[0].content == "(tool returned no output)"


def test_result_ordering_preserved():
    runner, _ = make_runner(
        [
            ProviderResponse(
                tool_calls=[
                    tool("c1", "adder", {"a": 1, "b": 1}),
                    tool("c2", "no_arg"),
                    tool("c3", "adder", {"a": 3, "b": 3}),
                ],
                finish_reason="tool_calls",
            ),
            final("done"),
        ]
    )
    result = runner.run(MESSAGES)
    assert [e.call_id for e in result.tool_executions] == ["c1", "c2", "c3"]
    assert [e.result.output for e in result.tool_executions] == ["2", "done", "6"]


def test_tool_call_id_preserved():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("call-alpha", "adder", {"a": 1, "b": 2})],
                             finish_reason="tool_calls"),
            final("3"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.tool_results[0] == ToolResultMessage(
        call_id="call-alpha", tool_name="adder", content="3", success=True
    )
    assert result.tool_results[0].call_id == "call-alpha"


def test_tool_name_preserved():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "no_arg")], finish_reason="tool_calls"),
            final("done"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.tool_results[0].tool_name == "no_arg"


def test_multiple_different_tools():
    runner, _ = make_runner(
        [
            ProviderResponse(
                tool_calls=[
                    tool("c1", "adder", {"a": 2, "b": 3}),
                    tool("c2", "no_arg"),
                    tool("c3", "fail"),
                ],
                finish_reason="tool_calls",
            ),
            final("done"),
        ]
    )
    result = runner.run(MESSAGES)
    assert [e.tool_name for e in result.tool_executions] == ["adder", "no_arg", "fail"]
    assert [e.result.success for e in result.tool_executions] == [True, True, False]
    assert result.completed is True


def test_repeated_tool_calls_across_iterations():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "adder", {"a": 1, "b": 1})],
                             finish_reason="tool_calls"),
            ProviderResponse(tool_calls=[tool("c2", "adder", {"a": 2, "b": 2})],
                             finish_reason="tool_calls"),
            final("bye"),
        ]
    )
    result = runner.run(MESSAGES)
    assert len(result.tool_executions) == 2
    assert [e.result.output for e in result.tool_executions] == ["2", "4"]
    # results accumulated: request 2 carried the first round, and both
    # rounds are reported at the end
    assert len(runner.provider.requests[1]["tool_results"]) == 1
    assert len(result.tool_results) == 2


def test_loop_is_bounded_not_recursive():
    responses = [
        ProviderResponse(tool_calls=[tool(f"c{i}", "adder", {"a": i, "b": i})],
                         finish_reason="tool_calls")
        for i in range(60)
    ]
    runner, _ = make_runner(responses, max_iterations=60)
    result = runner.run(MESSAGES)
    assert result.completed is False
    assert result.iterations == 60
    assert len(result.tool_executions) == 60


def test_no_session_state_mutation():
    assert "session" not in vars(runner_module)
    assert "handoff" not in vars(runner_module)
    assert "session_manager" not in vars(runner_module)
    runner, _ = make_runner([final("hi")])
    result = runner.run(MESSAGES)
    assert result.completed is True


def test_no_tool_execution_batch_or_response_mutation():
    response = ProviderResponse(
        tool_calls=[tool("c1", "adder", {"a": 1, "b": 1})],
        finish_reason="tool_calls",
    )
    snapshot = response.model_copy(deep=True)
    runner, _ = make_runner([response, final("done")])
    runner.run(MESSAGES)
    assert response == snapshot


def test_initial_messages_not_mutated_and_reported():
    runner, _ = make_runner([final("ok")])
    original = [{"role": "system", "content": "be helpful"},
                {"role": "user", "content": "work"}]
    result = runner.run(original)
    assert original == [{"role": "system", "content": "be helpful"},
                        {"role": "user", "content": "work"}]
    assert result.messages == original


def test_provider_receives_tool_definitions():
    runner, registry = make_runner([final("ok")], tools=[AdderTool(), NoArgTool()])
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert len(runner.provider.requests) == 1
    definitions = runner.provider.requests[0]["tools"]
    expected = [ToolDefinition.from_tool(t) for t in registry.list_tools()]
    assert definitions == expected
    assert [d.name for d in definitions] == ["adder", "no_arg"]


def test_provider_receives_tool_results_on_subsequent_iteration():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "adder", {"a": 4, "b": 5})],
                             finish_reason="tool_calls"),
            final("9"),
        ]
    )
    runner.run(MESSAGES)
    assert runner.provider.requests[0]["tool_results"] is None or \
        runner.provider.requests[0]["tool_results"] == []
    second = runner.provider.requests[1]["tool_results"]
    assert len(second) == 1
    assert second[0].call_id == "c1"
    assert second[0].content == "9"
    assert second[0].success is True


def test_zero_tool_executions_for_final_only_response():
    runner, _ = make_runner([final("nothing to do")])
    result = runner.run(MESSAGES)
    assert result.tool_executions == []
    assert result.tool_results == []
    assert result.completed is True


def test_iteration_count_matches_provider_calls():
    runner, _ = make_runner(
        [
            ProviderResponse(tool_calls=[tool("c1", "adder", {"a": 1, "b": 1})],
                             finish_reason="tool_calls"),
            ProviderResponse(tool_calls=[tool("c2", "adder", {"a": 2, "b": 2})],
                             finish_reason="tool_calls"),
            final("x"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.iterations == 3
    assert len(runner.provider.requests) == result.iterations


def test_accumulated_tool_execution_reporting():
    runner, _ = make_runner(
        [
            ProviderResponse(
                tool_calls=[
                    tool("c1", "adder", {"a": 1, "b": 1}),
                    tool("c2", "no_arg"),
                ],
                finish_reason="tool_calls",
            ),
            ProviderResponse(tool_calls=[tool("c3", "fail")], finish_reason="tool_calls"),
            final("final answer"),
        ]
    )
    result = runner.run(MESSAGES)
    assert len(result.tool_executions) == 3
    assert [e.call_id for e in result.tool_executions] == ["c1", "c2", "c3"]
    assert len(result.tool_results) == 3
    assert result.iterations == 3
    assert result.final_text == "final answer"
    assert result.completed is True


def test_agent_core_imports_no_provider_sdks():
    assert "openai" not in vars(runner_module)
    assert not hasattr(runner_module, "OpenAI")
    assert "anthropic" not in vars(runner_module)
    assert "gemini" not in vars(runner_module)
    assert "ollama" not in vars(runner_module)


def test_invalid_provider_response_type_handled():
    class WeirdProvider(ScriptedProvider):
        def send_normalized(self, messages, tools=None, tool_results=None):
            return "junk"

    registry = ToolRegistry()
    registry.register(AdderTool())
    runner = AgentRunner(provider=WeirdProvider([]), registry=registry)
    result = runner.run(MESSAGES)
    assert result.completed is False
    assert "expected a ProviderResponse" in (result.error or "")
    assert isinstance(result, AgentResult)


def test_empty_messages_raises():
    runner, _ = make_runner([final("x")])
    try:
        runner.run([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_unknown_tool_does_not_kill_agent():
    runner, _ = make_runner(
        [
            ProviderResponse(
                tool_calls=[
                    tool("c1", "not_real"),
                    tool("c2", "adder", {"a": 1, "b": 1}),
                    tool("c3", "not_real_again"),
                ],
                finish_reason="tool_calls",
            ),
            final("survived"),
        ]
    )
    result = runner.run(MESSAGES)
    assert result.completed is True
    assert len(result.tool_executions) == 3
    assert [e.result.success for e in result.tool_executions] == [False, True, False]
    assert len(result.tool_results) == 3


TEST_FUNCTIONS = [
    test_immediate_final_response,
    test_one_tool_call_then_final_response,
    test_multiple_tool_calls_in_one_response,
    test_multiple_sequential_tool_iterations,
    test_tool_failure_fed_back_to_provider,
    test_unknown_tool_fed_back_as_failure,
    test_max_iteration_limit_stops_without_extra_calls,
    test_max_iteration_exactly_reached,
    test_invalid_max_iterations_raise,
    test_provider_failure_is_surfaced,
    test_provider_failure_after_some_tool_runs,
    test_empty_final_response,
    test_empty_tool_result_fed_back,
    test_result_ordering_preserved,
    test_tool_call_id_preserved,
    test_tool_name_preserved,
    test_multiple_different_tools,
    test_repeated_tool_calls_across_iterations,
    test_loop_is_bounded_not_recursive,
    test_no_session_state_mutation,
    test_no_tool_execution_batch_or_response_mutation,
    test_initial_messages_not_mutated_and_reported,
    test_provider_receives_tool_definitions,
    test_provider_receives_tool_results_on_subsequent_iteration,
    test_zero_tool_executions_for_final_only_response,
    test_iteration_count_matches_provider_calls,
    test_accumulated_tool_execution_reporting,
    test_agent_core_imports_no_provider_sdks,
    test_invalid_provider_response_type_handled,
    test_empty_messages_raises,
    test_unknown_tool_does_not_kill_agent,
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