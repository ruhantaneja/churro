from pydantic import Field

from churro.tools import (
    DuplicateToolError,
    Tool,
    ToolArgs,
    ToolDefinitionError,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
)


class GreetArgs(ToolArgs):
    name: str
    greeting: str = "Hello"
    times: int = Field(default=1, ge=1)


class GreetTool(Tool[GreetArgs]):
    name = "greet"
    description = "Greet a person (test tool)."
    args_model = GreetArgs

    def _run(self, args: GreetArgs) -> ToolResult:
        lines = [f"{args.greeting}, {args.name}! ({i})" for i in range(1, args.times + 1)]
        return ToolResult(success=True, output="\n".join(lines))


class BoomTool(Tool[GreetArgs]):
    name = "boom"
    description = "Always raises during execution (test tool)."
    args_model = GreetArgs

    def _run(self, args: GreetArgs) -> ToolResult:
        raise RuntimeError("kaboom")


class NullReturningTool(Tool[GreetArgs]):
    name = "null"
    description = "Returns something that is not a ToolResult (test tool)."
    args_model = GreetArgs

    def _run(self, args: GreetArgs) -> ToolResult:
        return "this is not a ToolResult"


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GreetTool())
    registry.register(BoomTool())
    registry.register(NullReturningTool())
    return registry


def test_register_a_tool():
    registry = ToolRegistry()
    tool = GreetTool()
    registry.register(tool)
    assert registry.get("greet") is tool


def test_get_returns_registered_tool():
    registry = make_registry()
    assert isinstance(registry.get("greet"), GreetTool)
    assert isinstance(registry.get("boom"), BoomTool)


def test_get_unknown_tool_raises():
    registry = ToolRegistry()
    try:
        registry.get("nope")
    except UnknownToolError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected UnknownToolError")


def test_list_tools_returns_sorted_tools():
    registry = make_registry()
    names = [tool.name for tool in registry.list_tools()]
    assert names == sorted(names)
    assert names == ["boom", "greet", "null"]


def test_fresh_registry_is_empty():
    assert ToolRegistry().list_tools() == []


def test_duplicate_registration_rejected():
    registry = make_registry()
    try:
        registry.register(GreetTool())
    except DuplicateToolError as exc:
        assert "greet" in str(exc)
    else:
        raise AssertionError("expected DuplicateToolError")
    assert registry.get("greet").name == "greet"


def test_registering_non_tool_rejected():
    registry = ToolRegistry()
    try:
        registry.register("not a tool")
    except ToolDefinitionError:
        pass
    else:
        raise AssertionError("expected ToolDefinitionError")


def test_valid_execution_uses_defaults():
    result = make_registry().execute("greet", {"name": "Ada"})
    assert result.success is True
    assert result.output == "Hello, Ada! (1)"
    assert result.error == ""


def test_valid_execution_with_all_arguments():
    result = make_registry().execute(
        "greet", {"name": "Ada", "greeting": "Hi", "times": 2}
    )
    assert result.success is True
    assert result.output == "Hi, Ada! (1)\nHi, Ada! (2)"


def test_missing_required_argument_is_controlled():
    result = make_registry().execute("greet", {"greeting": "Yo"})
    assert result.success is False
    assert "name" in result.error


def test_wrong_argument_type_is_controlled():
    result = make_registry().execute("greet", {"name": 123})
    assert result.success is False
    assert "name" in result.error


def test_unknown_arguments_rejected():
    result = make_registry().execute("greet", {"name": "Ada", "bogus": 1})
    assert result.success is False
    assert "bogus" in result.error


def test_non_dict_arguments_controlled():
    result = make_registry().execute("greet", ["name", "Ada"])
    assert result.success is False
    assert result.error


def test_execution_failure_is_controlled():
    registry = make_registry()
    try:
        result = registry.execute("boom", {"name": "Ada"})
    except Exception:
        raise AssertionError("tool failure must not escape as an exception")
    assert result.success is False
    assert "kaboom" in result.error


def test_non_toolresult_return_is_controlled():
    result = make_registry().execute("null", {"name": "Ada"})
    assert result.success is False
    assert "ToolResult" in result.error


def test_arguments_dict_not_mutated():
    registry = ToolRegistry()
    registry.register(GreetTool())
    arguments = {"name": "Ada", "times": 3}
    before = dict(arguments)
    registry.execute("greet", arguments)
    assert arguments == before


def test_tool_subclass_requires_metadata():
    for tool_name, tool_description in (("", "x"), ("x", "")):
        try:
            type(
                "BrokenTool",
                (Tool,),
                {
                    "name": tool_name,
                    "description": tool_description,
                    "args_model": GreetArgs,
                    "_run": lambda self, args: ToolResult(success=True),
                },
            )
        except ToolDefinitionError:
            pass
        else:
            raise AssertionError("expected ToolDefinitionError")


def test_tool_input_schema_is_structured():
    schema = GreetTool().input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"name", "greeting", "times"}
    assert "name" in schema["required"]
    assert "greeting" not in schema["required"]
    assert schema.get("additionalProperties") is False


TEST_FUNCTIONS = [
    test_register_a_tool,
    test_get_returns_registered_tool,
    test_get_unknown_tool_raises,
    test_list_tools_returns_sorted_tools,
    test_fresh_registry_is_empty,
    test_duplicate_registration_rejected,
    test_registering_non_tool_rejected,
    test_valid_execution_uses_defaults,
    test_valid_execution_with_all_arguments,
    test_missing_required_argument_is_controlled,
    test_wrong_argument_type_is_controlled,
    test_unknown_arguments_rejected,
    test_non_dict_arguments_controlled,
    test_execution_failure_is_controlled,
    test_non_toolresult_return_is_controlled,
    test_arguments_dict_not_mutated,
    test_tool_subclass_requires_metadata,
    test_tool_input_schema_is_structured,
]


def main() -> None:
    failures = 0
    for fn in TEST_FUNCTIONS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    if failures:
        raise SystemExit(f"\n{failures} test(s) failed")
    print(f"\nAll {len(TEST_FUNCTIONS)} tests passed.")


if __name__ == "__main__":
    main()