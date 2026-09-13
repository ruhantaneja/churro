from pydantic import Field

from churro.tools import (
    DuplicateToolError,
    Tool,
    ToolArgs,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
)


class EchoArgs(ToolArgs):
    text: str
    times: int = Field(default=1, ge=1)


class EchoTool(Tool[EchoArgs]):
    name = "echo"
    description = "Repeat a message the given number of times (fake demo tool)."
    args_model = EchoArgs

    def _run(self, args: EchoArgs) -> ToolResult:
        return ToolResult(success=True, output="\n".join([args.text] * args.times))


class FizzArgs(ToolArgs):
    n: int = Field(ge=1)


class FizzTool(Tool[FizzArgs]):
    name = "fizz"
    description = "Always fails; demonstrates controlled tool errors (fake demo tool)."
    args_model = FizzArgs

    def _run(self, args: FizzArgs) -> ToolResult:
        raise RuntimeError("the fake tool exploded, but CHURRO stays up")


def main() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(FizzTool())

    print("Registered tools:")
    for tool in registry.list_tools():
        print(f"  {tool.name:8s} {tool.description}")
        print(f"    schema: {tool.input_schema}")

    print("\n-- valid execution -------------------------------------------")
    result = registry.execute("echo", {"text": "churro", "times": 3})
    print(f"success={result.success} error={result.error!r}")
    print(result.output)

    print("\n-- invalid arguments (controlled) -----------------------------")
    result = registry.execute("echo", {"times": -2})
    print(f"success={result.success} error={result.error!r}")

    print("\n-- unknown tool (controlled) ----------------------------------")
    result = registry.execute("read_file", {"path": "x.py"})
    print(f"success={result.success} error={result.error!r}")

    print("\n-- failing tool (controlled) ----------------------------------")
    result = registry.execute("fizz", {"n": 1})
    print(f"success={result.success} error={result.error!r}")

    print("\n-- definition errors ------------------------------------------")
    try:
        registry.register(EchoTool())
    except DuplicateToolError as exc:
        print(f"duplicate registration -> {exc}")

    try:
        registry.get("no_such_tool")
    except UnknownToolError as exc:
        print(f"unknown lookup      -> {exc}")


if __name__ == "__main__":
    main()