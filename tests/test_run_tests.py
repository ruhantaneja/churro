import importlib.util
import os
import sys
import tempfile
from pathlib import Path

from churro.tools import RunTestsTool, ToolRegistry
from churro.tools.run_tests import MAX_TIMEOUT_SECONDS
from churro.tools.tool import ToolResult


def make_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="churro runtests "))


def make_tool(root: Path | None = None, **kwargs) -> tuple[RunTestsTool, Path]:
    root = root or make_root()
    return RunTestsTool(root, **kwargs), root


def py_cmd(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def stdout_section(output: str) -> str:
    return output.split("STDOUT:\n", 1)[1].split("\n\nSTDERR:", 1)[0].strip()


def test_successful_test_command():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("print('suite ok')")})
    assert result.success is True
    assert result.error == ""
    assert "Exit code: 0" in result.output
    assert "Tests passed." in result.output
    assert "suite ok" in result.output


def test_failing_test_command():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("import sys;sys.exit(1)")})
    assert result.success is False
    assert "Exit code: 1" in result.output
    assert "Tests failed." in result.output
    assert "Tests failed (exit code 1)" in result.error


def test_captured_stdout():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("print('alpha');print('beta')")})
    assert result.success is True
    assert "alpha" in result.output
    assert "beta" in result.output


def test_captured_stderr():
    tool, _ = make_tool()
    result = tool.execute(
        {"command": py_cmd("import sys;print('err-line',file=sys.stderr)")}
    )
    assert result.success is True
    assert "err-line" in result.output
    section = result.output.split("STDOUT:\n", 1)[1].split("\n\nSTDERR:", 1)[0].strip()
    assert section == "(empty)"


def test_exit_code_represented():
    tool, _ = make_tool()
    ok = tool.execute({"command": py_cmd("pass")})
    assert ok.output.startswith("Test command:")
    assert "Exit code: 0" in ok.output
    bad = tool.execute({"command": py_cmd("import sys;sys.exit(5)")})
    assert "Exit code: 5" in bad.output


def test_timeout():
    tool, _ = make_tool()
    result = tool.execute(
        {
            "command": py_cmd(
                "import sys,time;print('tick');sys.stdout.flush();time.sleep(30)"
            ),
            "timeout_seconds": 1,
        }
    )
    assert result.success is False
    assert "timed out" in result.error.lower()
    assert "killed on timeout" in result.output
    assert "tick" in result.output


def test_timeout_seconds_must_be_positive():
    tool, _ = make_tool()
    for bad in (0, -5):
        result = tool.execute({"command": py_cmd("pass"), "timeout_seconds": bad})
        assert result.success is False
        assert "timeout_seconds" in result.error


def test_timeout_seconds_must_not_exceed_maximum():
    tool, _ = make_tool()
    assert MAX_TIMEOUT_SECONDS == 300
    result = tool.execute({"command": py_cmd("pass"), "timeout_seconds": 301})
    assert result.success is False
    assert "timeout_seconds" in result.error


def test_workspace_used_as_cwd():
    tool, root = make_tool()
    result = tool.execute({"command": py_cmd("import os;print(os.getcwd())")})
    assert result.success is True
    line = stdout_section(result.output)
    assert os.path.normcase(os.path.realpath(line)) == os.path.normcase(
        os.path.realpath(str(root))
    )


def test_default_pytest_detection():
    if importlib.util.find_spec("pytest") is None:
        return
    tool, root = make_tool()
    (root / "test_sample.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    result = tool.execute({})
    assert result.success is True
    assert "python -m pytest" in result.output
    assert "Tests passed." in result.output
    assert "passed" in result.output


def test_explicit_custom_command_used():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("print('custom-ran')")})
    assert result.success is True
    assert "custom-ran" in result.output
    assert "python -m pytest" not in result.output


def test_missing_pytest_unavailable_runner():
    tool, _ = make_tool(pytest_available=False)
    result = tool.execute({})
    assert result.success is False
    assert "pytest" in result.error.lower()
    assert "no test runner" in result.error.lower()
    assert result.output == ""


def test_command_execution_failure():
    tool, root = make_tool()
    result = tool.execute({"command": py_cmd("import sys;sys.exit(2)")})
    assert result.success is False
    assert "Exit code: 2" in result.output
    assert "Tests failed." in result.output
    missing = RunTestsTool(root.parent / "does_not_exist_xyz")
    failed = missing.execute({"command": py_cmd("pass")})
    assert failed.success is False
    assert failed.error


def test_output_truncation():
    tool, _ = make_tool(max_output_chars=200)
    code = ";".join(f"print('content line {i}')" for i in range(100))
    result = tool.execute({"command": py_cmd(code)})
    assert result.success is True
    assert "truncated" in result.output
    assert "Tests passed." in result.output
    assert "content line 99" not in result.output
    assert len(result.output) < 700


def test_toolresult_success_structure():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("pass")})
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error == ""
    assert isinstance(result.output, str) and result.output


def test_toolresult_failure_structure():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("import sys;sys.exit(1)")})
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert isinstance(result.output, str) and result.output
    assert isinstance(result.error, str) and result.error


def test_registry_dispatch():
    tool, root = make_tool()
    registry = ToolRegistry()
    registry.register(tool)
    assert any(t.name == "run_tests" for t in registry.list_tools())
    result = registry.execute("run_tests", {"command": py_cmd("print('via-registry')")})
    assert result.success is True
    assert "via-registry" in result.output


def test_workspace_not_modified_by_tool():
    tool, root = make_tool()
    (root / "keep.txt").write_text("stable", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = tool.execute({"command": py_cmd("pass")})
    assert result.success is True
    after = {p.name: p.read_bytes() for p in root.iterdir()}
    assert after == before


def test_no_arbitrary_cwd_argument():
    tool, _ = make_tool()
    other = make_root()
    result = tool.execute(
        {"command": py_cmd("print('x')"), "cwd": str(other)}
    )
    assert result.success is False
    assert "cwd" in result.error


def test_commands_run_relative_to_workspace():
    tool, root = make_tool()
    (root / "rel.txt").write_text("relative-test-read", encoding="utf-8")
    result = tool.execute({"command": py_cmd("print(open('rel.txt').read())")})
    assert result.success is True
    assert "relative-test-read" in result.output


TEST_FUNCTIONS = [
    test_successful_test_command,
    test_failing_test_command,
    test_captured_stdout,
    test_captured_stderr,
    test_exit_code_represented,
    test_timeout,
    test_timeout_seconds_must_be_positive,
    test_timeout_seconds_must_not_exceed_maximum,
    test_workspace_used_as_cwd,
    test_default_pytest_detection,
    test_explicit_custom_command_used,
    test_missing_pytest_unavailable_runner,
    test_command_execution_failure,
    test_output_truncation,
    test_toolresult_success_structure,
    test_toolresult_failure_structure,
    test_registry_dispatch,
    test_workspace_not_modified_by_tool,
    test_no_arbitrary_cwd_argument,
    test_commands_run_relative_to_workspace,
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