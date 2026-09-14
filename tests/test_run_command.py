import os
import sys
import tempfile
from pathlib import Path

from churro.tools import RunCommandTool, ToolRegistry
from churro.tools.run_command import MAX_TIMEOUT_SECONDS, RunCommandArgs
from churro.tools.tool import ToolResult


def make_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="churro run cmd "))


def make_tool(
    root: Path | None = None, max_output_chars: int = 20_000
) -> tuple[RunCommandTool, Path]:
    root = root or make_root()
    return RunCommandTool(root, max_output_chars=max_output_chars), root


def py_cmd(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def stdout_section(output: str) -> str:
    return output.split("STDOUT:\n", 1)[1].split("\n\nSTDERR:", 1)[0].strip()


def test_successful_command():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("print('hello')")})
    assert result.success is True
    assert result.error == ""
    assert "Exit code: 0" in result.output
    assert "hello" in result.output


def test_captured_stdout():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("print('alpha');print('beta')")})
    assert result.success is True
    assert "alpha" in result.output
    assert "beta" in result.output


def test_captured_stderr():
    tool, _ = make_tool()
    result = tool.execute(
        {"command": py_cmd("import sys;print('oops',file=sys.stderr)")}
    )
    assert result.success is True
    assert "oops" in result.output


def test_empty_stdout_reported():
    tool, _ = make_tool()
    result = tool.execute(
        {"command": py_cmd("import sys;sys.stderr.write('only-err')")}
    )
    assert result.success is True
    section = result.output.split("STDOUT:\n", 1)[1].split("\n\nSTDERR:", 1)[0].strip()
    assert section == "(empty)"
    assert "only-err" in result.output


def test_non_zero_exit_is_controlled_failure():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("import sys;sys.exit(3)")})
    assert result.success is False
    assert "Exit code: 3" in result.output
    assert "Command exited with code 3" in result.error


def test_command_not_found_is_controlled_failure():
    tool, _ = make_tool()
    result = tool.execute({"command": "definitely_not_a_real_command_xyz_123"})
    assert result.success is False
    assert result.error
    assert "Exit code:" in result.output


def test_workspace_is_subprocess_cwd():
    tool, root = make_tool()
    result = tool.execute({"command": py_cmd("import os;print(os.getcwd())")})
    assert result.success is True
    line = stdout_section(result.output)
    expected = os.path.normcase(os.path.realpath(str(root)))
    actual = os.path.normcase(os.path.realpath(line))
    assert actual == expected


def test_relative_filesystem_access_uses_workspace():
    tool, root = make_tool()
    (root / "rel.txt").write_text("relative-reading-works", encoding="utf-8")
    result = tool.execute({"command": py_cmd("print(open('rel.txt').read())")})
    assert result.success is True
    assert "relative-reading-works" in result.output


def test_default_timeout_used_when_omitted():
    tool, _ = make_tool()
    assert RunCommandArgs(command="x").timeout_seconds == 30
    result = tool.execute({"command": py_cmd("print('default-ok')")})
    assert result.success is True


def test_timeout_kills_command():
    tool, _ = make_tool()
    result = tool.execute(
        {"command": py_cmd("import time;time.sleep(30)"), "timeout_seconds": 1}
    )
    assert result.success is False
    assert "timed out" in result.error.lower()
    assert "killed on timeout" in result.output


def test_timeout_seconds_must_be_positive():
    tool, _ = make_tool()
    for bad in (0, -5):
        result = tool.execute(
            {"command": py_cmd("print('x')"), "timeout_seconds": bad}
        )
        assert result.success is False
        assert "timeout_seconds" in result.error


def test_timeout_seconds_must_not_exceed_maximum():
    tool, _ = make_tool()
    assert MAX_TIMEOUT_SECONDS == 300
    result = tool.execute(
        {"command": py_cmd("print('x')"), "timeout_seconds": 301}
    )
    assert result.success is False
    assert "timeout_seconds" in result.error


def test_timeout_seconds_must_be_integer():
    tool, _ = make_tool()
    for bad in ("5", 2.5, True):
        result = tool.execute(
            {"command": py_cmd("print('x')"), "timeout_seconds": bad}
        )
        assert result.success is False
        assert "timeout_seconds" in result.error


def test_empty_command_rejected():
    tool, _ = make_tool()
    result = tool.execute({"command": ""})
    assert result.success is False
    assert "command" in result.error


def test_whitespace_command_rejected():
    tool, _ = make_tool()
    result = tool.execute({"command": "   \t"})
    assert result.success is False
    assert "command" in result.error


def test_output_truncation_indicated():
    tool, _ = make_tool(max_output_chars=64)
    result = tool.execute({"command": py_cmd("print('x'*200)")})
    assert result.success is True
    assert "truncated" in result.output
    section = stdout_section(result.output)
    assert len(section) < 200
    assert "Exit code: 0" in result.output


def test_stdout_and_stderr_distinguishable():
    tool, _ = make_tool()
    result = tool.execute(
        {
            "command": (
                py_cmd(
                    "import sys;print('ONSTDOUT');print('ONSTDERR',file=sys.stderr)"
                )
            )
        }
    )
    assert result.success is True
    stderr_section = result.output.split("STDERR:\n", 1)[1].rstrip("\n")
    assert "ONSTDERR" in stderr_section
    assert "ONSTDOUT" not in stderr_section
    assert "ONSTDOUT" in result.output


def test_exit_code_represented_in_output():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("print('done')")})
    assert result.output.startswith("Exit code: 0\n")
    failed = tool.execute({"command": py_cmd("import sys;sys.exit(7)")})
    assert failed.output.startswith("Exit code: 7\n")


def test_toolresult_success_structure():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("print('struct')")})
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
    root = make_root()
    registry = ToolRegistry()
    registry.register(RunCommandTool(root))
    assert any(t.name == "run_command" for t in registry.list_tools())
    result = registry.execute("run_command", {"command": py_cmd("print('via-registry')")})
    assert result.success is True
    assert "via-registry" in result.output


def test_command_cannot_change_configured_workspace():
    tool, root = make_tool()
    before = tool._workspace
    tool.execute({"command": py_cmd("import os;os.chdir('..');print('moved')")})
    assert tool._workspace == before
    result = tool.execute({"command": py_cmd("import os;print(os.getcwd())")})
    line = stdout_section(result.output)
    expected = os.path.normcase(os.path.realpath(str(root)))
    assert os.path.normcase(os.path.realpath(line)) == expected


def test_cannot_select_arbitrary_cwd():
    tool, root = make_tool()
    other = make_root()
    result = tool.execute(
        {"command": py_cmd("import os;print(os.getcwd())"), "cwd": str(other)}
    )
    assert result.success is False
    assert "cwd" in result.error
    normal = tool.execute({"command": py_cmd("import os;print(os.getcwd())")})
    assert normal.success is True
    assert os.path.normcase(
        os.path.realpath(stdout_section(normal.output))
    ) == os.path.normcase(os.path.realpath(str(root)))


def test_workspace_path_with_spaces():
    tool, root = make_tool()
    assert " " in root.name
    result = tool.execute({"command": py_cmd("import os;print(os.getcwd())")})
    assert result.success is True
    line = stdout_section(result.output)
    assert os.path.normcase(os.path.realpath(line)) == os.path.normcase(
        os.path.realpath(str(root))
    )


def test_describes_arbitrary_command_execution():
    tool, _ = make_tool()
    assert "arbitrary" in tool.description
    assert "not sandboxed" in tool.description.lower()
    assert "trust" in tool.description.lower()


def test_keyboard_interrupt_reraises_from_run_captured():
    """Step 18.9: Ctrl+C while a tool subprocess runs must propagate.

    ``run_captured`` must re-raise KeyboardInterrupt instead of capturing
    it as a generic subprocess error, so the CLI can abort the agent.
    """
    import subprocess

    from churro.tools.subprocess_runner import run_captured

    interrupted = KeyboardInterrupt("interrupted")

    class FakeProc:
        pid = 999999999

        def communicate(self, timeout):
            raise interrupted

        def kill(self):
            pass

        def wait(self, timeout=5):
            return 0

    original_open = subprocess.Popen
    subprocess.Popen = lambda *args, **kwargs: FakeProc()
    try:
        try:
            run_captured([sys.executable, "-c", "pass"], make_root(), 30)
        except KeyboardInterrupt as exc:
            assert exc is interrupted
        else:
            raise AssertionError("expected KeyboardInterrupt to propagate")
    finally:
        subprocess.Popen = original_open


TEST_FUNCTIONS = [
    test_successful_command,
    test_captured_stdout,
    test_captured_stderr,
    test_empty_stdout_reported,
    test_non_zero_exit_is_controlled_failure,
    test_command_not_found_is_controlled_failure,
    test_workspace_is_subprocess_cwd,
    test_relative_filesystem_access_uses_workspace,
    test_default_timeout_used_when_omitted,
    test_timeout_kills_command,
    test_timeout_seconds_must_be_positive,
    test_timeout_seconds_must_not_exceed_maximum,
    test_timeout_seconds_must_be_integer,
    test_empty_command_rejected,
    test_whitespace_command_rejected,
    test_output_truncation_indicated,
    test_stdout_and_stderr_distinguishable,
    test_exit_code_represented_in_output,
    test_toolresult_success_structure,
    test_toolresult_failure_structure,
    test_registry_dispatch,
    test_command_cannot_change_configured_workspace,
    test_cannot_select_arbitrary_cwd,
    test_workspace_path_with_spaces,
    test_describes_arbitrary_command_execution,
    test_keyboard_interrupt_reraises_from_run_captured,
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