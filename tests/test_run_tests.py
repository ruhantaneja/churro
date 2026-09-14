import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import churro

from churro.tools import RunTestsTool, ToolRegistry
from churro.tools import subprocess_runner
from churro.tools.run_tests import (
    INFRASTRUCTURE_FAILURE_PREFIX,
    MAX_TIMEOUT_SECONDS,
    TEST_SUITE_FAILURE_MESSAGE,
)
from churro.tools.subprocess_runner import CapturedResult
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


def write_plain_files(root: Path, names: list[str]) -> list[Path]:
    (root / "tests").mkdir(exist_ok=True)
    created = []
    for name in names:
        path = root / "tests" / name
        path.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        created.append(path)
    return created


class RecordingSubprocess:
    """Fake for churro.tools.subprocess_runner.run_captured.

    Records every call and returns per-key results (absolute file
    basename for plain runs, "pytest" for the pytest invocation, or the
    exact shell string for explicit commands). Missing keys succeed with
    exit code 0 so multi-file suites pass by default.
    """

    def __init__(self) -> None:
        self.results: dict = {}
        self.calls: list[tuple] = []

    def __call__(self, argv, cwd, timeout_seconds, shell=False, env_extra=None):
        self.calls.append((argv, Path(cwd), timeout_seconds, bool(shell), env_extra))
        if shell:
            return self.results.get(argv, CapturedResult(returncode=0))
        key = Path(argv[-1]).name if isinstance(argv, (list, tuple)) else argv
        return self.results.get(key, CapturedResult(returncode=0))

    def executed_names(self) -> list[str]:
        names = []
        for argv, _, _, shell, _ in self.calls:
            if not shell and isinstance(argv, list) and len(argv) == 2:
                names.append(Path(argv[-1]).name)
        return names

    def shell_commands(self) -> list[str]:
        return [argv for argv, _, _, shell, _ in self.calls if shell]


# --- explicit command behavior (real subprocess, unchanged semantics) ---

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


def test_explicit_command_uses_shell_string():
    stub = RecordingSubprocess()
    result = RunTestsTool(make_root(), run_subprocess=stub).execute(
        {"command": "echo explicit-ran"}
    )
    assert result.success is True
    assert "explicit-ran" in result.output
    assert stub.shell_commands() == ["echo explicit-ran"]


def test_explicit_custom_command_used():
    tool, _ = make_tool()
    result = tool.execute({"command": py_cmd("print('custom-ran')")})
    assert result.success is True
    assert "custom-ran" in result.output
    assert "python -m pytest" not in result.output


# --- timeouts and validation (preserved) ---

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
    assert result.error.startswith(INFRASTRUCTURE_FAILURE_PREFIX)


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


# --- workspace safety for explicit commands (preserved) ---

def test_workspace_used_as_cwd():
    tool, root = make_tool()
    result = tool.execute({"command": py_cmd("import os;print(os.getcwd())")})
    assert result.success is True
    line = stdout_section(result.output)
    assert os.path.normcase(os.path.realpath(line)) == os.path.normcase(
        os.path.realpath(str(root))
    )


def test_commands_run_relative_to_workspace():
    tool, root = make_tool()
    (root / "rel.txt").write_text("relative-test-read", encoding="utf-8")
    result = tool.execute({"command": py_cmd("print(open('rel.txt').read())")})
    assert result.success is True
    assert "relative-test-read" in result.output


def test_no_arbitrary_cwd_argument():
    tool, _ = make_tool()
    other = make_root()
    result = tool.execute(
        {"command": py_cmd("print('x')"), "cwd": str(other)}
    )
    assert result.success is False
    assert "cwd" in result.error


def test_command_execution_failure():
    tool, root = make_tool()
    result = tool.execute({"command": py_cmd("import sys;sys.exit(2)")})
    assert result.success is False
    assert "Exit code: 2" in result.output
    assert "Tests failed." in result.output
    missing = RunTestsTool(root.parent / "does_not_exist_xyz")
    failed = missing.execute({"command": py_cmd("pass")})
    assert failed.success is False
    assert failed.error.startswith(INFRASTRUCTURE_FAILURE_PREFIX)


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


def test_args_dictionary_not_mutated():
    args = {"command": py_cmd("print('x')"), "timeout_seconds": 30}
    snapshot = dict(args)
    tool, _ = make_tool()
    result = tool.execute(args)
    assert result.success is True
    assert args == snapshot
    empty = {}
    result2 = RunTestsTool(make_root(), pytest_available=False).execute(empty)
    assert result2.success is False
    assert empty == {}


# --- pytest convention detection ---

def test_pytest_available_and_clearly_pytest_uses_pytest():
    root = make_root()
    (root / "pytest.ini").write_text("", encoding="utf-8")
    stub = RecordingSubprocess()
    stub.results["pytest"] = CapturedResult(returncode=0, stdout="2 passed")
    tool = RunTestsTool(root, pytest_available=True, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert result.error == ""
    assert "python -m pytest" in result.output
    assert "Tests passed." in result.output
    assert "2 passed" in result.output
    argv, cwd, timeout_seconds, shell, _ = stub.calls[0]
    assert list(argv) == [sys.executable, "-m", "pytest"]
    assert shell is False
    assert timeout_seconds == 120


def test_pytest_configured_in_pyproject_uses_pytest():
    root = make_root()
    (root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8"
    )
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=True, run_subprocess=stub)
    result = tool.execute({})
    assert "python -m pytest" in result.output
    assert stub.executed_names() == []


def test_pytest_unavailable_uses_plain_discovery_even_with_marker():
    root = make_root()
    (root / "pytest.ini").write_text("", encoding="utf-8")
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert stub.executed_names() == ["test_a.py"]
    assert "pytest" not in [n for n in stub.executed_names()]
    for argv, *_ in stub.calls:
        assert "pytest" not in argv


def test_pytest_available_with_conventional_tests_dir_uses_pytest():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    stub.results["pytest"] = CapturedResult(returncode=0, stdout="1 passed")
    tool = RunTestsTool(root, pytest_available=True, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert result.error == ""
    assert "python -m pytest" in result.output
    assert "Tests passed." in result.output
    assert "1 passed" in result.output
    argv, cwd, timeout_seconds, shell, _ = stub.calls[0]
    assert list(argv) == [sys.executable, "-m", "pytest"]
    assert shell is False
    assert timeout_seconds == 120


def test_pytest_available_but_no_conventional_tests_dir_uses_plain():
    root = make_root()
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=True, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert stub.calls == []
    assert "No test files discovered" in result.output


def test_pytest_available_but_tests_dir_has_no_test_py_uses_plain():
    root = make_root()
    tests_dir = root / "tests"
    tests_dir.mkdir()
    for name in ("helper.py", "notes.txt"):
        (tests_dir / name).write_text("x", encoding="utf-8")
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=True, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert "python -m pytest" not in result.output
    assert "No test files discovered" in result.output


def test_pytest_unavailable_with_conventional_tests_dir_uses_plain():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert stub.executed_names() == ["test_a.py"]


def test_pytest_style_test_function_now_executed_by_pytest():
    root = make_root()
    (root / "tests").mkdir()
    (root / "tests" / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    stub = RecordingSubprocess()
    stub.results["pytest"] = CapturedResult(
        returncode=1,
        stderr=(
            "tests/test_calculator.py F\n\n"
            "_____________ test_add _____________\n"
            "    def test_add():\n"
            ">       assert add(2, 3) == 5\n"
            "E       assert -1 == 5\n"
            "\nAssertionError\n"
        ),
    )
    tool = RunTestsTool(root, pytest_available=True, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert "Test suite executed but tests failed (pytest exit code 1)" in result.error
    assert "Tests failed." in result.output
    assert "assert add(2, 3) == 5" in result.output
    assert stub.executed_names() == []


# --- plain-Python discovery ---

def test_plain_discovery_discovers_only_test_py_in_tests():
    root = make_root()
    (root / "tests").mkdir()
    for name in ("test_keep.py", "helper.py", "notes.txt"):
        (root / "tests" / name).write_text("x", encoding="utf-8")
    (root / "tests" / "sub").mkdir()
    (root / "tests" / "sub" / "test_inner.py").write_text("x", encoding="utf-8")
    (root / "test_root_level.py").write_text("x", encoding="utf-8")
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert stub.executed_names() == ["test_keep.py"]
    assert "1 passed" in result.output


def test_plain_discovery_deterministic_ordering():
    root = make_root()
    write_plain_files(root, ["test_z.py", "test_m.py", "test_a.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert stub.executed_names() == ["test_a.py", "test_m.py", "test_z.py"]
    assert "3 files discovered" in result.output
    assert "3 files executed" in result.output


def test_plain_discovery_successful_file():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert "1 passed" in result.output
    assert "0 failed" in result.output
    assert "0 infrastructure failures" in result.output


def test_plain_discovery_failing_file():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    stub.results["test_a.py"] = CapturedResult(returncode=1)
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert "0 passed" in result.output
    assert "1 failed" in result.output
    assert "tests/test_a.py (exit code 1)" in result.output


def test_plain_discovery_multiple_files():
    root = make_root()
    write_plain_files(root, ["test_a.py", "test_b.py", "test_c.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert "3 files discovered" in result.output
    assert "3 files executed" in result.output
    assert "3 passed" in result.output
    assert "0 failed" in result.output


def test_plain_discovery_mixed_pass_fail():
    root = make_root()
    write_plain_files(root, ["test_a.py", "test_b.py", "test_c.py"])
    stub = RecordingSubprocess()
    stub.results["test_b.py"] = CapturedResult(
        returncode=2, stdout="assert failed", stderr="boom"
    )
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert "2 passed" in result.output
    assert "1 failed" in result.output
    assert "tests/test_b.py (exit code 2)" in result.output
    assert "assert failed" in result.output
    assert "boom" in result.output
    assert result.error.startswith(TEST_SUITE_FAILURE_MESSAGE)


def test_plain_discovery_all_tests_fail():
    root = make_root()
    write_plain_files(root, ["test_a.py", "test_b.py"])
    stub = RecordingSubprocess()
    stub.results["test_a.py"] = CapturedResult(returncode=3)
    stub.results["test_b.py"] = CapturedResult(returncode=1)
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert "0 passed" in result.output
    assert "2 failed" in result.output
    assert result.error == (
        f"{TEST_SUITE_FAILURE_MESSAGE}: 2 of 2 file(s) exited non-zero"
    )


def test_plain_discovery_uses_sys_executable():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    argv, cwd, _, shell, _ = stub.calls[0]
    assert shell is False
    assert argv[0] == sys.executable
    assert Path(argv[1]).name == "test_a.py"
    assert cwd == root


# --- infrastructure vs test failure ---

def test_subprocess_launch_failure_is_infrastructure():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    stub.results["test_a.py"] = CapturedResult(
        error="Command or working directory not found"
    )
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert result.error.startswith(INFRASTRUCTURE_FAILURE_PREFIX)
    assert "could not be executed" in result.error
    assert "1 infrastructure failures" in result.output
    assert "tests/test_a.py: could not start" in result.output
    assert TEST_SUITE_FAILURE_MESSAGE not in result.error


def test_discovery_timeout_is_infrastructure():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    stub.results["test_a.py"] = CapturedResult(stdout="started-test", timed_out=True)
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert result.error.startswith(INFRASTRUCTURE_FAILURE_PREFIX)
    assert "timed out" in result.error.lower()
    assert "1 infrastructure failures" in result.output
    assert "started-test" in result.output
    assert TEST_SUITE_FAILURE_MESSAGE not in result.error


def test_test_failure_not_mistaken_for_infrastructure():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    stub.results["test_a.py"] = CapturedResult(returncode=1, stderr="AssertionError")
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert result.error.startswith(TEST_SUITE_FAILURE_MESSAGE)
    assert result.error.count(INFRASTRUCTURE_FAILURE_PREFIX) == 0
    assert "0 infrastructure failures" in result.output
    assert "AssertionError" in result.output


def test_mixed_test_and_infrastructure_failures():
    root = make_root()
    write_plain_files(root, ["test_a.py", "test_b.py"])
    stub = RecordingSubprocess()
    stub.results["test_a.py"] = CapturedResult(returncode=1)
    stub.results["test_b.py"] = CapturedResult(
        error="Failed to launch command: FileNotFoundError"
    )
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert "0 passed" in result.output
    assert "1 failed" in result.output
    assert "1 infrastructure failures" in result.output
    assert TEST_SUITE_FAILURE_MESSAGE in result.error
    assert INFRASTRUCTURE_FAILURE_PREFIX in result.error


# --- output capture, cap, deterministic aggregate ---

def test_discovery_captures_stdout_and_stderr():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    stub.results["test_a.py"] = CapturedResult(
        returncode=1, stdout="printed-out", stderr="error-err"
    )
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert "printed-out" in result.output
    assert "error-err" in result.output


def test_discovery_output_cap():
    root = make_root()
    write_plain_files(root, ["test_a.py", "test_b.py"])
    stub = RecordingSubprocess()
    big = "x" * 4000
    stub.results["test_a.py"] = CapturedResult(returncode=1, stdout=big, stderr="")
    stub.results["test_b.py"] = CapturedResult(returncode=1, stdout="", stderr=big)
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub, max_output_chars=200)
    result = tool.execute({})
    assert result.success is False
    assert "truncated" in result.output
    assert "2 files discovered" in result.output
    assert len(result.output) < 500


def test_plain_discovery_deterministic_aggregate_output():
    root = make_root()
    write_plain_files(root, ["test_a.py", "test_b.py"])
    stub = RecordingSubprocess()
    stub.results["test_b.py"] = CapturedResult(returncode=1, stderr="boom")
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    lines = result.output.splitlines()
    assert lines[0] == "Test suite:"
    assert lines[1] == "2 files discovered"
    assert lines[2] == "2 files executed"
    assert lines[3] == "1 passed"
    assert lines[4] == "1 failed"
    assert lines[5] == "0 infrastructure failures"
    assert "Failed:" in lines
    assert "tests/test_b.py (exit code 1)" in lines
    assert "[--- tests/test_b.py (exit code 1) ---]" in result.output
    assert result.output.endswith(
        "Overall: 1 passed, 1 failed, 0 infrastructure failure(s)."
    )


# --- empty / missing test directories ---

def test_empty_test_directory_is_infrastructure():
    root = make_root()
    (root / "tests").mkdir()
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert result.error.startswith(INFRASTRUCTURE_FAILURE_PREFIX)
    assert "no test files discovered" in result.error
    assert "0 files discovered" in result.output
    assert "0 infrastructure failures" in result.output
    assert stub.calls == []


def test_no_test_directory_is_infrastructure():
    root = make_root()
    tool = RunTestsTool(root, pytest_available=False)
    result = tool.execute({})
    assert result.success is False
    assert result.error.startswith(INFRASTRUCTURE_FAILURE_PREFIX)
    assert "no test files discovered" in result.error


# --- workspace safety / no network / reuse / no pytest dependency ---

def test_plain_discovery_does_not_leave_workspace():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    other = make_root()
    write_plain_files(other, ["test_b.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert stub.executed_names() == ["test_a.py"]


def test_plain_discovery_no_network_calls():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    for argv, cwd, _, shell, _ in stub.calls:
        assert shell is False
        assert list(argv)[:1] == [sys.executable]
        assert Path(argv[-1]).is_relative_to(root)
        assert "http" not in " ".join(argv).lower()


def test_no_pytest_dependency_required():
    root = make_root()
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    tool = RunTestsTool(root, pytest_available=False, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is True
    assert "python -m pytest" not in result.output


def test_default_subprocess_runner_is_run_captured():
    tool, _ = make_tool()
    assert tool._run_subprocess is subprocess_runner.run_captured


def test_pytest_path_launch_failure_is_infrastructure():
    root = make_root()
    (root / "pytest.ini").write_text("", encoding="utf-8")
    stub = RecordingSubprocess()
    stub.results["pytest"] = CapturedResult(error="command not found")
    tool = RunTestsTool(root, pytest_available=True, run_subprocess=stub)
    result = tool.execute({})
    assert result.success is False
    assert result.error.startswith(INFRASTRUCTURE_FAILURE_PREFIX)
    assert "pytest" in result.error


@contextmanager
def set_pythonpath(value: str | None):
    previous = os.environ.get("PYTHONPATH")
    if value is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous


def plain_env_and_argv(root: Path, existing_pythonpath: str | None = None) -> tuple:
    write_plain_files(root, ["test_a.py"])
    stub = RecordingSubprocess()
    with set_pythonpath(existing_pythonpath):
        result = RunTestsTool(root, pytest_available=False, run_subprocess=stub).execute({})
    assert result.success is True
    argv, cwd, timeout, shell, env = stub.calls[0]
    assert shell is False
    return argv, env


# --- PYTHONPATH for auto-discovered tests ---

def test_auto_discovery_env_includes_workspace_root_in_pythonpath():
    root = make_root()
    argv, env = plain_env_and_argv(root)
    assert argv[0] == sys.executable
    assert Path(argv[1]).name == "test_a.py"
    expected = os.pathsep.join([str(root)])
    assert env == {"PYTHONPATH": expected}


def test_auto_discovery_preserves_existing_pythonpath():
    root = make_root()
    argv, env = plain_env_and_argv(root, "libA" + os.pathsep + "libB")
    assert env == {
        "PYTHONPATH": os.pathsep.join([str(root), "libA", "libB"])
    }


def test_auto_discovery_no_duplicate_workspace_root():
    root = make_root()
    existing = str(root) + os.pathsep + "libA"
    argv, env = plain_env_and_argv(root, existing)
    assert env == {"PYTHONPATH": existing}


def test_auto_discovery_uses_platform_pathsep():
    root = make_root()
    argv, env = plain_env_and_argv(root, "libA" + os.pathsep + "libB")
    assert env["PYTHONPATH"] == os.pathsep.join([str(root), "libA", "libB"])
    assert len(env["PYTHONPATH"].split(os.pathsep)) == 3


def test_auto_discovery_does_not_mutate_parent_env():
    root = make_root()
    before = dict(os.environ)
    argv, env = plain_env_and_argv(root, "libY")
    assert dict(os.environ) == before


def test_all_discovered_files_receive_same_env():
    root = make_root()
    write_plain_files(root, ["test_a.py", "test_b.py"])
    stub = RecordingSubprocess()
    with set_pythonpath("libX" + os.pathsep + str(root) + os.pathsep + "libZ"):
        result = RunTestsTool(root, pytest_available=False, run_subprocess=stub).execute({})
    assert result.success is True
    envs = [call[4] for call in stub.calls]
    assert len(envs) == 2
    assert envs[0] == envs[1] == {
        "PYTHONPATH": os.pathsep.join(["libX", str(root), "libZ"])
    }


def test_explicit_command_env_unchanged():
    stub = RecordingSubprocess()
    result = RunTestsTool(make_root(), run_subprocess=stub).execute(
        {"command": "echo explicit-ran"}
    )
    assert result.success is True
    assert "explicit-ran" in result.output
    assert len(stub.calls) == 1
    assert stub.calls[0][4] is None
    assert stub.calls[0][3] is True


def test_discovered_test_can_import_churro():
    churro_root = str(Path(churro.__file__).resolve().parents[1])
    root = make_root()
    (root / "tests").mkdir()
    (root / "tests" / "test_probe.py").write_text(
        "import churro\nprint('probe imported churro')\n", encoding="utf-8"
    )
    with set_pythonpath(churro_root):
        result = RunTestsTool(root, pytest_available=False).execute({})
    assert result.success is True
    assert "1 passed" in result.output
    assert "0 failed" in result.output


TEST_FUNCTIONS = [
    test_successful_test_command,
    test_failing_test_command,
    test_captured_stdout,
    test_captured_stderr,
    test_exit_code_represented,
    test_explicit_command_uses_shell_string,
    test_explicit_custom_command_used,
    test_timeout,
    test_timeout_seconds_must_be_positive,
    test_timeout_seconds_must_not_exceed_maximum,
    test_workspace_used_as_cwd,
    test_commands_run_relative_to_workspace,
    test_no_arbitrary_cwd_argument,
    test_command_execution_failure,
    test_output_truncation,
    test_toolresult_success_structure,
    test_toolresult_failure_structure,
    test_registry_dispatch,
    test_workspace_not_modified_by_tool,
    test_args_dictionary_not_mutated,
    test_pytest_available_and_clearly_pytest_uses_pytest,
    test_pytest_configured_in_pyproject_uses_pytest,
    test_pytest_unavailable_uses_plain_discovery_even_with_marker,
    test_pytest_available_with_conventional_tests_dir_uses_pytest,
    test_pytest_available_but_no_conventional_tests_dir_uses_plain,
    test_pytest_available_but_tests_dir_has_no_test_py_uses_plain,
    test_pytest_unavailable_with_conventional_tests_dir_uses_plain,
    test_pytest_style_test_function_now_executed_by_pytest,
    test_plain_discovery_discovers_only_test_py_in_tests,
    test_plain_discovery_deterministic_ordering,
    test_plain_discovery_successful_file,
    test_plain_discovery_failing_file,
    test_plain_discovery_multiple_files,
    test_plain_discovery_mixed_pass_fail,
    test_plain_discovery_all_tests_fail,
    test_plain_discovery_uses_sys_executable,
    test_subprocess_launch_failure_is_infrastructure,
    test_discovery_timeout_is_infrastructure,
    test_test_failure_not_mistaken_for_infrastructure,
    test_mixed_test_and_infrastructure_failures,
    test_discovery_captures_stdout_and_stderr,
    test_discovery_output_cap,
    test_plain_discovery_deterministic_aggregate_output,
    test_empty_test_directory_is_infrastructure,
    test_no_test_directory_is_infrastructure,
    test_plain_discovery_does_not_leave_workspace,
    test_plain_discovery_no_network_calls,
    test_no_pytest_dependency_required,
    test_default_subprocess_runner_is_run_captured,
    test_pytest_path_launch_failure_is_infrastructure,
    test_auto_discovery_env_includes_workspace_root_in_pythonpath,
    test_auto_discovery_preserves_existing_pythonpath,
    test_auto_discovery_no_duplicate_workspace_root,
    test_auto_discovery_uses_platform_pathsep,
    test_auto_discovery_does_not_mutate_parent_env,
    test_all_discovered_files_receive_same_env,
    test_explicit_command_env_unchanged,
    test_discovered_test_can_import_churro,
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