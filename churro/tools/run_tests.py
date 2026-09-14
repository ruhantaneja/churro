import importlib.util
import os
import sys
from pathlib import Path

from pydantic import Field

from churro.tools.subprocess_runner import run_captured
from churro.tools.tool import Tool, ToolArgs, ToolResult
from churro.tools.workspace import iter_files

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 300
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_TEST_COMMAND_DISPLAY = "python -m pytest"
DISPLAY_CAP_CHARS = 200

PYTEST_MARKER_FILES = ("pytest.ini", "conftest.py")
PYTEST_CONFIG_MARKERS = (
    ("pyproject.toml", "[tool.pytest.ini_options]"),
    ("setup.cfg", "[tool:pytest]"),
    ("tox.ini", "[pytest]"),
)
TEST_SUBDIRECTORY = "tests"
TEST_FILE_PREFIX = "test_"
TEST_FILE_SUFFIX = ".py"

INFRASTRUCTURE_FAILURE_PREFIX = "Test infrastructure failure: "
TEST_SUITE_FAILURE_MESSAGE = "Test suite executed but tests failed"


class RunTestsArgs(ToolArgs):
    command: str | None = Field(strict=True, default=None)
    timeout_seconds: int = Field(
        strict=True, default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=MAX_TIMEOUT_SECONDS
    )


class RunTestsTool(Tool[RunTestsArgs]):
    name = "run_tests"
    description = (
        "Run the project's tests. This is the primary and specialized tool "
        "for running, verifying, checking, and diagnosing the project's test "
        "suite, and it automatically handles test discovery and the test "
        "environment for you. When the user asks to run, verify, check, or "
        "diagnose tests, use run_tests rather than run_command or manually "
        "constructed pytest/python test commands. run_tests reports test "
        "failures and the captured output back to the agent. A provided "
        "command is arbitrary, trusted shell execution (like run_command) "
        "and is NOT sandboxed; the workspace is always the working "
        "directory. With no command, the project's test convention is "
        "discovered automatically: `python -m pytest` (current interpreter) "
        "is used when the project clearly uses pytest (pytest.ini, "
        "conftest.py, or pytest configuration in pyproject.toml/setup.cfg/"
        "tox.ini) or when the project uses the conventional tests/test_*.py "
        "layout and pytest is available; otherwise each file matching "
        "tests/test_*.py is run with the current Python interpreter "
        "(sys.executable) as its own subprocess (with the workspace root "
        "added to PYTHONPATH so project imports resolve), in deterministic "
        "sorted order, and per-file results are aggregated. "
        "Execution/infrastructure failures (cannot start, timeout, no tests "
        "found) are reported separately from test failures (tests ran but "
        "failed). The timeout applies to each subprocess individually."
    )
    args_model = RunTestsArgs

    def __init__(
        self,
        workspace: str | Path,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        pytest_available: bool | None = None,
        run_subprocess=None,
    ) -> None:
        self._workspace = Path(workspace).resolve()
        self._max_output_chars = int(max_output_chars)
        if pytest_available is None:
            pytest_available = importlib.util.find_spec("pytest") is not None
        self._pytest_available = bool(pytest_available)
        self._run_subprocess = run_subprocess or run_captured

    def _run(self, args: RunTestsArgs) -> ToolResult:
        custom = (args.command or "").strip()
        if custom:
            return self._run_explicit(custom, args.timeout_seconds)
        return self._run_auto(args.timeout_seconds)

    def _run_explicit(self, command: str, timeout_seconds: int) -> ToolResult:
        display = self._display_command(command)
        captured = self._run_subprocess(
            command, self._workspace, timeout_seconds, shell=True
        )
        if captured.error:
            return ToolResult(
                success=False,
                error=(
                    f"{INFRASTRUCTURE_FAILURE_PREFIX}cannot start command:\n"
                    f"{captured.error}"
                ),
            )
        stdout, stderr = self._cap_streams(captured.stdout, captured.stderr)
        if captured.timed_out:
            return ToolResult(
                success=False,
                output=self._compose(
                    display, "killed on timeout", stdout, stderr, "Command timed out."
                ),
                error=(
                    f"{INFRASTRUCTURE_FAILURE_PREFIX}command timed out after "
                    f"{timeout_seconds} seconds"
                ),
            )
        if captured.returncode == 0:
            return ToolResult(
                success=True,
                output=self._compose(
                    display, str(captured.returncode), stdout, stderr, "Tests passed."
                ),
            )
        return ToolResult(
            success=False,
            output=self._compose(
                display, str(captured.returncode), stdout, stderr, "Tests failed."
            ),
            error=f"Tests failed (exit code {captured.returncode})",
        )

    def _run_auto(self, timeout_seconds: int) -> ToolResult:
        if self._use_pytest():
            return self._run_pytest(timeout_seconds)
        return self._run_plain_suite(timeout_seconds)

    def _use_pytest(self) -> bool:
        if not self._pytest_available:
            return False
        if self._project_clearly_uses_pytest():
            return True
        # Conventional pytest layout: pytest is installed and the project
        # has tests/test_*.py files, which is how pytest-style tests declare
        # themselves. Plain-script execution would never run their test
        # bodies (a function definition alone exits 0), so use pytest.
        return bool(self._discover_plain_test_files())

    def _project_clearly_uses_pytest(self) -> bool:
        for name in PYTEST_MARKER_FILES:
            if (self._workspace / name).is_file():
                return True
        for name, marker in PYTEST_CONFIG_MARKERS:
            path = self._workspace / name
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if marker in content:
                return True
        return False

    def _run_pytest(self, timeout_seconds: int) -> ToolResult:
        argv = [sys.executable, "-m", "pytest"]
        captured = self._run_subprocess(
            argv, self._workspace, timeout_seconds, shell=False
        )
        if captured.error:
            return ToolResult(
                success=False,
                error=(
                    f"{INFRASTRUCTURE_FAILURE_PREFIX}cannot start pytest: "
                    f"{captured.error}"
                ),
            )
        stdout, stderr = self._cap_streams(captured.stdout, captured.stderr)
        if captured.timed_out:
            return ToolResult(
                success=False,
                output=self._compose(
                    DEFAULT_TEST_COMMAND_DISPLAY,
                    "killed on timeout",
                    stdout,
                    stderr,
                    "Command timed out.",
                ),
                error=(
                    f"{INFRASTRUCTURE_FAILURE_PREFIX}pytest timed out after "
                    f"{timeout_seconds} seconds"
                ),
            )
        if captured.returncode == 0:
            return ToolResult(
                success=True,
                output=self._compose(
                    DEFAULT_TEST_COMMAND_DISPLAY,
                    str(captured.returncode),
                    stdout,
                    stderr,
                    "Tests passed.",
                ),
            )
        return ToolResult(
            success=False,
            output=self._compose(
                DEFAULT_TEST_COMMAND_DISPLAY,
                str(captured.returncode),
                stdout,
                stderr,
                "Tests failed.",
            ),
            error=f"{TEST_SUITE_FAILURE_MESSAGE} (pytest exit code {captured.returncode})",
        )

    def _run_plain_suite(self, timeout_seconds: int) -> ToolResult:
        files = self._discover_plain_test_files()
        if not files:
            return ToolResult(
                success=False,
                output=(
                    self._compose_suite(
                        0, 0, 0, 0, 0, [], [], []
                    )
                    + "\n\n"
                    + "No test files discovered (searched tests/ for test_*.py)."
                ),
                error=(
                    f"{INFRASTRUCTURE_FAILURE_PREFIX}no test files discovered: "
                    "nothing matching tests/test_*.py"
                ),
            )

        passed: list[Path] = []
        failed: list[tuple[Path, int, str, str]] = []
        infra: list[tuple[Path, str]] = []
        blocks: list[str] = []
        env = self._test_env()
        for file in files:
            captured = self._run_subprocess(
                [sys.executable, str(file)],
                self._workspace,
                timeout_seconds,
                shell=False,
                env_extra=env,
            )
            if captured.error:
                message = f"could not start: {captured.error}"
                infra.append((file, message))
                blocks.append(
                    self._failure_block(file, "could not start", "", captured.error)
                )
            elif captured.timed_out:
                message = f"timed out after {timeout_seconds} seconds"
                infra.append((file, message))
                blocks.append(
                    self._failure_block(
                        file, "killed on timeout", captured.stdout, captured.stderr
                    )
                )
            elif captured.returncode == 0:
                passed.append(file)
            else:
                failed.append((file, captured.returncode, captured.stdout, captured.stderr))
                blocks.append(
                    self._failure_block(
                        file,
                        f"exit code {captured.returncode}",
                        captured.stdout,
                        captured.stderr,
                    )
                )

        failure_lines = [f"{self._relative(p)} (exit code {code})" for p, code, _, _ in failed]
        infra_lines = [f"{self._relative(p)}: {message}" for p, message in infra]
        output = self._compose_suite(
            len(files),
            len(files),
            len(passed),
            len(failed),
            len(infra),
            failure_lines,
            infra_lines,
            blocks,
        )

        if failed and infra:
            return ToolResult(
                success=False,
                output=output,
                error=(
                    f"{TEST_SUITE_FAILURE_MESSAGE}: {len(failed)} file(s) exited "
                    f"non-zero; {INFRASTRUCTURE_FAILURE_PREFIX}{len(infra)} file(s) "
                    "could not be executed"
                ),
            )
        if failed:
            return ToolResult(
                success=False,
                output=output,
                error=(
                    f"{TEST_SUITE_FAILURE_MESSAGE}: {len(failed)} of {len(files)} "
                    "file(s) exited non-zero"
                ),
            )
        if infra:
            first = infra_lines[0] if infra_lines else ""
            return ToolResult(
                success=False,
                output=output,
                error=(
                    f"{INFRASTRUCTURE_FAILURE_PREFIX}{len(infra)} file(s) could not "
                    f"be executed ({first})"
                ),
            )
        return ToolResult(success=True, output=output)

    def _discover_plain_test_files(self) -> list[Path]:
        tests_dir = self._workspace / TEST_SUBDIRECTORY
        if not tests_dir.is_dir():
            return []
        return [
            path
            for path in iter_files(tests_dir, recursive=False)
            if path.name.startswith(TEST_FILE_PREFIX)
            and path.name.endswith(TEST_FILE_SUFFIX)
        ]

    def _test_env(self) -> dict[str, str]:
        """Environment for auto-discovered test subprocesses.

        The workspace root is prepended to ``PYTHONPATH`` so a test file
        under ``tests/`` can import the project, while every existing
        ``PYTHONPATH`` entry is preserved and nothing is duplicated. This
        only extends the child environment; the parent process environment
        is never mutated.
        """
        workspace = str(self._workspace)
        entries = [
            entry
            for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if entry
        ]
        if workspace not in entries:
            entries.insert(0, workspace)
        return {"PYTHONPATH": os.pathsep.join(entries)}

    def _compose_suite(
        self,
        discovered: int,
        executed: int,
        passed: int,
        failed: int,
        infra: int,
        failure_lines: list[str],
        infra_lines: list[str],
        blocks: list[str],
    ) -> str:
        lines = [
            "Test suite:",
            f"{discovered} files discovered",
            f"{executed} files executed",
            f"{passed} passed",
            f"{failed} failed",
            f"{infra} infrastructure failures",
        ]
        if failure_lines:
            lines += ["", "Failed:"]
            lines += failure_lines
        if infra_lines:
            lines += ["", "Infrastructure failures:"]
            lines += infra_lines
        if blocks:
            lines += ["", "Relevant output:"]
            for block in blocks:
                lines += ["", block]
        lines += ["", f"Overall: {passed} passed, {failed} failed, {infra} infrastructure failure(s)."]
        text = "\n".join(lines).rstrip()
        return self._cap_one(text, self._max_output_chars)

    def _failure_block(self, file: Path, label: str, stdout: str, stderr: str) -> str:
        header = f"[--- {self._relative(file)} ({label}) ---]"
        parts = [header, "STDOUT:", stdout if stdout else "(empty)", "STDERR:", stderr if stderr else "(empty)"]
        return "\n".join(parts)

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self._workspace).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _display_command(custom: str) -> str:
        if len(custom) <= DISPLAY_CAP_CHARS:
            return custom
        return custom[:DISPLAY_CAP_CHARS] + (
            f"\n... [command truncated at {DISPLAY_CAP_CHARS} characters]"
        )

    @staticmethod
    def _compose(
        display: str,
        code_label: str,
        stdout: str,
        stderr: str,
        summary: str,
    ) -> str:
        lines = ["Test command:", display, "", f"Exit code: {code_label}", ""]
        for label, text in (("STDOUT", stdout), ("STDERR", stderr)):
            lines.append(f"{label}:")
            lines.append(text if text else "(empty)")
            lines.append("")
        lines.append(summary)
        return "\n".join(lines).rstrip("\n")

    def _cap_streams(self, stdout: str, stderr: str) -> tuple[str, str]:
        total = len(stdout) + len(stderr)
        budget = self._max_output_chars
        if total <= budget or total == 0:
            return stdout, stderr
        out_cap = budget * len(stdout) // total
        err_cap = budget - out_cap
        return (
            self._cap_one(stdout, out_cap),
            self._cap_one(stderr, err_cap),
        )

    @staticmethod
    def _cap_one(text: str, cap: int) -> str:
        if text and len(text) > cap:
            return text[:cap] + f"\n... [output truncated at {cap} characters]"
        return text