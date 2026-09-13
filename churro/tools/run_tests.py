import importlib.util
import sys
from pathlib import Path

from pydantic import Field

from churro.tools.subprocess_runner import run_captured
from churro.tools.tool import Tool, ToolArgs, ToolResult

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 300
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_TEST_COMMAND_DISPLAY = "python -m pytest"
DISPLAY_CAP_CHARS = 200


class RunTestsArgs(ToolArgs):
    command: str | None = Field(strict=True, default=None)
    timeout_seconds: int = Field(
        strict=True, default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=MAX_TIMEOUT_SECONDS
    )


class RunTestsTool(Tool[RunTestsArgs]):
    name = "run_tests"
    description = (
        "Run the project's test suite from the workspace and report results. "
        "With no command, runs `python -m pytest` (using the current Python "
        "interpreter) when pytest is available, else fails cleanly. A "
        "provided command is arbitrary, trusted shell execution (like "
        "run_command) and is NOT sandboxed; the workspace is always the "
        "working directory. Only Python/pytest detection is supported."
    )
    args_model = RunTestsArgs

    def __init__(
        self,
        workspace: str | Path,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        pytest_available: bool | None = None,
    ) -> None:
        self._workspace = Path(workspace).resolve()
        self._max_output_chars = int(max_output_chars)
        if pytest_available is None:
            pytest_available = importlib.util.find_spec("pytest") is not None
        self._pytest_available = bool(pytest_available)

    def _run(self, args: RunTestsArgs) -> ToolResult:
        custom = (args.command or "").strip()
        if custom:
            argv = custom
            display = self._display_command(custom)
            shell = True
        else:
            if not self._pytest_available:
                return ToolResult(
                    success=False,
                    error="No test runner detected: pytest is not available",
                )
            argv = [sys.executable, "-m", "pytest"]
            display = DEFAULT_TEST_COMMAND_DISPLAY
            shell = False

        captured = run_captured(
            argv, self._workspace, args.timeout_seconds, shell=shell
        )
        if captured.error:
            return ToolResult(success=False, error=f"Failed to run tests:\n{captured.error}")

        stdout, stderr = self._cap_streams(captured.stdout, captured.stderr)
        if captured.timed_out:
            return ToolResult(
                success=False,
                output=self._compose(
                    display, "killed on timeout", stdout, stderr, "Command timed out."
                ),
                error=f"Tests timed out after {args.timeout_seconds} seconds",
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