from pathlib import Path

from pydantic import Field, field_validator

from churro.tools.subprocess_runner import run_captured
from churro.tools.tool import Tool, ToolArgs, ToolResult

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300
DEFAULT_MAX_OUTPUT_CHARS = 20_000


class RunCommandArgs(ToolArgs):
    command: str = Field(strict=True, min_length=1)
    timeout_seconds: int = Field(
        strict=True, default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=MAX_TIMEOUT_SECONDS
    )

    @field_validator("command")
    @classmethod
    def _command_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must not be empty or whitespace")
        return value


class RunCommandTool(Tool[RunCommandArgs]):
    name = "run_command"
    description = (
        "Execute a shell command in the workspace directory and return its "
        "output. DANGER: this runs arbitrary, trusted commands with real shell "
        "access and is NOT sandboxed; workspace confinement only sets the "
        "initial working directory. A command can read or write any file or "
        "network resource the process can reach. Only use with inputs you trust."
    )
    args_model = RunCommandArgs

    def __init__(
        self,
        workspace: str | Path,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self._workspace = Path(workspace).resolve()
        self._max_output_chars = int(max_output_chars)

    def _run(self, args: RunCommandArgs) -> ToolResult:
        captured = run_captured(
            args.command.strip(),
            self._workspace,
            args.timeout_seconds,
            shell=True,
        )
        if captured.error:
            return ToolResult(success=False, error=captured.error)
        if captured.timed_out:
            return self._timeout_result(
                args.timeout_seconds, captured.stdout, captured.stderr
            )
        return self._completed_result(
            captured.returncode, captured.stdout, captured.stderr
        )

    def _completed_result(self, code: int, stdout: str, stderr: str) -> ToolResult:
        stdout, stderr = self._cap_streams(stdout, stderr)
        output = self._format_output(str(code), stdout, stderr)
        if code == 0:
            return ToolResult(success=True, output=output)
        return ToolResult(
            success=False,
            output=output,
            error=f"Command exited with code {code}",
        )

    def _timeout_result(self, timeout: int, stdout: str, stderr: str) -> ToolResult:
        stdout, stderr = self._cap_streams(stdout, stderr)
        output = self._format_output("killed on timeout", stdout, stderr)
        return ToolResult(
            success=False,
            output=output,
            error=f"Command timed out after {timeout} seconds",
        )

    def _cap_streams(self, stdout: str, stderr: str) -> tuple[str, str]:
        total = len(stdout) + len(stderr)
        if total <= self._max_output_chars or total == 0:
            return stdout, stderr
        out_cap = self._max_output_chars * len(stdout) // total
        err_cap = self._max_output_chars - out_cap
        return (
            self._cap_one(stdout, out_cap),
            self._cap_one(stderr, err_cap),
        )

    @staticmethod
    def _cap_one(text: str, cap: int) -> str:
        if text and len(text) > cap:
            return text[:cap] + f"\n... [output truncated at {cap} characters]"
        return text

    @staticmethod
    def _format_output(code: str, stdout: str, stderr: str) -> str:
        parts = [f"Exit code: {code}", ""]
        for label, text in (("STDOUT", stdout), ("STDERR", stderr)):
            parts.append(f"{label}:")
            parts.append(text if text else "(empty)")
            parts.append("")
        return "\n".join(parts).rstrip("\n")