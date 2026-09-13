import os
import shutil
from pathlib import Path

from pydantic import Field

from churro.tools.subprocess_runner import run_captured
from churro.tools.tool import Tool, ToolArgs, ToolResult
from churro.tools.workspace import (
    WorkspacePathError,
    relative_display,
    resolve_within_workspace,
)

DEFAULT_MAX_OUTPUT_CHARS = 20_000
_GIT_TIMEOUT_SECONDS = 300

_GIT_DIFF_ENV = {
    "GIT_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
}


class GitDiffArgs(ToolArgs):
    staged: bool = False
    path: str = "."


class GitDiffTool(Tool[GitDiffArgs]):
    name = "git_diff"
    description = (
        "Inspect uncommitted changes in the workspace with Git. Returns the "
        "working-tree diff by default, or the staged diff with staged=true. "
        "The optional path restricts the diff to a workspace-relative path. "
        "Strictly read-only: it never modifies files, the index, or Git "
        "state. Git ignores untracked files by design, so new files do not "
        "appear in the diff."
    )
    args_model = GitDiffArgs

    def __init__(
        self,
        workspace: str | Path,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        git_executable: str | None = None,
    ) -> None:
        self._workspace = Path(workspace).resolve()
        self._max_output_chars = int(max_output_chars)
        self._git_executable = git_executable or shutil.which("git")

    def _run(self, args: GitDiffArgs) -> ToolResult:
        try:
            resolved = resolve_within_workspace(self._workspace, args.path)
        except WorkspacePathError as exc:
            return ToolResult(success=False, error=str(exc))

        if self._git_executable is None:
            return ToolResult(success=False, error="Git executable not found")

        rel = relative_display(self._workspace, resolved)
        git_args = ["--no-pager", "diff"]
        if args.staged:
            git_args.append("--cached")
        git_args.extend(["--", rel])

        captured = run_captured(
            [self._git_executable, *git_args],
            self._workspace,
            _GIT_TIMEOUT_SECONDS,
            env_extra=_GIT_DIFF_ENV,
        )
        if captured.error:
            return ToolResult(success=False, error=captured.error)
        if captured.timed_out:
            return ToolResult(success=False, error="git diff timed out")

        code, out, err = captured.returncode, captured.stdout, captured.stderr
        diff = self._redact(out) if out else ""
        if code != 0:
            message = self._redact(err) if err else f"git exited with code {code}"
            return ToolResult(
                success=False,
                error=f"git diff failed (exit code {code}):\n{message}",
            )

        if not diff.strip() and not resolved.exists():
            return ToolResult(
                success=False,
                error=f"Path does not exist: {rel}",
            )

        if not diff.strip():
            diff = "No changes."
        return ToolResult(success=True, output=self._compose(diff))

    def _compose(self, diff: str) -> str:
        if len(diff) <= self._max_output_chars:
            return f"Git diff:\n{diff}"
        kept = diff[: self._max_output_chars]
        marker = f"\n... [diff truncated; showing first {self._max_output_chars} characters]"
        return f"Git diff:\n{kept}{marker}"

    def _redact(self, text: str) -> str:
        candidates = set()
        workspace = str(self._workspace)
        candidates.update({workspace, workspace.replace(os.sep, "/")})
        fs = os.path.normcase(self._workspace)
        candidates.add(fs)
        candidates.add(fs.replace(os.sep, "/"))
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate:
                text = text.replace(candidate, "<workspace>")
        return text