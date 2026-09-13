from pathlib import Path

from pydantic import Field

from churro.tools.tool import Tool, ToolArgs, ToolResult
from churro.tools.workspace import (
    WorkspacePathError,
    relative_display,
    resolve_within_workspace,
)


class ReadFileArgs(ToolArgs):
    path: str = Field(strict=True)


class ReadFileTool(Tool[ReadFileArgs]):
    name = "read_file"
    description = "Read a UTF-8 text file relative to the configured workspace root."
    args_model = ReadFileArgs

    def __init__(self, workspace: str | Path) -> None:
        self._workspace = Path(workspace).resolve()

    def _run(self, args: ReadFileArgs) -> ToolResult:
        try:
            target = resolve_within_workspace(self._workspace, args.path)
        except WorkspacePathError as exc:
            return ToolResult(success=False, error=str(exc))

        display = relative_display(self._workspace, target)
        if not target.exists():
            return ToolResult(success=False, error=f"File does not exist: {display}")
        if target.is_dir():
            return ToolResult(
                success=False, error=f"Path is a directory, not a file: {display}"
            )

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                success=False, error=f"File is not valid UTF-8 text: {display}"
            )
        except OSError as exc:
            return ToolResult(
                success=False,
                error=f"Cannot read file: {display} ({type(exc).__name__}: {exc})",
            )

        return ToolResult(success=True, output=content)