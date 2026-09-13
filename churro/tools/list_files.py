from pathlib import Path

from pydantic import Field

from churro.tools.tool import Tool, ToolArgs, ToolResult
from churro.tools.workspace import (
    WorkspacePathError,
    relative_display,
    resolve_within_workspace,
)


class ListFilesArgs(ToolArgs):
    path: str = Field(default=".", strict=True)
    recursive: bool = False


class ListFilesTool(Tool[ListFilesArgs]):
    name = "list_files"
    description = (
        "List files and directories inside the workspace as workspace-relative paths."
    )
    args_model = ListFilesArgs

    def __init__(self, workspace: str | Path) -> None:
        self._workspace = Path(workspace).resolve()

    def _format(self, entry: Path) -> str:
        label = "DIR" if entry.is_dir() else "FILE"
        rel = relative_display(self._workspace, entry)
        if label == "DIR":
            rel = rel + "/"
        return f"{label:<6}{rel}"

    def _walk(self, directory: Path, lines: list[str], recursive: bool) -> None:
        for entry in sorted(
            directory.iterdir(), key=lambda child: (child.name.lower(), child.name)
        ):
            lines.append(self._format(entry))
            if recursive and entry.is_dir() and not entry.is_symlink():
                self._walk(entry, lines, recursive)

    def _run(self, args: ListFilesArgs) -> ToolResult:
        try:
            target = resolve_within_workspace(self._workspace, args.path)
        except WorkspacePathError as exc:
            return ToolResult(success=False, error=str(exc))

        display = relative_display(self._workspace, target)
        if not target.exists():
            return ToolResult(
                success=False, error=f"Directory does not exist: {display}"
            )
        if not target.is_dir():
            return ToolResult(
                success=False, error=f"Path is not a directory: {display}"
            )

        try:
            lines: list[str] = []
            self._walk(target, lines, args.recursive)
        except OSError as exc:
            return ToolResult(
                success=False,
                error=f"Cannot list directory: {display} ({type(exc).__name__}: {exc})",
            )

        return ToolResult(success=True, output="\n".join(lines))