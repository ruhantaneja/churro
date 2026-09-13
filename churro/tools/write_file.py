import os
import stat
import tempfile
from pathlib import Path

from pydantic import Field

from churro.tools.tool import Tool, ToolArgs, ToolResult
from churro.tools.workspace import (
    WorkspacePathError,
    relative_display,
    resolve_within_workspace,
)

_TEMP_PREFIX = ".churro-write-"
_TEMP_SUFFIX = ".tmp"


class WriteFileArgs(ToolArgs):
    path: str = Field(strict=True, min_length=1)
    content: str = Field(strict=True)


class WriteFileTool(Tool[WriteFileArgs]):
    name = "write_file"
    description = (
        "Create or replace a UTF-8 text file inside the workspace, preserving "
        "content exactly and never writing outside the workspace."
    )
    args_model = WriteFileArgs

    def __init__(self, workspace: str | Path) -> None:
        self._workspace = Path(workspace).resolve()

    def _run(self, args: WriteFileArgs) -> ToolResult:
        try:
            target = resolve_within_workspace(self._workspace, args.path)
        except WorkspacePathError as exc:
            return ToolResult(success=False, error=str(exc))

        display = relative_display(self._workspace, target)

        if target.exists() and target.is_dir():
            return ToolResult(
                success=False, error=f"Path is a directory, not a file: {display}"
            )

        parent = target.parent
        if not parent.is_dir():
            return ToolResult(
                success=False,
                error=(
                    f"Parent directory does not exist: "
                    f"{relative_display(self._workspace, parent)}"
                ),
            )

        created = not target.exists()
        try:
            return self._write_atomic(target, display, args.content, created)
        except OSError as exc:
            return ToolResult(
                success=False,
                error=f"Cannot write file: {display} ({type(exc).__name__}: {exc})",
            )

    def _write_atomic(
        self, target: Path, display: str, content: str, created: bool
    ) -> ToolResult:
        """Write *content* to *target* atomically via temp-file + replace.

        For an existing target the original permission mode is restored on
        the replacement file so that existing access-control bits are not
        silently stripped.
        """
        mode = None
        if not created:
            mode = stat.S_IMODE(target.stat().st_mode)

        temp_path: str | None = None
        handle = None
        replaced = False
        try:
            fd, temp_path = tempfile.mkstemp(
                dir=str(target.parent), prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX
            )
            try:
                handle = os.fdopen(fd, "w", encoding="utf-8", newline="")
            except BaseException:
                os.close(fd)
                raise
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            if mode is not None:
                os.chmod(temp_path, mode)
            os.replace(temp_path, target)
            replaced = True
        finally:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            if not replaced and temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        action = "Created" if created else "Updated"
        return ToolResult(success=True, output=f"{action} {display}")
