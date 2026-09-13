from collections.abc import Iterator
from pathlib import Path

from pydantic import Field

from churro.tools.tool import Tool, ToolArgs, ToolResult
from churro.tools.workspace import (
    WorkspacePathError,
    iter_files,
    relative_display,
    resolve_within_workspace,
)


class SearchFilesArgs(ToolArgs):
    query: str = Field(strict=True, min_length=1)
    path: str = Field(default=".", strict=True)
    recursive: bool = True
    max_results: int = Field(default=50, ge=1)


class SearchFilesTool(Tool[SearchFilesArgs]):
    name = "search_files"
    description = (
        "Search UTF-8 text files inside the workspace for a literal, "
        "case-sensitive string and report file, line number, and matching line."
    )
    args_model = SearchFilesArgs

    def __init__(self, workspace: str | Path) -> None:
        self._workspace = Path(workspace).resolve()

    def _iter_matches(self, directory: Path, query: str, recursive: bool) -> Iterator[str]:
        for entry in iter_files(directory, recursive):
            try:
                target = resolve_within_workspace(self._workspace, str(entry))
            except WorkspacePathError:
                continue
            try:
                with open(target, "r", encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, start=1):
                        if query in line:
                            rel = relative_display(self._workspace, entry)
                            yield f"{rel}:{line_no}:{line.rstrip(chr(13) + chr(10))}"
            except UnicodeDecodeError:
                continue
            except OSError:
                continue

    def _run(self, args: SearchFilesArgs) -> ToolResult:
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

        matches: list[str] = []
        matches_iter = self._iter_matches(target, args.query, args.recursive)
        truncated = False
        for match in matches_iter:
            matches.append(match)
            if len(matches) >= args.max_results:
                truncated = next(matches_iter, None) is not None
                matches_iter.close()
                break
        if truncated:
            matches.append(f"... results truncated at {args.max_results} matches")

        return ToolResult(success=True, output="\n".join(matches))