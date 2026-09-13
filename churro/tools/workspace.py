from collections.abc import Iterator
from pathlib import Path


class WorkspacePathError(Exception):
    """A requested path could not be resolved or would leave the workspace."""


def resolve_within_workspace(root: Path, raw_path: str) -> Path:
    """Resolve a requested path inside the workspace and reject escapes.

    The candidate is rendered as a real path (following symlinks) and then
    checked against the resolved workspace root, so relative traversal
    (``..``), absolute paths elsewhere, and symlinks pointing outside are
    all rejected. Inputs are interpreted relative to ``root`` unless they
    are absolute; absolute inputs are allowed only when they stay inside
    it. The returned error never exposes an absolute host path.
    """
    requested = Path(raw_path)
    candidate = requested if requested.is_absolute() else root / requested
    try:
        resolved = candidate.resolve()
    except (OSError, ValueError) as exc:
        raise WorkspacePathError(
            f"Cannot resolve path {raw_path!r} ({type(exc).__name__}: {exc})"
        ) from None
    if not resolved.is_relative_to(root):
        raise WorkspacePathError(f"Path {raw_path!r} is outside the workspace")
    return resolved


def relative_display(root: Path, resolved: Path) -> str:
    """Present a resolved path relative to root (never an absolute path)."""
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return resolved.name
    if rel == Path("."):
        return "."
    return rel.as_posix()


def iter_files(directory: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield regular files beneath ``directory`` in a stable sorted order.

    Subdirectories are visited depth-first in name order and symlinked
    directories are never descended, so traversal is deterministic, cannot
    loop, and cannot leave the workspace through a directory symlink.
    """
    for entry in sorted(
        directory.iterdir(), key=lambda child: (child.name.lower(), child.name)
    ):
        if entry.is_file():
            yield entry
        elif recursive and entry.is_dir() and not entry.is_symlink():
            yield from iter_files(entry, recursive)