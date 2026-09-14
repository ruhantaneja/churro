"""Deterministic, content-free snapshot of a workspace for context discovery.

The index is built purely from directory traversal and file-name metadata
(no file contents are ever read), so it is cheap, safe, and deterministic.
The summary it renders is injected into the system prompt so the agent
starts with a compact map of the project instead of spending iterations on
exploration. Detailed investigation still goes through the filesystem tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_TEMP_PREFIX = ".churro-write-"

# Directories and files that are never surfaced in the index. These are
# generated/runtime artifacts or sensitive files (.env), and are excluded
# at any depth beneath a workspace.
_SKIP_EXACT: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".pytest_cache",
        "sessions",
        "node_modules",
        "venv",
        ".venv",
        "env",
        ".env",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "dist",
        "build",
        ".eggs",
        ".idea",
        ".vscode",
    }
)

# File extension -> human-readable language name.
_EXT_LANG: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "JavaScript",
    ".tsx": "TypeScript",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".rb": "Ruby",
    ".php": "PHP",
    ".c": "C",
    ".cpp": "C++",
    ".cs": "C#",
    ".swift": "Swift",
    ".sh": "Shell",
    ".bash": "Shell",
}

# Well-known configuration / manifest file names.
_CONFIG_NAMES: frozenset[str] = frozenset(
    {
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".gitignore",
        ".gitattributes",
        "tox.ini",
        "pytest.ini",
        "conftest.py",
        "tsconfig.json",
        ".env.example",
        ".editorconfig",
        "Pipfile",
        "poetry.lock",
    }
)

# Documentation file name prefixes (matched against the file stem).
_DOC_PREFIXES: tuple[str, ...] = (
    "readme",
    "changelog",
    "contributing",
    "license",
)

# A category falls back to extension-grouped counts above this many files.
_MAX_CATEGORY_FILES = 20
# Hard cap on rendered summary lines (safety valve for pathologically large
# or unusual workspaces).
_MAX_SUMMARY_LINES = 80


def _is_skipped(name: str) -> bool:
    if name in _SKIP_EXACT:
        return True
    if name.endswith(".egg-info"):
        return True
    if name.startswith(_TEMP_PREFIX):
        return True
    return False


def _is_doc(rel_parts: list[str]) -> bool:
    if "docs" in rel_parts[:-1] or "doc" in rel_parts[:-1]:
        return True
    stem = Path(rel_parts[-1]).stem.lower()
    return any(stem.startswith(prefix) for prefix in _DOC_PREFIXES)


def _is_config(name: str) -> bool:
    if name in _CONFIG_NAMES:
        return True
    return name.startswith("requirements") and name.endswith(".txt")


def _is_test(rel_parts: list[str]) -> bool:
    if "tests" in rel_parts[:-1] or "test" in rel_parts[:-1]:
        return True
    stem = Path(rel_parts[-1]).stem.lower()
    if stem.startswith("test_") or stem.startswith("test-"):
        return True
    if stem.endswith("_test") or stem.endswith(".test") or stem.endswith(".spec"):
        return True
    return False


def _walk(root: Path) -> tuple[list[str], list[str], list[str], list[str], int]:
    """Walk ``root`` once, yielding relative posix paths by category.

    Returns ``(source, test, config, doc, directory_count)`` where paths are
    workspace-relative using forward slashes. Contents are never read; only
    directory traversal and name classification happen.
    """
    source: list[str] = []
    test: list[str] = []
    config: list[str] = []
    doc: list[str] = []
    directory_count = 0

    def visit(directory: Path) -> None:
        nonlocal directory_count
        try:
            children = sorted(
                directory.iterdir(), key=lambda child: (child.name.lower(), child.name)
            )
        except OSError:
            return
        for entry in children:
            name = entry.name
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            if is_dir:
                if _is_skipped(name) or entry.is_symlink():
                    continue
                directory_count += 1
                visit(entry)
                continue
            if _is_skipped(name):
                continue
            rel = (entry.relative_to(root)).as_posix()
            rel_parts = rel.split("/")
            if _is_config(name):
                config.append(rel)
            elif _is_doc(rel_parts):
                doc.append(rel)
            elif _is_test(rel_parts):
                test.append(rel)
            else:
                source.append(rel)

    visit(root)
    return source, test, config, doc, directory_count


@dataclass(frozen=True)
class RepoIndex:
    """Immutable snapshot of a workspace's layout (relative paths only)."""

    root_name: str
    total_files: int
    total_dirs: int
    directories: tuple[str, ...]
    source_files: tuple[str, ...]
    test_files: tuple[str, ...]
    config_files: tuple[str, ...]
    doc_files: tuple[str, ...]
    languages: tuple[str, ...]


def build_index(root: Path) -> RepoIndex:
    """Build a ``RepoIndex`` for ``root`` without reading any file contents."""
    root = Path(root)
    source, test, config, doc, total_dirs = _walk(root)
    all_files = source + test + config + doc

    seen: list[str] = []
    for rel in all_files:
        lang = _EXT_LANG.get(Path(rel).suffix)
        if lang and lang not in seen:
            seen.append(lang)
    languages = tuple(sorted(seen))

    directories: list[str] = []
    for entry in sorted(
        root.iterdir(), key=lambda child: (child.name.lower(), child.name)
    ):
        if not _is_skipped(entry.name):
            try:
                if entry.is_dir() and not entry.is_symlink():
                    directories.append(entry.name)
            except OSError:
                pass

    return RepoIndex(
        root_name=root.name or str(root),
        total_files=len(all_files),
        total_dirs=total_dirs,
        directories=tuple(directories),
        source_files=tuple(source),
        test_files=tuple(test),
        config_files=tuple(config),
        doc_files=tuple(doc),
        languages=tuple(languages),
    )


def _render_category(lines: list[str], label: str, files: list[str]) -> None:
    if not files:
        return
    if len(files) <= _MAX_CATEGORY_FILES:
        lines.append(f"{label}:")
        lines.extend(f"  {rel}" for rel in files)
    else:
        groups: dict[str, int] = {}
        for rel in files:
            ext = Path(rel).suffix
            group = _EXT_LANG.get(ext) or (ext or "no-extension")
            groups[group] = groups.get(group, 0) + 1
        lines.append(f"{label} ({len(files)} files):")
        lines.extend(f"  {group}: {count}" for group, count in sorted(groups.items()))


def format_summary(index: RepoIndex) -> str:
    """Render a compact prompt-friendly overview (never more than 80 lines)."""
    counts = {
        "source": len(index.source_files),
        "test": len(index.test_files),
        "config": len(index.config_files),
        "doc": len(index.doc_files),
    }
    lang = ", ".join(index.languages) or "none"
    lines = [
        f"Project: {index.root_name}",
        f"Total files: {index.total_files} ("
        + ", ".join(f"{n} {name}" for name, n in counts.items() if n)
        + f") | Languages: {lang}",
    ]
    if index.directories:
        lines.append("Directories:")
        lines.extend(f"  {d}/" for d in index.directories)

    _render_category(lines, "Source files", list(index.source_files))
    _render_category(lines, "Test files", list(index.test_files))
    _render_category(lines, "Config", list(index.config_files))
    _render_category(lines, "Docs", list(index.doc_files))

    if len(lines) > _MAX_SUMMARY_LINES:
        lines = lines[: _MAX_SUMMARY_LINES - 1] + ["... summary truncated"]
    return "\n".join(lines)


def build_summary(root: Path) -> str:
    """Convenience: return ``format_summary(build_index(root))``."""
    return format_summary(build_index(root))