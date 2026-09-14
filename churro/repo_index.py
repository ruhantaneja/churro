"""Deterministic, content-free snapshot of a workspace for context discovery.

The index is built purely from directory traversal and file-name metadata
(no file contents are ever read), so it is cheap, safe, and deterministic.
The summary it renders is injected into the system prompt so the agent
starts with a compact map of the project instead of spending iterations on
exploration. Detailed investigation still goes through the filesystem tools.

Step 21 adds a shallow, module-level structural layer: Python packages,
entry points, top-level symbols, imports, and local import relationships.
This layer reads only the Python source/test files the index already
classified, runs conservative regex analysis (no execution, no AST, no
importlib), and never leaves the workspace.
"""

from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
# Step 21: repository structure intelligence
#
# Extends Step 20 from "what files exist" to "what are the important parts
# and how are they related". All analysis is deterministic and content is
# read only for the Python source/test files the index already classified.
# ---------------------------------------------------------------------------

# Hard cap on rendered structure lines (safety valve for large workspaces).
_STRUCTURE_MAX_LINES = 60
# Cap on how many "most imported" modules are shown.
_MOST_IMPORTED_MAX = 8
# Cap on how many symbols / local imports a single module line lists.
_MAX_INLINE_ITEMS = 8

_FROM_IMPORT_RE = re.compile(r"^from\s+([\w.]+)\s+import\b")
_IMPORT_RE = re.compile(r"^import\s+(.+)")
_IMPORT_NAME_RE = re.compile(r"^([\w.]+)")
_CLASS_RE = re.compile(r"^class\s+(\w+)", re.MULTILINE)
_FUNCTION_RE = re.compile(r"^(?:async\s+)?def\s+(\w+)", re.MULTILINE)


@dataclass(frozen=True)
class ModuleInfo:
    """Structural summary of one analyzed Python module.

    ``path`` is workspace-relative with forward slashes. ``imports`` are the
    dotted module names exactly as written (deduplicated, sorted);
    ``imports_local`` are the workspace-relative paths of the imports that
    resolve to files in the repository index. ``classes`` and ``functions``
    are top-level only (nested symbols are intentionally excluded).
    """

    path: str
    lines: int
    classes: tuple[str, ...]
    functions: tuple[str, ...]
    imports: tuple[str, ...]
    imports_local: tuple[str, ...]


@dataclass(frozen=True)
class RepoStructure:
    """Deterministic, module-level map of a Python workspace.

    ``packages`` are the workspace-relative directories that contain an
    ``__init__.py`` (the root package is ``"."``). ``entry_points`` are the
    ``__main__.py`` workspace-relative paths. ``modules`` holds every local
    Python source/test file that could be read and analyzed, sorted by path.
    ``imported_by`` maps each local Python module path to the (sorted) paths
    of modules that import it — the centrality view.
    """

    packages: tuple[str, ...]
    entry_points: tuple[str, ...]
    modules: tuple[ModuleInfo, ...]
    imported_by: dict[str, tuple[str, ...]]


def _read_python_file(path: Path) -> str | None:
    """Read a Python file as UTF-8, or ``None`` if it cannot be decoded."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _count_lines(content: str) -> int:
    """Count the newline-separated lines in ``content``."""
    return len(content.splitlines())


def _extract_imports(content: str) -> tuple[str, ...]:
    """Extract dotted import names conservatively (regex, no execution).

    Handles ``import x``, ``import x, y as z``, ``from x import y``, and the
    common multiline ``from x import (`` opening line. Names are deduplicated
    and returned sorted.
    """
    names: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        from_match = _FROM_IMPORT_RE.match(stripped)
        if from_match is not None:
            names.append(from_match.group(1))
            continue
        import_match = _IMPORT_RE.match(stripped)
        if import_match is None:
            continue
        tail = import_match.group(1).strip()
        if tail.startswith("("):
            continue
        for piece in tail.split(","):
            piece = piece.strip()
            if not piece:
                continue
            name_match = _IMPORT_NAME_RE.match(piece)
            if name_match is not None:
                names.append(name_match.group(1))
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return tuple(sorted(seen))


def _extract_symbols(content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(classes, functions)`` declared at column zero only.

    Because the patterns anchor at the start of a line without any leading
    indentation, nested classes and functions are never reported.
    """
    classes = tuple(cls for cls in _CLASS_RE.findall(content))
    functions = tuple(fn for fn in _FUNCTION_RE.findall(content))
    return classes, functions


def _resolve_import_to_path(module_name: str, known: frozenset[str]) -> str | None:
    """Resolve a dotted import name to an existing workspace-relative path.

    Tries ``<module>.py`` first, then ``<pkg>/__init__.py``. Returns ``None``
    when the import does not name a file in the repository index (stdlib,
    third-party, or unknown), so external imports are never mapped locally.
    """
    if not module_name:
        return None
    dotted = module_name
    module_candidate = dotted.replace(".", "/") + ".py"
    if module_candidate in known:
        return module_candidate
    package_candidate = dotted.replace(".", "/") + "/__init__.py"
    if package_candidate in known:
        return package_candidate
    return None


def build_structure(root: Path, index: RepoIndex | None = None) -> RepoStructure:
    """Build module-level structural intelligence for ``root``.

    Derives everything from the existing ``RepoIndex`` (built once here when
    none is supplied) so the workspace is walked at most once. Only the
    Python source/test files already classified by the index are read, and
    only their line counts, top-level symbols, and import statements are
    extracted via deterministic regexes. Nothing is executed or imported; no
    network calls and no filesystem writes occur.
    """
    root = Path(root)
    if index is None:
        index = build_index(root)

    source = list(index.source_files)
    test = list(index.test_files)
    py_files = sorted(rel for rel in source + test if rel.endswith(".py"))
    known = frozenset(py_files)

    all_indexed = source + test
    packages: list[str] = []
    entry_points: list[str] = []
    for rel in all_indexed:
        if rel.endswith("__init__.py"):
            parent = rel[: -len("__init__.py")].rstrip("/")
            packages.append(parent or ".")
        if rel.endswith("__main__.py"):
            entry_points.append(rel)

    modules: list[ModuleInfo] = []
    importers: dict[str, set[str]] = {rel: set() for rel in py_files}

    for rel in py_files:
        content = _read_python_file(root / rel)
        if content is None:
            continue
        classes, functions = _extract_symbols(content)
        imports = _extract_imports(content)
        local_imports = tuple(
            sorted(
                resolved
                for name in imports
                if (resolved := _resolve_import_to_path(name, known)) is not None
            )
        )
        modules.append(
            ModuleInfo(
                path=rel,
                lines=_count_lines(content),
                classes=classes,
                functions=functions,
                imports=imports,
                imports_local=local_imports,
            )
        )
        for target in local_imports:
            importers[target].add(rel)

    imported_by = {rel: tuple(sorted(importers[rel])) for rel in sorted(importers)}

    return RepoStructure(
        packages=tuple(sorted(packages)),
        entry_points=tuple(entry_points),
        modules=tuple(modules),
        imported_by=imported_by,
    )


def _render_inline(names: tuple[str, ...], cap: int = _MAX_INLINE_ITEMS) -> str:
    """Render ``names`` compactly, capping long lists to keep lines short."""
    shown = ", ".join(names[:cap])
    extra = len(names) - cap
    if extra > 0:
        return f"{shown} ... +{extra} more"
    return shown


def _render_module_line(module: ModuleInfo) -> str:
    detail = f"  {module.path}: {module.lines} lines"
    if module.classes:
        detail += f"; classes: {_render_inline(module.classes)}"
    if module.functions:
        detail += f"; functions: {_render_inline(module.functions)}"
    if module.imports_local:
        detail += f"; local: {_render_inline(module.imports_local)}"
    return detail


def format_structure(structure: RepoStructure) -> str:
    """Render structural intelligence as a compact prompt block.

    Capped at ``_STRUCTURE_MAX_LINES`` lines with a truncation marker. Paths
    are always workspace-relative. Returns an empty string when the
    workspace has no meaningful Python package hierarchy, so Step 20's
    overview remains the fallback.
    """
    if not structure.packages or not structure.modules:
        return ""

    lines: list[str] = []

    lines.append("Packages:")
    for package in structure.packages:
        lines.append(f"  {package}/" if package != "." else "  .")

    if structure.entry_points:
        lines.append("")
        lines.append("Entry points:")
        lines.extend(f"  {entry}" for entry in structure.entry_points)

    ranked = sorted(
        (
            (rel, importers)
            for rel, importers in structure.imported_by.items()
            if importers
        ),
        key=lambda item: (-len(item[1]), item[0]),
    )
    if ranked:
        lines.append("")
        lines.append("Most-imported (structurally central):")
        for rel, importers in ranked[:_MOST_IMPORTED_MAX]:
            label = "file" if len(importers) == 1 else "files"
            lines.append(f"  {rel}: imported by {len(importers)} {label}")

    lines.append("")
    lines.append("Modules:")
    for module in structure.modules:
        lines.append(_render_module_line(module))

    if len(lines) > _STRUCTURE_MAX_LINES:
        lines = lines[: _STRUCTURE_MAX_LINES - 1] + ["... structure truncated"]
    return "\n".join(lines)


def build_full_summary(root: Path) -> str:
    """Convenience: Step 20 overview plus Step 21 structure, joined.

    Walks the workspace once and returns ``format_summary`` followed by
    ``format_structure`` (when non-empty), separated by a blank line.
    """
    index = build_index(root)
    summary = format_summary(index)
    structure = format_structure(build_structure(root, index))
    if structure:
        return f"{summary}\n\n{structure}"
    return summary