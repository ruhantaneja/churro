"""Tests for the repository index (churro.repo_index).

Verifies workspace containment, deterministic output, classification
rules, skip logic, and the prompt-friendly summary formatter, plus the
Step 21 structural layer (packages, entry points, imports, symbols,
centrality, and its compact rendering).
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from churro.repo_index import (
    ModuleInfo,
    RepoIndex,
    RepoStructure,
    _count_lines,
    _extract_imports,
    _extract_symbols,
    _is_skipped,
    _resolve_import_to_path,
    build_full_summary,
    build_index,
    build_structure,
    build_summary,
    format_structure,
    format_summary,
)


def root() -> Path:
    return Path(tempfile.mkdtemp())


# ── Skip logic ──────────────────────────────────────────────────────

def test_skip_exact_match():
    assert _is_skipped("__pycache__")
    assert _is_skipped(".git")
    assert _is_skipped("node_modules")
    assert _is_skipped(".env")


def test_skip_prefix():
    assert _is_skipped(".churro-write-abc123")
    assert not _is_skipped(".churro-write")
    assert _is_skipped(".churro-write-")


def test_skip_egg_info():
    assert _is_skipped("myproject.egg-info")
    assert not _is_skipped("myproject")


def test_normal_name_not_skipped():
    assert not _is_skipped("src")
    assert not _is_skipped("README.md")
    assert not _is_skipped("pyproject.toml")


# ── Empty workspace ─────────────────────────────────────────────────

def test_empty_workspace():
    r = root()
    idx = build_index(r)
    assert idx.root_name == r.name
    assert idx.total_files == 0
    assert idx.total_dirs == 0
    assert idx.source_files == ()
    assert idx.test_files == ()
    assert idx.config_files == ()
    assert idx.doc_files == ()
    assert idx.languages == ()
    assert idx.directories == ()


# ── Single file / classification ───────────────────────────────────

def test_single_source_file():
    r = root()
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 1
    assert idx.source_files == ("app.py",)
    assert idx.test_files == ()
    assert idx.config_files == ()
    assert idx.doc_files == ()
    assert idx.languages == ("Python",)


def test_pytest_config_file():
    r = root()
    (r / "conftest.py").write_text("", encoding="utf-8")
    (r / "pytest.ini").write_text("", encoding="utf-8")
    (r / "tox.ini").write_text("", encoding="utf-8")
    idx = build_index(r)
    assert len(idx.config_files) == 3
    assert idx.source_files == ()


def test_setup_py():
    r = root()
    (r / "setup.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.config_files == ("setup.py",)


def test_requirements_any():
    for name in ("requirements.txt", "requirements-dev.txt",
                 "requirements-test.txt"):
        r = root()
        (r / name).write_text("x", encoding="utf-8")
        idx = build_index(r)
        assert idx.config_files == (name,), name
        assert idx.total_files == 1


def test_dockerfile():
    r = root()
    (r / "Dockerfile").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.config_files == ("Dockerfile",)


def test_docker_compose():
    for name in ("docker-compose.yml", "docker-compose.yaml"):
        r = root()
        (r / name).write_text("x", encoding="utf-8")
        idx = build_index(r)
        assert idx.config_files == (name,)


def test_readme():
    r = root()
    (r / "README.md").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.doc_files == ("README.md",)


def test_readme_case_insensitive():
    r = root()
    (r / "readme").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.doc_files == ("readme",)


def test_changelog():
    r = root()
    (r / "CHANGELOG.md").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.doc_files == ("CHANGELOG.md",)


def test_license_no_extension():
    r = root()
    (r / "LICENSE").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.doc_files == ("LICENSE",)


def test_contributing():
    r = root()
    (r / "CONTRIBUTING.md").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.doc_files == ("CONTRIBUTING.md",)


def test_docs_dir_classified_as_doc():
    r = root()
    (r / "docs").mkdir()
    (r / "docs" / "guide.md").write_text("x", encoding="utf-8")
    (r / "docs" / "conf.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert len(idx.doc_files) == 2


def test_test_stem_prefix():
    r = root()
    (r / "test_app.py").write_text("x", encoding="utf-8")
    (r / "test_utils.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.test_files == ("test_app.py", "test_utils.py")
    assert idx.source_files == ()


def test_test_stem_suffix():
    r = root()
    (r / "app_test.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.test_files == ("app_test.py",)


def test_test_spec_suffix():
    r = root()
    (r / "app.spec.js").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.test_files == ("app.spec.js",)


def test_test_dot_test_suffix():
    r = root()
    (r / "app.test.ts").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.test_files == ("app.test.ts",)


def test_test_file_under_tests_dir():
    r = root()
    (r / "tests").mkdir()
    (r / "tests" / "helper.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.test_files == ("tests/helper.py",)
    assert idx.source_files == ()


def test_test_file_under_test_dir():
    r = root()
    (r / "test").mkdir()
    (r / "test" / "bar.js").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.test_files == ("test/bar.js",)


def test_non_test_under_non_tests_dir():
    r = root()
    (r / "src").mkdir()
    (r / "src" / "helper.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.source_files == ("src/helper.py",)
    assert idx.test_files == ()


# ── Language detection ──────────────────────────────────────────────

def test_python():
    r = root()
    (r / "a.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.languages == ("Python",)


def test_mixed_languages():
    r = root()
    (r / "a.py").write_text("x", encoding="utf-8")
    (r / "b.js").write_text("x", encoding="utf-8")
    (r / "c.rs").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.languages == ("JavaScript", "Python", "Rust")


def test_test_files_contribute_language():
    r = root()
    (r / "tests").mkdir()
    (r / "tests" / "test_a.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.languages == ("Python",)


def test_config_non_language_extension_ignored():
    r = root()
    (r / "pyproject.toml").write_text("x", encoding="utf-8")
    (r / "a.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.languages == ("Python",)


def test_no_extension_ignored():
    r = root()
    (r / "Makefile").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.languages == ()


# ── Skip directories (recursive exclusion) ─────────────────────────

def test_skip_pycache():
    r = root()
    pyc = r / "__pycache__"
    pyc.mkdir()
    (pyc / "module.cpython-312.pyc").write_bytes(b"\x00")
    (r / "main.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 1
    assert idx.source_files == ("main.py",)


def test_skip_git():
    r = root()
    git = r / ".git"
    refs = git / "refs"
    refs.mkdir(parents=True)
    (refs / "heads").mkdir()
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 1


def test_skip_sessions():
    r = root()
    (r / "sessions").mkdir()
    (r / "sessions" / "abc.json").write_text("x", encoding="utf-8")
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 1


def test_skip_churro_write_temp():
    r = root()
    scratch = r / ".churro-write-tmp-001"
    scratch.mkdir()
    (scratch / "scratch.py").write_text("x", encoding="utf-8")
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 1
    assert idx.source_files == ("app.py",)


def test_skip_egg_info_dir():
    r = root()
    egg = r / "myproject.egg-info"
    egg.mkdir()
    (egg / "SOURCES.txt").write_text("x", encoding="utf-8")
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 1


def test_skip_dotenv_file():
    r = root()
    (r / ".env").write_text("SECRET=x", encoding="utf-8")
    (r / ".env.example").write_text("SECRET=", encoding="utf-8")
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 2
    assert ".env" not in idx.config_files
    assert ".env" not in idx.source_files
    assert ".env" not in idx.doc_files
    assert ".env" not in idx.test_files
    assert ".env.example" in idx.config_files


def test_skip_node_modules():
    r = root()
    nm = r / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("x", encoding="utf-8")
    (r / "app.js").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 1


def test_skip_nested_pycache():
    r = root()
    nested = r / "src" / "__pycache__"
    nested.mkdir(parents=True)
    (nested / "mod.cpython-312.pyc").write_bytes(b"\x00")
    (r / "src").mkdir(parents=True, exist_ok=True)
    (r / "src" / "main.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 1
    assert idx.source_files == ("src/main.py",)


# ── Directories ─────────────────────────────────────────────────────

def test_directories_top_level_only():
    r = root()
    (r / "src").mkdir()
    (r / "src" / "nested").mkdir()
    (r / "src" / "nested" / "deep").mkdir()
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.directories == ("src",)


def test_directories_sorted():
    r = root()
    (r / "z_mod").mkdir()
    (r / "a_mod").mkdir()
    (r / "m_mod").mkdir()
    idx = build_index(r)
    assert idx.directories == ("a_mod", "m_mod", "z_mod")


def test_total_dirs_count():
    r = root()
    (r / "a").mkdir()
    (r / "b").mkdir()
    (r / "a" / "c").mkdir()
    (r / "__pycache__").mkdir()
    idx = build_index(r)
    # a, b, a/c are counted. __pycache__ is skipped.
    assert idx.total_dirs == 3


def test_directories_skip_node_modules():
    r = root()
    (r / "node_modules").mkdir()
    (r / "src").mkdir()
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.directories == ("src",)


# ── Deterministic output ───────────────────────────────────────────

def test_deterministic_index():
    r = root()
    (r / "z.py").write_text("x", encoding="utf-8")
    (r / "A.py").write_text("y", encoding="utf-8")
    (r / "tests").mkdir()
    (r / "tests" / "test_z.py").write_text("z", encoding="utf-8")
    idx1 = build_index(r)
    idx2 = build_index(r)
    assert idx1 == idx2


def test_deterministic_summary():
    r = root()
    (r / "z.py").write_text("x", encoding="utf-8")
    (r / "A.py").write_text("y", encoding="utf-8")
    s1 = build_summary(r)
    s2 = build_summary(r)
    assert s1 == s2


# ── Format summary ─────────────────────────────────────────────────

def test_format_summary_sections():
    r = root()
    (r / "app.py").write_text("x", encoding="utf-8")
    (r / "tests").mkdir()
    (r / "tests" / "test_app.py").write_text("x", encoding="utf-8")
    (r / "README.md").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert f"Project: {r.name}" in s
    assert "Source files:" in s
    assert "Test files:" in s
    assert "Docs:" in s


def test_format_summary_empty_categories_not_shown():
    r = root()
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert "Source files:" in s
    assert "Test files" not in s
    assert "Config" not in s
    assert "Docs" not in s


def test_format_summary_lines_under_cap():
    r = root()
    (r / "app.py").write_text("x", encoding="utf-8")
    (r / "README.md").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert len(s.split("\n")) <= 80


def test_format_summary_no_absolute_paths():
    r = root()
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert "C:" not in s
    assert "D:" not in s


def test_format_summary_language_in_header():
    r = root()
    (r / "a.py").write_text("x", encoding="utf-8")
    (r / "b.js").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert "Languages: JavaScript, Python" in s


# ── No accidental file reads ───────────────────────────────────────

def test_no_file_reads_during_index():
    r = root()
    (r / "app.py").write_text("content", encoding="utf-8")
    with patch.object(Path, "read_text", side_effect=OSError("no reads")), \
         patch.object(Path, "read_bytes", side_effect=OSError("no reads")):
        idx = build_index(r)
        assert idx.total_files == 1
        assert idx.source_files == ("app.py",)


# ── Large workspace (> _MAX_CATEGORY_FILES) ────────────────────────

def test_large_workspace_groups_source_files():
    r = root()
    for i in range(25):
        (r / f"mod{i}.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.total_files == 25
    assert len(idx.source_files) == 25
    s = format_summary(idx)
    assert "25 files" in s
    assert "Python: 25" in s


def test_format_summary_total_files_by_category():
    r = root()
    for i in range(3):
        (r / f"src{i}.py").write_text("x", encoding="utf-8")
    (r / "tests").mkdir()
    for i in range(2):
        (r / "tests" / f"test_{i}.py").write_text("x", encoding="utf-8")
    (r / "README.md").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert "3 source" in s
    assert "2 test" in s
    assert "1 doc" in s


# ── Mixed categories same workspace ────────────────────────────────

def test_full_python_project():
    r = root()
    (r / "setup.py").write_text("x", encoding="utf-8")
    (r / "pyproject.toml").write_text("x", encoding="utf-8")
    (r / "README.md").write_text("x", encoding="utf-8")
    (r / "src").mkdir()
    (r / "src" / "app.py").write_text("x", encoding="utf-8")
    (r / "tests").mkdir()
    (r / "tests" / "test_app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert idx.source_files == ("src/app.py",)
    assert idx.test_files == ("tests/test_app.py",)
    assert sorted(idx.config_files) == ["pyproject.toml", "setup.py"]
    assert idx.doc_files == ("README.md",)
    assert idx.directories == ("src", "tests")
    assert idx.languages == ("Python",)
    assert idx.total_files == 5
    assert idx.total_dirs == 2


# ── Doc files under docs/ dir ──────────────────────────────────────

def test_doc_under_docs_dir_any_extension():
    r = root()
    (r / "docs").mkdir()
    (r / "docs" / "api.py").write_text("x", encoding="utf-8")
    (r / "docs" / "guide.md").write_text("x", encoding="utf-8")
    idx = build_index(r)
    assert len(idx.doc_files) == 2
    assert "docs/api.py" in idx.doc_files
    assert "docs/guide.md" in idx.doc_files


# ── Relative only ──────────────────────────────────────────────────

def test_index_paths_are_posix_relative():
    r = root()
    (r / "src").mkdir()
    (r / "src" / "main.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    for rel in idx.source_files + idx.test_files + idx.config_files + idx.doc_files:
        assert "\\" not in rel
        assert "/" in rel or rel == Path(rel).name


# ── Format summary with no config/docs ────────────────────────────

def test_format_summary_no_config_no_docs():
    r = root()
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert "Config" not in s
    assert "Docs" not in s


def test_format_summary_config_present():
    r = root()
    (r / "pyproject.toml").write_text("x", encoding="utf-8")
    (r / "app.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert "Config:" in s


# ── count on pathologically large category triggers grouping ───────

def test_format_summary_grouping_for_large_test_category():
    r = root()
    (r / "tests").mkdir()
    for i in range(25):
        (r / "tests" / f"test_{i}.py").write_text("x", encoding="utf-8")
    idx = build_index(r)
    s = format_summary(idx)
    assert "25 files" in s


# ── full project with everything ───────────────────────────────────

def test_format_summary_full_project():
    r = root()
    (r / "pyproject.toml").write_text("x", encoding="utf-8")
    (r / "setup.py").write_text("x", encoding="utf-8")
    (r / "README.md").write_text("x", encoding="utf-8")
    (r / "src").mkdir()
    (r / "src" / "app.py").write_text("x", encoding="utf-8")
    (r / "src" / "app.js").write_text("x", encoding="utf-8")
    (r / "tests").mkdir()
    (r / "tests" / "test_app.py").write_text("x", encoding="utf-8")
    s = build_summary(r)
    assert "Project:" in s
    assert "Languages:" in s
    assert len(s.split("\n")) <= 80


# ── Step 21: repository structure intelligence ─────────────────────

def write_rel(root: Path, rel: str, content: str = "") -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def make_package(root: Path, *files: str) -> None:
    for rel in files:
        write_rel(root, rel)


def test_structure_empty_workspace():
    r = root()
    structure = build_structure(r)
    assert structure.packages == ()
    assert structure.entry_points == ()
    assert structure.modules == ()
    assert structure.imported_by == {}
    assert format_structure(structure) == ""


def test_structure_no_packages_empty_section():
    r = root()
    write_rel(r, "pricing.py", "def line_total(q, p):\n    return q * p\n")
    structure = build_structure(r)
    assert structure.packages == ()
    assert format_structure(structure) == ""


def test_structure_package_detection():
    r = root()
    make_package(r, "app/__init__.py", "app/mod.py")
    structure = build_structure(r)
    assert structure.packages == ("app",)


def test_structure_nested_packages():
    r = root()
    make_package(
        r,
        "pkg/__init__.py",
        "pkg/sub/__init__.py",
        "pkg/sub/deep/__init__.py",
        "pkg/sub/deep/mod.py",
    )
    structure = build_structure(r)
    assert structure.packages == ("pkg", "pkg/sub", "pkg/sub/deep")


def test_structure_entry_points():
    r = root()
    make_package(r, "app/__init__.py", "app/__main__.py")
    structure = build_structure(r)
    assert structure.entry_points == ("app/__main__.py",)


def test_structure_no_entry_points():
    r = root()
    make_package(r, "app/__init__.py", "app/mod.py")
    structure = build_structure(r)
    assert structure.entry_points == ()


def test_structure_line_count():
    r = root()
    write_rel(
        r,
        "app/mod.py",
        "import os\n\n\ndef f():\n    return 1\n\n\n",
    )
    structure = build_structure(r)
    module = structure.modules[0]
    assert module.path == "app/mod.py"
    assert module.lines == 7


def test_structure_binary_file_skipped():
    r = root()
    make_package(r, "app/__init__.py")
    target = r / "app" / "binmod.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\xff\xfe\x00\x01 broken")
    structure = build_structure(r)
    assert [m.path for m in structure.modules] == ["app/__init__.py"]
    assert structure.packages == ("app",)


def test_structure_simple_import():
    assert _extract_imports("import os\n") == ("os",)


def test_structure_from_import():
    assert _extract_imports("from pathlib import Path\n") == ("pathlib",)


def test_structure_multiline_from_import():
    content = (
        "from churro.tools import (\n"
        "    Tool,\n"
        "    ToolResult,\n"
        ")\n"
    )
    assert _extract_imports(content) == ("churro.tools",)


def test_structure_import_aliases_and_csv():
    content = "import os, sys\nimport pandas as pd\nimport collections as c\n"
    assert _extract_imports(content) == ("collections", "os", "pandas", "sys")


def test_structure_deduplicated_imports():
    content = "import os\nimport os as o\nfrom os import path\n"
    assert _extract_imports(content) == ("os",)


def test_structure_local_import_resolution():
    r = root()
    write_rel(r, "pricing.py", "def line_total(q, p):\n    return q * p\n")
    write_rel(r, "shopping_cart.py", "from pricing import line_total\n")
    structure = build_structure(r)
    module = next(m for m in structure.modules if m.path == "shopping_cart.py")
    assert module.imports_local == ("pricing.py",)
    assert "pricing" in module.imports


def test_structure_dotted_local_import_resolution():
    r = root()
    make_package(
        r,
        "pkg/__init__.py",
        "pkg/tools/__init__.py",
        "pkg/tools/tool.py",
    )
    write_rel(
        r,
        "pkg/consumer.py",
        "from pkg.tools.tool import Tool\n",
    )
    write_rel(r, "pkg/tools/tool.py", "class Tool:\n    pass\n")
    structure = build_structure(r)
    module = next(m for m in structure.modules if m.path == "pkg/consumer.py")
    assert module.imports_local == ("pkg/tools/tool.py",)


def test_structure_package_import_resolution():
    r = root()
    make_package(r, "pkg/__init__.py")
    write_rel(r, "entry.py", "import pkg\n")
    structure = build_structure(r)
    module = next(m for m in structure.modules if m.path == "entry.py")
    assert module.imports_local == ("pkg/__init__.py",)


def test_structure_external_import_not_resolved():
    r = root()
    make_package(r, "app/__init__.py")
    write_rel(r, "app/mod.py", "import os\nimport pytest\nimport openai\n")
    structure = build_structure(r)
    module = next(m for m in structure.modules if m.path == "app/mod.py")
    assert module.imports_local == ()
    assert module.imports == ("openai", "os", "pytest")


def test_structure_unknown_import_no_failure():
    r = root()
    make_package(r, "app/__init__.py")
    write_rel(r, "app/mod.py", "import totally_unknown_thing\n")
    structure = build_structure(r)
    assert structure.modules[0].imports_local == ()


def test_structure_top_level_classes():
    content = "class Foo(BaseModel):\n    pass\n\nclass Bar:\n    pass\n"
    classes, functions = _extract_symbols(content)
    assert classes == ("Foo", "Bar")
    assert functions == ()


def test_structure_top_level_functions():
    content = "def one():\n    pass\n\nasync def two():\n    pass\n"
    classes, functions = _extract_symbols(content)
    assert classes == ()
    assert functions == ("one", "two")


def test_structure_nested_symbols_excluded():
    content = (
        "class Outer:\n"
        "    class Inner:\n"
        "        pass\n"
        "    def method(self):\n"
        "        pass\n"
        "        def inner_fn(self):\n"
        "            pass\n"
        "\n"
        "def top():\n"
        "    def nested():\n"
        "        pass\n"
    )
    classes, functions = _extract_symbols(content)
    assert classes == ("Outer",)
    assert functions == ("top",)


def test_structure_imported_by():
    r = root()
    write_rel(r, "base.py", "def b():\n    pass\n")
    write_rel(r, "left.py", "from base import b\n")
    write_rel(r, "right.py", "import base\n")
    structure = build_structure(r)
    assert structure.imported_by["base.py"] == ("left.py", "right.py")


def test_structure_imported_by_sorted():
    r = root()
    write_rel(r, "base.py", "")
    write_rel(r, "z_user.py", "import base\n")
    write_rel(r, "a_user.py", "import base\n")
    structure = build_structure(r)
    assert structure.imported_by["base.py"] == ("a_user.py", "z_user.py")


def test_structure_centrality_order_in_format():
    r = root()
    make_package(r, "pkg/__init__.py")
    write_rel(r, "central.py", "")
    write_rel(r, "one.py", "import central\n")
    write_rel(r, "two.py", "import central\n")
    write_rel(r, "solo.py", "")
    write_rel(r, "user.py", "import solo\n")
    structure = build_structure(r)
    text = format_structure(structure)
    assert text.index("central.py: imported by 2 files") < text.index(
        "solo.py: imported by 1 file"
    )


def test_structure_format_sections():
    r = root()
    make_package(r, "app/__init__.py", "app/__main__.py")
    write_rel(r, "app/mod.py", "import os\n")
    text = format_structure(build_structure(r))
    assert "Packages:" in text
    assert "app/" in text
    assert "Entry points:" in text
    assert "app/__main__.py" in text
    assert "Most-imported" not in text
    assert "Modules:" in text
    assert "app/mod.py" in text


def test_structure_format_line_cap():
    r = root()
    make_package(r, "pkg/__init__.py")
    for i in range(70):
        write_rel(r, f"pkg/mod{i}.py", f"def f{i}():\n    pass\n")
    structure = build_structure(r)
    text = format_structure(structure)
    lines = text.splitlines()
    assert len(lines) <= 60
    assert lines[-1] == "... structure truncated"


def test_structure_format_no_absolute_paths():
    r = root()
    make_package(r, "app/__init__.py")
    write_rel(r, "app/mod.py", "import os\n")
    text = format_structure(build_structure(r))
    assert "C:" not in text
    assert "D:" not in text


def test_structure_deterministic():
    r = root()
    make_package(r, "pkg/__init__.py")
    write_rel(r, "pkg/a.py", "import os\nfrom b import x\n\ndef f():\n    pass\n")
    write_rel(r, "pkg/b.py", "def x():\n    pass\n")
    s1 = build_structure(r)
    s2 = build_structure(r)
    assert s1 == s2
    assert format_structure(s1) == format_structure(s2)


def test_structure_prebuilt_index_reuse():
    import churro.repo_index as repo_index

    r = root()
    make_package(r, "pkg/__init__.py")
    write_rel(r, "pkg/mod.py", "import os\n")
    index = repo_index.build_index(r)
    with patch.object(
        repo_index, "build_index", side_effect=AssertionError("must not re-walk")
    ):
        structure = build_structure(r, index=index)
    assert structure.packages == ("pkg",)
    assert any(m.path == "pkg/mod.py" for m in structure.modules)


def test_structure_non_python_project_fallback():
    r = root()
    write_rel(r, "app.js", "const x = 1;\n")
    write_rel(r, "tests/test_app.js", "describe('x', () => {})\n")
    structure = build_structure(r)
    assert structure.packages == ()
    assert structure.modules == ()
    assert format_structure(structure) == ""


def test_structure_python_without_packages_fallback():
    r = root()
    write_rel(r, "pricing.py", "def line_total(q, p):\n    return q * p\n")
    write_rel(r, "shopping_cart.py", "from pricing import line_total\n")
    write_rel(r, "tests/test_pricing.py", "from pricing import line_total\n")
    structure = build_structure(r)
    assert structure.packages == ()
    assert format_structure(structure) == ""


def test_structure_mixed_source_test_imports():
    r = root()
    write_rel(r, "pricing.py", "def line_total(q, p):\n    return q * p\n")
    write_rel(r, "tests/__init__.py", "")
    write_rel(r, "tests/test_pricing.py", "from pricing import line_total\n")
    structure = build_structure(r)
    assert structure.packages == ("tests",)
    test_module = next(
        m for m in structure.modules if m.path == "tests/test_pricing.py"
    )
    assert test_module.imports_local == ("pricing.py",)
    assert structure.imported_by["pricing.py"] == ("tests/test_pricing.py",)


def test_structure_only_reads_python_files():
    import pathlib

    r = root()
    make_package(r, "pkg/__init__.py")
    write_rel(r, "pkg/app.py", "x = 1\n")
    write_rel(r, "pkg/app.js", "x = 1;\n")
    write_rel(r, "README.md", "# docs\n")
    write_rel(r, "pyproject.toml", "[project]\n")
    read_suffixes: list[str] = []
    original = pathlib.Path.read_text

    def tracking(self, *args, **kwargs):
        read_suffixes.append(self.suffix)
        return original(self, *args, **kwargs)

    with patch("pathlib.Path.read_text", tracking):
        build_structure(r)
    assert read_suffixes, "expected at least one Python file read"
    assert all(suffix == ".py" for suffix in read_suffixes), read_suffixes


def test_structure_module_info_fields():
    r = root()
    make_package(r, "pkg/__init__.py")
    write_rel(
        r,
        "pkg/mod.py",
        "import os\nfrom pathlib import Path\n\nclass K:\n    pass\n\ndef f():\n    pass\n",
    )
    module = next(m for m in build_structure(r).modules if m.path == "pkg/mod.py")
    assert module.lines == 8
    assert module.classes == ("K",)
    assert module.functions == ("f",)
    assert module.imports == ("os", "pathlib")
    assert module.imports_local == ()


def test_structure_source_and_test_modules_analyzed():
    r = root()
    make_package(r, "pkg/__init__.py", "tests/__init__.py")
    write_rel(r, "pkg/mod.py", "def f():\n    pass\n")
    write_rel(r, "tests/test_mod.py", "from pkg.mod import f\n")
    structure = build_structure(r)
    paths = [m.path for m in structure.modules]
    assert "pkg/mod.py" in paths
    assert "tests/test_mod.py" in paths


def test_build_full_summary_combines():
    r = root()
    make_package(r, "pkg/__init__.py")
    write_rel(r, "pkg/mod.py", "import os\n")
    text = build_full_summary(r)
    assert "Project:" in text
    assert "Packages:" in text
    assert "pkg/" in text


def test_structure_real_churro():
    repo = Path(__file__).resolve().parents[1]
    structure = build_structure(repo)
    assert "churro" in structure.packages
    assert "churro/__main__.py" in structure.entry_points
    main_module = next(m for m in structure.modules if m.path == "churro/main.py")
    assert any(path.endswith(".py") for path in main_module.imports_local)
    text = format_structure(structure)
    assert "Packages:" in text
    assert "churro/main.py" in text
    assert "C:" not in text
    assert "D:" not in text


def test_structure_imported_by_unknown_writer():
    r = root()
    make_package(r, "pkg/__init__.py")
    write_rel(r, "pkg/base.py", "")
    write_rel(r, "pkg/user.py", "from pkg.base import x\n")
    write_rel(r, "pkg/orphan.py", "")
    structure = build_structure(r)
    assert "pkg/base.py" in structure.imported_by
    assert structure.imported_by["pkg/base.py"] == ("pkg/user.py",)
    assert structure.imported_by["pkg/orphan.py"] == ()


TEST_FUNCTIONS = [
    # Skip logic
    test_skip_exact_match,
    test_skip_prefix,
    test_skip_egg_info,
    test_normal_name_not_skipped,
    # Empty workspace
    test_empty_workspace,
    # Classification
    test_single_source_file,
    test_pytest_config_file,
    test_setup_py,
    test_requirements_any,
    test_dockerfile,
    test_docker_compose,
    test_readme,
    test_readme_case_insensitive,
    test_changelog,
    test_license_no_extension,
    test_contributing,
    test_docs_dir_classified_as_doc,
    test_test_stem_prefix,
    test_test_stem_suffix,
    test_test_spec_suffix,
    test_test_dot_test_suffix,
    test_test_file_under_tests_dir,
    test_test_file_under_test_dir,
    test_non_test_under_non_tests_dir,
    # Language detection
    test_python,
    test_mixed_languages,
    test_test_files_contribute_language,
    test_config_non_language_extension_ignored,
    test_no_extension_ignored,
    # Skip directories
    test_skip_pycache,
    test_skip_git,
    test_skip_sessions,
    test_skip_churro_write_temp,
    test_skip_egg_info_dir,
    test_skip_dotenv_file,
    test_skip_node_modules,
    test_skip_nested_pycache,
    # Directories
    test_directories_top_level_only,
    test_directories_sorted,
    test_total_dirs_count,
    test_directories_skip_node_modules,
    # Deterministic
    test_deterministic_index,
    test_deterministic_summary,
    # Format summary
    test_format_summary_sections,
    test_format_summary_empty_categories_not_shown,
    test_format_summary_lines_under_cap,
    test_format_summary_no_absolute_paths,
    test_format_summary_language_in_header,
    # No file reads
    test_no_file_reads_during_index,
    # Large workspace
    test_large_workspace_groups_source_files,
    test_format_summary_total_files_by_category,
    # Mixed categories
    test_full_python_project,
    # Doc files under docs/
    test_doc_under_docs_dir_any_extension,
    # Relative only
    test_index_paths_are_posix_relative,
    # Summary present/absent
    test_format_summary_no_config_no_docs,
    test_format_summary_config_present,
    # Grouping
    test_format_summary_grouping_for_large_test_category,
    # Full project
    test_format_summary_full_project,
    # ── Step 21 structure intelligence ──
    test_structure_empty_workspace,
    test_structure_no_packages_empty_section,
    test_structure_package_detection,
    test_structure_nested_packages,
    test_structure_entry_points,
    test_structure_no_entry_points,
    test_structure_line_count,
    test_structure_binary_file_skipped,
    test_structure_simple_import,
    test_structure_from_import,
    test_structure_multiline_from_import,
    test_structure_import_aliases_and_csv,
    test_structure_deduplicated_imports,
    test_structure_local_import_resolution,
    test_structure_dotted_local_import_resolution,
    test_structure_package_import_resolution,
    test_structure_external_import_not_resolved,
    test_structure_unknown_import_no_failure,
    test_structure_top_level_classes,
    test_structure_top_level_functions,
    test_structure_nested_symbols_excluded,
    test_structure_imported_by,
    test_structure_imported_by_sorted,
    test_structure_centrality_order_in_format,
    test_structure_format_sections,
    test_structure_format_line_cap,
    test_structure_format_no_absolute_paths,
    test_structure_deterministic,
    test_structure_prebuilt_index_reuse,
    test_structure_non_python_project_fallback,
    test_structure_python_without_packages_fallback,
    test_structure_mixed_source_test_imports,
    test_structure_only_reads_python_files,
    test_structure_module_info_fields,
    test_structure_source_and_test_modules_analyzed,
    test_build_full_summary_combines,
    test_structure_real_churro,
    test_structure_imported_by_unknown_writer,
]


def main() -> None:
    failures = 0
    for fn in TEST_FUNCTIONS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    if failures:
        raise SystemExit(f"\n{failures} test(s) failed")
    print(f"\nAll {len(TEST_FUNCTIONS)} tests passed.")


if __name__ == "__main__":
    main()
