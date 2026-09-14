"""Tests for the repository index (churro.repo_index).

Verifies workspace containment, deterministic output, classification
rules, skip logic, and the prompt-friendly summary formatter.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from churro.repo_index import (
    RepoIndex,
    _is_skipped,
    build_index,
    build_summary,
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
