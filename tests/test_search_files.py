import tempfile
from pathlib import Path

from churro.tools import ReadFileTool, SearchFilesTool, ToolRegistry
from churro.tools.tool import ToolResult


def make_root() -> Path:
    return Path(tempfile.mkdtemp())


def make_tool(root: Path | None = None) -> tuple[SearchFilesTool, Path]:
    root = root or make_root()
    return SearchFilesTool(root), root


def split_lines(result: ToolResult) -> list[str]:
    return result.output.split("\n") if result.output else []


def test_single_matching_file():
    tool, root = make_tool()
    (root / "a.txt").write_text("needle in a haystack\nnothing here\n", encoding="utf-8")
    result = tool.execute({"query": "needle"})
    assert result.success is True
    assert split_lines(result) == ["a.txt:1:needle in a haystack"]


def test_multiple_matching_files():
    tool, root = make_tool()
    (root / "a.txt").write_text("first match here\n", encoding="utf-8")
    (root / "b.txt").write_text("second match here\n", encoding="utf-8")
    result = tool.execute({"query": "match"})
    assert result.success is True
    assert split_lines(result) == [
        "a.txt:1:first match here",
        "b.txt:1:second match here",
    ]


def test_multiple_matches_within_one_file():
    tool, root = make_tool()
    lines = ["zero\n", "hit one\n", "two\n", "hit two\n", "four\n", "five\n"]
    (root / "m.txt").write_text("".join(lines), encoding="utf-8")
    result = tool.execute({"query": "hit"})
    assert split_lines(result) == ["m.txt:2:hit one", "m.txt:4:hit two"]


def test_no_matches():
    tool, root = make_tool()
    (root / "a.txt").write_text("nothing interesting\n", encoding="utf-8")
    result = tool.execute({"query": "zzzzz"})
    assert result.success is True
    assert result.output == ""


def test_case_sensitive_behavior():
    tool, root = make_tool()
    (root / "c.txt").write_text("Apple pie\napple pie\nAn Apple on top\n", encoding="utf-8")
    result = tool.execute({"query": "Apple"})
    assert result.success is True
    assert split_lines(result) == [
        "c.txt:1:Apple pie",
        "c.txt:3:An Apple on top",
    ]


def test_default_path_is_workspace_root():
    tool, root = make_tool()
    (root / "root.txt").write_text("query here\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "nested.txt").write_text("query here too\n", encoding="utf-8")

    explicit = tool.execute({"query": "query", "path": "."})
    implicit = tool.execute({"query": "query"})
    assert explicit.success is True
    assert implicit.success is True
    assert implicit.output == explicit.output
    assert "root.txt" in implicit.output
    assert "sub/nested.txt" in implicit.output


def test_search_inside_nested_directory():
    tool, root = make_tool()
    (root / "top.txt").write_text("needle\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "inner.txt").write_text("needle inside\n", encoding="utf-8")

    result = tool.execute({"query": "needle", "path": "sub"})
    assert result.success is True
    assert split_lines(result) == ["sub/inner.txt:1:needle inside"]


def test_recursive_search():
    tool, root = make_tool()
    (root / "l1.txt").write_text("find me\n", encoding="utf-8")
    (root / "d").mkdir()
    (root / "d" / "l2.txt").write_text("find me too\n", encoding="utf-8")
    (root / "d" / "e").mkdir()
    (root / "d" / "e" / "l3.txt").write_text("find me as well\n", encoding="utf-8")

    result = tool.execute({"query": "find me", "recursive": True})
    assert split_lines(result) == [
        "d/e/l3.txt:1:find me as well",
        "d/l2.txt:1:find me too",
        "l1.txt:1:find me",
    ]


def test_non_recursive_search():
    tool, root = make_tool()
    (root / "l1.txt").write_text("find me\n", encoding="utf-8")
    (root / "d").mkdir()
    (root / "d" / "l2.txt").write_text("find me too\n", encoding="utf-8")

    result = tool.execute({"query": "find me", "recursive": False})
    assert result.success is True
    assert split_lines(result) == ["l1.txt:1:find me"]


def test_max_results_limit():
    tool, root = make_tool()
    (root / "many.txt").write_text(
        "".join(f"needle line {i}\n" for i in range(10)), encoding="utf-8"
    )
    result = tool.execute({"query": "needle", "max_results": 3})
    assert result.success is True
    lines = split_lines(result)
    match_lines = [line for line in lines if not line.startswith("...")]
    assert match_lines == [
        "many.txt:1:needle line 0",
        "many.txt:2:needle line 1",
        "many.txt:3:needle line 2",
    ]
    assert len(match_lines) == 3


def test_truncation_indication():
    tool, root = make_tool()
    (root / "many.txt").write_text(
        "".join(f"needle line {i}\n" for i in range(5)), encoding="utf-8"
    )
    result = tool.execute({"query": "needle", "max_results": 3})
    lines = split_lines(result)
    assert len(lines) == 4
    assert lines == [
        "many.txt:1:needle line 0",
        "many.txt:2:needle line 1",
        "many.txt:3:needle line 2",
        "... results truncated at 3 matches",
    ]


def test_no_truncation_when_exact_match_count():
    tool, root = make_tool()
    (root / "three.txt").write_text("hit\nhit again\nhit thrice\n", encoding="utf-8")
    result = tool.execute({"query": "hit", "max_results": 3})
    lines = split_lines(result)
    assert len(lines) == 3
    assert all(not line.startswith("...") for line in lines)
    assert "..." not in result.output


def test_deterministic_result_ordering():
    tool, root = make_tool()
    (root / "b.txt").write_text("hit\n", encoding="utf-8")
    (root / "A.txt").write_text("hit\n", encoding="utf-8")
    (root / "m").mkdir()
    (root / "m" / "c.txt").write_text("hit\n", encoding="utf-8")

    first = tool.execute({"query": "hit", "recursive": True})
    second = tool.execute({"query": "hit", "recursive": True})
    assert first.success is True
    assert first.output == second.output

    lines = split_lines(first)
    assert lines[0].startswith("A.txt")
    assert lines[1].startswith("b.txt")
    assert lines[2].startswith("m/c.txt")


def test_missing_search_path():
    tool, _ = make_tool()
    result = tool.execute({"query": "x", "path": "no_such_dir"})
    assert result.success is False
    assert "does not exist" in result.error


def test_search_path_is_a_file():
    tool, root = make_tool()
    (root / "f.txt").write_text("x\n", encoding="utf-8")
    result = tool.execute({"query": "x", "path": "f.txt"})
    assert result.success is False
    assert "not a directory" in result.error


def test_traversal_escape_rejected():
    root = make_root()
    outside = root.parent / f"leak_{root.name}.txt"
    outside.write_text("secret query\n", encoding="utf-8")
    try:
        result = SearchFilesTool(root).execute({"query": "secret query", "path": "../"})
        assert result.success is False
        assert "outside" in result.error
    finally:
        outside.unlink(missing_ok=True)


def test_absolute_path_outside_workspace_rejected():
    tool, _ = make_tool()
    outside = make_root()
    result = tool.execute({"query": "x", "path": str(outside)})
    assert result.success is False
    assert "outside" in result.error


def test_absolute_path_inside_workspace_allowed():
    tool, root = make_tool()
    (root / "in.txt").write_text("needle\n", encoding="utf-8")
    result = tool.execute({"query": "needle", "path": str(root)})
    assert result.success is True
    assert split_lines(result) == ["in.txt:1:needle"]


def test_symlink_escape_protection():
    root = make_root()
    outside = make_root()
    (outside / "secret.txt").write_text("forbidden needle\n", encoding="utf-8")
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        return

    try:
        direct = SearchFilesTool(root).execute({"query": "forbidden", "path": "link"})
        assert direct.success is False
        assert "outside" in direct.error

        recursive = SearchFilesTool(root).execute(
            {"query": "forbidden needle", "recursive": True}
        )
        assert recursive.success is True
        assert "secret.txt" not in recursive.output
        assert "forbidden" not in recursive.output
    finally:
        outside.unlink(missing_ok=True)


def test_unicode_file_contents():
    tool, root = make_tool()
    (root / "uni.txt").write_text(
        "日本語の行\nhéllo wörld\nもう一つの行\n", encoding="utf-8"
    )
    result = tool.execute({"query": "日本"})
    assert result.success is True
    assert split_lines(result) == ["uni.txt:1:日本語の行"]

    other = tool.execute({"query": "héllo"})
    assert split_lines(other) == ["uni.txt:2:héllo wörld"]


def test_invalid_utf8_file_is_skipped():
    tool, root = make_tool()
    (root / "bin.dat").write_bytes(bytes(range(256)))
    (root / "ok.txt").write_text("needle here\n", encoding="utf-8")

    result = tool.execute({"query": "needle"})
    assert result.success is True
    lines = split_lines(result)
    assert lines == ["ok.txt:1:needle here"]
    assert all("bin.dat" not in line for line in lines)


def test_empty_query_rejected():
    tool, _ = make_tool()
    result = tool.execute({"query": ""})
    assert result.success is False
    assert result.error


def test_invalid_max_results_rejected():
    tool, _ = make_tool()
    for bad in (0, -3, 1.5, "many"):
        result = tool.execute({"query": "x", "max_results": bad})
        assert result.success is False, f"expected failure for max_results={bad!r}"
        assert result.error


def test_toolresult_structure():
    tool, root = make_tool()
    (root / "s.txt").write_text("needle\n", encoding="utf-8")

    ok = tool.execute({"query": "needle"})
    assert isinstance(ok, ToolResult)
    assert ok.success is True
    assert ok.output == "s.txt:1:needle"
    assert ok.error == ""

    missing = tool.execute({"query": "needle", "path": "nope"})
    assert isinstance(missing, ToolResult)
    assert missing.success is False
    assert missing.output == ""
    assert isinstance(missing.error, str)
    assert missing.error


def test_registry_dispatch():
    root = make_root()
    (root / "greet.txt").write_text("hello world\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(SearchFilesTool(root))
    registry.register(ReadFileTool(root))

    found = registry.execute("search_files", {"query": "hello"})
    assert found.success is True
    assert found.output == "greet.txt:1:hello world"

    read = registry.execute("read_file", {"path": "greet.txt"})
    assert read.success is True
    assert read.output == "hello world\n"


def test_filesystem_not_modified():
    tool, root = make_tool()
    (root / "keep.txt").write_text("needle\nsecond line\n", encoding="utf-8")

    before_names = sorted(p.name for p in root.iterdir())
    before_content = (root / "keep.txt").read_bytes()

    result = tool.execute({"query": "needle", "recursive": True})
    assert result.success is True

    assert sorted(p.name for p in root.iterdir()) == before_names
    assert (root / "keep.txt").read_bytes() == before_content


TEST_FUNCTIONS = [
    test_single_matching_file,
    test_multiple_matching_files,
    test_multiple_matches_within_one_file,
    test_no_matches,
    test_case_sensitive_behavior,
    test_default_path_is_workspace_root,
    test_search_inside_nested_directory,
    test_recursive_search,
    test_non_recursive_search,
    test_max_results_limit,
    test_truncation_indication,
    test_no_truncation_when_exact_match_count,
    test_deterministic_result_ordering,
    test_missing_search_path,
    test_search_path_is_a_file,
    test_traversal_escape_rejected,
    test_absolute_path_outside_workspace_rejected,
    test_absolute_path_inside_workspace_allowed,
    test_symlink_escape_protection,
    test_unicode_file_contents,
    test_invalid_utf8_file_is_skipped,
    test_empty_query_rejected,
    test_invalid_max_results_rejected,
    test_toolresult_structure,
    test_registry_dispatch,
    test_filesystem_not_modified,
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