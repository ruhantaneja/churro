import os
import tempfile
from pathlib import Path

from churro.tools import ListFilesTool, ReadFileTool, ToolRegistry
from churro.tools.tool import ToolResult


def make_root() -> Path:
    return Path(tempfile.mkdtemp())


def make_tool(root: Path | None = None) -> tuple[ListFilesTool, Path]:
    root = root or make_root()
    return ListFilesTool(root), root


def split_lines(result: ToolResult) -> list[str]:
    return result.output.split("\n") if result.output else []


def test_empty_workspace():
    tool, _ = make_tool()
    result = tool.execute({})
    assert result.success is True
    assert result.output == ""
    recursive = tool.execute({"path": ".", "recursive": True})
    assert recursive.success is True
    assert recursive.output == ""


def test_files_in_workspace_root():
    tool, root = make_tool()
    (root / "a.txt").write_text("x", encoding="utf-8")
    (root / "b.txt").write_text("y", encoding="utf-8")
    result = tool.execute({"path": "."})
    assert result.success is True
    assert split_lines(result) == ["FILE  a.txt", "FILE  b.txt"]


def test_directories_in_workspace_root():
    tool, root = make_tool()
    (root / "d1").mkdir()
    (root / "d2").mkdir()
    result = tool.execute({"path": "."})
    assert result.success is True
    assert split_lines(result) == ["DIR   d1/", "DIR   d2/"]


def test_mixed_files_and_directories():
    tool, root = make_tool()
    (root / "sub").mkdir()
    (root / "z.txt").write_text("x", encoding="utf-8")
    result = tool.execute({"path": "."})
    assert split_lines(result) == ["DIR   sub/", "FILE  z.txt"]


def test_default_path_is_workspace_root():
    tool, root = make_tool()
    (root / "only.txt").write_text("x", encoding="utf-8")
    explicit = tool.execute({"path": "."})
    implicit = tool.execute({})
    assert explicit.output == "FILE  only.txt"
    assert implicit.output == explicit.output


def test_listing_a_nested_directory():
    tool, root = make_tool()
    (root / "sub").mkdir()
    (root / "sub" / "a.txt").write_text("x", encoding="utf-8")
    (root / "sub" / "inner").mkdir()
    result = tool.execute({"path": "sub"})
    assert result.success is True
    assert split_lines(result) == ["FILE  sub/a.txt", "DIR   sub/inner/"]


def test_recursive_listing():
    tool, root = make_tool()
    (root / "r.txt").write_text("x", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "c.txt").write_text("y", encoding="utf-8")
    (root / "sub" / "deep").mkdir()
    (root / "sub" / "deep" / "d.txt").write_text("z", encoding="utf-8")

    result = tool.execute({"path": ".", "recursive": True})
    assert result.success is True
    assert split_lines(result) == [
        "FILE  r.txt",
        "DIR   sub/",
        "FILE  sub/c.txt",
        "DIR   sub/deep/",
        "FILE  sub/deep/d.txt",
    ]


def test_non_recursive_excludes_descendants():
    tool, root = make_tool()
    (root / "r.txt").write_text("x", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "deep").mkdir()
    (root / "sub" / "deep" / "d.txt").write_text("z", encoding="utf-8")

    result = tool.execute({"path": ".", "recursive": False})
    assert result.success is True
    assert split_lines(result) == ["FILE  r.txt", "DIR   sub/"]
    assert "deep" not in result.output
    assert "d.txt" not in result.output


def test_deterministic_ordering():
    tool, root = make_tool()
    (root / "b2.txt").write_text("x", encoding="utf-8")
    (root / "A1.txt").write_text("y", encoding="utf-8")
    (root / "m").mkdir()
    (root / "m" / "c.txt").write_text("z", encoding="utf-8")

    first = tool.execute({"recursive": True})
    second = tool.execute({"recursive": True})
    assert first.output == second.output

    lines = split_lines(first)
    assert "FILE  A1.txt" in lines
    assert "FILE  b2.txt" in lines
    assert lines.index("FILE  A1.txt") < lines.index("FILE  b2.txt")


def test_missing_directory():
    tool, _ = make_tool()
    result = tool.execute({"path": "no_such_dir"})
    assert result.success is False
    assert "does not exist" in result.error


def test_path_pointing_to_a_file():
    tool, root = make_tool()
    (root / "f.txt").write_text("x", encoding="utf-8")
    result = tool.execute({"path": "f.txt"})
    assert result.success is False
    assert "not a directory" in result.error


def test_traversal_escape_rejected():
    root = make_root()
    outside = root.parent / f"outside_{root.name}.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        result = ListFilesTool(root).execute({"path": "../" + outside.name})
        assert result.success is False
        assert "outside" in result.error
    finally:
        outside.unlink(missing_ok=True)


def test_absolute_path_outside_workspace_rejected():
    tool, _ = make_tool()
    outside = make_root()
    result = tool.execute({"path": str(outside)})
    assert result.success is False
    assert "outside" in result.error


def test_symlink_escape_rejected():
    root = make_root()
    outside = make_root()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        return

    try:
        direct = ListFilesTool(root).execute({"path": "link"})
        assert direct.success is False
        assert "outside" in direct.error

        recursive = ListFilesTool(root).execute({"path": ".", "recursive": True})
        assert recursive.success is True
        assert "link/" in recursive.output
        assert "secret.txt" not in recursive.output
    finally:
        outside.unlink(missing_ok=True)


def test_absolute_path_inside_workspace_allowed():
    tool, root = make_tool()
    (root / "yes.txt").write_text("x", encoding="utf-8")
    result = tool.execute({"path": str(root)})
    assert result.success is True
    assert split_lines(result) == ["FILE  yes.txt"]


def test_unicode_filenames():
    tool, root = make_tool()
    (root / "héllo.txt").write_text("x", encoding="utf-8")
    (root / "日本語.txt").write_text("y", encoding="utf-8")

    first = tool.execute({"path": "."})
    second = tool.execute({"path": "."})
    assert first.output == second.output
    lines = split_lines(first)
    assert "FILE  héllo.txt" in lines
    assert "FILE  日本語.txt" in lines


def test_output_contains_relative_paths_only():
    tool, root = make_tool()
    (root / "sub").mkdir()
    (root / "sub" / "x.txt").write_text("x", encoding="utf-8")

    result = tool.execute({"path": ".", "recursive": True})
    assert result.success is True
    assert str(root) not in result.output
    assert "\\" not in result.output

    for line in split_lines(result):
        assert line.startswith(("FILE  ", "DIR   "))
        part = line[6:]
        assert not os.path.isabs(part)


def test_toolresult_success_and_failure_structure():
    tool, root = make_tool()
    (root / "a.txt").write_text("x", encoding="utf-8")

    ok = tool.execute({"path": "."})
    assert isinstance(ok, ToolResult)
    assert ok.success is True
    assert ok.output == "FILE  a.txt"
    assert ok.error == ""

    missing = tool.execute({"path": "nope"})
    assert isinstance(missing, ToolResult)
    assert missing.success is False
    assert missing.output == ""
    assert isinstance(missing.error, str)
    assert missing.error


def test_registry_dispatch():
    root = make_root()
    (root / "a.txt").write_text("hello from a", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(ListFilesTool(root))
    registry.register(ReadFileTool(root))

    listed = registry.execute("list_files", {"path": "."})
    assert listed.success is True
    assert listed.output == "FILE  a.txt"

    read = registry.execute("read_file", {"path": "a.txt"})
    assert read.success is True
    assert read.output == "hello from a"


def test_does_not_modify_filesystem():
    tool, root = make_tool()
    (root / "keep.txt").write_text("stable", encoding="utf-8")
    (root / "d").mkdir()

    before_names = sorted(p.name for p in root.iterdir())
    before_content = (root / "keep.txt").read_bytes()

    result = tool.execute({"path": ".", "recursive": True})
    assert result.success is True

    after_names = sorted(p.name for p in root.iterdir())
    assert after_names == before_names
    assert (root / "keep.txt").read_bytes() == before_content


def test_invalid_arguments_controlled():
    tool, _ = make_tool()

    bad_path = tool.execute({"path": 123})
    assert bad_path.success is False
    assert "path" in bad_path.error

    unknown = tool.execute({"path": ".", "bogus": 1})
    assert unknown.success is False
    assert "bogus" in unknown.error


TEST_FUNCTIONS = [
    test_empty_workspace,
    test_files_in_workspace_root,
    test_directories_in_workspace_root,
    test_mixed_files_and_directories,
    test_default_path_is_workspace_root,
    test_listing_a_nested_directory,
    test_recursive_listing,
    test_non_recursive_excludes_descendants,
    test_deterministic_ordering,
    test_missing_directory,
    test_path_pointing_to_a_file,
    test_traversal_escape_rejected,
    test_absolute_path_outside_workspace_rejected,
    test_symlink_escape_rejected,
    test_absolute_path_inside_workspace_allowed,
    test_unicode_filenames,
    test_output_contains_relative_paths_only,
    test_toolresult_success_and_failure_structure,
    test_registry_dispatch,
    test_does_not_modify_filesystem,
    test_invalid_arguments_controlled,
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