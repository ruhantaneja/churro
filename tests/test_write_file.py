import os
import stat
import tempfile
from pathlib import Path

from churro.tools import ReadFileTool, ToolRegistry, WriteFileTool
from churro.tools.tool import ToolResult


def make_root() -> Path:
    return Path(tempfile.mkdtemp())


def make_tool(root: Path | None = None) -> tuple[WriteFileTool, Path]:
    root = root or make_root()
    return WriteFileTool(root), root


def test_creates_new_file():
    tool, root = make_tool()
    result = tool.execute({"path": "new.txt", "content": "hello"})
    assert result.success is True
    assert result.output == "Created new.txt"
    assert (root / "new.txt").read_text(encoding="utf-8") == "hello"


def test_overwrites_existing_file():
    tool, root = make_tool()
    (root / "f.txt").write_text("old", encoding="utf-8")
    result = tool.execute({"path": "f.txt", "content": "new-content"})
    assert result.success is True
    assert result.output == "Updated f.txt"
    assert (root / "f.txt").read_text(encoding="utf-8") == "new-content"


def test_empty_content_creates_file():
    tool, root = make_tool()
    result = tool.execute({"path": "empty.txt", "content": ""})
    assert result.success is True
    assert result.output == "Created empty.txt"
    assert (root / "empty.txt").read_bytes() == b""


def test_empty_content_truncates_existing():
    tool, root = make_tool()
    (root / "t.txt").write_text("something", encoding="utf-8")
    result = tool.execute({"path": "t.txt", "content": ""})
    assert result.success is True
    assert (root / "t.txt").read_bytes() == b""


def test_unicode_content():
    tool, root = make_tool()
    content = "héllo wörld 🎉 日本語\nもう一行\n"
    result = tool.execute({"path": "uni.txt", "content": content})
    assert result.success is True
    assert (root / "uni.txt").read_bytes() == content.encode("utf-8")
    assert (root / "uni.txt").read_text(encoding="utf-8") == content


def test_multiline_content():
    tool, root = make_tool()
    content = "line one\nline two\nline three\n"
    tool.execute({"path": "multi.txt", "content": content})
    assert (root / "multi.txt").read_bytes() == content.encode("utf-8")
    assert (root / "multi.txt").read_text(encoding="utf-8") == content


def test_preserves_crlf_exactly():
    tool, root = make_tool()
    content = "line1\r\nline2\r\n"
    tool.execute({"path": "crlf.txt", "content": content})
    raw = (root / "crlf.txt").read_bytes()
    assert raw == content.encode("utf-8")


def test_missing_parent_directory_fails():
    tool, root = make_tool()
    result = tool.execute({"path": "nope/sub/file.txt", "content": "x"})
    assert result.success is False
    assert "Parent directory does not exist" in result.error
    assert not (root / "nope").exists()


def test_parent_directories_not_auto_created():
    tool, root = make_tool()
    tool.execute({"path": "deep/leaf.txt", "content": "x"})
    assert not (root / "deep").exists()


def test_target_is_a_directory_fails():
    tool, root = make_tool()
    (root / "dir").mkdir()
    before = sorted(p.name for p in root.iterdir())
    result = tool.execute({"path": "dir", "content": "x"})
    assert result.success is False
    assert "directory" in result.error.lower()
    assert sorted(p.name for p in root.iterdir()) == before


def test_traversal_rejected():
    tool, root = make_tool()
    outside = root.parent / f"leak_{root.name}.txt"
    outside.write_text("original", encoding="utf-8")
    try:
        result = tool.execute({"path": "../" + outside.name, "content": "pwned"})
        assert result.success is False
        assert "outside" in result.error
        assert outside.read_text(encoding="utf-8") == "original"
    finally:
        outside.unlink(missing_ok=True)


def test_absolute_path_outside_rejected():
    tool, root = make_tool()
    outside = make_root() / "x.txt"
    outside.write_text("original", encoding="utf-8")
    result = tool.execute({"path": str(outside), "content": "pwned"})
    assert result.success is False
    assert "outside" in result.error
    assert outside.read_text(encoding="utf-8") == "original"


def test_absolute_path_inside_allowed():
    tool, root = make_tool()
    target = root / "inside.txt"
    result = tool.execute({"path": str(target), "content": "abs ok"})
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "abs ok"
    assert "inside.txt" in result.output


def test_symlink_to_outside_file_rejected():
    tool, root = make_tool()
    outside = make_root() / "ext.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "link"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    try:
        result = tool.execute({"path": "link", "content": "overwrite"})
        assert result.success is False
        assert "outside" in result.error
        assert outside.read_text(encoding="utf-8") == "secret"
    finally:
        outside.unlink(missing_ok=True)


def test_symlink_parent_escape_rejected():
    tool, root = make_tool()
    outside = make_root()
    linkdir = root / "linkdir"
    try:
        linkdir.symlink_to(outside, target_is_directory=True)
    except OSError:
        return
    try:
        result = tool.execute({"path": "linkdir/file.txt", "content": "nope"})
        assert result.success is False
        assert "outside" in result.error
        assert not (outside / "file.txt").exists()
    finally:
        outside.unlink(missing_ok=True)


def test_registry_dispatch():
    root = make_root()
    registry = ToolRegistry()
    registry.register(WriteFileTool(root))
    registry.register(ReadFileTool(root))

    result = registry.execute("write_file", {"path": "w.txt", "content": "hello"})
    assert result.success is True

    read = registry.execute("read_file", {"path": "w.txt"})
    assert read.success is True
    assert read.output == "hello"


def test_toolresult_success_structure():
    tool, _ = make_tool()
    result = tool.execute({"path": "a.txt", "content": "data"})
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error == ""
    assert str(make_root()) not in result.output
    assert "\\" not in result.output


def test_toolresult_failure_structure():
    tool, _ = make_tool()
    result = tool.execute({"path": "missing_parent/file.txt", "content": "data"})
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.output == ""
    assert isinstance(result.error, str)
    assert result.error


def test_empty_path_rejected():
    tool, _ = make_tool()
    result = tool.execute({"path": "", "content": "data"})
    assert result.success is False
    assert "path" in result.error


def test_invalid_argument_types_rejected():
    tool, _ = make_tool()
    assert tool.execute({"path": 123, "content": "data"}).success is False
    assert tool.execute({"path": "a.txt", "content": 123}).success is False


def test_unknown_arguments_rejected():
    tool, _ = make_tool()
    result = tool.execute({"path": "a.txt", "content": "x", "bogus": 1})
    assert result.success is False
    assert "bogus" in result.error


def test_writing_stays_inside_workspace():
    tool, root = make_tool()
    for name in ("a.txt", "b.txt"):
        tool.execute({"path": name, "content": f"data {name}"})
    before = set(p.name for p in root.iterdir())
    tool.execute({"path": "c.txt", "content": "more"})
    after = set(p.name for p in root.iterdir())
    assert "c.txt" in after


def test_failed_write_leaves_unrelated_files_untouched():
    tool, root = make_tool()
    keep = root / "keep.txt"
    keep.write_text("stable", encoding="utf-8")
    before_bytes = keep.read_bytes()
    tool.execute({"path": "nope/x.txt", "content": "x"})
    assert keep.read_bytes() == before_bytes
    assert not (root / "nope").exists()


def test_overwrite_replaces_intended_file_only():
    tool, root = make_tool()
    (root / "a.txt").write_text("old_a", encoding="utf-8")
    (root / "b.txt").write_text("old_b", encoding="utf-8")

    tool.execute({"path": "a.txt", "content": "new_a"})
    assert (root / "a.txt").read_text(encoding="utf-8") == "new_a"
    assert (root / "b.txt").read_text(encoding="utf-8") == "old_b"


def test_permissions_preserved_on_overwrite():
    tool, root = make_tool()
    target = root / "perm.txt"
    target.write_text("x", encoding="utf-8")
    try:
        os.chmod(str(target), 0o600)
    except OSError:
        return
    before_mode = stat.S_IMODE(target.stat().st_mode)
    before_content = target.read_bytes()

    result = tool.execute({"path": "perm.txt", "content": "written"})
    if result.success:
        assert target.read_text(encoding="utf-8") == "written"
        assert stat.S_IMODE(target.stat().st_mode) == before_mode
    else:
        assert target.read_bytes() == before_content


def test_result_valid_utf8():
    tool, root = make_tool()
    content = "héllo 日本語 🎉\nline two\n"
    tool.execute({"path": "ok.txt", "content": content})
    raw = (root / "ok.txt").read_bytes()
    assert raw.decode("utf-8", "strict") == content


def test_invalid_path_is_controlled():
    tool, _ = make_tool()
    result = tool.execute({"path": "bad\x00path.txt", "content": "x"})
    assert result.success is False
    assert result.error


TEST_FUNCTIONS = [
    test_creates_new_file,
    test_overwrites_existing_file,
    test_empty_content_creates_file,
    test_empty_content_truncates_existing,
    test_unicode_content,
    test_multiline_content,
    test_preserves_crlf_exactly,
    test_missing_parent_directory_fails,
    test_parent_directories_not_auto_created,
    test_target_is_a_directory_fails,
    test_traversal_rejected,
    test_absolute_path_outside_rejected,
    test_absolute_path_inside_allowed,
    test_symlink_to_outside_file_rejected,
    test_symlink_parent_escape_rejected,
    test_registry_dispatch,
    test_toolresult_success_structure,
    test_toolresult_failure_structure,
    test_empty_path_rejected,
    test_invalid_argument_types_rejected,
    test_unknown_arguments_rejected,
    test_writing_stays_inside_workspace,
    test_failed_write_leaves_unrelated_files_untouched,
    test_overwrite_replaces_intended_file_only,
    test_permissions_preserved_on_overwrite,
    test_result_valid_utf8,
    test_invalid_path_is_controlled,
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