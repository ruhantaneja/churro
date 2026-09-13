import tempfile
from pathlib import Path

from churro.tools import ReadFileTool, ToolRegistry
from churro.tools.tool import ToolResult


def make_root() -> Path:
    return Path(tempfile.mkdtemp())


def make_tool(root: Path | None = None) -> tuple[ReadFileTool, Path]:
    root = root or make_root()
    return ReadFileTool(root), root


def test_reads_normal_utf8_file():
    tool, root = make_tool()
    (root / "hello.txt").write_text("hello world\n", encoding="utf-8")
    result = tool.execute({"path": "hello.txt"})
    assert result.success is True
    assert result.output == "hello world\n"
    assert result.error == ""


def test_reads_empty_file():
    tool, root = make_tool()
    (root / "empty.txt").touch()
    result = tool.execute({"path": "empty.txt"})
    assert result.success is True
    assert result.output == ""


def test_missing_file():
    tool, _ = make_tool()
    result = tool.execute({"path": "missing.txt"})
    assert result.success is False
    assert "does not exist" in result.error
    assert "missing.txt" in result.error


def test_directory_instead_of_file():
    tool, root = make_tool()
    (root / "sub").mkdir()
    result = tool.execute({"path": "sub"})
    assert result.success is False
    assert "directory" in result.error.lower()


def test_invalid_path_is_controlled():
    tool, _ = make_tool()
    result = tool.execute({"path": "bad\x00path.txt"})
    assert result.success is False
    assert result.error


def test_binary_file_not_valid_utf8_is_controlled():
    tool, root = make_tool()
    (root / "bin.dat").write_bytes(bytes(range(256)))
    result = tool.execute({"path": "bin.dat"})
    assert result.success is False
    assert "utf-8" in result.error.lower()


def test_relative_path_inside_workspace():
    tool, root = make_tool()
    (root / "a.txt").write_text("data", encoding="utf-8")
    result = tool.execute({"path": "./a.txt"})
    assert result.success is True
    assert result.output == "data"


def test_nested_file_inside_workspace():
    tool, root = make_tool()
    (root / "sub" / "deep").mkdir(parents=True)
    (root / "sub" / "deep" / "n.txt").write_text("nested", encoding="utf-8")

    forward = tool.execute({"path": "sub/deep/n.txt"})
    assert forward.success is True
    assert forward.output == "nested"

    backslash = tool.execute({"path": "sub\\deep\\n.txt"})
    assert backslash.success is True
    assert backslash.output == "nested"


def test_traversal_escape_rejected():
    root = make_root()
    secret = root.parent / f"secret_{root.name}.txt"
    secret.write_text("top secret", encoding="utf-8")
    tool = ReadFileTool(root)

    result = tool.execute({"path": "../" + secret.name})
    assert result.success is False
    assert "outside" in result.error

    result = tool.execute({"path": "../" + secret.parent.name + "/" + secret.name})
    assert result.success is False
    assert "outside" in result.error
    secret.unlink(missing_ok=True)


def test_absolute_path_outside_workspace_rejected():
    tool, root = make_tool()
    outside = make_root() / "s.txt"
    outside.write_text("x", encoding="utf-8")
    result = tool.execute({"path": str(outside)})
    assert result.success is False
    assert "outside" in result.error


def test_absolute_path_inside_workspace_allowed():
    tool, root = make_tool()
    (root / "in.txt").write_text("inside", encoding="utf-8")
    result = tool.execute({"path": str(root / "in.txt")})
    assert result.success is True
    assert result.output == "inside"


def test_symlink_escape_rejected():
    root = make_root()
    outside = root.parent / f"real_{root.name}.txt"
    outside.write_text("secret via symlink", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        outside.unlink(missing_ok=True)
        return

    try:
        result = ReadFileTool(root).execute({"path": "link.txt"})
        assert result.success is False
        assert "outside" in result.error
    finally:
        outside.unlink(missing_ok=True)


def test_workspace_root_itself_rejected():
    tool, _ = make_tool()
    result = tool.execute({"path": "."})
    assert result.success is False
    assert "directory" in result.error.lower()


def test_empty_path_maps_to_root_directory():
    tool, _ = make_tool()
    result = tool.execute({"path": ""})
    assert result.success is False
    assert "directory" in result.error.lower()


def test_unicode_content():
    tool, root = make_tool()
    content = "héllo wörld 🎉 ünïcode — 日本語 中文\n第二行 continues\n最后一行"
    (root / "uni.txt").write_text(content, encoding="utf-8")
    result = tool.execute({"path": "uni.txt"})
    assert result.success is True
    assert result.output == content


def test_large_file():
    tool, root = make_tool()
    lines = [f"line {i:06d} of a large file\n" for i in range(5000)]
    content = "".join(lines)
    (root / "big.txt").write_text(content, encoding="utf-8")
    result = tool.execute({"path": "big.txt"})
    assert result.success is True
    assert result.output == content
    assert len(result.output) > 100_000


def test_toolresult_structure():
    tool, root = make_tool()
    (root / "s.txt").write_text("ok", encoding="utf-8")

    ok = tool.execute({"path": "s.txt"})
    assert isinstance(ok, ToolResult)
    assert ok.success is True
    assert ok.output == "ok"
    assert ok.error == ""

    missing = tool.execute({"path": "missing.txt"})
    assert isinstance(missing, ToolResult)
    assert missing.success is False
    assert missing.output == ""
    assert isinstance(missing.error, str)
    assert missing.error


def test_file_not_modified():
    tool, root = make_tool()
    target = root / "m.txt"
    original = "stable content that must not change\nsecond line"
    target.write_text(original, encoding="utf-8")
    before = target.read_bytes()

    result = tool.execute({"path": "m.txt"})
    assert result.success is True
    assert result.output == original
    assert target.read_bytes() == before
    assert target.read_text(encoding="utf-8") == original


def test_missing_path_argument_is_controlled():
    tool, _ = make_tool()
    result = tool.execute({})
    assert result.success is False
    assert result.error


def test_non_string_path_is_controlled():
    tool, _ = make_tool()
    result = tool.execute({"path": 123})
    assert result.success is False
    assert "path" in result.error


def test_unknown_argument_is_controlled():
    tool, _ = make_tool()
    result = tool.execute({"path": "x.txt", "bogus": 1})
    assert result.success is False
    assert "bogus" in result.error


def test_works_with_registry():
    tool, root = make_tool()
    (root / "r.txt").write_text("via registry", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(tool)
    assert registry.get("read_file") is tool

    result = registry.execute("read_file", {"path": "r.txt"})
    assert result.success is True
    assert result.output == "via registry"


def test_definition_and_schema():
    tool, _ = make_tool()
    assert tool.name == "read_file"
    assert tool.description

    schema = tool.input_schema
    assert schema["type"] == "object"
    assert schema["properties"]["path"]["type"] == "string"
    assert schema["required"] == ["path"]
    assert schema["additionalProperties"] is False


TEST_FUNCTIONS = [
    test_reads_normal_utf8_file,
    test_reads_empty_file,
    test_missing_file,
    test_directory_instead_of_file,
    test_invalid_path_is_controlled,
    test_binary_file_not_valid_utf8_is_controlled,
    test_relative_path_inside_workspace,
    test_nested_file_inside_workspace,
    test_traversal_escape_rejected,
    test_absolute_path_outside_workspace_rejected,
    test_absolute_path_inside_workspace_allowed,
    test_symlink_escape_rejected,
    test_workspace_root_itself_rejected,
    test_empty_path_maps_to_root_directory,
    test_unicode_content,
    test_large_file,
    test_toolresult_structure,
    test_file_not_modified,
    test_missing_path_argument_is_controlled,
    test_non_string_path_is_controlled,
    test_unknown_argument_is_controlled,
    test_works_with_registry,
    test_definition_and_schema,
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