import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from churro.tools import GitDiffTool, ToolRegistry
from churro.tools.tool import ToolResult

GIT = shutil.which("git")


def make_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="churro git "))


def make_tool(root: Path | None = None, **kwargs) -> tuple[GitDiffTool, Path]:
    root = root or make_root()
    return GitDiffTool(root, **kwargs), root


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [GIT, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def init_repo(root: Path) -> Path:
    git(["init", "-q", root.name], root.parent)
    (root / "tracked.txt").write_text("line one\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "nested.txt").write_text("nested\n", encoding="utf-8")
    git(["add", "."], root)
    git(["commit", "-q", "-m", "initial"], root)
    return root


def test_git_repo_with_no_changes():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    result = tool.execute({})
    assert result.success is True
    assert result.output == "Git diff:\nNo changes."


def test_modified_tracked_file():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("line one modified\n", encoding="utf-8")
    result = tool.execute({})
    assert result.success is True
    assert "-line one" in result.output
    assert "+line one modified" in result.output
    assert "tracked.txt" in result.output


def test_new_untracked_file_not_in_diff():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "brand-new.txt").write_text("untracked content\n", encoding="utf-8")
    result = tool.execute({})
    assert result.success is True
    assert result.output == "Git diff:\nNo changes."
    assert "brand-new" not in result.output
    assert "untracked content" not in result.output


def test_staged_change():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("line one\nline staged\n", encoding="utf-8")
    git(["add", "tracked.txt"], root)
    staged = tool.execute({"staged": True})
    assert staged.success is True
    assert "+line staged" in staged.output
    working = tool.execute({"staged": False})
    assert working.success is True
    assert working.output == "Git diff:\nNo changes."


def test_path_restricted_diff():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("one changed\n", encoding="utf-8")
    (root / "sub" / "nested.txt").write_text("nested changed\n", encoding="utf-8")
    result = tool.execute({"path": "sub/nested.txt"})
    assert result.success is True
    assert "nested changed" in result.output
    assert "tracked.txt" not in result.output
    assert "one changed" not in result.output


def test_nested_path_restriction_with_default_path():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (root / "sub" / "nested.txt").write_text("also changed\n", encoding="utf-8")
    full = tool.execute({})
    assert full.success is True
    assert "changed" in full.output
    assert "also changed" in full.output


def test_default_path_is_workspace_root():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("modified\n", encoding="utf-8")
    dots = tool.execute({"path": "."})
    none = tool.execute({})
    assert dots.success is True
    assert dots.output == none.output
    assert "modified" in dots.output


def test_nonexistent_path_is_controlled_failure():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    result = tool.execute({"path": "never_existed.txt"})
    assert result.success is False
    assert "Path does not exist" in result.error
    assert "never_existed.txt" in result.error


def test_traversal_rejected():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    result = tool.execute({"path": "../outside.txt"})
    assert result.success is False
    assert "outside" in result.error


def test_absolute_path_outside_rejected():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    outside = make_root() / "external.txt"
    outside.write_text("x", encoding="utf-8")
    result = tool.execute({"path": str(outside)})
    assert result.success is False
    assert "outside" in result.error


def test_absolute_path_inside_workspace_allowed():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("abs edited\n", encoding="utf-8")
    result = tool.execute({"path": str(root / "tracked.txt")})
    assert result.success is True
    assert "abs edited" in result.output
    assert str(root) not in result.output


def test_workspace_is_git_cwd():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "sub" / "nested.txt").write_text("cwd verified\n", encoding="utf-8")
    (root / "tracked.txt").write_text("outside sub\n", encoding="utf-8")
    inner = GitDiffTool(root / "sub")
    result = inner.execute({})
    assert result.success is True
    assert "cwd verified" in result.output
    assert "outside sub" not in result.output
    assert str(root / "sub") not in result.output


def test_non_git_workspace_is_controlled_failure():
    if GIT is None:
        return
    tool, root = make_tool()
    root.mkdir(exist_ok=True)
    (root / "some.txt").write_text("x", encoding="utf-8")
    result = tool.execute({})
    assert result.success is False
    assert "not a git repository" in result.error.lower() or "git" in result.error.lower()


def test_git_unavailable_or_launch_failure():
    tool, root = make_tool()
    init_repo(root)
    bogus = GitDiffTool(root, git_executable=str(Path(tempfile.gettempdir()) / "no_git_here"))
    result = bogus.execute({})
    assert result.success is False
    assert result.error


def test_git_diff_arguments_reject_empty_command():
    tool, root = make_tool()
    init_repo(root)
    result = tool.execute({"bogus": 1})
    assert result.success is False
    assert "bogus" in result.error


def test_output_preserved():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("one\n--\ntwo\n", encoding="utf-8")
    (root / "sub" / "nested.txt").write_text("nested-x\n", encoding="utf-8")
    result = tool.execute({"path": "tracked.txt"})
    assert result.success is True
    assert "--- a/tracked.txt" in result.output or "--- tracked.txt" in result.output
    assert "+++ b/tracked.txt" in result.output or "+++ tracked.txt" in result.output
    assert "--" in result.output


def test_output_truncation():
    if GIT is None:
        return
    tool, root = make_tool(max_output_chars=200)
    init_repo(root)
    (root / "tracked.txt").write_text("".join(f"content line {i}\n" for i in range(500)), encoding="utf-8")
    result = tool.execute({})
    assert result.success is True
    assert "truncated" in result.output
    assert len(result.output) < 450
    assert result.output.startswith("Git diff:\n")
    assert "content line 499" not in result.output


def test_toolresult_success_structure():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    result = tool.execute({})
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error == ""
    assert isinstance(result.output, str) and result.output


def test_toolresult_failure_structure():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    result = tool.execute({"path": "does_not_exist.txt"})
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert isinstance(result.output, str)
    assert result.output == ""
    assert result.error


def test_registry_dispatch():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("reg edited\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(tool)
    assert any(t.name == "git_diff" for t in registry.list_tools())
    result = registry.execute("git_diff", {})
    assert result.success is True
    assert "reg edited" in result.output


def test_does_not_modify_files_or_git_state():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("state before\n", encoding="utf-8")
    (root / "sub" / "nested.txt").write_text("nested state\n", encoding="utf-8")
    bytes_before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    status_before = git(["status", "--porcelain"], root).stdout
    result = tool.execute({})
    assert result.success is True
    bytes_after = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    status_after = git(["status", "--porcelain"], root).stdout
    assert bytes_after == bytes_before
    assert status_after == status_before


def test_does_not_alter_configured_workspace():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    before = tool._workspace
    tool.execute({})
    tool.execute({"staged": True})
    assert tool._workspace == before
    assert str(tool._workspace) == str(Path(root).resolve())


def test_no_absolute_host_paths_in_output_or_errors():
    if GIT is None:
        return
    tool, root = make_tool()
    init_repo(root)
    (root / "tracked.txt").write_text("leak check\n", encoding="utf-8")
    for args in ({}, {"staged": True}, {"path": "tracked.txt"}):
        result = tool.execute(args)
        assert result.success is True
        assert str(root) not in result.output
        assert os.path.normcase(str(root)) not in result.output
    failed = tool.execute({"path": "../escape"})
    assert failed.success is False
    assert str(root) not in failed.error
    assert os.path.normcase(str(root)) not in failed.error


TEST_FUNCTIONS = [
    test_git_repo_with_no_changes,
    test_modified_tracked_file,
    test_new_untracked_file_not_in_diff,
    test_staged_change,
    test_path_restricted_diff,
    test_nested_path_restriction_with_default_path,
    test_default_path_is_workspace_root,
    test_nonexistent_path_is_controlled_failure,
    test_traversal_rejected,
    test_absolute_path_outside_rejected,
    test_absolute_path_inside_workspace_allowed,
    test_workspace_is_git_cwd,
    test_non_git_workspace_is_controlled_failure,
    test_git_unavailable_or_launch_failure,
    test_git_diff_arguments_reject_empty_command,
    test_output_preserved,
    test_output_truncation,
    test_toolresult_success_structure,
    test_toolresult_failure_structure,
    test_registry_dispatch,
    test_does_not_modify_files_or_git_state,
    test_does_not_alter_configured_workspace,
    test_no_absolute_host_paths_in_output_or_errors,
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