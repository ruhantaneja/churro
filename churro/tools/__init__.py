from churro.tools.git_diff import GitDiffTool
from churro.tools.list_files import ListFilesTool
from churro.tools.read_file import ReadFileTool
from churro.tools.registry import ToolRegistry
from churro.tools.executor import ToolExecutionBatch, ToolExecutionResult, ToolExecutor
from churro.tools.result_messages import (
    result_content,
    result_message_from_execution,
    result_messages_from_batch,
)
from churro.tools.run_command import RunCommandTool
from churro.tools.run_tests import RunTestsTool
from churro.tools.search_files import SearchFilesTool
from churro.tools.tool import (
    DuplicateToolError,
    Tool,
    ToolArgs,
    ToolDefinitionError,
    ToolError,
    ToolResult,
    UnknownToolError,
)
from pathlib import Path

from churro.providers.responses import ToolResultMessage
from churro.tools.write_file import WriteFileTool


def default_tool_registry(workspace_root: str | Path) -> ToolRegistry:
    """Build the standard workspace-bound tool registry.

    This is the single registration point for the current CHURRO tool set;
    the CLI passes its result straight to the AgentRunner. Each tool is
    workspace-confined to ``workspace_root`` and never registered twice.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace_root))
    registry.register(ListFilesTool(workspace_root))
    registry.register(SearchFilesTool(workspace_root))
    registry.register(WriteFileTool(workspace_root))
    registry.register(RunCommandTool(workspace_root))
    registry.register(GitDiffTool(workspace_root))
    registry.register(RunTestsTool(workspace_root))
    return registry

__all__ = [
    "DuplicateToolError",
    "GitDiffTool",
    "ListFilesTool",
    "ReadFileTool",
    "RunCommandTool",
    "RunTestsTool",
    "SearchFilesTool",
    "Tool",
    "ToolArgs",
    "ToolDefinitionError",
    "ToolError",
    "ToolExecutionBatch",
    "ToolExecutionResult",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolResultMessage",
    "UnknownToolError",
    "WriteFileTool",
    "default_tool_registry",
    "result_content",
    "result_message_from_execution",
    "result_messages_from_batch",
]