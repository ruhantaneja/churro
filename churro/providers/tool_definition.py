from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    """A tool exposed to an LLM, described independently of any provider.

    ``input_schema`` is the JSON Schema for the tool's arguments, normally
    derived from a churro ``Tool``'s ``input_schema`` (see
    :meth:`ToolDefinition.from_tool`).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_tool(cls, tool: Any) -> "ToolDefinition":
        """Build a definition from a churro Tool (name/description/schema).
        """
        return cls(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )