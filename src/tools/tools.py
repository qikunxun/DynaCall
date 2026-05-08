from typing import List, Optional, Any

from src.tools.base import Tool, ToolBase, tool


class InvalidTool(ToolBase):
    """Tool that is run when invalid tool name is encountered by agent."""

    def __init__(self, available_tool_names: List[str] = None):
        self.name = "invalid_tool"
        self.description = "Called when tool name is invalid. Suggests valid tool names."
        self.available_tool_names = available_tool_names or []

    def run(
        self,
        requested_tool_name: str,
        available_tool_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        tool_names = available_tool_names or self.available_tool_names
        available_tool_names_str = ", ".join(tool_names)
        return (
            f"{requested_tool_name} is not a valid tool, "
            f"try one of [{available_tool_names_str}]."
        )

    async def arun(
        self,
        requested_tool_name: str,
        available_tool_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        tool_names = available_tool_names or self.available_tool_names
        available_tool_names_str = ", ".join(tool_names)
        return (
            f"{requested_tool_name} is not a valid tool, "
            f"try one of [{available_tool_names_str}]."
        )


__all__ = ["InvalidTool", "ToolBase", "tool", "Tool", "StructuredTool"]