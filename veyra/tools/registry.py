from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from ..security.permissions import PermissionLevel, SecurityGuard


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
        permission_level: PermissionLevel = PermissionLevel.SAFE,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.permission_level = permission_level

    def execute(self, **kwargs: Any) -> Any:
        return self.handler(**kwargs)


class ToolRegistry:
    """Registry for structured tool definitions and secure execution."""

    def __init__(self, security_guard: SecurityGuard | None = None) -> None:
        self.tools: dict[str, Tool] = {}
        self.security = security_guard or SecurityGuard()
        self._register_default_tools()

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        self.security.register_tool(tool.name, tool.permission_level)

    def get_tool(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validates permission and executes the named tool."""
        tool = self.get_tool(name)
        if not tool:
            return {"status": "error", "error": f"Tool '{name}' not found."}

        if not self.security.check_permission(name, arguments):
            return {
                "status": "denied",
                "error": f"Permission denied for tool '{name}' ({tool.permission_level.value}).",
            }

        try:
            result = tool.execute(**arguments)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        """Returns JSON-schema descriptions of all available tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "permission_level": t.permission_level.value,
            }
            for t in self.tools.values()
        ]

    def _register_default_tools(self) -> None:
        def read_file(path: str) -> str:
            p = Path(path)
            if not p.exists() or not p.is_file():
                return f"File '{path}' does not exist."
            return p.read_text(encoding="utf-8", errors="replace")[:10000]

        def list_directory(path: str = ".") -> list[str]:
            p = Path(path)
            if not p.exists() or not p.is_dir():
                return [f"Directory '{path}' does not exist."]
            return [f.name for f in p.iterdir()][:100]

        def calculate(expression: str) -> str:
            # Safe math evaluation using restricted globals
            allowed = {"__builtins__": None}
            try:
                val = eval(expression, allowed, {})
                return str(val)
            except Exception as e:
                return f"Calculation error: {e}"

        def write_file(path: str, content: str) -> str:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to '{path}'."

        self.register(
            Tool(
                name="read_file",
                description="Reads up to 10,000 characters from a local text file.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                handler=read_file,
                permission_level=PermissionLevel.SAFE,
            )
        )
        self.register(
            Tool(
                name="list_directory",
                description="Lists files and folders inside a directory.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                handler=list_directory,
                permission_level=PermissionLevel.SAFE,
            )
        )
        self.register(
            Tool(
                name="calculate",
                description="Safely calculates a mathematical expression.",
                parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                handler=calculate,
                permission_level=PermissionLevel.SAFE,
            )
        )
        self.register(
            Tool(
                name="write_file",
                description="Writes content to a file on disk.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
                handler=write_file,
                permission_level=PermissionLevel.CAUTION,
            )
        )
