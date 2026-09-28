from __future__ import annotations

from enum import Enum
from typing import Callable


class PermissionLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGEROUS = "DANGEROUS"


class SecurityGuard:
    """Security governance and permission enforcement for tool calls and computer actions."""

    def __init__(
        self,
        approval_callback: Callable[[str, str, dict], bool] | None = None,
        auto_approve_safe: bool = True,
    ) -> None:
        self.approval_callback = approval_callback
        self.auto_approve_safe = auto_approve_safe

        # Predefined tool safety tiers
        self.tool_permissions: dict[str, PermissionLevel] = {
            "read_file": PermissionLevel.SAFE,
            "list_directory": PermissionLevel.SAFE,
            "calculate": PermissionLevel.SAFE,
            "search_memory": PermissionLevel.SAFE,
            "write_file": PermissionLevel.CAUTION,
            "edit_file": PermissionLevel.CAUTION,
            "install_package": PermissionLevel.CAUTION,
            "delete_file": PermissionLevel.DANGEROUS,
            "run_shell": PermissionLevel.DANGEROUS,
            "system_reboot": PermissionLevel.DANGEROUS,
        }

    def register_tool(self, name: str, level: PermissionLevel) -> None:
        self.tool_permissions[name] = level

    def check_permission(self, tool_name: str, arguments: dict) -> bool:
        """Determines if a tool execution is authorized."""
        level = self.tool_permissions.get(tool_name, PermissionLevel.DANGEROUS)

        if level == PermissionLevel.SAFE and self.auto_approve_safe:
            return True

        if self.approval_callback is not None:
            return self.approval_callback(tool_name, level.value, arguments)

        # By default, require explicit approval for CAUTION or DANGEROUS
        return False
