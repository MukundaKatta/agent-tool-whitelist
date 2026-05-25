"""agent-tool-whitelist: enforce an allowlist of tool names for agent loops.

Public API:
    ToolWhitelist(allowed, *, deny_message, raise_on_deny)
    ToolNotAllowedError — raised when a blocked tool is called
    ToolWhitelistError  — configuration errors
    tool_guard(whitelist) — decorator
"""

from .core import (
    ToolNotAllowedError,
    ToolWhitelist,
    ToolWhitelistError,
    tool_guard,
)

__all__ = [
    "ToolWhitelist",
    "ToolNotAllowedError",
    "ToolWhitelistError",
    "tool_guard",
]
__version__ = "0.1.0"
