"""Enforce that only allowed tools can be called in an agent loop."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class ToolNotAllowedError(Exception):
    """Raised when a tool call is blocked by the whitelist."""


class ToolWhitelistError(Exception):
    """Raised for configuration errors."""


# ---------------------------------------------------------------------------
# ToolWhitelist
# ---------------------------------------------------------------------------

@dataclass
class ToolWhitelist:
    """Enforce an allowlist of tool names for an agent loop.

    Args:
        allowed: collection of permitted tool names (exact match).
        deny_message: message template for the error/warning. Use {name} for
            the disallowed tool name.
        raise_on_deny: if True (default), raise ToolNotAllowedError when a
            blocked tool is called. If False, log and return False.

    Usage::

        whitelist = ToolWhitelist(["web_search", "read_file"])

        # Before executing a tool call
        whitelist.check("web_search")    # ok
        whitelist.check("exec_shell")    # raises ToolNotAllowedError

        # Filter a list of tool calls (drop disallowed ones)
        safe = whitelist.filter_calls(tool_calls)
    """

    allowed: list[str]
    deny_message: str = "Tool '{name}' is not in the allowed list"
    raise_on_deny: bool = True
    _allowed_set: set[str] = field(default_factory=set, init=False, repr=False)
    _denied: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.allowed:
            raise ToolWhitelistError("allowed list cannot be empty")
        # Deduplicate while preserving first-seen order so that ``allowed`` and
        # ``_allowed_set`` never drift out of sync.
        self.allowed = list(dict.fromkeys(self.allowed))
        self._allowed_set = set(self.allowed)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def is_allowed(self, name: str) -> bool:
        """Return True if the tool name is in the allowed list."""
        return name in self._allowed_set

    def check(self, name: str) -> bool:
        """Check whether a tool name is allowed.

        Returns True if allowed. Raises ToolNotAllowedError (or returns False
        if raise_on_deny=False) if the tool is not allowed.
        """
        if name in self._allowed_set:
            return True
        self._denied.append(name)
        if self.raise_on_deny:
            raise ToolNotAllowedError(self.deny_message.format(name=name))
        return False

    # ------------------------------------------------------------------
    # Batch helpers for Anthropic/OpenAI tool_use blocks
    # ------------------------------------------------------------------

    def check_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Check all tool calls in a list. Raises on the first blocked tool.

        Expects dicts with a ``name`` or ``function.name`` field (covers both
        Anthropic tool_use blocks and OpenAI function_call dicts).
        """
        for call in tool_calls:
            name = _extract_name(call)
            if name:
                self.check(name)

    def filter_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return only the allowed tool calls from the list. Never raises.

        Disallowed calls are silently dropped and recorded in denied_names.
        """
        result: list[dict[str, Any]] = []
        for call in tool_calls:
            name = _extract_name(call)
            if name and name in self._allowed_set:
                result.append(call)
            elif name:
                self._denied.append(name)
        return result

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, name: str) -> None:
        """Add a tool name to the allowed list.

        Idempotent: adding a name that is already allowed is a no-op and does
        not create a duplicate entry in ``allowed``.
        """
        if name not in self._allowed_set:
            self.allowed.append(name)
            self._allowed_set.add(name)

    def remove(self, name: str) -> None:
        """Remove a tool name from the allowed list."""
        if name in self._allowed_set:
            self.allowed = [n for n in self.allowed if n != name]
            self._allowed_set.discard(name)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    @property
    def denied_names(self) -> list[str]:
        """All tool names that were denied since creation (or last reset)."""
        return list(self._denied)

    def reset_denied(self) -> None:
        """Clear the denied names log."""
        self._denied.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_name(call: dict[str, Any]) -> str:
    """Extract tool name from Anthropic tool_use or OpenAI function_call dict."""
    # Anthropic: {"type": "tool_use", "name": "...", ...}
    if "name" in call:
        return str(call["name"])
    # OpenAI function: {"function": {"name": "...", ...}, ...}
    fn = call.get("function")
    if isinstance(fn, dict) and "name" in fn:
        return str(fn["name"])
    return ""


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def tool_guard(whitelist: ToolWhitelist) -> Callable[[F], F]:
    """Decorator that checks the tool name before calling the function.

    The wrapped function must take the tool name as its first positional
    argument. ``whitelist.check`` is invoked before the function body runs, so
    a blocked tool is rejected before any side effects occur:

    * If the whitelist raises on deny (the default), a blocked tool raises
      :class:`ToolNotAllowedError` and the wrapped function is never called.
    * If the whitelist was created with ``raise_on_deny=False``, a blocked tool
      is recorded in ``denied_names``, the wrapped function is skipped, and the
      wrapper returns ``None``.

    Usage::

        @tool_guard(whitelist)
        def run_tool(name: str, args: dict) -> Any:
            ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(name: str, *args: Any, **kwargs: Any) -> Any:
            if not whitelist.check(name):
                # Only reached when raise_on_deny=False; the call is blocked,
                # recorded in denied_names, and the function is not executed.
                return None
            return fn(name, *args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator
