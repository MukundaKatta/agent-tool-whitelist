"""Tests for agent-tool-whitelist."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from agent_tool_whitelist import (
    ToolNotAllowedError,
    ToolWhitelist,
    ToolWhitelistError,
    tool_guard,
)


# ---------------------------------------------------------------------------
# is_allowed
# ---------------------------------------------------------------------------

def test_is_allowed_true():
    w = ToolWhitelist(["search", "read"])
    assert w.is_allowed("search") is True


def test_is_allowed_false():
    w = ToolWhitelist(["search", "read"])
    assert w.is_allowed("exec_shell") is False


def test_is_allowed_empty_string():
    w = ToolWhitelist(["search"])
    assert w.is_allowed("") is False


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def test_check_allowed_returns_true():
    w = ToolWhitelist(["search"])
    assert w.check("search") is True


def test_check_blocked_raises():
    w = ToolWhitelist(["search"])
    with pytest.raises(ToolNotAllowedError):
        w.check("exec_shell")


def test_check_error_message_contains_name():
    w = ToolWhitelist(["search"])
    with pytest.raises(ToolNotAllowedError, match="exec_shell"):
        w.check("exec_shell")


def test_check_custom_deny_message():
    w = ToolWhitelist(["search"], deny_message="blocked: {name}")
    with pytest.raises(ToolNotAllowedError, match="blocked: exec_shell"):
        w.check("exec_shell")


def test_check_raise_on_deny_false_returns_false():
    w = ToolWhitelist(["search"], raise_on_deny=False)
    result = w.check("exec_shell")
    assert result is False


def test_check_raise_on_deny_false_no_exception():
    w = ToolWhitelist(["search"], raise_on_deny=False)
    w.check("anything")  # should not raise


# ---------------------------------------------------------------------------
# check_calls (Anthropic tool_use blocks)
# ---------------------------------------------------------------------------

def test_check_calls_all_allowed():
    w = ToolWhitelist(["search", "read"])
    calls = [
        {"type": "tool_use", "name": "search", "input": {}},
        {"type": "tool_use", "name": "read", "input": {}},
    ]
    w.check_calls(calls)  # no exception


def test_check_calls_blocked_raises():
    w = ToolWhitelist(["search"])
    calls = [{"type": "tool_use", "name": "exec_shell", "input": {}}]
    with pytest.raises(ToolNotAllowedError, match="exec_shell"):
        w.check_calls(calls)


def test_check_calls_openai_format():
    w = ToolWhitelist(["search"])
    calls = [{"function": {"name": "search", "arguments": "{}"}}]
    w.check_calls(calls)  # no exception


def test_check_calls_openai_blocked():
    w = ToolWhitelist(["search"])
    calls = [{"function": {"name": "exec_shell", "arguments": "{}"}}]
    with pytest.raises(ToolNotAllowedError):
        w.check_calls(calls)


def test_check_calls_empty_list():
    w = ToolWhitelist(["search"])
    w.check_calls([])  # no exception


# ---------------------------------------------------------------------------
# filter_calls
# ---------------------------------------------------------------------------

def test_filter_keeps_allowed():
    w = ToolWhitelist(["search", "read"])
    calls = [
        {"name": "search", "input": {}},
        {"name": "exec_shell", "input": {}},
        {"name": "read", "input": {}},
    ]
    result = w.filter_calls(calls)
    assert len(result) == 2
    assert result[0]["name"] == "search"
    assert result[1]["name"] == "read"


def test_filter_drops_blocked():
    w = ToolWhitelist(["search"])
    calls = [{"name": "exec_shell"}]
    result = w.filter_calls(calls)
    assert result == []


def test_filter_never_raises():
    w = ToolWhitelist(["search"])
    calls = [{"name": "exec_shell"}, {"name": "rm_rf"}]
    result = w.filter_calls(calls)  # no exception
    assert result == []


def test_filter_empty_list():
    w = ToolWhitelist(["search"])
    assert w.filter_calls([]) == []


def test_filter_records_denied_names():
    w = ToolWhitelist(["search"])
    w.filter_calls([{"name": "exec_shell"}, {"name": "delete_file"}])
    assert "exec_shell" in w.denied_names
    assert "delete_file" in w.denied_names


# ---------------------------------------------------------------------------
# denied_names audit
# ---------------------------------------------------------------------------

def test_denied_names_empty_initially():
    w = ToolWhitelist(["search"])
    assert w.denied_names == []


def test_denied_names_populated_by_check():
    w = ToolWhitelist(["search"], raise_on_deny=False)
    w.check("bad_tool")
    assert "bad_tool" in w.denied_names


def test_denied_names_populated_by_filter():
    w = ToolWhitelist(["search"])
    w.filter_calls([{"name": "evil"}])
    assert "evil" in w.denied_names


def test_reset_denied_clears():
    w = ToolWhitelist(["search"], raise_on_deny=False)
    w.check("bad")
    w.reset_denied()
    assert w.denied_names == []


# ---------------------------------------------------------------------------
# add / remove
# ---------------------------------------------------------------------------

def test_add_new_tool():
    w = ToolWhitelist(["search"])
    w.add("read")
    assert w.is_allowed("read") is True


def test_remove_tool():
    w = ToolWhitelist(["search", "read"])
    w.remove("read")
    assert w.is_allowed("read") is False
    assert w.is_allowed("search") is True


def test_remove_nonexistent_no_error():
    w = ToolWhitelist(["search"])
    w.remove("nonexistent")  # no exception


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_empty_allowed_raises():
    with pytest.raises(ToolWhitelistError):
        ToolWhitelist([])


# ---------------------------------------------------------------------------
# tool_guard decorator
# ---------------------------------------------------------------------------

def test_tool_guard_allows():
    w = ToolWhitelist(["search"])

    @tool_guard(w)
    def run(name, args):
        return f"ran:{name}"

    assert run("search", {}) == "ran:search"


def test_tool_guard_blocks():
    w = ToolWhitelist(["search"])

    @tool_guard(w)
    def run(name, args):
        return "should not get here"

    with pytest.raises(ToolNotAllowedError):
        run("exec_shell", {})


def test_tool_guard_preserves_name():
    w = ToolWhitelist(["search"])

    @tool_guard(w)
    def dispatch(name, args):
        pass

    assert dispatch.__name__ == "dispatch"
