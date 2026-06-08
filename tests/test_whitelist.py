"""Tests for agent-tool-whitelist.

These tests use only the Python standard-library ``unittest`` framework so they
run with no third-party dependencies::

    python3 -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_tool_whitelist import (  # noqa: E402
    ToolNotAllowedError,
    ToolWhitelist,
    ToolWhitelistError,
    tool_guard,
)


class IsAllowedTests(unittest.TestCase):
    def test_is_allowed_true(self):
        w = ToolWhitelist(["search", "read"])
        self.assertIs(w.is_allowed("search"), True)

    def test_is_allowed_false(self):
        w = ToolWhitelist(["search", "read"])
        self.assertIs(w.is_allowed("exec_shell"), False)

    def test_is_allowed_empty_string(self):
        w = ToolWhitelist(["search"])
        self.assertIs(w.is_allowed(""), False)


class CheckTests(unittest.TestCase):
    def test_check_allowed_returns_true(self):
        w = ToolWhitelist(["search"])
        self.assertIs(w.check("search"), True)

    def test_check_blocked_raises(self):
        w = ToolWhitelist(["search"])
        with self.assertRaises(ToolNotAllowedError):
            w.check("exec_shell")

    def test_check_error_message_contains_name(self):
        w = ToolWhitelist(["search"])
        with self.assertRaises(ToolNotAllowedError) as ctx:
            w.check("exec_shell")
        self.assertIn("exec_shell", str(ctx.exception))

    def test_check_custom_deny_message(self):
        w = ToolWhitelist(["search"], deny_message="blocked: {name}")
        with self.assertRaises(ToolNotAllowedError) as ctx:
            w.check("exec_shell")
        self.assertEqual(str(ctx.exception), "blocked: exec_shell")

    def test_check_raise_on_deny_false_returns_false(self):
        w = ToolWhitelist(["search"], raise_on_deny=False)
        self.assertIs(w.check("exec_shell"), False)

    def test_check_raise_on_deny_false_no_exception(self):
        w = ToolWhitelist(["search"], raise_on_deny=False)
        w.check("anything")  # should not raise


class CheckCallsTests(unittest.TestCase):
    def test_check_calls_all_allowed(self):
        w = ToolWhitelist(["search", "read"])
        calls = [
            {"type": "tool_use", "name": "search", "input": {}},
            {"type": "tool_use", "name": "read", "input": {}},
        ]
        w.check_calls(calls)  # no exception

    def test_check_calls_blocked_raises(self):
        w = ToolWhitelist(["search"])
        calls = [{"type": "tool_use", "name": "exec_shell", "input": {}}]
        with self.assertRaises(ToolNotAllowedError) as ctx:
            w.check_calls(calls)
        self.assertIn("exec_shell", str(ctx.exception))

    def test_check_calls_openai_format(self):
        w = ToolWhitelist(["search"])
        calls = [{"function": {"name": "search", "arguments": "{}"}}]
        w.check_calls(calls)  # no exception

    def test_check_calls_openai_blocked(self):
        w = ToolWhitelist(["search"])
        calls = [{"function": {"name": "exec_shell", "arguments": "{}"}}]
        with self.assertRaises(ToolNotAllowedError):
            w.check_calls(calls)

    def test_check_calls_empty_list(self):
        w = ToolWhitelist(["search"])
        w.check_calls([])  # no exception

    def test_check_calls_stops_at_first_blocked(self):
        # The second call is also blocked but the first should raise first,
        # and only the first blocked name should be recorded.
        w = ToolWhitelist(["search"])
        calls = [{"name": "first_bad"}, {"name": "second_bad"}]
        with self.assertRaises(ToolNotAllowedError):
            w.check_calls(calls)
        self.assertEqual(w.denied_names, ["first_bad"])

    def test_check_calls_ignores_unnamed_call(self):
        # A malformed entry with no extractable name must not raise.
        w = ToolWhitelist(["search"])
        w.check_calls([{"input": {}}, {"name": "search"}])


class FilterCallsTests(unittest.TestCase):
    def test_filter_keeps_allowed(self):
        w = ToolWhitelist(["search", "read"])
        calls = [
            {"name": "search", "input": {}},
            {"name": "exec_shell", "input": {}},
            {"name": "read", "input": {}},
        ]
        result = w.filter_calls(calls)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "search")
        self.assertEqual(result[1]["name"], "read")

    def test_filter_drops_blocked(self):
        w = ToolWhitelist(["search"])
        self.assertEqual(w.filter_calls([{"name": "exec_shell"}]), [])

    def test_filter_never_raises(self):
        w = ToolWhitelist(["search"])
        calls = [{"name": "exec_shell"}, {"name": "rm_rf"}]
        self.assertEqual(w.filter_calls(calls), [])

    def test_filter_empty_list(self):
        w = ToolWhitelist(["search"])
        self.assertEqual(w.filter_calls([]), [])

    def test_filter_records_denied_names(self):
        w = ToolWhitelist(["search"])
        w.filter_calls([{"name": "exec_shell"}, {"name": "delete_file"}])
        self.assertIn("exec_shell", w.denied_names)
        self.assertIn("delete_file", w.denied_names)

    def test_filter_openai_format(self):
        w = ToolWhitelist(["search"])
        calls = [
            {"function": {"name": "search", "arguments": "{}"}},
            {"function": {"name": "exec_shell", "arguments": "{}"}},
        ]
        result = w.filter_calls(calls)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "search")

    def test_filter_ignores_unnamed_call(self):
        w = ToolWhitelist(["search"])
        result = w.filter_calls([{"input": {}}, {"name": "search"}])
        self.assertEqual(len(result), 1)
        self.assertEqual(w.denied_names, [])


class DeniedNamesTests(unittest.TestCase):
    def test_denied_names_empty_initially(self):
        w = ToolWhitelist(["search"])
        self.assertEqual(w.denied_names, [])

    def test_denied_names_populated_by_check(self):
        w = ToolWhitelist(["search"], raise_on_deny=False)
        w.check("bad_tool")
        self.assertIn("bad_tool", w.denied_names)

    def test_denied_names_populated_by_filter(self):
        w = ToolWhitelist(["search"])
        w.filter_calls([{"name": "evil"}])
        self.assertIn("evil", w.denied_names)

    def test_denied_names_returns_a_copy(self):
        # Mutating the returned list must not corrupt internal state.
        w = ToolWhitelist(["search"], raise_on_deny=False)
        w.check("bad")
        snapshot = w.denied_names
        snapshot.append("tampered")
        self.assertNotIn("tampered", w.denied_names)

    def test_reset_denied_clears(self):
        w = ToolWhitelist(["search"], raise_on_deny=False)
        w.check("bad")
        w.reset_denied()
        self.assertEqual(w.denied_names, [])


class AddRemoveTests(unittest.TestCase):
    def test_add_new_tool(self):
        w = ToolWhitelist(["search"])
        w.add("read")
        self.assertIs(w.is_allowed("read"), True)

    def test_add_is_idempotent(self):
        w = ToolWhitelist(["search"])
        w.add("read")
        w.add("read")
        self.assertEqual(w.allowed.count("read"), 1)

    def test_remove_tool(self):
        w = ToolWhitelist(["search", "read"])
        w.remove("read")
        self.assertIs(w.is_allowed("read"), False)
        self.assertIs(w.is_allowed("search"), True)

    def test_remove_nonexistent_no_error(self):
        w = ToolWhitelist(["search"])
        w.remove("nonexistent")  # no exception


class ValidationTests(unittest.TestCase):
    def test_empty_allowed_raises(self):
        with self.assertRaises(ToolWhitelistError):
            ToolWhitelist([])

    def test_duplicate_allowed_names_collapse(self):
        w = ToolWhitelist(["search", "search", "read"])
        self.assertIs(w.is_allowed("search"), True)
        self.assertIs(w.is_allowed("read"), True)
        # Duplicates are removed but first-seen order is preserved.
        self.assertEqual(w.allowed, ["search", "read"])


class ToolGuardTests(unittest.TestCase):
    def test_tool_guard_allows(self):
        w = ToolWhitelist(["search"])

        @tool_guard(w)
        def run(name, args):
            return f"ran:{name}"

        self.assertEqual(run("search", {}), "ran:search")

    def test_tool_guard_blocks(self):
        w = ToolWhitelist(["search"])

        @tool_guard(w)
        def run(name, args):
            return "should not get here"

        with self.assertRaises(ToolNotAllowedError):
            run("exec_shell", {})

    def test_tool_guard_preserves_name(self):
        w = ToolWhitelist(["search"])

        @tool_guard(w)
        def dispatch(name, args):
            pass

        self.assertEqual(dispatch.__name__, "dispatch")

    def test_tool_guard_preserves_docstring(self):
        w = ToolWhitelist(["search"])

        @tool_guard(w)
        def dispatch(name, args):
            """Run a tool by name."""

        self.assertEqual(dispatch.__doc__, "Run a tool by name.")

    def test_tool_guard_passes_through_args_and_kwargs(self):
        w = ToolWhitelist(["search"])

        @tool_guard(w)
        def run(name, *args, **kwargs):
            return (name, args, kwargs)

        self.assertEqual(
            run("search", 1, 2, flag=True),
            ("search", (1, 2), {"flag": True}),
        )

    def test_tool_guard_skips_blocked_when_not_raising(self):
        # With raise_on_deny=False the guard must NOT execute the wrapped
        # function for a blocked tool; it returns None and records the denial.
        w = ToolWhitelist(["search"], raise_on_deny=False)
        calls = []

        @tool_guard(w)
        def run(name, args):
            calls.append(name)
            return "executed"

        result = run("exec_shell", {})
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertIn("exec_shell", w.denied_names)

    def test_tool_guard_runs_allowed_when_not_raising(self):
        w = ToolWhitelist(["search"], raise_on_deny=False)

        @tool_guard(w)
        def run(name, args):
            return "executed"

        self.assertEqual(run("search", {}), "executed")


if __name__ == "__main__":
    unittest.main()
