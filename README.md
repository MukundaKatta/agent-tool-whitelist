# agent-tool-whitelist

Enforce an allowlist of tool names in LLM agent loops.

Zero dependencies. Python 3.10+. MIT.

## Install

```bash
pip install agent-tool-whitelist
```

## Usage

```python
from agent_tool_whitelist import ToolWhitelist, ToolNotAllowedError

whitelist = ToolWhitelist(["web_search", "read_file", "write_file"])

# Check before executing
whitelist.check("web_search")   # ok
whitelist.check("exec_shell")   # raises ToolNotAllowedError
```

## Filter a tool_use list (Anthropic or OpenAI format)

```python
# Anthropic content blocks
safe_calls = whitelist.filter_calls(response.content)

# OpenAI function calls
safe_calls = whitelist.filter_calls(response.choices[0].message.tool_calls)
```

`filter_calls` never raises — it silently drops blocked calls and records them
in `whitelist.denied_names`.

## Raise vs filter

```python
# Raise immediately on first blocked call (default)
whitelist.check_calls(tool_calls)   # ToolNotAllowedError on first blocked

# Filter silently, return only allowed calls
safe = whitelist.filter_calls(tool_calls)
```

## Decorator

The wrapped function must take the tool name as its first positional argument.
The whitelist check runs *before* the function body, so blocked tools never
reach your dispatch logic.

```python
from agent_tool_whitelist import tool_guard

@tool_guard(whitelist)
def dispatch_tool(name: str, args: dict) -> Any:
    return run_tool(name, args)

dispatch_tool("web_search", {...})  # ok
dispatch_tool("exec_shell", {...})  # raises ToolNotAllowedError
```

If the whitelist was created with `raise_on_deny=False`, the guard does not
raise: it records the blocked tool in `denied_names`, skips the wrapped
function entirely, and returns `None`.

```python
whitelist = ToolWhitelist(["web_search"], raise_on_deny=False)

@tool_guard(whitelist)
def dispatch_tool(name: str, args: dict) -> Any:
    return run_tool(name, args)

dispatch_tool("exec_shell", {...})  # returns None, run_tool is never called
assert "exec_shell" in whitelist.denied_names
```

## Audit

```python
blocked = whitelist.denied_names   # all tools blocked since creation
whitelist.reset_denied()           # clear the log
```

## API

### `ToolWhitelist(allowed, *, deny_message="Tool '{name}' is not in the allowed list", raise_on_deny=True)`

```python
whitelist.is_allowed(name) -> bool
whitelist.check(name) -> bool        # raises or returns False
whitelist.check_calls(calls) -> None # raises on first blocked
whitelist.filter_calls(calls) -> list[dict]
whitelist.add(name) -> None
whitelist.remove(name) -> None
whitelist.denied_names -> list[str]
whitelist.reset_denied() -> None
```

## Development

The test suite uses only the Python standard library (`unittest`) — no third-party
dependencies are required to run it:

```bash
python3 -m unittest discover -s tests
```

CI runs the same suite across Python 3.10–3.13 and verifies the package builds
and imports from the built wheel (see `.github/workflows/ci.yml`).

## License

MIT
