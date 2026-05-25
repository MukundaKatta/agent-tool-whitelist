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

```python
from agent_tool_whitelist import tool_guard

@tool_guard(whitelist)
def dispatch_tool(name: str, args: dict) -> Any:
    return run_tool(name, args)

dispatch_tool("web_search", {...})  # ok
dispatch_tool("exec_shell", {...})  # raises ToolNotAllowedError
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

## License

MIT
