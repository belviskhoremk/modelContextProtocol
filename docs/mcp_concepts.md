# MCP Concepts: Study Notes

## Why MCP?

Models are only as good as the context they are given. The **Model Context
Protocol (MCP)** is an open protocol that standardizes how LLM applications
connect to external tools and data sources.

Everything MCP does can be done without it. Without a standard, though, AI
development is fragmented: every app rebuilds the same integrations (Google
Drive, a database, GitHub...) again and again. With MCP, an integration is
written **once as a server** and reused by any MCP-compatible app:

```
MCP-compatible app ──► Data-store MCP server      ──► databases
                   ──► CRM MCP server             ──► CRM systems
                   ──► Version-control MCP server ──► Git hosting

Claude Desktop ┐
AI agent       ├──► Google Drive MCP server ──► Google Drive
IDE assistant  ┘
```

## Architecture: client-server

```
┌──────────── HOST (LLM application) ────────────┐
│  MCP client ◄── MCP protocol ──► MCP server A  │
│  MCP client ◄── MCP protocol ──► MCP server B  │
└────────────────────────────────────────────────┘
```

- **Host**: the LLM application that wants to access data (a chatbot, an
  IDE, Claude Desktop...).
- **MCP server**: a lightweight program that exposes specific capabilities.
- **MCP client**: lives inside the host and keeps a **1:1 connection** with
  one server. It invokes tools, queries resources and fills in prompts.

## The three primitives a server exposes

### 1. Tools: functions the model can call

```python
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number to add
        b: Second number to add
    """
    return a + b
```

The function name, docstring and type hints become the tool's name,
description and JSON schema. The **model** decides when to call a tool.

### 2. Resources: read-only data

The server exposes data at a URI. The **application** decides when to read
it (for example, when the user types `@topic`).

```python
@mcp.resource("docs://documents", mime_type="application/json")
def list_docs():
    """Return the list of document names."""

@mcp.resource("docs://documents/{doc_id}", mime_type="text/plain")
def fetch_doc(doc_id: str):
    """Return the contents of one document (a templated resource)."""
```

### 3. Prompts: reusable prompt templates

Predefined, well-engineered prompts that the **user** picks, so they don't
have to write complex prompts themselves.

```python
@mcp.prompt(name="format", description="Rewrite a document in Markdown")
def format_document(doc_id: str) -> list[base.Message]:
    ...
```

| Primitive | Controlled by | Example |
|-----------|---------------|---------|
| Tool | the model | `search_papers(topic)` |
| Resource | the application | `papers://folders` |
| Prompt | the user | `/prompt generate_search_prompt topic=rag` |

## Communication lifecycle

1. **Initialization**: the client and server exchange capabilities
   (`session.initialize()`).
2. **Message exchange**: requests (`list_tools`, `call_tool`,
   `read_resource`, ...), responses and notifications.
3. **Termination**: the connection is closed.

## Transports

| Transport | Use | Notes |
|-----------|-----|-------|
| **stdio** | Local servers | The client launches the server as a subprocess and talks over stdin/stdout |
| HTTP + SSE | Remote (legacy) | Stateful connection |
| **Streamable HTTP** | Remote (recommended) | Supports stateless and stateful connections |

> ⚠️ With stdio, **stdout is the protocol channel**. A server must never
> `print()` to stdout. Log to stderr instead, or the client receives
> corrupted messages.

## Low-level client (what the chatbots do under the hood)

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Tell the client how to launch the server.
server_params = StdioServerParameters(command="uv", args=["run", "02_research_server.py"])

async def run():
    # Launch the server as a subprocess; get read/write streams to it.
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                   # 1:1 handshake
            tools = await session.list_tools()           # discover tools
            result = await session.call_tool("search_papers", arguments={"topic": "rag"})
```

## Testing a server with the MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run 02_research_server.py
```

`npx` downloads and runs the Inspector on the fly, so nothing needs to be
installed globally. It opens a web UI where you can list and call the server's
tools, resources and prompts.
