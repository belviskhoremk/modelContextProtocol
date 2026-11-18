"""Lesson 2: A first MCP server, exposing tools.

``FastMCP`` turns plain Python functions into MCP tools: the function name,
docstring and type hints become the tool's name, description and JSON schema.
No hand-written schema like in lesson 1 is needed.

Any MCP client (our chatbots, Claude Desktop, an IDE...) can now use these
tools without writing any glue code.

Run (normally the client starts it for you):
    uv run 02_research_server.py

Inspect it in a browser:
    npx @modelcontextprotocol/inspector uv run 02_research_server.py
"""

from mcp.server.fastmcp import FastMCP

import research_tools

mcp = FastMCP("research")

# Register the shared functions as MCP tools.
mcp.tool()(research_tools.search_papers)
mcp.tool()(research_tools.extract_info)


if __name__ == "__main__":
    # stdio transport: the client launches this script as a subprocess and
    # talks to it over stdin/stdout (so never print() to stdout here).
    mcp.run(transport="stdio")
