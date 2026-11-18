"""Lesson 4: A chatbot connected to several MCP servers.

Servers are listed in ``server_config.json`` (the same format Claude Desktop
uses). Besides our research server, we connect to two **reference servers**
maintained by the MCP project:

* ``filesystem``: read/write files in the current directory (``.``);
* ``fetch``: download a web page and convert it to Markdown.

``npx`` / ``uvx`` download and run them on the fly, with no install needed.

Each server gets its own ``ClientSession``. We remember which session owns
each tool so that tool calls are routed to the right server. An
``AsyncExitStack`` keeps every connection open and closes them all at the end.

Run:
    uv run 04_chatbot_multi_server.py
"""

import asyncio
import json
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm import make_llm, run_agent, tool_result_text

CONFIG_FILE = "server_config.json"


class MCPChatBot:
    def __init__(self):
        self.llm = make_llm()
        self.exit_stack = AsyncExitStack()
        self.available_tools: list[dict] = []
        self.tool_to_session: dict[str, ClientSession] = {}

    async def connect_to_server(self, name: str, config: dict) -> None:
        """Start one server, open a session and register its tools."""
        try:
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(StdioServerParameters(**config))
            )
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = (await session.list_tools()).tools
            print(f"Connected to {name} with tools:", [t.name for t in tools])
            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append(
                    {"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema}
                )
        except Exception as exc:  # one broken server should not stop the others
            print(f"Failed to connect to {name}: {exc}")

    async def connect_to_servers(self) -> None:
        """Connect to every server listed in the config file."""
        with open(CONFIG_FILE) as f:
            servers = json.load(f).get("mcpServers", {})
        for name, config in servers.items():
            await self.connect_to_server(name, config)

    async def call_tool(self, name: str, args: dict) -> str:
        """Route a tool call to the server that provides the tool."""
        result = await self.tool_to_session[name].call_tool(name, arguments=args)
        return tool_result_text(result)

    async def chat_loop(self) -> None:
        print("\nMCP Chatbot started! Type your queries or 'quit' to exit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break
            if not query:
                continue
            try:
                print("\n" + await run_agent(self.llm, self.available_tools, query, self.call_tool))
            except Exception as exc:
                print(f"\nError: {exc}")

    async def cleanup(self) -> None:
        """Close every session and stop every server subprocess."""
        await self.exit_stack.aclose()


async def main() -> None:
    chatbot = MCPChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
