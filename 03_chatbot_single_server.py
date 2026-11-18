"""Lesson 3: A chatbot (MCP host) connected to one MCP server.

The chatbot no longer knows anything about the tools. It:

1. launches the research server (lesson 2) as a subprocess over stdio;
2. asks it which tools it offers (``list_tools``);
3. hands those tool definitions to the LLM;
4. forwards every tool call to the server (``call_tool``).

Run:
    uv run 03_chatbot_single_server.py
"""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm import make_llm, run_agent, tool_result_text

# How to start the server we want to talk to.
SERVER = StdioServerParameters(command="uv", args=["run", "02_research_server.py"])


class MCPChatBot:
    def __init__(self):
        self.llm = make_llm()
        self.session: ClientSession | None = None
        self.available_tools: list[dict] = []

    async def call_tool(self, name: str, args: dict) -> str:
        """Execute a tool on the MCP server."""
        result = await self.session.call_tool(name, arguments=args)
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

    async def run(self) -> None:
        # stdio_client launches the server and returns read/write streams;
        # ClientSession speaks the MCP protocol over them.
        async with stdio_client(SERVER) as (read, write), ClientSession(read, write) as session:
            self.session = session
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("Connected to server with tools:", [tool.name for tool in tools])
            self.available_tools = [
                {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                for t in tools
            ]
            await self.chat_loop()


if __name__ == "__main__":
    asyncio.run(MCPChatBot().run())
