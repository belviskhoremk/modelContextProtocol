import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import List, TypedDict, Dict

import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

nest_asyncio.apply()  # -> Necessary for event loops in python

_ = load_dotenv(find_dotenv())
together_api_key = os.getenv("TOGETHER_API_KEY")
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

"""
We first need to setup a json file to tell the Client how we want to connect to each individual server
For the reference servers, the commands `npx` and `uvx` directly install the files of the servers to your local environment (you don't need to install them ahead of time). Note for the `filesystem`, the `.` is provided as the third argument and it means "current directory". This means that you're allowing for the `fetch` server to interact with the files and directories that are within the current directory.
"""


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()  # To manage multiple async context managers , our connexions for reading and writing and managing entire connexion to the session
        self.llm = ChatOpenAI(
            base_url='https://api.fireworks.ai/inference/v1',
            api_key=fireworks_api_key,
            model="accounts/fireworks/models/gpt-oss-20b",
            temperature=0.0
        )
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )  # new
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )  # new
            await session.initialize()
            self.sessions.append(session)

            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])

            for tool in tools:  # new
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):  # new
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query):
        messages = [HumanMessage(content=query)]

        # Use ainvoke for async operation
        response = await self.llm.bind_tools(self.available_tools).ainvoke(messages)

        process_query_loop = True
        while process_query_loop:
            # Check if response has tool calls
            if response.tool_calls:
                print(f"AI Response: {response.content}")

                # Add the AI message with tool calls to conversation
                messages.append(response)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # Call tool through the session
                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name, arguments=tool_args)

                    # Add tool result to messages
                    tool_message = ToolMessage(
                        content=str(result.content),
                        tool_call_id=tool_id
                    )
                    messages.append(tool_message)

                # Get next response from the model (async)
                response = await self.llm.bind_tools(self.available_tools).ainvoke(messages)

            else:
                # No tool calls, just text response
                print("Final response:", response.content)
                process_query_loop = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()
        print("Cleaned up all sessions.")


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
