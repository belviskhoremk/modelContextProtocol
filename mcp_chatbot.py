import os

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import ToolMessage, HumanMessage
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
from langchain_openai import ChatOpenAI


nest_asyncio.apply()    # -> Necessary for event loops in python

_ = load_dotenv(find_dotenv())
together_api_key = os.getenv("TOGETHER_API_KEY")
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.llm = ChatOpenAI(
            base_url='https://api.fireworks.ai/inference/v1',
            api_key=fireworks_api_key,
            model="accounts/fireworks/models/gpt-oss-20b",
            temperature=0.0
        )
        self.available_tools: List[dict] = []

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
                    result = await self.session.call_tool(tool_name, arguments=tool_args)

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

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "mcp_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response.tools]

                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())