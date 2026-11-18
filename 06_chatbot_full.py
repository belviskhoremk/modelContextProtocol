"""Lesson 6: A chatbot that uses tools, resources and prompts.

Connects to every server in ``server_config.json`` (including the lesson 5
server) and adds a small command syntax so the *user* controls resources and
prompts, while the *model* controls tools:

    @folders                  read resource papers://folders
    @<topic>                  read resource papers://<topic>
    /prompts                  list the prompts offered by the servers
    /prompt <name> k=v ...    fill a prompt template and send it to the model
    anything else             normal chat, with tool calling

Run:
    uv run 06_chatbot_full.py
"""

import asyncio
import json
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm import make_llm, run_agent, tool_result_text

CONFIG_FILE = "server_config.json"

HELP = """Commands:
  @folders                   list topics with saved papers
  @<topic>                   show the papers saved for a topic
  /prompts                   list available prompts
  /prompt <name> <arg=value> run a prompt template
  quit                       exit"""


class MCPChatBot:
    def __init__(self):
        self.llm = make_llm()
        self.exit_stack = AsyncExitStack()
        self.available_tools: list[dict] = []
        self.available_prompts: list[dict] = []
        # Maps each tool name, prompt name and resource URI to the session that owns it.
        self.sessions: dict[str, ClientSession] = {}

    # --- Connection -----------------------------------------------------------
    async def connect_to_server(self, name: str, config: dict) -> None:
        """Start one server and register its tools, prompts and resources."""
        try:
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(StdioServerParameters(**config))
            )
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            # The handshake result tells us what the server supports.
            capabilities = (await session.initialize()).capabilities
        except Exception as exc:
            print(f"Failed to connect to {name}: {exc}")
            return

        tools = (await session.list_tools()).tools
        for tool in tools:
            self.sessions[tool.name] = session
            self.available_tools.append(
                {"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema}
            )

        # Prompts and resources are optional server capabilities.
        if capabilities and capabilities.prompts:
            for prompt in (await session.list_prompts()).prompts:
                self.sessions[prompt.name] = session
                self.available_prompts.append(
                    {"name": prompt.name, "description": prompt.description, "arguments": prompt.arguments}
                )
        if capabilities and capabilities.resources:
            for resource in (await session.list_resources()).resources:
                self.sessions[str(resource.uri)] = session
            # Templated resources (e.g. papers://{topic}) are listed separately.
            for template in (await session.list_resource_templates()).resourceTemplates:
                self.sessions[template.uriTemplate] = session

        print(f"Connected to {name} with tools:", [t.name for t in tools])

    async def connect_to_servers(self) -> None:
        with open(CONFIG_FILE) as f:
            servers = json.load(f).get("mcpServers", {})
        for name, config in servers.items():
            await self.connect_to_server(name, config)

    # --- Tools ----------------------------------------------------------------
    async def call_tool(self, name: str, args: dict) -> str:
        result = await self.sessions[name].call_tool(name, arguments=args)
        return tool_result_text(result)

    async def ask(self, query: str) -> None:
        print("\n" + await run_agent(self.llm, self.available_tools, query, self.call_tool))

    # --- Resources ------------------------------------------------------------
    async def get_resource(self, uri: str) -> None:
        """Read a resource and print it."""
        # Concrete URIs (papers://folders) are registered directly; others
        # (papers://<topic>) match a template such as papers://{topic}.
        session = self.sessions.get(uri) or next(
            (s for key, s in self.sessions.items()
             if "{" in key and uri.startswith(key.split("{")[0])),
            None,
        )
        if session is None:
            print(f"Resource '{uri}' not found.")
            return
        result = await session.read_resource(uri)
        print("\n" + (result.contents[0].text if result.contents else "No content available."))

    # --- Prompts --------------------------------------------------------------
    def list_prompts(self) -> None:
        if not self.available_prompts:
            print("No prompts available.")
            return
        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            for arg in prompt["arguments"] or []:
                print(f"    - {arg.name}{'' if arg.required else ' (optional)'}")

    async def execute_prompt(self, name: str, args: dict) -> None:
        """Fill the prompt template on the server, then send it to the model."""
        session = self.sessions.get(name)
        if session is None:
            print(f"Prompt '{name}' not found.")
            return
        result = await session.get_prompt(name, arguments=args)
        text = "\n".join(m.content.text for m in result.messages if hasattr(m.content, "text"))
        print(f"\nExecuting prompt '{name}'...")
        await self.ask(text)

    # --- Chat loop ------------------------------------------------------------
    async def handle(self, query: str) -> None:
        """Dispatch one line of user input."""
        if query.startswith("@"):  # resource
            topic = query[1:]
            await self.get_resource("papers://folders" if topic == "folders" else f"papers://{topic}")
        elif query.startswith("/"):  # command
            command, *rest = query.split()
            if command == "/prompts":
                self.list_prompts()
            elif command == "/prompt" and rest:
                args = dict(arg.split("=", 1) for arg in rest[1:] if "=" in arg)
                await self.execute_prompt(rest[0], args)
            else:
                print(HELP)
        else:  # normal chat
            await self.ask(query)

    async def chat_loop(self) -> None:
        print("\nMCP Chatbot started!\n" + HELP)
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break
            if not query:
                continue
            try:
                await self.handle(query)
            except Exception as exc:
                print(f"\nError: {exc}")

    async def cleanup(self) -> None:
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
