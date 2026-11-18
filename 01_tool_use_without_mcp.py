"""Lesson 1: Tool use *without* MCP.

Before introducing MCP, we wire tools to an LLM by hand:

1. describe each tool with a JSON schema (``TOOLS``);
2. map tool names to Python functions (``TOOL_FUNCTIONS``);
3. when the model asks for a tool, run the function ourselves and send the
   result back.

This works, but every application has to repeat this glue code for every tool.
That repetition is exactly what MCP removes (see lesson 2).

Run:
    uv run 01_tool_use_without_mcp.py
"""

import asyncio
import json

from llm import make_llm, run_agent
from research_tools import extract_info, search_papers

# Tool definitions sent to the model: name, description and a JSON schema of
# the arguments. The model uses them to decide when and how to call a tool.
TOOLS = [
    {
        "name": "search_papers",
        "description": "Search for papers on arXiv based on a topic and store their information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic to search for"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to retrieve",
                    "default": 5,
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "extract_info",
        "description": "Search for information about a specific paper across all topic directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "The ID of the paper to look for"},
            },
            "required": ["paper_id"],
        },
    },
]

# Which Python function implements each tool.
TOOL_FUNCTIONS = {
    "search_papers": search_papers,
    "extract_info": extract_info,
}


async def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Run a tool locally and turn its result into text for the model."""
    try:
        result = TOOL_FUNCTIONS[tool_name](**tool_args)
    except Exception as exc:  # report the error to the model instead of crashing
        return f"Error executing tool {tool_name}: {exc}"

    if result is None:
        return "The operation completed but didn't return any results."
    if isinstance(result, list):
        return ", ".join(result)
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)


async def chat_loop() -> None:
    llm = make_llm()
    print("Type your queries or 'quit' to exit.")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue
        try:
            print("\n" + await run_agent(llm, TOOLS, query, execute_tool))
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    asyncio.run(chat_loop())
