"""LLM setup and the tool-calling loop shared by the chatbots.

The chatbots use an open model hosted on **Fireworks AI** through its
OpenAI-compatible API, via LangChain's ``ChatOpenAI``.
"""

import os
from collections.abc import Awaitable, Callable

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-20b"

# An async function that runs a tool: (tool_name, tool_args) -> result text.
ToolCaller = Callable[[str, dict], Awaitable[str]]


def make_llm() -> ChatOpenAI:
    """Create the chat model from the FIREWORKS_* environment variables."""
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatOpenAI(
        base_url=FIREWORKS_BASE_URL,
        api_key=api_key,
        model=os.getenv("FIREWORKS_MODEL", DEFAULT_MODEL),
        temperature=0.0,
    )


async def run_agent(llm: ChatOpenAI, tools: list[dict], query: str, call_tool: ToolCaller) -> str:
    """Answer ``query``, letting the model call ``tools`` as many times as it needs.

    The loop:
      1. send the conversation to the model, with the tool definitions;
      2. if the reply contains tool calls, run each one, append the results
         as ``ToolMessage``s and go back to 1;
      3. otherwise the reply is the final answer.

    Args:
        llm: The chat model.
        tools: Tool definitions as {"name", "description", "input_schema"} dicts.
        query: The user's question.
        call_tool: Coroutine that executes a tool and returns its result as text.

    Returns:
        The model's final answer.
    """
    model = llm.bind_tools(tools)
    messages = [HumanMessage(content=query)]

    while True:
        response = await model.ainvoke(messages)
        if not response.tool_calls:
            return response.content

        messages.append(response)  # the assistant turn that requested the tools
        for tool_call in response.tool_calls:
            print(f"→ Calling tool {tool_call['name']} with {tool_call['args']}")
            result = await call_tool(tool_call["name"], tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))


def tool_result_text(result) -> str:
    """Flatten an MCP ``CallToolResult`` into plain text for the model."""
    return "\n".join(item.text for item in result.content if hasattr(item, "text"))
