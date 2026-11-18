# 🔌 Learning the Model Context Protocol (MCP)

A step-by-step project for learning the
[Model Context Protocol](https://modelcontextprotocol.io/) in Python. It goes
from wiring tools to an LLM by hand, to a chatbot that talks to several MCP
servers and uses all three MCP primitives: **tools, resources and prompts**.

The running example is a **research assistant**: it searches arXiv, saves
paper metadata and answers questions about it. Reference servers add
filesystem access and web fetching.

## Lessons

Each lesson is a runnable script that builds on the previous one.

| # | File | What you learn |
|---|------|----------------|
| – | [`docs/mcp_concepts.md`](docs/mcp_concepts.md) | **Start here**: why MCP, host/client/server architecture, tools vs resources vs prompts, transports |
| 1 | [`01_tool_use_without_mcp.py`](01_tool_use_without_mcp.py) | Tool calling *without* MCP: hand-written JSON schemas and dispatch. The glue code MCP removes |
| 2 | [`02_research_server.py`](02_research_server.py) | A first MCP server with `FastMCP`: plain functions become tools |
| 3 | [`03_chatbot_single_server.py`](03_chatbot_single_server.py) | An MCP host: launch a server over stdio, discover its tools, route tool calls |
| 4 | [`04_chatbot_multi_server.py`](04_chatbot_multi_server.py) | Connect to **several servers** from `server_config.json`, including the reference `filesystem` and `fetch` servers |
| 5 | [`05_research_server_full.py`](05_research_server_full.py) | Add **resources** (`papers://folders`, templated `papers://{topic}`) and a **prompt** template |
| 6 | [`06_chatbot_full.py`](06_chatbot_full.py) | A chatbot using tools, resources (`@topic`) and prompts (`/prompt`) |

Shared code:

- [`research_tools.py`](research_tools.py): the arXiv `search_papers` and
  `extract_info` functions used by every lesson.
- [`llm.py`](llm.py): the LLM client and the tool-calling loop
  (model → tool calls → results → model → ... → answer).

## Architecture (lesson 6)

```
┌──────────────── 06_chatbot_full.py (host) ────────────────┐
│  LLM (Fireworks, via LangChain)                           │
│  ClientSession ◄─stdio─► 05_research_server_full.py       │  tools, resources, prompt
│  ClientSession ◄─stdio─► @modelcontextprotocol/server-filesystem │  read/write files
│  ClientSession ◄─stdio─► mcp-server-fetch                 │  fetch web pages
└───────────────────────────────────────────────────────────┘
```

## Setup

**Requirements:** [uv](https://docs.astral.sh/uv/), Python 3.10+ (pinned to
3.12 in `.python-version`), Node.js (for the `npx` filesystem server), and a
[Fireworks AI](https://fireworks.ai/) API key.

```bash
git clone https://github.com/belviskhoremk/modelContextProtocol.git
cd modelContextProtocol
uv sync
cp .env.example .env    # add your FIREWORKS_API_KEY
```

The chatbots use Fireworks' OpenAI-compatible API through `langchain-openai`.
Set `FIREWORKS_MODEL` to use another model that supports tool calling.

## Usage

```bash
uv run 06_chatbot_full.py
```

```text
Query: find 3 recent papers about retrieval augmented generation
→ Calling tool search_papers with {'topic': 'retrieval augmented generation', 'max_results': 3}
...
Query: @folders                                      # read a resource
Query: @retrieval_augmented_generation               # papers saved for a topic
Query: /prompts                                      # list prompt templates
Query: /prompt generate_search_prompt topic=mcp num_papers=3
Query: save a summary of these papers to summary.md  # filesystem server
Query: quit
```

Paper metadata is saved under `papers/<topic>/papers_info.json`.

### Inspect a server

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is a web
UI for calling a server's tools, resources and prompts directly:

```bash
npx @modelcontextprotocol/inspector uv run 05_research_server_full.py
```

### Use the server in Claude Desktop

Add it to `claude_desktop_config.json` (use absolute paths):

```json
{
  "mcpServers": {
    "research": {
      "command": "uv",
      "args": ["--directory", "/path/to/modelContextProtocol", "run", "05_research_server_full.py"]
    }
  }
}
```

## Key takeaways

- **Tools** are chosen by the model, **resources** by the application,
  **prompts** by the user.
- With the stdio transport, **stdout is the protocol channel**, so servers must
  log to stderr, never `print()`.
- One `ClientSession` per server. Keep a map from each tool, prompt or resource
  to its session to route requests.
- `AsyncExitStack` keeps many connections open and closes them cleanly.

## Resources

- [MCP documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Reference servers](https://github.com/modelcontextprotocol/servers)
- DeepLearning.AI short course *MCP: Build Rich-Context AI Apps with Anthropic*:
  the research-assistant example is based on it
