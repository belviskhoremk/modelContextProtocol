"""Lesson 5: An MCP server exposing tools, resources and prompts.

On top of the lesson 2 tools, this server adds:

* **Resources** (read-only data the *application* decides to load):
    - ``papers://folders``: list of topics with saved papers;
    - ``papers://{topic}``: details of the papers saved for one topic
      (a *templated* resource: ``{topic}`` is a parameter).
* **Prompt** (a template the *user* picks):
    - ``generate_search_prompt``: a well-engineered research prompt, so
      users get good results without writing complex prompts themselves.

Run:
    uv run 05_research_server_full.py

Inspect it:
    npx @modelcontextprotocol/inspector uv run 05_research_server_full.py
"""

from mcp.server.fastmcp import FastMCP

import research_tools
from research_tools import PAPER_DIR, load_papers, topic_dir

mcp = FastMCP("research")

# --- Tools (same as lesson 2) -------------------------------------------------
mcp.tool()(research_tools.search_papers)
mcp.tool()(research_tools.extract_info)


# --- Resources ----------------------------------------------------------------
@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """List all available topic folders in the papers directory."""
    folders = sorted(p.parent.name for p in PAPER_DIR.glob("*/papers_info.json"))

    content = "# Available Topics\n\n"
    if not folders:
        return content + "No topics found.\n"
    content += "".join(f"- {folder}\n" for folder in folders)
    return content + "\nUse @<topic> to access papers in that topic.\n"


@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """Get detailed information about papers on a specific topic.

    Args:
        topic: The research topic to retrieve papers for.
    """
    papers = load_papers(topic_dir(topic) / "papers_info.json")
    if not papers:
        return f"# No papers found for topic: {topic}\n\nTry searching for papers on this topic first."

    content = f"# Papers on {topic.replace('_', ' ').title()}\n\nTotal papers: {len(papers)}\n\n"
    for paper_id, info in papers.items():
        content += (
            f"## {info['title']}\n"
            f"- **Paper ID**: {paper_id}\n"
            f"- **Authors**: {', '.join(info['authors'])}\n"
            f"- **Published**: {info['published']}\n"
            f"- **PDF URL**: [{info['pdf_url']}]({info['pdf_url']})\n\n"
            f"### Summary\n{info['summary'][:500]}...\n\n---\n\n"
        )
    return content


# --- Prompts ------------------------------------------------------------------
@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """Generate a prompt to find and discuss academic papers on a specific topic."""
    return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. Follow these instructions:
    1. First, search for papers using search_papers(topic='{topic}', max_results={num_papers})
    2. For each paper found, extract and organize the following information:
       - Paper title
       - Authors
       - Publication date
       - Brief summary of the key findings
       - Main contributions or innovations
       - Methodologies used
       - Relevance to the topic '{topic}'

    3. Provide a comprehensive summary that includes:
       - Overview of the current state of research in '{topic}'
       - Common themes and trends across the papers
       - Key research gaps or areas for future investigation
       - Most impactful or influential papers in this area

    4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.

    Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""


if __name__ == "__main__":
    mcp.run(transport="stdio")
