"""arXiv research helpers shared by every lesson.

Two plain functions that the lessons expose as tools, first by hand
(lesson 1), then through MCP servers (lessons 2 and 5):

* ``search_papers``: search arXiv and save the results' metadata to
  ``papers/<topic>/papers_info.json`` (papers are not downloaded);
* ``extract_info``: look up a saved paper by id across all topics.

Logging goes to **stderr**: MCP stdio servers use stdout for the protocol.
"""

import json
import logging
import sys
from pathlib import Path

import arxiv

PAPER_DIR = Path("papers")

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("research")


def topic_dir(topic: str) -> Path:
    """Folder where papers of ``topic`` are stored, e.g. "Deep Learning" -> papers/deep_learning."""
    return PAPER_DIR / topic.lower().replace(" ", "_")


def load_papers(path: Path) -> dict:
    """Read a papers_info.json file, returning {} if it is missing or invalid."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def search_papers(topic: str, max_results: int = 5) -> list[str]:
    """Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for.
        max_results: Maximum number of results to retrieve (default: 5).

    Returns:
        List of paper IDs found in the search.
    """
    logger.info("Searching arXiv for %r (max_results=%d)", topic, max_results)
    search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    folder = topic_dir(topic)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / "papers_info.json"

    # Merge new results into what was already saved for this topic.
    papers_info = load_papers(file_path)
    paper_ids = []
    for paper in arxiv.Client().results(search):
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)
        papers_info[paper_id] = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
        }

    file_path.write_text(json.dumps(papers_info, indent=2))
    logger.info("Saved %d papers to %s", len(paper_ids), file_path)
    return paper_ids


def extract_info(paper_id: str) -> str:
    """Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for.

    Returns:
        JSON string with paper information if found, error message if not found.
    """
    for file_path in PAPER_DIR.glob("*/papers_info.json"):
        papers_info = load_papers(file_path)
        if paper_id in papers_info:
            return json.dumps(papers_info[paper_id], indent=2)
    return f"There's no saved information related to paper {paper_id}."
