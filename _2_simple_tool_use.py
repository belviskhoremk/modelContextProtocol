from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import arxiv, json, os
from typing import List
import logging

_ = load_dotenv(find_dotenv())
together_api_key = os.getenv("TOGETHER_API_KEY")
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
PAPER_DIR = 'papers'

logger = logging.getLogger(__name__)

"""
We are going to create two simple tools that load contents from arxiv and extract info to display to 
the user
"""

"""
The first tool searches for relevant arXiv papers based on a topic and stores the papers' 
info in a JSON file (title, authors, summary, paper url and the publication date). 
The JSON files are organized by topics in the papers directory. The tool does not download the papers.
"""


def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """

    logger.info(f"Searching for papers on topic: {topic} (max_results={max_results})")
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = client.results(search)
        logger.info("arXiv search completed successfully.")
    except Exception as e:
        logger.error(f"Error during arXiv search: {e}")
        return []

    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    try:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created or already exists: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return []

    file_path = os.path.join(path, "papers_info.json")

    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
        logger.info(f"Loaded existing papers info from {file_path}")
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}
        logger.info(f"No existing or valid papers info found at {file_path}, starting fresh.")
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {e}")
        return []

    paper_ids = []
    try:
        for paper in papers:
            paper_id = paper.get_short_id()
            paper_ids.append(paper_id)
            paper_info = {
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary,
                'pdf_url': paper.pdf_url,
                'published': str(paper.published.date())
            }
            papers_info[paper_id] = paper_info
        logger.info(f"Processed {len(paper_ids)} papers.")
    except Exception as e:
        logger.error(f"Error processing papers: {e}")
        return []

    try:
        with open(file_path, "w") as json_file:
            json.dump(papers_info, json_file, indent=2)
        logger.info(f"Results are saved in: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save papers info to {file_path}: {e}")
        return []

    return paper_ids

# print(search_papers("LLM"))

"""
The second tool looks for information about a specific paper across all topic directories inside the `papers` directory.
"""
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """
    import datetime

    year = datetime.datetime.now().year
    logger.info(f"[{year}] Starting search for paper_id: {paper_id}")

    try:
        for item in os.listdir(PAPER_DIR):
            item_path = os.path.join(PAPER_DIR, item)
            if os.path.isdir(item_path):
                file_path = os.path.join(item_path, "papers_info.json")
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, "r") as json_file:
                            papers_info = json.load(json_file)
                            if paper_id in papers_info:
                                logger.info(f"[{year}] Found paper_id {paper_id} in {file_path}")
                                return json.dumps(papers_info[paper_id], indent=2)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logger.error(f"[{year}] Error reading {file_path}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"[{year}] Unexpected error reading {file_path}: {str(e)}")
                        continue
    except Exception as e:
        logger.error(f"[{year}] Error listing directories in {PAPER_DIR}: {str(e)}")
        return f"Error searching for paper {paper_id}: {str(e)}"

    logger.info(f"[{year}] No information found for paper_id: {paper_id}")
    return f"There's no saved information related to paper {paper_id}."

print(extract_info('2412.18022v1'))


tools = [
    {
        "name": "search_papers",
        "description": "Search for papers on arXiv based on a topic and store their information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to search for"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to retrieve",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "extract_info",
        "description": "Search for information about a specific paper across all topic directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The ID of the paper to look for"
                }
            },
            "required": ["paper_id"]
        }
    }
]

mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}


def execute_tool(tool_name, tool_args):
    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
    try:
        result = mapping_tool_function[tool_name](**tool_args)
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return f"Error executing tool {tool_name}: {e}"

    if result is None:
        logger.info(f"Tool {tool_name} returned None.")
        result = "The operation completed but didn't return any results."

    elif isinstance(result, list):
        logger.info(f"Tool {tool_name} returned a list with {len(result)} items.")
        result = ', '.join(result)

    elif isinstance(result, dict):
        logger.info(f"Tool {tool_name} returned a dictionary.")
        result = json.dumps(result, indent=2)

    else:
        logger.info(f"Tool {tool_name} returned a value of type {type(result).__name__}.")
        result = str(result)
    return result

"""
When Using Anthropic


client = anthropic.Anthropic()
def process_query(query):
    
    messages = [{'role': 'user', 'content': query}]
    
    response = client.messages.create(max_tokens = 2024,
                                  model = 'claude-3-7-sonnet-20250219', 
                                  tools = tools,
                                  messages = messages)
    
    process_query = True
    while process_query:
        assistant_content = []

        for content in response.content:
            if content.type == 'text':
                
                print(content.text)
                assistant_content.append(content)
                
                if len(response.content) == 1:
                    process_query = False
            
            elif content.type == 'tool_use':
                
                assistant_content.append(content)
                messages.append({'role': 'assistant', 'content': assistant_content})
                
                tool_id = content.id
                tool_args = content.input
                tool_name = content.name
                print(f"Calling tool {tool_name} with args {tool_args}")
                
                result = execute_tool(tool_name, tool_args)
                messages.append({"role": "user", 
                                  "content": [
                                      {
                                          "type": "tool_result",
                                          "tool_use_id": tool_id,
                                          "content": result
                                      }
                                  ]
                                })
                response = client.messages.create(max_tokens = 2024,
                                  model = 'claude-3-7-sonnet-20250219', 
                                  tools = tools,
                                  messages = messages) 
                
                if len(response.content) == 1 and response.content[0].type == "text":
                    print(response.content[0].text)
                    process_query = False
"""

llm = ChatOpenAI(
    base_url='https://api.fireworks.ai/inference/v1',
    api_key=fireworks_api_key,
    model="accounts/fireworks/models/llama4-maverick-instruct-basic",
    temperature=0.0
)


def process_query(query):
    messages = [HumanMessage(content=query)]

    response = llm.bind_tools(tools).invoke(messages)
    print("RESPONSE\n", response)

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

                # Execute the tool
                result = execute_tool(tool_name, tool_args)

                # Add tool result to messages
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id
                )
                messages.append(tool_message)

            # Get next response from the model
            response = llm.bind_tools(tools).invoke(messages)

        else:
            # No tool calls, just text response
            print("Final response:", response.content)
            process_query_loop = False


def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")

chat_loop()