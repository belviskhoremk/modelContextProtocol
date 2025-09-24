"""
It helps to connect applications to external resources like gdrive, etc.
Models are only as good as the context given to them

MCP protocol is an open source protocol that standardizes how your LLM applications connect to and work with
your tools and data sources. It standardizes how AI apps interact with external systems.

Everything we do with MCP can be done without. but the thing is, without mcp, we have a fragmented AI development
where we have to rebuild the same integration over and over again

e.g.
MCP compatible app  -> Data Store MCP Server(query and fetch data, etc.) -> Dta stores(databases, etc.)
                    -> CRM MCP Server   -> CRM Systems
                    -> Version control MCP server(Commit changes, etc.)  -> Version control software

MCP servers are reusable by various AI apps

MCP Compatible AI assistant
MCP compatible AI agent             -> Google Drive MCP server  -> Google Drive
MCP compatible Desktop app



MCP has a CLIENT-SERVER architecture

HOST
MCP CLIENT  <-  MCP PROTOCOL    -> MCP SERVER
MCP CLIENT  <-  MCP PROTOCOL    -> MCP SERVER

The host are LLM apps that want to access data through MCP
MCP servers are lightweight programs that each expose specific capabilities through MCP
MCP clients maintain 1:1 connections with servers, inside the host application

MCP Client onvokes tools, query resources, interpolates prompts
MCP Server exposes toos, resources and prompt templates(predefines, and the client can access them.)



MCP provides  SDK for building servers and clients.

e.g.
@mcp.tool
def add(a: int, b: int) -> int:
    '''
    Args
        a: First number to add
        b: Second number to add
    Returns:
        The sum of the tqo numbers
    '''
    return a + b


For resources, we allow the server to expose data to the client.
We specify a url/location where the client goes to find the data
e.g.
@mcp.resource("docs://documents", mime_type = "application/json")
def list_docs():
    #Return a list of documents names

@mcp.resource("docs://documents/{doc_id}", mime_type = "text/plain")
def fetch_docs(doc_id: str):
    #Return the contents of a doc

For prompt, we give it a nma and description.
They define a set of User and Assistant messages that can be used by the client
e.g:
@mcp.prompt(name='format', description='Rewrites the contents of a documetn in Markdown format')
def format_document(doc_id: str) -> list[base.Message]:
    #Return a list of messages


communication lifecycle
initialization
message exchange
termination

MCP Transports
we use stdio for locally

remotely, we use http+SSE(Server send events)->stateful connection or
streamable http->stateless/stateful connection->recommended and used

"""