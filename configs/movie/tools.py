from src.tools.tools import Tool
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia


_SEARCH_DESCRIPTION = (
    'search(["entity: str"]) -> str:\n'
    ' - Executes an exact search for the entity on Wikipedia.\n'
    ' - Returns the first paragraph if the entity is found.\n'
)

def generate_tools(args):
    web_searcher = ReActWikipedia()
    if args.model_type == "vllm":
        # vLLM models are more prone to context overflow on long wiki snippets.
        docstore = DocstoreExplorer(web_searcher, char_limit=400, one_sentence=True)
    else:
        docstore = DocstoreExplorer(web_searcher)

    return [
        Tool(
            name="search",
            func=docstore.asearch,
            description=_SEARCH_DESCRIPTION,
            stringify_rule=lambda call_args: f"search({call_args[0]})",
        ),
    ]
