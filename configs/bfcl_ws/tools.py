from src.tools.tools import Tool
from configs.bfcl_ws.urlfetch import URLFetch
from src.docstore.google_search import WebSearchAPI
from src.dynacall.llm_adapters import create_llm_adapter
import ast
import json
import re

_SEARCH_ENGINE_DESCRIPTION = (
    'search_engine(["query: str"])-> list:\n'
    " - Web discovery tool: returns search results with title, href, and snippet body.\n"
)

_FETCH_URLS_DESCRIPTION = (
    "fetch_urls(query: str, search_results: Optional[list]) -> str:\n"
    " - Takes search results and uses an internal LLM to select grounded high-relevance URLs.\n"
    " - Fetches selected URLs and returns merged page contents.\n"
    " - Use this after search_engine before extracting the final answer/entity.\n"
)

_SEMANTIC_MAP_DESCRIPTION = (
    'semantic_map(["global_question: str", "local_request: str", "plan_context: str", ["observation_1", ...], "output_schema: str"]) -> str:\n'
    " - GAIA-style typed semantic extraction/normalization tool.\n"
    " - Also accepts legacy calls like semantic_map([instruction, inputs, output_schema]).\n"
    " - Supported schemas: string, number, boolean, list[string], json{...}.\n"
)


def run_search_engine_factory(web_search_api):
    async def run_search_engine(query, max_results=10):
        return await web_search_api.asearch(
            keywords=query,
            max_results=max_results,
            region="wt-wt",
        )

    return run_search_engine


def run_fetch_urls_factory(url_fetch_chain):
    def _split_compound_arg(text: str):
        s = str(text).strip()
        depth = 0
        in_str = False
        quote = ""
        escape = False
        for i, ch in enumerate(s):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if in_str:
                if ch == quote:
                    in_str = False
                continue
            if ch in {"'", '"'}:
                in_str = True
                quote = ch
                continue
            if ch in {"[", "(", "{"}:
                depth += 1
                continue
            if ch in {"]", ")", "}"}:
                depth = max(0, depth - 1)
                continue
            if ch == "," and depth == 0:
                return s[:i].strip(), s[i + 1 :].strip()
        return s, None

    def _normalize_value(raw):
        if raw is None:
            return None
        if isinstance(raw, (list, dict)):
            return raw
        text = str(raw).strip()
        if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
            text = text[1:-1]
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(text)
            except Exception:
                continue
        return text

    async def run_fetch_urls(*args):
        query = None
        context = None
        if len(args) == 1 and isinstance(args[0], list):
            values = args[0]
            if len(values) >= 1:
                query = values[0]
            if len(values) >= 2:
                context = values[1]
        elif len(args) >= 1:
            query = args[0]
            if len(args) >= 2:
                context = args[1]
        if context is None and isinstance(query, str):
            left, right = _split_compound_arg(query)
            if right is not None:
                query = _normalize_value(left)
                context = _normalize_value(right)
        if query is None:
            query = ""
        result = await url_fetch_chain.acall(
            {
                "query": query,
                "search_results": context,
            }
        )
        return result["content"]

    return run_fetch_urls


# def run_semantic_map_factory(llm_adapter):
#     def _coerce_scalar_output(raw: str):
#         text = str(raw or "").strip()
#         for loader in (json.loads, ast.literal_eval):
#             try:
#                 parsed = loader(text)
#             except Exception:
#                 continue
#             if isinstance(parsed, dict):
#                 for v in parsed.values():
#                     return str(v or "").strip()
#             if isinstance(parsed, list):
#                 if not parsed:
#                     return ""
#                 return str(parsed[0] or "").strip()
#             return str(parsed or "").strip()
#         # Fallback for "key:value" style.
#         if ":" in text:
#             return text.split(":", 1)[1].strip()
#         return text

#     async def run_semantic_map(*args):
#         if len(args) == 1:
#             values = args[0] if isinstance(args[0], list) else [args[0]]
#         else:
#             values = list(args)

#         if len(values) >= 3 and isinstance(values[0], (list, tuple)):
#             first_item = list(values[0])
#             if len(first_item) == 1 and isinstance(first_item[0], str):
#                 values = [first_item[0], values[1], values[2], *values[3:]]

#         if len(values) == 1 and isinstance(values[0], tuple):
#             values = list(values[0])

#         if len(values) < 3:
#             return "semantic_map expects either [instruction, inputs, output_schema] or [global_question, local_request, plan_context, inputs, output_schema]."

#         if len(values) >= 5:
#             local_request = str(values[1]).strip()
#             inputs = values[3]
#             output_schema = str(values[4]).strip() or "string"
#         else:
#             local_request = str(values[0]).strip()
#             inputs = values[1]
#             output_schema = str(values[2]).strip() or "string"

#         if not isinstance(inputs, list):
#             inputs = [inputs]

#         rendered_inputs = []
#         for idx, item in enumerate(inputs, start=1):
#             rendered_inputs.append(f"Input {idx}:\n{str(item)}")

#         prompt = (
#             "You are semantic_map, a typed semantic extraction operator.\n"
#             "Use only provided evidence. Do not guess.\n"
#             "Return the shortest valid value matching output schema.\n"
#             "Output discipline:\n"
#             "- Always return one scalar entity/value only.\n"
#             "- Never return JSON objects, field labels, or key:value prose lines.\n"
#             "- If multiple values are requested, return the single most relevant next-hop entity only.\n"
#             "No markdown fences, no explanations.\n\n"
#             f"Current Request:\n{local_request}\n\n"
#             f"Output schema:\n{output_schema}\n\n"
#             f"{chr(10).join(rendered_inputs)}\n\n"
#             "Return only the final value."
#         )
#         raw = await llm_adapter.apredict(prompt)
#         if raw.startswith("Error:"):
#             return raw

#         raw = raw.strip()
#         return _coerce_scalar_output(raw)

#     return run_semantic_map

def run_semantic_map_factory(llm_adapter):
    async def run_semantic_map(*args):
        if len(args) == 1:
            values = args[0] if isinstance(args[0], list) else [args[0]]
        else:
            values = list(args)

        if len(values) >= 3 and isinstance(values[0], (list, tuple)):
            first_item = list(values[0])
            if len(first_item) == 1 and isinstance(first_item[0], str):
                values = [first_item[0], values[1], values[2], *values[3:]]

        if len(values) == 1 and isinstance(values[0], tuple):
            values = list(values[0])

        if len(values) < 3:
            return "semantic_map expects either [instruction, inputs, output_schema] or [global_question, local_request, plan_context, inputs, output_schema]."

        context = {}
        if len(values) >= 6:
            try:
                context = json.loads(str(values[5]))
            except Exception:
                context = {"raw_context": str(values[5])}

        if len(values) >= 5:
            global_question = str(values[0]).strip()
            local_request = str(values[1]).strip()
            plan_context = str(values[2]).strip()
            inputs = values[3]
            output_schema = str(values[4]).strip() or "string"
        else:
            local_request = str(values[0]).strip()
            inputs = values[1]
            output_schema = str(values[2]).strip() or "string"
            global_question = str(context.get("global_question", "") or "").strip()
            plan_context = str(context.get("plan_context", "") or local_request).strip()

        context.setdefault("global_question", global_question)
        context.setdefault("local_question", local_request)
        context.setdefault("plan_context", plan_context)

        if not isinstance(inputs, list):
            inputs = [inputs]

        rendered_inputs = []
        for idx, item in enumerate(inputs, start=1):
            rendered_inputs.append(f"Input {idx}:\n{str(item)}")

        prompt = (
            "You are semantic_map, a typed semantic operator with global task awareness.\n"
            "Use only provided evidence. Do not guess.\n"
            "Return the shortest valid value matching output schema.\n"
            "Output discipline:\n"
            "- Always return one scalar entity/value only.\n"
            "- Never return JSON objects, field labels, or key:value prose lines.\n"
            "- If multiple values are requested, return the single most relevant next-hop entity only.\n"
            "No markdown fences, no explanations.\n\n"
            f"Global Question:\n{context.get('global_question', global_question)}\n\n"
            f"Local Request:\n{context.get('local_question', local_request)}\n\n"
            f"Plan Context:\n{context.get('plan_context', plan_context)}\n\n"
            f"History of all observations:\n{json.dumps(context.get('history_observations', []), ensure_ascii=False)}\n\n"
            f"Current plan fragment:\n{json.dumps(context.get('current_plan', []), ensure_ascii=False)}\n\n"
            f"Output schema:\n{output_schema}\n\n"
            f"{chr(10).join(rendered_inputs)}\n\n"
            "Return only the final value."
        )
        raw = await llm_adapter.apredict(prompt)
        if raw.startswith("Error:"):
            return raw

        return raw.strip()

    return run_semantic_map

def get_model_new(model_type, model_name, **kwargs):
    return create_llm_adapter(model_type, model_name, **kwargs)


def generate_tools(args, model_name):
    llm = get_model_new(
        model_type=args.model_type,
        model_name=model_name,
        vllm_port=args.vllm_port,
        stream=False,
        temperature=0,
    )

    web_search_api = WebSearchAPI()
    url_fetch_chain = URLFetch(
        include_summary=False,
        max_urls=5,
        fetch_mode="truncate",
        web_search_api=web_search_api,
    )
    search_engine_func = run_search_engine_factory(web_search_api)
    fetch_urls_func = run_fetch_urls_factory(url_fetch_chain)
    semantic_map_func = run_semantic_map_factory(llm)

    return [
        Tool(
            name="search_engine",
            func=search_engine_func,
            description=_SEARCH_ENGINE_DESCRIPTION,
            stringify_rule=lambda args: f"search_engine({args[0]})",
        ),
        Tool(
            name="fetch_urls",
            func=fetch_urls_func,
            description=_FETCH_URLS_DESCRIPTION,
            stringify_rule=lambda args: f"fetch_urls({args[0]})",
        ),
        Tool(
            name="semantic_map",
            func=semantic_map_func,
            description=_SEMANTIC_MAP_DESCRIPTION,
            stringify_rule=lambda args: f"semantic_map({args[0]})",
        ),
    ]
