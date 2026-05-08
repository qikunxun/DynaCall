import json
import os
import asyncio

from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia
from src.tools.tools import Tool
from src.dynacall.llm_adapters import create_llm_adapter

from .mathtool import calculate_with_steps

_SEARCH_DESCRIPTION = (
    'Search(["entity: str"]) -> str:\n'
    " - Executes an exact search for the entity on Wikipedia.\n"
    " - Returns the first paragraph if the entity is found.\n"
)

_CALCULATE_DESCRIPTION = (
    'Calculate(["expression: str", ["dependency_1", ...]]) -> str:\n'
    " - Evaluates a pure numerical expression with numbers and operators.\n"
    " - Supported operations include +, -, *, /, **, abs(...), min(...), and max(...).\n"
    " - The optional second argument is a list of upstream references used by the expression.\n"
    " - Use semantic_map first if you need to extract or normalize values from text.\n"
    " - Never pass raw search results directly. Calculate only works on numerical expressions.\n"
    " - The expression must not introduce variables.\n"
)

_SEMANTIC_MAP_DESCRIPTION = (
    'semantic_map(["instruction: str", ["input_1", "input_2", ...], "output_schema: str"]) -> str:\n'
    " - A typed local semantic transformation agent for short extraction, normalization, and comparison steps.\n"
    " - Use it to extract a short value from Wikipedia search results before another tool continues.\n"
    " - It must only do a local transformation on the provided inputs. It must not solve the whole question.\n"
    ' - Supported schemas: "string", "number", "boolean", "list[string]".\n'
)


def _format_semantic_value(schema: str, value):
    normalized = schema.strip().lower()

    if normalized == "list[number]":
        if isinstance(value, list):
            items = value
        else:
            text = str(value).strip()
            try:
                parsed = json.loads(text)
                items = parsed if isinstance(parsed, list) else [text]
            except Exception:
                items = [part.strip() for part in text.split(",") if part.strip()]
        formatted = []
        for item in items:
            try:
                number = float(str(item).strip())
                if number.is_integer():
                    formatted.append(int(number))
                else:
                    formatted.append(float(str(round(number, 6)).rstrip("0").rstrip(".")))
            except Exception:
                formatted.append(str(item).strip())
        return formatted

    if normalized == "number":
        try:
            number = float(str(value).strip())
            if number.is_integer():
                return str(int(number))
            return str(round(number, 6)).rstrip("0").rstrip(".")
        except Exception:
            return str(value).strip()

    if normalized == "boolean":
        text = str(value).strip().lower()
        return "true" if text in {"true", "yes", "1"} else "false"

    if normalized == "list[string]":
        if isinstance(value, list):
            return ", ".join(str(v).strip() for v in value if str(v).strip())
        text = str(value).strip()
        if not text:
            return ""
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return ", ".join(str(v).strip() for v in parsed if str(v).strip())
            except Exception:
                pass
        return text

    return str(value).strip()


def run_calculate_factory():
    def _extract_expression(args):
        if len(args) == 1 and isinstance(args[0], list):
            values = args[0]
        else:
            values = list(args)
        if not values:
            return ""
        return str(values[0]).strip()

    def run_calculate(*args):
        expression = _extract_expression(args)
        if not expression:
            return "Error: Missing expression."
        try:
            return str(calculate_with_steps(expression)).strip()
        except Exception:
            return "Error: Invalid expression. Try with a different expression."

    async def arun_calculate(*args):
        expression = _extract_expression(args)
        if not expression:
            return "Error: Missing expression."
        try:
            return str(calculate_with_steps(expression)).strip()
        except Exception:
            return "Error: Invalid expression. Try with a different expression."

    run_calculate.async_func = arun_calculate
    return run_calculate


def run_semantic_map_factory(llm_adapter):
    # Restore full context injection so semantic_map can use global question,
    # plan/history/provenance when deciding the correct numeric target.
    MINIMAL_SEMANTIC_CONTEXT = False

    def _compact_text(value, limit=240):
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " ..."

    def _meaningful_context_text(value):
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and parsed.get("mode") == "batch":
                    return ""
            except Exception:
                pass
        return text

    def _compact_current_plan(context_payload, max_items=8):
        rows = []
        for item in context_payload.get("current_plan", []) or []:
            tool = str(item.get("tool", "") or "")
            action = str(item.get("action", "") or "")
            if not tool or tool == "semantic_map":
                continue
            rows.append(
                f"{item.get('task_id')}: {tool} - {_compact_text(action, 180)}"
            )
            if len(rows) >= max_items:
                break
        return "\n".join(rows)

    def _compact_guard_history(context_payload, max_items=4):
        rows = []
        for item in context_payload.get("history_observations", []) or []:
            tool = str(item.get("tool", "") or "")
            if tool not in {"branch", "replan"}:
                continue
            rows.append(
                f"{item.get('task_id')}: {tool} observation={_compact_text(item.get('observation'), 220)}"
            )
            if len(rows) >= max_items:
                break
        return "\n".join(rows)

    def _compact_downstream_usage(context_payload):
        rows = []
        for item in context_payload.get("current_plan", []) or []:
            tool = str(item.get("tool", "") or "")
            if tool not in {"Calculate"}:
                continue
            rows.append(_compact_text(item.get("action"), 260))
            if len(rows) >= 2:
                break
        return "\n".join(rows)

    async def run_semantic_map(*args):
        if len(args) == 1 and isinstance(args[0], list):
            values = args[0]
        else:
            values = list(args)

        if len(values) < 3:
            return "semantic_map expects [instruction, inputs, output_schema]."

        if len(values) >= 5:
            global_question = str(values[0]).strip()
            instruction = str(values[1]).strip()
            raw_inputs = values[3] if isinstance(values[3], list) else [values[3]]
            output_schema = str(values[4]).strip()
            context = values[5] if len(values) > 5 else None
        else:
            instruction = str(values[0]).strip()
            raw_inputs = values[1] if isinstance(values[1], list) else [values[1]]
            output_schema = str(values[2]).strip()
            context = values[3] if len(values) > 3 else None
            global_question = ""

        context_payload = {}
        if isinstance(context, str) and context.strip():
            try:
                context_payload = json.loads(context)
            except Exception:
                context_payload = {}

        global_question = global_question or str(context_payload.get("global_question", "")).strip()
        local_question = str(context_payload.get("local_question", "")).strip()
        plan_context = _meaningful_context_text(context_payload.get("plan_context", ""))
        compact_plan = _compact_current_plan(context_payload)
        compact_guard_history = _compact_guard_history(context_payload)
        compact_downstream_usage = _compact_downstream_usage(context_payload)
        provenance = context_payload.get("inputs_with_provenance", []) or []

        batch_spec = None
        if instruction.startswith("{"):
            try:
                parsed_instruction = json.loads(instruction)
                if isinstance(parsed_instruction, dict) and parsed_instruction.get("mode") == "batch":
                    batch_spec = parsed_instruction
            except Exception:
                batch_spec = None

        if batch_spec is not None:
            items = batch_spec.get("items", [])
            result_format_lines = [
                f"Result {idx}: <value>" for idx in range(1, len(items) + 1)
            ]
            prompt_lines = [
                "You are a typed local semantic transformation agent.",
                "Think shortly and locally.",
                "Perform only the requested semantic transformation.",
                "Do not solve the whole task, do not call tools, and do not invent missing facts.",
                "Each Result i must be exactly one numeric literal only (digits, optional leading '-' and decimal point).",
                "Negative numbers are allowed when grounded by the input.",
                "Never output operators or expression fragments such as '+', '-', '*', '/', '(/)', '((+)/(+))', or parenthesized templates.",
                "Never output equations, units, labels, or explanations.",
                f"Return exactly {len(items)} result lines, one result per item, in this exact format:",
                *result_format_lines,
                "No explanations, markdown, bullets, or extra text.",
            ]
            if global_question and not MINIMAL_SEMANTIC_CONTEXT:
                prompt_lines.append(f"Global Question:\n{global_question}")
            shared_batch_context = [
                "Shared Batch Context:\n"
                "- Item i must use Input i and produce Result i.\n"
                "- Preserve item order exactly.\n"
                "- Results will be consumed independently by downstream tools via Result i references."
            ]
            if local_question and not local_question.startswith("{") and not MINIMAL_SEMANTIC_CONTEXT:
                shared_batch_context.append(f"Local Question:\n{_compact_text(local_question, 500)}")
            if plan_context and not MINIMAL_SEMANTIC_CONTEXT:
                shared_batch_context.append(f"Plan Context:\n{_compact_text(plan_context, 700)}")
            if compact_plan and not MINIMAL_SEMANTIC_CONTEXT:
                shared_batch_context.append(f"Current Plan Skeleton:\n{compact_plan}")
            if compact_guard_history and not MINIMAL_SEMANTIC_CONTEXT:
                shared_batch_context.append(f"Guard/Replan State:\n{compact_guard_history}")
            if compact_downstream_usage and not MINIMAL_SEMANTIC_CONTEXT:
                shared_batch_context.append(f"Downstream Usage:\n{compact_downstream_usage}")
            prompt_lines.append("\n\n".join(shared_batch_context))

            # Keep provenance compact and non-redundant: one shared index instead of
            # repeating multi-line tool/action blocks for every item.
            input_source_rows = []
            for idx in range(1, len(items) + 1):
                source = "unknown"
                if idx - 1 < len(provenance):
                    meta = provenance[idx - 1]
                    source_tool = str(meta.get("tool", "")).strip()
                    source = source_tool or source
                input_source_rows.append(f"Input {idx} source: {source}")
            if input_source_rows and not MINIMAL_SEMANTIC_CONTEXT:
                prompt_lines.append("Input Source Index:\n" + "\n".join(input_source_rows))

            for idx, item in enumerate(items, start=1):
                prompt_lines.append(f"Item {idx} instruction: {item.get('instruction', '')}")
                prompt_lines.append(f"Item {idx} schema: {item.get('output_schema', '')}")
                depends_on_result = item.get("depends_on_result")
                if depends_on_result:
                    prompt_lines.append(f"Item {idx} input depends on Result {depends_on_result}.")
                if idx - 1 < len(raw_inputs):
                    prompt_lines.append(f"Item {idx} input:\n{str(raw_inputs[idx - 1]).strip()}")
            prompt_lines.append("Output result lines in item order.")
            payload = await llm_adapter.apredict("\n\n".join(prompt_lines))
            if payload is None:
                return ""
            payload = str(payload).strip()
            if payload.startswith("Error:"):
                return payload
            if payload.startswith("```"):
                payload = payload.strip("`")
                if payload.startswith("json"):
                    payload = payload[4:].strip()
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, list):
                    return "\n".join(
                        f"Result {idx}: {_format_semantic_value(str(items[idx - 1].get('output_schema', 'string')), value)}"
                        for idx, value in enumerate(parsed, start=1)
                    )
            except Exception:
                pass
            if "Result 1:" in payload:
                return payload
            return "\n".join(
                f"Result {idx}: {line.strip()}"
                for idx, line in enumerate(payload.splitlines(), start=1)
                if line.strip()
            )

        rendered_inputs = []
        for idx, item in enumerate(raw_inputs, start=1):
            rendered_inputs.append(f"Input {idx}:\n{str(item).strip()}")

        rendered_provenance = []
        for idx, item in enumerate(provenance, start=1):
            rendered_provenance.append(
                f"Input {idx} provenance:\n"
                f"tool={item.get('tool', '')}\n"
                f"action={item.get('action', '')}"
            )

        prompt_parts = [
            "You are a typed local semantic transformation agent.",
            "Think shortly and locally.",
            "Perform only the requested semantic transformation.",
            "Do not solve the whole task, do not call tools, and do not invent missing facts.",
            "Return exactly one numeric literal (digits, optional leading '-' and decimal point) and nothing else.",
            "Negative numbers are allowed when grounded by the input.",
            "Never output operators or expression fragments such as '+', '-', '*', '/', '(/)', '((+)/(+))', or parenthesized templates.",
            "Never output equations, units, labels, or explanations.",
            "Do not include explanations, justification, markdown, or any extra text.",
            "Your output will be parsed automatically. Extra text is an error.",
        ]
        if global_question and not MINIMAL_SEMANTIC_CONTEXT:
            prompt_parts.append(f"Global Question:\n{global_question}")
        shared_context = []
        if local_question and local_question != instruction and not MINIMAL_SEMANTIC_CONTEXT:
            shared_context.append(f"Local Question:\n{_compact_text(local_question, 500)}")
        if plan_context and plan_context != instruction and not MINIMAL_SEMANTIC_CONTEXT:
            shared_context.append(f"Plan Context:\n{_compact_text(plan_context, 700)}")
        if compact_plan and not MINIMAL_SEMANTIC_CONTEXT:
            shared_context.append(f"Current Plan Skeleton:\n{compact_plan}")
        if compact_guard_history and not MINIMAL_SEMANTIC_CONTEXT:
            shared_context.append(f"Guard/Replan State:\n{compact_guard_history}")
        if compact_downstream_usage and not MINIMAL_SEMANTIC_CONTEXT:
            shared_context.append(f"Downstream Usage:\n{compact_downstream_usage}")
        if shared_context:
            prompt_parts.append("Shared Context:\n" + "\n\n".join(shared_context))
        prompt_parts.append(f"Instruction:\n{instruction}")
        prompt_parts.append(f"Output schema:\n{output_schema}")
        if rendered_provenance and not MINIMAL_SEMANTIC_CONTEXT:
            prompt_parts.append("Inputs with provenance:\n" + "\n\n".join(rendered_provenance))
        prompt_parts.append("Rendered inputs:\n" + "\n\n".join(rendered_inputs))
        prompt_parts.append("Output:")

        payload = await llm_adapter.apredict("\n\n".join(prompt_parts))
        if payload is None:
            return ""
        payload = str(payload).strip()
        if payload.startswith("Error:"):
            return payload
        if payload.startswith("```"):
            payload = payload.strip("`")
            if payload.startswith("json"):
                payload = payload[4:].strip()

        normalized = output_schema.strip().lower()
        if normalized == "list[number]":
            try:
                parsed = json.loads(payload)
                return _format_semantic_value(output_schema, parsed)
            except Exception:
                return _format_semantic_value(output_schema, payload)
        if normalized == "list[string]":
            items = [part.strip() for part in payload.split(",") if part.strip()]
            return _format_semantic_value(output_schema, items)
        if normalized in {"string", "number", "boolean"}:
            return _format_semantic_value(output_schema, payload.splitlines()[0].strip())
        return _format_semantic_value(output_schema, payload)

    return run_semantic_map


web_searcher = ReActWikipedia()
docstore = DocstoreExplorer(web_searcher)


def get_model_new(model_type, model_name, **kwargs):
    return create_llm_adapter(model_type, model_name, **kwargs)


def run_search_factory(timeout_seconds: float = 10.0):
    async def run_search(*args):
        if len(args) == 1 and isinstance(args[0], list):
            values = args[0]
        else:
            values = list(args)
        if not values:
            return "Search error: missing query."
        query = str(values[0]).strip()
        if not query:
            return "Search error: empty query."
        try:
            return await asyncio.wait_for(docstore.asearch(query), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return f"Search timeout after {int(timeout_seconds)}s"
        except Exception as e:
            return f"Search error: {e}"

    return run_search


def generate_tools(args, model_name):
    llm = get_model_new(
        model_type=args.model_type,
        model_name=model_name,
        vllm_port=args.vllm_port,
        stream=False,
        temperature=0,
        api_key=os.environ.get("API_KEY"),
        api_base=os.environ.get("API_BASE"),
    )

    semantic_map = run_semantic_map_factory(llm)
    calculate = run_calculate_factory()
    search = run_search_factory(timeout_seconds=10.0)

    return [
        Tool(
            name="Search",
            func=search,
            coroutine=search,
            description=_SEARCH_DESCRIPTION,
            stringify_rule=lambda args: f'Search(["{args[0]}"])',
        ),
        Tool(
            name="semantic_map",
            func=semantic_map,
            coroutine=semantic_map,
            description=_SEMANTIC_MAP_DESCRIPTION,
            stringify_rule=lambda args: "semantic_map(<instruction>, <inputs>, <schema>)",
        ),
        Tool(
            name="Calculate",
            func=calculate,
            coroutine=calculate.async_func,
            description=_CALCULATE_DESCRIPTION,
            stringify_rule=lambda args: f"Calculate({list(args)})",
        ),
    ]
