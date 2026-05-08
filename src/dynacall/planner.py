# planner.py
"""DynaCall Planner."""

import json
import re
import ast
import asyncio
from typing import Any, Optional, Sequence, List, Dict
from src.dynacall.task import Task
from src.utils.logger_utils import log

# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"

JOIN_DESCRIPTION = (
    "join():\n"
    " - Collects and combines results from prior actions.\n"
    " - A LLM agent is called upon invoking join to finalize the user query.\n"
    " - Use join only as the terminal action when the existing observations are already sufficient to answer.\n"
    " - The final planning round should usually be a plan containing only join.\n"
)

BRANCH_DESCRIPTION = (
    "branch([predicate: str | dict, if_true: list[int], if_false: list[int]]):\n"
    " - Evaluates a condition over previous observations and activates one of two branches.\n"
    " - Use this when the next step depends on whether earlier evidence contains a usable signal.\n"
    " - Supported predicates include is_empty($id), is_nonempty($id), contains_url($id), contains_number($id), "
    'contains_json_field($id, "field"), list_length_ge($id, n), tool_error($id), matches_regex($id, "pattern"), '
    'and contains_str($id, "substring").\n'
    " - Predicates can be combined with &/| for lightweight boolean logic.\n"
    ' - For edge cases, you may use a structured condition object like {"op":"llm_judge","instruction":"...","inputs":["$id"]}.\n'
    ' - Structured conditions may also use {"op":"and","args":[...]} and {"op":"or","args":[...]} to combine subconditions.\n'
    " - The if_true and if_false fields are lists of action ids to activate.\n"
)

REPLAN_DESCRIPTION = (
    "replan([reason: str | dict]):\n"
    " - Requests replanning when current evidence is insufficient or the current branch fails.\n"
    " - Use local replan to preserve executed observations and modify only the still-unexecuted suffix.\n"
    " - Use global replan to discard all current observations and restart from scratch.\n"
    ' - Preferred JSON node form: {"type":"replan","scope":"local"|"global","reason":"short reason"}.\n'
    " - If scope is omitted, local is the default.\n"
)

def default_dependency_rule(idx, args: str):
    """默认依赖规则"""
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers

def _parse_action_args(args: str) -> tuple:
    """解析动作参数 - 原始实现"""
    if args == "":
        return ()
    try:
        args = ast.literal_eval(args)
    except:
        args = args
    if not isinstance(args, list) and not isinstance(args, tuple):
        args = (args,)
    return args

def _find_tool(tool_name: str, tools: Sequence[Any]) -> Any:
    """查找工具 - 原始实现"""
    for tool in tools:
        if hasattr(tool, 'name') and tool.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found.")

def _get_dependencies_from_graph(idx: int, tool_name: str, args: Sequence[Any]) -> List[int]:
    """从图中获取依赖关系 - 原始实现"""
    if tool_name == "join":
        # depends on the previous step
        dependencies = list(range(1, idx))
    elif tool_name == "replan":
        # replan is controlled by branch/reasoning context and should not wait on every
        # previous task in a branched plan; it can inspect already completed observations
        # from the scheduler when triggered.
        dependencies = []
    else:
        # define dependencies based on the dependency rule
        args_str = str(args)
        dependencies = [i for i in range(1, idx) if default_dependency_rule(i, args_str)]

    return dependencies

def instantiate_task(tools: Sequence[Any], idx: int, tool_name: str, args: str, thought: str) -> Task:
    """实例化任务 - 原始实现"""
    dependencies = _get_dependencies_from_graph(idx, tool_name, args)
    parsed_args = _parse_action_args(args)
    
    if tool_name in {"join", "branch", "replan"}:
        # join does not have a tool
        tool_func = lambda *x: None
        stringify_rule = None
    else:
        tool = _find_tool(tool_name, tools)
        tool_func = tool.func
        stringify_rule = tool.stringify_rule if hasattr(tool, 'stringify_rule') else None
    
    return Task(
        idx=idx,
        name=tool_name,
        tool=tool_func,
        args=parsed_args,
        dependencies=dependencies,
        stringify_rule=stringify_rule,
        thought=thought,
        is_join=tool_name == "join",
        is_branch=tool_name == "branch",
        is_replan=tool_name == "replan",
    )

class JSONPlanParser:
    """JSON plan parser that flattens a structured plan into Task objects."""

    def __init__(self, tools: Sequence[Any]):
        self.tools = tools
        self.tool_names = {getattr(tool, "name", None) for tool in tools if getattr(tool, "name", None)}

    def _strip_think_blocks(self, text: str) -> str:
        """Remove leading or wrapper <think>...</think> blocks before JSON parsing."""
        cleaned = text
        # Remove one or more leading think blocks.
        while True:
            updated = re.sub(r"^\s*<think>.*?</think>\s*", "", cleaned, flags=re.S | re.I)
            if updated == cleaned:
                break
            cleaned = updated

        # Handle the common wrapper shape: <think>...</think>{...json...}
        wrapper = re.match(
            r"^\s*<think>.*?</think>\s*(.*)$",
            cleaned,
            flags=re.S | re.I,
        )
        if wrapper:
            return wrapper.group(1).strip()
        return cleaned

    def _extract_json_payload(self, text: str) -> str:
        stripped = self._strip_think_blocks(text).strip()
        fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", stripped, flags=re.S | re.I)
        if fenced:
            stripped = fenced.group(1).strip()
        if not stripped:
            raise ValueError("Planner output is empty")
        if stripped[0] not in "{[":
            first_obj = stripped.find("{")
            first_arr = stripped.find("[")
            starts = [idx for idx in (first_obj, first_arr) if idx != -1]
            if not starts:
                raise ValueError("Planner output must start with a top-level JSON object or array")
            stripped = stripped[min(starts):].lstrip()
            if not stripped or stripped[0] not in "{[":
                raise ValueError("Planner output must start with a top-level JSON object or array")

        stripped = self._repair_string_concatenation(stripped)
        stripped = self._repair_quoted_json_arrays(stripped)
        stripped = self._repair_nested_singleton_args_arrays(stripped)
        stripped = self._repair_unquoted_plan_references(stripped)
        stripped = self._repair_duplicate_object_closers(stripped)
        stripped = self._repair_unescaped_embedded_json_literals(stripped)

        decoder = json.JSONDecoder()
        try:
            _, end = decoder.raw_decode(stripped)
        except json.JSONDecodeError as exc:
            repaired = self._repair_near_error(stripped, exc, decoder)
            if repaired is not None:
                stripped = repaired
                _, end = decoder.raw_decode(stripped)
            else:
                raise ValueError(f"Planner output is not valid top-level JSON: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Planner output is not valid top-level JSON: {exc}") from exc

        trailing = stripped[end:].strip()
        if trailing:
            trailing = re.sub(r"^\s*```(?:json)?", "", trailing, flags=re.I).strip()
            trailing = re.sub(r"```\s*$", "", trailing).strip()
            if trailing:
                return stripped[:end].strip()

        return stripped[:end].strip()

    def _repair_near_error(
        self,
        text: str,
        exc: json.JSONDecodeError,
        decoder: json.JSONDecoder,
    ) -> Optional[str]:
        """Apply very small local repairs near a JSON parse error.

        Keep this narrow and mechanical:
        - insert a comma at the error point when two JSON values appear adjacent
        - remove one extra closing bracket/brace near the error point
        """
        pos = max(0, min(exc.pos, len(text)))
        candidates: List[str] = []

        prev_char = text[pos - 1] if pos > 0 else ""
        curr_char = text[pos] if pos < len(text) else ""

        if prev_char in '}]"' and curr_char in '{["':
            candidates.append(text[:pos] + "," + text[pos:])

        window_start = max(0, pos - 3)
        window_end = min(len(text), pos + 4)
        for idx in range(window_start, window_end):
            if text[idx] in "]}":
                candidates.append(text[:idx] + text[idx + 1 :])

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                _, end = decoder.raw_decode(candidate)
                trailing = candidate[end:].strip()
                if not trailing or trailing.startswith("```"):
                    return candidate
            except Exception:
                continue
        return None

    def _repair_string_concatenation(self, text: str) -> str:
        """Repair common planner mistake: JSON string concatenation like "a" + "$b".

        This keeps the top-level parser strict while allowing a narrow, mechanical
        normalization of string-literal concatenations into one JSON string.
        """
        string_lit = r'"(?:\\.|[^"\\])*"'
        pattern = re.compile(rf"({string_lit})\s*\+\s*({string_lit})")
        adjacent_pattern = re.compile(rf"({string_lit})\s+({string_lit})")

        def repl(match: re.Match) -> str:
            left = json.loads(match.group(1))
            right = json.loads(match.group(2))
            return json.dumps(left + right, ensure_ascii=False)

        repaired = text
        while True:
            updated = pattern.sub(repl, repaired)
            updated = adjacent_pattern.sub(repl, updated)
            if updated == repaired:
                return repaired
            repaired = updated

    def _repair_quoted_json_arrays(self, text: str) -> str:
        """Repair a common planner formatting bug where a JSON array is wrapped in quotes.

        Example:
          "...", "["$x","$y"]", "string"
        becomes
          "...", ["$x","$y"], "string"

        Keep this narrow: only unwrap bracketed arrays that are already composed of
        quoted JSON string elements.
        """
        pattern = re.compile(
            r'"\[(\s*"(?:\\.|[^"\\])*"\s*(?:,\s*"(?:\\.|[^"\\])*"\s*)*)\]"'
        )

        repaired = text
        while True:
            updated = pattern.sub(lambda m: "[" + m.group(1) + "]", repaired)
            if updated == repaired:
                return repaired
            repaired = updated

    def _repair_unescaped_embedded_json_literals(self, text: str) -> str:
        """Repair unescaped quotes in embedded JSON-like literals inside string fields.

        Common malformed planner output examples:
        - "Return JSON {"author":"...","novel":"..."} ..."
        - "json{"author":"string","novel":"string"}"

        These are intended to be plain string content, but inner quotes are often not
        escaped, which breaks top-level JSON parsing. We only escape quotes inside the
        innermost {...} after the marker prefixes.
        """

        def _escape_inner_quotes(match: re.Match) -> str:
            marker = match.group(1)
            body = match.group(2)
            body = re.sub(r'(?<!\\)"', r'\\"', body)
            return f"{marker}{{{body}}}"

        repaired = text
        repaired = re.sub(
            r"(Return JSON\s*)\{([^{}]*)\}",
            _escape_inner_quotes,
            repaired,
            flags=re.I,
        )
        repaired = re.sub(
            r"(json)\{([^{}]*)\}",
            _escape_inner_quotes,
            repaired,
            flags=re.I,
        )
        return repaired

    def _repair_nested_singleton_args_arrays(self, text: str) -> str:
        """Repair a common planner bug where args becomes [[\"...\"]] or [[\"...\"].

        The planner occasionally wraps a single string arg in an extra list, and in
        malformed cases omits the second closing bracket before the next object.
        Narrowly normalize only the `args` field shape.
        """
        repaired = text
        string_lit = r'"(?:\\.|[^"\\])*"'

        wrapped_pattern = re.compile(
            rf'("args"\s*:\s*)\[\[\s*({string_lit})\s*\]\]'
        )
        repaired = wrapped_pattern.sub(r'\1[\2]', repaired)

        missing_closer_pattern = re.compile(
            rf'("args"\s*:\s*)\[\[\s*({string_lit})\s*\](\s*\}})(?=\s*,\s*\{{)'
        )
        repaired = missing_closer_pattern.sub(r'\1[\2]\3', repaired)
        return repaired

    def _repair_duplicate_object_closers(self, text: str) -> str:
        """Repair a common malformed-JSON pattern with one extra `}` before a sibling.

        Example:
          ..."}]}},{"type":"branch",...
        becomes
          ..."}]},{"type":"branch",...
        """
        patterns = [
            re.compile(r'(\}\]\})(\}+)(,\s*\{)'),
            re.compile(r'(\}\}\])(\}+)(,\s*\{)'),
        ]

        repaired = text
        while True:
            updated = repaired
            for pattern in patterns:
                updated = pattern.sub(lambda m: m.group(1) + m.group(3), updated)
            if updated == repaired:
                return repaired
            repaired = updated

    def _repair_unquoted_plan_references(self, text: str) -> str:
        """Repair planner JSON where $refs appear bare instead of inside JSON strings.

        Example:
          {"args":["https://x", $s1]}
        becomes
          {"args":["https://x", "$s1"]}

        Keep this narrow by only quoting bare $ident / $123 tokens that appear
        immediately after JSON structural punctuation and before JSON delimiters.
        """
        ref_token = r'(\$[A-Za-z_][A-Za-z0-9_]*|\$\d+)'
        patterns = [
            re.compile(rf'(?P<prefix>[\[,:\s])(?P<ref>{ref_token})(?P<suffix>\s*[\],}}])'),
        ]

        def is_inside_json_string(src: str, pos: int) -> bool:
            in_string = False
            escaped = False
            for ch in src[:pos]:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
            return in_string

        repaired = text
        while True:
            updated = repaired
            for pattern in patterns:
                pieces = []
                last = 0
                changed = False
                for match in pattern.finditer(updated):
                    if is_inside_json_string(updated, match.start("ref")):
                        continue
                    pieces.append(updated[last:match.start()])
                    pieces.append(
                        f'{match.group("prefix")}"{match.group("ref")}"{match.group("suffix")}'
                    )
                    last = match.end()
                    changed = True
                if changed:
                    pieces.append(updated[last:])
                    updated = "".join(pieces)
            if updated == repaired:
                return repaired
            repaired = updated

    def parse(self, text: str) -> Dict[int, Task]:
        payload = self._extract_json_payload(text)
        data = json.loads(payload)
        data = self._normalize_payload(data)

        if isinstance(data, list):
            top_level_nodes = data
        elif isinstance(data, dict):
            if data.get("kind") == "seq":
                top_level_nodes = data.get("steps", [])
            elif data.get("type") == "seq":
                top_level_nodes = data.get("steps", [])
            else:
                top_level_nodes = [data]
        else:
            raise ValueError("Top-level JSON plan must be an object or list")

        graph_dict: Dict[int, Task] = {}
        next_idx = 1
        named_ids: Dict[str, int] = {}

        def resolve_named_refs(value: Any) -> Any:
            if isinstance(value, str):
                def repl(match):
                    ref_name = match.group(1)
                    if ref_name.isdigit():
                        return match.group(0)
                    if ref_name not in named_ids:
                        raise ValueError(f"Unknown plan reference: ${ref_name}")
                    suffix = match.group(2) or ""
                    return f"${named_ids[ref_name]}{suffix}"

                return re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*|\d+)\}?(\[\d+\])?", repl, value)
            if isinstance(value, list):
                return [resolve_named_refs(item) for item in value]
            if isinstance(value, tuple):
                return tuple(resolve_named_refs(item) for item in value)
            if isinstance(value, dict):
                return {k: resolve_named_refs(v) for k, v in value.items()}
            return value

        def add_node(node: Dict[str, Any]) -> List[int]:
            nonlocal next_idx
            if isinstance(node, int):
                return [node]
            if isinstance(node, str) and node.isdigit():
                return [int(node)]
            if not isinstance(node, dict):
                raise ValueError(f"Unsupported JSON plan node payload: {type(node).__name__}")

            node_kind = node.get("kind")
            node_type = node_kind or node.get("type", "tool")
            thought = node.get("thought", "")

            tool_shorthand_types = set(self.tool_names)
            tool_shorthand_types.update({"join", "branch", "replan"})
            if node_type in tool_shorthand_types and node_type != "tool":
                normalized_node = dict(node)
                if node_type in {"join", "branch", "replan"}:
                    normalized_node["kind"] = node_type
                else:
                    normalized_node["kind"] = "tool"
                    normalized_node["tool"] = node_type
                node = normalized_node
                node_type = node["kind"]

            if node_type == "seq":
                subtree_ids: List[int] = []
                for step in node.get("steps", []):
                    subtree_ids.extend(add_node(step))
                return subtree_ids

            if node_type == "tool":
                tool_name = node.get("tool")
                if not tool_name:
                    tool_name = node.get("name") or node.get("action") or node.get("tool_name")
                if not tool_name and isinstance(node.get("id"), str) and node["id"] in self.tool_names:
                    tool_name = node["id"]
                if not tool_name:
                    raise ValueError(f"Tool node is missing tool name: {node}")
                if tool_name in {"branch", "replan", "join"}:
                    normalized_node = dict(node)
                    normalized_node["kind"] = str(tool_name)
                    normalized_node.pop("type", None)
                    normalized_node.pop("tool", None)
                    return add_node(normalized_node)
                idx = next_idx
                next_idx += 1
                node_id = node.get("id")
                if node_id:
                    named_ids[str(node_id)] = idx
                task = instantiate_task(
                    tools=self.tools,
                    idx=idx,
                    tool_name=tool_name,
                    args=repr(resolve_named_refs(node.get("args", []))),
                    thought=thought,
                )
                graph_dict[idx] = task
                return [idx]

            if node_type == "join":
                idx = next_idx
                next_idx += 1
                node_id = node.get("id")
                if node_id:
                    named_ids[str(node_id)] = idx
                task = instantiate_task(
                    tools=self.tools,
                    idx=idx,
                    tool_name="join",
                    args="",
                    thought=thought,
                )
                graph_dict[idx] = task
                return [idx]

            if node_type == "replan":
                idx = next_idx
                next_idx += 1
                node_id = node.get("id")
                if node_id:
                    named_ids[str(node_id)] = idx
                replan_payload = {
                    "reason": resolve_named_refs(node.get("reason", "Evidence is insufficient.")),
                    "scope": str(node.get("scope", "local")).strip().lower() or "local",
                }
                task = instantiate_task(
                    tools=self.tools,
                    idx=idx,
                    tool_name="replan",
                    args=repr([replan_payload]),
                    thought=thought,
                )
                graph_dict[idx] = task
                return [idx]

            if node_type == "branch":
                idx = next_idx
                next_idx += 1
                node_id = node.get("id")
                if node_id:
                    named_ids[str(node_id)] = idx
                then_ids = []
                for child in node.get("then", []):
                    then_ids.extend(add_node(child))
                else_ids = []
                for child in node.get("else", []):
                    else_ids.extend(add_node(child))
                for child_id in then_ids + else_ids:
                    child_task = graph_dict.get(child_id)
                    if child_task is None:
                        continue
                    deps = list(getattr(child_task, "dependencies", []) or [])
                    if idx not in deps:
                        deps.append(idx)
                        child_task.dependencies = sorted(set(deps))
                task = instantiate_task(
                    tools=self.tools,
                    idx=idx,
                    tool_name="branch",
                    args=repr([resolve_named_refs(node.get("condition", "")), then_ids, else_ids]),
                    thought=thought,
                )
                graph_dict[idx] = task
                return [idx] + then_ids + else_ids

            raise ValueError(f"Unsupported JSON plan node type: {node_type}")

        for top_level_node in top_level_nodes:
            add_node(top_level_node)
        return {idx: graph_dict[idx] for idx in sorted(graph_dict)}

    def _normalize_payload(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._normalize_payload(item) for item in value]
        if not isinstance(value, dict):
            return value

        node = {k: self._normalize_payload(v) for k, v in value.items()}
        kind = node.get("kind")
        node_type = kind or node.get("type")

        if isinstance(node_type, str) and node_type in self.tool_names and node_type not in {"join", "branch", "replan"}:
            if node.get("tool") is None:
                node["tool"] = node_type
            node["kind"] = "tool"

        if node.get("kind") == "tool":
            if "tool" not in node or node.get("tool") is None:
                for fallback_key in ("name", "action", "tool_name", "type"):
                    fallback = node.get(fallback_key)
                    if isinstance(fallback, str) and fallback in self.tool_names:
                        node["tool"] = fallback
                        break
            args = node.get("args", [])
            if args is None:
                args = []
            elif not isinstance(args, list):
                args = [args]
            if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 1:
                args = [args[0][0]]
            node["args"] = args

        if node.get("kind") == "branch":
            if not isinstance(node.get("then"), list):
                node["then"] = [] if node.get("then") is None else [node["then"]]
            if not isinstance(node.get("else"), list):
                node["else"] = [] if node.get("else") is None else [node["else"]]

        if node.get("kind") == "replan":
            scope = str(node.get("scope", "local")).strip().lower()
            node["scope"] = scope if scope in {"local", "global"} else "local"
            reason = node.get("reason", "Evidence is insufficient.")
            if not isinstance(reason, str):
                reason = str(reason)
            node["reason"] = reason.strip() or "Evidence is insufficient."

        return node

def generate_planner_prompt(tools: Sequence[Any], example_prompt: str, is_replan: bool = False) -> str:
    """生成规划器提示"""
    prefix = (
        "Given a user query, create a short-horizon JSON plan to solve it reliably. "
        f"Use only the following {len(tools) + 2} action types:\n"
    )

    # 工具列表
    for i, tool in enumerate(tools):
        description = getattr(tool, 'description', f"Tool {tool.name}" if hasattr(tool, 'name') else str(tool))
        prefix += f"{i+1}. {description}\n"

    # Join 操作
    prefix += f"{i+2}. {JOIN_DESCRIPTION}\n"
    prefix += f"{i+3}. {BRANCH_DESCRIPTION}\n"
    prefix += f"{i+4}. {REPLAN_DESCRIPTION}\n\n"

    # 指导原则
    prefix += (
        "Output format:\n"
        ' - Return exactly one JSON array and nothing else.\n'
        ' - The top-level plan must be a JSON array of step objects.\n'
        ' - A tool call must be represented as {"kind":"tool","tool":"ToolName","args":[...]}.\n'
        ' - A branch step must be represented as {"kind":"branch","condition":"predicate","then":[...],"else":[...]}.\n'
        ' - A replanning step must be represented as {"kind":"replan","scope":"local"|"global","reason":"short reason"}.\n'
        ' - If the current observations are already sufficient to answer, return [{"kind":"join"}].\n'
        ' - Otherwise return only the next executable plan fragment and omit join.\n'
        ' - You may add an optional "id" field to any node. Prefer this for tools whose outputs will be reused later.\n'
        "Simplicity policy:\n"
        " - Keep the plan as short and flat as possible.\n"
        " - Prefer short-horizon planning over long decompositions.\n"
        " - Planning should be stage-aware: early stage = probe, later stage = short execution.\n"
        " - At the beginning, when information is scarce, prefer branch + replan guarded probing structures around a single new tool observation.\n"
        " - Early plans should aggressively branch on boundary conditions and route quickly into local/global replan instead of forcing one brittle path.\n"
        " - Near the answer, when the object and operands are already verified, reduce branch usage and finish with a short direct path to join.\n"
        " - Do not pre-commit to a downstream chain before the first observation identifies a concrete object, source, file, candidate set, or exact operands.\n"
        " - In early uncertainty, keep probes short; after a concrete source/object is verified, emit a short executable chain (typically 2-5 actions) to complete read/extract/compute.\n"
        " - Control-flow nodes such as branch and replan may wrap that tool action.\n"
        " - The only exception is the terminal join-only round, which must contain only join.\n"
        " - Do not emit multiple independent tool actions in the same round.\n"
        " - Prefer a linear sequence.\n"
        " - Add id only when that output is reused later; otherwise omit id to reduce JSON size.\n"
        "Guidelines:\n"
        " - Each action described above contains input/output types and description.\n"
        " - You must strictly adhere to the input and output types for each action.\n"
        " - Inputs for actions can either be constants or outputs from preceding actions. Prefer $name where name matches a previous node's id field.\n"
        " - Numeric references like $1 are allowed, but named references are preferred in JSON plans.\n"
        " - Args must be valid JSON values only. Do not write expressions like \"text\" + \"$name\" inside JSON.\n"
        " - Every tool node must include an explicit string field \"tool\" with the exact tool name.\n"
        " - Do not use the tool name as kind; kind must stay \"tool\" for tool calls.\n"
        " - Args must be a JSON array. For one-argument tools use [arg], never [[arg]] and never a bare scalar.\n"
        " - If you need a later tool to use an extracted value, pass the reference itself as a separate arg or let the later tool consume $name directly.\n"
        " - Use join only for the terminal plan; non-terminal plans should not include join.\n"
        " - Do not optimize for maximum parallelism if it makes the plan longer or more assumption-heavy.\n"
        " - Each step must either gather a missing observation, verify an object, or transform one compact artifact into the next.\n"
        " - Avoid planning long downstream chains from unverified search results or noisy text.\n"
        " - If the first step is exploratory retrieval and returns a concrete grounded source, continue with the minimum downstream read/extract steps in the same plan instead of stopping at URL discovery.\n"
        " - Prefer branch predicates such as is_empty, is_nonempty, tool_error, contains_url, contains_number, contains_json_field, list_length_ge, matches_regex, contains_str, or llm_judge to detect whether the current observation supports the next move.\n"
        " - Prefer local replan when earlier observations remain useful; prefer global replan when the whole route is contaminated, blocked, or misguided.\n"
        " - Use semantic_map only when a later step needs a typed compact artifact from prior observations, typically in a complex chain such as tool -> semantic_map -> tool.\n"
        " - In simple cases, prefer using the raw tool observation directly instead of inserting semantic_map.\n"
        " - When you do use semantic_map, prefer the 5-argument form semantic_map([global_question, local_request, plan_context, observations, output_schema]).\n"
        " - Its effective context includes the global question, the local sub-question, the history of observations, and the current plan fragment.\n"
        " - Only use the provided action types.\n"
        " - For python actions, args must contain exactly one Python string.\n"
        " - The python code itself must be syntactically valid Python.\n"
        " - If the logic needs loops or multi-step conditionals, encode them with \\n inside the Python string.\n"
        " - If a python action computes or extracts the final scalar/text result, it must print that final result to stdout.\n"
        " - For calculator actions, use exposed helper functions directly: ceil(x), floor(x), sqrt(x), abs(x), round(x), min(...), max(...). Do not write math.ceil(x).\n"
        " - Never include comments, markdown fences, or prose outside the JSON array.\n\n"
    )

    if is_replan:
        prefix += (
            ' - You are given "Previous Plan" with executed observations. Use that information to create the next JSON plan under "Current Plan".\n'
            " - In the Current Plan, NEVER repeat already executed actions from the Previous Plan unless replanning explicitly requires a different query.\n"
            " - Use local replan when you only need to repair the remaining suffix. Use global replan when earlier observations should be discarded and the route should restart from scratch.\n"
        )
    prefix += "Here are some examples:\n\n"
    prefix += example_prompt

    return prefix

class Planner:
    """规划器主类"""
    
    def __init__(
        self,
        llm: Any,
        example_prompt: str,
        example_prompt_replan: str,
        tools: Sequence[Any],
        stop: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.example_prompt = example_prompt
        self.example_prompt_replan = example_prompt_replan
        self.tools = tools
        self.stop = None
        self.output_parser = JSONPlanParser(tools)
    
    async def run_llm(
        self,
        inputs: Dict[str, Any],
        is_replan: bool = False,
        callbacks = None,
        stream: bool = False,
    ) -> Any:
        """运行 LLM"""
        # 生成系统提示
        system_prompt = generate_planner_prompt(
            tools=self.tools,
            example_prompt=self.example_prompt_replan if is_replan else self.example_prompt,
            is_replan=is_replan,
        )
        
        # 构建人类提示
        if is_replan:
            assert "context" in inputs, "If replanning, context must be provided"
            human_prompt = f"Question: {inputs['input']}\n{inputs['context']}\n"
        else:
            human_prompt = f"Question: {inputs['input']}"
        
        full_prompt = f"{system_prompt}\n\n{human_prompt}"
        
        try:
            # 检查 LLM 是否支持 stream
            if stream:
                # 流式生成 - 返回异步生成器
                if hasattr(self.llm, 'agenerate_stream'):
                    return self.llm.agenerate_stream(
                        prompts=[{"text": full_prompt}],
                        stop=self.stop,
                        callbacks=callbacks,
                    )
                elif hasattr(self.llm, '_call_async_stream'):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": human_prompt}
                    ]
                    return self.llm._call_async_stream(
                        messages,
                        callbacks=callbacks,
                        stop=self.stop,
                    )
                else:
                    # 如果不支持 stream，创建简单的生成器
                    async def simple_stream():
                        result = await self._run_non_stream(full_prompt, callbacks)
                        yield result
                    return simple_stream()
            else:
                # 非流式生成
                return await self._run_non_stream(full_prompt, callbacks)
                
        except Exception as e:
            log(f"Error in planner.run_llm: {str(e)}")
            if stream:
                async def error_stream():
                    yield f"Error: {str(e)}"
                return error_stream()
            else:
                return f"Error: {str(e)}"
    
    async def _run_non_stream(self, prompt: str, callbacks = None) -> str:
        """运行非流式 LLM"""
        # 创建类似 LangChain 的 StringPromptValue 格式
        class SimplePromptValue:
            def __init__(self, text):
                self.text = text
        
        prompt_value = SimplePromptValue(text=prompt)
        
        if hasattr(self.llm, 'agenerate_prompt'):
            response = await self.llm.agenerate_prompt(
                prompts=[prompt_value],
                stop=self.stop,
                callbacks=callbacks,
            )
            
            # 解析响应
            if isinstance(response, dict) and 'generations' in response:
                if response['generations'] and response['generations'][0]:
                    gen = response['generations'][0][0]
                    if hasattr(gen, 'text'):
                        return gen.text
                    elif isinstance(gen, dict) and 'text' in gen:
                        return gen['text']
            
            return str(response)
        elif hasattr(self.llm, 'apredict'):
            return await self.llm.apredict(
                prompt,
                callbacks=callbacks,
                stop=self.stop,
            )
        elif hasattr(self.llm, '_call_async'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            return await self.llm._call_async(
                messages,
                callbacks=callbacks,
                stop=self.stop,
            )
        else:
            return f"LLM adapter does not support non-stream generation"
    
    async def plan(
        self,
        inputs: Dict[str, Any],
        is_replan: bool = False,
        callbacks = None,
        **kwargs: Any,
    ) -> Dict[int, Task]:
        """创建计划 - 非流式版本"""
        llm_response = await self.run_llm(
            inputs=inputs,
            is_replan=is_replan,
            callbacks=callbacks,
            stream=False,
        )
        
        if isinstance(llm_response, str):
            response_text = llm_response
        else:
            response_text = str(llm_response)
        
        log("Planner response: \n", response_text, block=True)
        
        return self.output_parser.parse(response_text)

    async def plan_with_raw(
        self,
        inputs: Dict[str, Any],
        is_replan: bool = False,
        callbacks = None,
        **kwargs: Any,
    ) -> tuple[Dict[int, Task], str]:
        """创建计划并返回原始 planner 文本"""
        llm_response = await self.run_llm(
            inputs=inputs,
            is_replan=is_replan,
            callbacks=callbacks,
            stream=False,
        )

        response_text = llm_response if isinstance(llm_response, str) else str(llm_response)
        log("Planner response: \n", response_text, block=True)
        return self.output_parser.parse(response_text), response_text
    
    async def aplan(
        self,
        inputs: Dict[str, Any],
        task_queue: asyncio.Queue,
        is_replan: bool = False,
        callbacks = None,
        **kwargs: Any,
    ) -> None:
        """异步创建计划 - 流式版本"""
        try:
            tasks = await self.plan(
                inputs=inputs,
                is_replan=is_replan,
                callbacks=callbacks,
            )
            for _, task in tasks.items():
                await task_queue.put(task)
            await task_queue.put(None)
        except Exception as e:
            log(f"Error in planner.aplan: {str(e)}")
            await task_queue.put(None)
    
    async def batch_aplan(
        self,
        inputs_list: List[Dict[str, Any]],
        task_queues: List[asyncio.Queue],
        is_replan: bool = False,
        callbacks: List = None,
    ) -> List[Any]:
        """批量异步规划"""
        tasks = []
        for i, inputs in enumerate(inputs_list):
            task = self.aplan(
                inputs=inputs,
                task_queue=task_queues[i],
                is_replan=is_replan,
                callbacks=callbacks[i] if callbacks and i < len(callbacks) else None,
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
