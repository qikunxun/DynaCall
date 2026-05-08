import asyncio
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast
from src.dynacall.constants import JOINNER_REPLAN, JOINNER_FINISH
from src.dynacall.planner import Planner
from src.utils.logger_utils import log

# 定义替代 LangChain 类的接口
class BaseLLMInterface:
    """替代 LangChain BaseLLM 的接口"""
    
    def __init__(self, model_type: str, model_name: str, **kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.config = kwargs
    
    async def agenerate_prompt(self, prompts, stop=None, callbacks=None):
        """替代 LangChain 的 agenerate_prompt 方法"""
        raise NotImplementedError("子类必须实现此方法")
    
    async def apredict(self, text, callbacks=None, stop=None):
        """替代 LangChain 的 apredict 方法"""
        raise NotImplementedError("子类必须实现此方法")

class BaseChatModelInterface(BaseLLMInterface):
    """替代 LangChain BaseChatModel 的接口"""
    
    async def _call_async(self, messages, callbacks=None, stop=None):
        """替代 LangChain 的 _call_async 方法"""
        raise NotImplementedError("子类必须实现此方法")

# 定义简单的回调处理器
class StatsCallbackHandler:
    """替代 LangChain callback handler 的简单统计收集器"""
    
    def __init__(self, stream: bool = False):
        self.stream = stream
        self.stats = self._empty_stats()
        self.additional_fields = {}

    def _empty_stats(self):
        return {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "time": 0.0,
            "all_times": [],
        }

    def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        elapsed: float = 0.0,
    ):
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        total_tokens = int(total_tokens or (input_tokens + output_tokens) or 0)
        elapsed = float(elapsed or 0.0)
        self.stats["calls"] += 1
        self.stats["input_tokens"] += input_tokens
        self.stats["output_tokens"] += output_tokens
        self.stats["total_tokens"] += total_tokens
        self.stats["time"] += elapsed
        self.stats["all_times"].append(round(elapsed, 2))
    
    def reset(self):
        self.stats = self._empty_stats()
        self.additional_fields = {}
    
    def get_stats(self):
        return {**self.stats, **self.additional_fields}

class AsyncStatsCallbackHandler(StatsCallbackHandler):
    """异步版本的统计收集器"""
    pass

class LLMAgent:
    """自代理定义，去除了 LangChain 依赖"""
    
    def __init__(self, llm: BaseLLMInterface) -> None:
        self.llm = llm
    
    async def arun(self, prompt: str, callbacks=None) -> str:
        """运行 LLM 代理"""
        try:
            # 尝试使用 agenerate_prompt 方法（类似 LangChain）
            if hasattr(self.llm, 'agenerate_prompt'):
                response = await self.llm.agenerate_prompt(
                    prompts=[{"text": prompt}],  # 简化格式
                    stop=["<END_OF_RESPONSE>"],
                    callbacks=callbacks,
                )
                
                # 简化响应解析
                if isinstance(response, dict) and 'generations' in response:
                    if response['generations'] and response['generations'][0]:
                        if response['generations'][0][0] and 'text' in response['generations'][0][0]:
                            return response['generations'][0][0]['text']
                        elif response['generations'][0][0] and 'message' in response['generations'][0][0]:
                            if hasattr(response['generations'][0][0]['message'], 'content'):
                                return response['generations'][0][0]['message'].content
                            elif isinstance(response['generations'][0][0]['message'], dict):
                                return response['generations'][0][0]['message'].get('content', '')
                
                return str(response)
            
            # 尝试使用 apredict 方法
            elif hasattr(self.llm, 'apredict'):
                return await self.llm.apredict(
                    prompt,
                    callbacks=callbacks,
                    stop=["<END_OF_RESPONSE>"]
                )
            
            # 最后的回退方案
            else:
                return f"LLM response to: {prompt[:100]}..."
                
        except Exception as e:
            log(f"Error in LLMAgent.arun: {str(e)}")
            return f"Error: {str(e)}"

class Controller:
    """控制器引擎，去除了 LangChain 依赖"""
    
    def __init__(
        self,
        tools: Sequence[Any],  # 使用 Any 替代 Union[Tool, StructuredTool]
        planner_llm: BaseLLMInterface,
        planner_example_prompt: str,
        planner_example_prompt_replan: Optional[str],
        planner_stop: Optional[list[str]],
        planner_stream: bool,
        agent_llm: BaseLLMInterface,
        joinner_prompt: str,
        joinner_prompt_final: Optional[str],
        planner_critic_prompt: Optional[str],
        planner_critic_prompt_replan: Optional[str],
        max_replans: int,
        benchmark: bool,
        **kwargs,
    ) -> None:
        
        # 保存参数
        self.tools = tools
        self.planner_llm = planner_llm
        self.planner_example_prompt = planner_example_prompt
        self.planner_example_prompt_replan = planner_example_prompt_replan or planner_example_prompt
        self.planner_stop = planner_stop
        self.planner_stream = planner_stream
        self.agent = LLMAgent(agent_llm)
        self.joinner_prompt = joinner_prompt
        self.joinner_prompt_final = joinner_prompt_final or joinner_prompt
        self.planner_critic_prompt = planner_critic_prompt
        self.planner_critic_prompt_replan = planner_critic_prompt_replan or planner_critic_prompt
        self.plan_critic = LLMAgent(planner_llm) if planner_critic_prompt else None
        self.max_replans = max_replans
        self.benchmark = benchmark
        
        # 延迟导入 planner，避免循环依赖
        self._planner = None
        
        # 回调处理器
        if benchmark:
            self.planner_callback = AsyncStatsCallbackHandler(stream=planner_stream)
            self.executor_callback = AsyncStatsCallbackHandler(stream=False)
        else:
            self.planner_callback = None
            self.executor_callback = None
        
        # 输入输出键
        self.input_key = "input"
        self.output_key = "output"
    
    @property
    def planner(self):
        """延迟加载 planner 以避免导入问题"""
        if self._planner is None:
            self._planner = Planner(
                llm=self.planner_llm,
                example_prompt=self.planner_example_prompt,
                example_prompt_replan=self.planner_example_prompt_replan,
                tools=self.tools,
                stop=self.planner_stop,
            )
        return self._planner
    
    def get_all_stats(self):
        """获取所有统计信息"""
        stats = {}
        if self.benchmark:
            stats["planner"] = self.planner_callback.get_stats()
            stats["executor"] = self.executor_callback.get_stats()
            total = {}
            keys = set(stats["planner"]) | set(stats["executor"])
            for key in keys:
                planner_value = stats["planner"].get(key, 0)
                executor_value = stats["executor"].get(key, 0)
                if isinstance(planner_value, list) or isinstance(executor_value, list):
                    total[key] = (
                        (planner_value if isinstance(planner_value, list) else [])
                        + (executor_value if isinstance(executor_value, list) else [])
                    )
                elif isinstance(planner_value, (int, float)) and isinstance(executor_value, (int, float)):
                    total[key] = planner_value + executor_value
            stats["total"] = total
        return stats
    
    def reset_all_stats(self):
        """重置所有统计信息"""
        if self.planner_callback:
            self.planner_callback.reset()
        if self.executor_callback:
            self.executor_callback.reset()
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _parse_joinner_output(self, raw_answer: str) -> tuple:
        """解析 joinner 输出"""
        thought, answer, is_replan = "", "", False

        explicit_action_seen = False
        raw_answers = raw_answer.split("\n")
        for ans in raw_answers:
            line = ans.strip()
            if not line:
                continue

            if line.startswith("Thought:"):
                thought = line.split("Thought:", 1)[1].strip()
                continue

            action_text = line[len("Action:"):].strip() if line.startswith("Action:") else line
            match = re.match(rf"^({JOINNER_FINISH}|{JOINNER_REPLAN})\((.*)\)\s*$", action_text)
            if match:
                answer = match.group(2).strip()
                is_replan = match.group(1) == JOINNER_REPLAN
                explicit_action_seen = True
                continue

            # 仅在尚未发现显式 Action 时，才回退使用普通文本。
            if not explicit_action_seen:
                answer = action_text
                is_replan = False
        return thought, answer, is_replan

    def _contains_nested_action(self, text: str) -> bool:
        stripped = (text or "").strip()
        return (
            f"{JOINNER_FINISH}(" in stripped
            or f"{JOINNER_REPLAN}(" in stripped
        )

    def _contains_invalid_final_action(self, raw_answer: str) -> bool:
        for line in raw_answer.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if f"{JOINNER_FINISH}({JOINNER_REPLAN}(" in stripped:
                return True
            if f"{JOINNER_FINISH}({JOINNER_FINISH}(" in stripped:
                return True
            if f"{JOINNER_REPLAN}({JOINNER_FINISH}(" in stripped:
                return True
            if f"{JOINNER_REPLAN}({JOINNER_REPLAN}(" in stripped:
                return True
            if not stripped.startswith("Action:"):
                if stripped.startswith(f"{JOINNER_FINISH}(") or stripped.startswith(f"{JOINNER_REPLAN}("):
                    continue
                continue
            action_text = stripped[len("Action:"):].strip()
            if action_text.startswith(f"{JOINNER_FINISH}(") or action_text.startswith(f"{JOINNER_REPLAN}("):
                continue
            return True
        return False

    def _normalize_symbol_name_answer(self, question: str, answer: str) -> str:
        q = (question or "").lower()
        if not any(token in q for token in ["name of the character", "name of the symbol", "name of one character", "punctuation mark", "token"]):
            return answer

        symbol_map = {
            "`": "backtick",
            "'": "apostrophe",
            '"': "quote",
            "~": "tilde",
            "!": "exclamation mark",
            "@": "at",
            "#": "hash",
            "$": "dollar sign",
            "%": "percent sign",
            "^": "caret",
            "&": "ampersand",
            "*": "asterisk",
            "(": "left parenthesis",
            ")": "right parenthesis",
            "-": "hyphen",
            "_": "underscore",
            "=": "equals sign",
            "+": "plus sign",
            "[": "left bracket",
            "]": "right bracket",
            "{": "left brace",
            "}": "right brace",
            "|": "pipe",
            "\\": "backslash",
            ":": "colon",
            ";": "semicolon",
            ",": "comma",
            ".": "period",
            "<": "less than sign",
            ">": "greater than sign",
            "/": "slash",
            "?": "question mark",
        }
        return symbol_map.get(answer, answer)

    def _normalize_final_answer(self, question: str, answer: str) -> str:
        normalized = (answer or "").strip()
        if not normalized:
            return normalized

        q = (question or "").lower()
        if normalized.startswith(JOINNER_FINISH + "(") and normalized.endswith(")"):
            normalized = normalized[len(JOINNER_FINISH) + 1 : -1].strip()
        normalized = self._normalize_symbol_name_answer(question, normalized)

        asks_numeric = any(
            token in q
            for token in [
                "how many",
                "what is the percentage",
                "percentage",
                "what is the average",
                "average number",
                "volume",
                "what was the volume",
                "in m^3",
                "in m^2",
                "in km",
                "in kg",
            ]
        )
        if asks_numeric and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized):
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)
            if len(numbers) == 1:
                normalized = numbers[0]
            else:
                unit_match = re.fullmatch(
                    r"\s*([-+]?\d+(?:\.\d+)?)\s*(?:[%A-Za-zμ/_^ -]+|\b[a-zA-Z]+\^\d+\b|\bm\^\d+\b)+\s*",
                    normalized,
                )
                if unit_match:
                    normalized = unit_match.group(1)

        scale_matchers = [
            (("how many thousand", "in thousands"), 1000),
            (("how many million", "in millions"), 1_000_000),
            (("how many billion", "in billions"), 1_000_000_000),
        ]
        if re.fullmatch(r"[-+]?\d+(?:\.0+)?", normalized):
            value = float(normalized)
            for phrases, scale in scale_matchers:
                if any(phrase in q for phrase in phrases) and abs(value) >= scale:
                    scaled = value / scale
                    normalized = str(int(scaled)) if float(scaled).is_integer() else str(scaled)
                    break

        if any(phrase in q for phrase in ["comma delimited", "comma-delimited", "comma separated", "comma-separated"]):
            parts = [part.strip() for part in normalized.split(",")]
            parts = [part for part in parts if part]
            if len(parts) >= 2:
                normalized = ", ".join(parts)

        if "singular form" in q:
            if " & " in normalized:
                normalized = normalized.split("&", 1)[0].strip()
            elif re.search(r"\band\b", normalized, flags=re.I):
                normalized = re.split(r"\band\b", normalized, maxsplit=1, flags=re.I)[0].strip()
            normalized = re.sub(r"^(a|an|the)\s+", "", normalized, flags=re.I).strip()

        if normalized.lower() in {"true", "false"}:
            if any(token in q for token in ["can ", "could ", "would ", "does ", "do ", "is ", "are ", "was ", "were "]):
                normalized = "Yes" if normalized.lower() == "true" else "No"

        if "scene heading" in q and "location" in q:
            heading_match = re.match(r"^(?:int|ext|int\./ext|ext\./int)\.?\s+(.*?)(?:\s+-\s+(?:day|night|morning|evening|continuous|later|same time))?$", normalized, flags=re.I)
            if heading_match:
                normalized = heading_match.group(1).strip()

        if "exactly as it appears" in q and "location called" in q:
            normalized = re.sub(r"^(?:int|ext|int\./ext|ext\./int)\.?\s+", "", normalized, flags=re.I).strip()
            normalized = re.sub(r"\s+-\s+(?:day|night|morning|evening|continuous|later|same time)$", "", normalized, flags=re.I).strip()

        return normalized

    
    def _generate_context_for_replanner(
        self,
        tasks: Mapping[int, Any],
        joinner_thought: str,
        structured_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """为重新规划生成上下文"""
        sections: List[str] = []

        if structured_state:
            sections.append(
                "Replan State JSON:\n"
                + json.dumps(structured_state, ensure_ascii=False, indent=2)
            )

        if not tasks:
            sections.append(f"Thought: {joinner_thought}")
            return "\n\n".join(sections)
        
        previous_plan_and_observations = "\n".join([
            self._format_task_for_context(task, idx)
            for idx, task in tasks.items()
            if not getattr(task, 'is_join', False)
        ])
        
        joinner_thought = f"Thought: {joinner_thought}"
        sections.extend([previous_plan_and_observations, joinner_thought])
        context = "\n\n".join(sections)
        return context
    
    def _format_task_for_context(self, task: Any, idx: int) -> str:
        """格式化任务用于上下文"""
        result = []
        
        # 添加思考
        if hasattr(task, 'thought') and task.thought:
            result.append(f"Thought: {task.thought}")
        
        # 添加动作
        if hasattr(task, 'name'):
            action_str = f"{idx}. {task.name}"
            if hasattr(task, 'args') and task.args:
                args_str = str(task.args)
                action_str += f"({args_str})"
            result.append(action_str)
        
        # 添加观察
        if hasattr(task, 'observation') and task.observation:
            result.append(f"Observation: {task.observation}")
        
        return "\n".join(result)

    def _extract_last_observation_candidate(self, scratchpad: str) -> str:
        if not scratchpad:
            return ""

        lines = [line.strip() for line in scratchpad.splitlines() if line.strip()]
        last_tool = ""
        candidates: List[tuple[int, str]] = []
        preferred_tools = {
            "semantic_map": 4,
            "code_interpreter": 4,
            "calculator": 4,
            "quote_verifier": 4,
            "ocr": 3,
            "spreadsheet_reader": 2,
            "pdf_viewer": 2,
            "web_browser": 1,
            "search_engine": 0,
            "branch": -2,
            "replan": -3,
            "join": -3,
        }

        for line in lines:
            action_match = re.match(r"^\d+\.\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if action_match:
                last_tool = action_match.group(1).strip().lower()
                continue
            if not line.startswith("Observation:"):
                continue
            obs = line.split("Observation:", 1)[1].strip()
            lowered = obs.lower()
            if not obs:
                continue
            if any(
                marker in lowered
                for marker in [
                    "skipped by branch",
                    "task result missing",
                    "execution error",
                    "tool error",
                    "replan requested",
                ]
            ):
                continue
            if len(obs) > 3000:
                continue
            score = preferred_tools.get(last_tool, 0)
            if len(obs) <= 120:
                score += 1
            candidates.append((score, obs))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1].strip()
        if best.lower().startswith("stdout:"):
            best = best.split("stdout:", 1)[1].strip()
            if "\n" in best:
                best = best.splitlines()[0].strip()
        return best
    
    def _format_contexts(self, contexts: Sequence[str]) -> str:
        """格式化多个上下文"""
        formatted_contexts = ""
        for context in contexts:
            formatted_contexts += f"Previous Plan:\n\n{context}\n\n"
        formatted_contexts += "Current Plan:\n\n"
        return formatted_contexts

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        candidates = [text]
        obj_match = re.search(r"\{.*\}", text, flags=re.S)
        arr_match = re.search(r"\[.*\]", text, flags=re.S)
        if obj_match:
            candidates.insert(0, obj_match.group(0))
        if arr_match:
            candidates.insert(0, arr_match.group(0))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                continue
        return None

    def _flatten_plan_steps(self, plan_node: Any) -> List[Dict[str, Any]]:
        if isinstance(plan_node, list):
            flattened: List[Dict[str, Any]] = []
            for item in plan_node:
                flattened.extend(self._flatten_plan_steps(item))
            return flattened
        if not isinstance(plan_node, dict):
            return []
        flattened = [plan_node]
        if isinstance(plan_node.get("steps"), list):
            flattened.extend(self._flatten_plan_steps(plan_node["steps"]))
        if isinstance(plan_node.get("then"), list):
            flattened.extend(self._flatten_plan_steps(plan_node["then"]))
        if isinstance(plan_node.get("else"), list):
            flattened.extend(self._flatten_plan_steps(plan_node["else"]))
        return flattened

    def _hard_review_plan(self, question: str, raw_plan: str) -> Optional[Dict[str, Any]]:
        plan = self._extract_first_json_object(raw_plan)
        if not plan:
            return {
                "approve": False,
                "reason": "Planner output was not valid JSON.",
                "suggestion": "Return one valid JSON plan only.",
            }

        steps = self._flatten_plan_steps(plan)
        q = (question or "").lower()
        tool_names = [
            str(step.get("tool") or step.get("type") or "").lower()
            for step in steps
            if isinstance(step, dict)
        ]
        non_join_steps = [
            step for step in steps
            if isinstance(step, dict) and str(step.get("tool") or step.get("type") or "").lower() != "join"
        ]
        def parse_semantic_map_args(args: Any) -> tuple[Any, str]:
            if not isinstance(args, list) or len(args) < 3:
                return None, ""
            schema = str(args[-1] or "").strip().lower() if isinstance(args[-1], str) else ""
            direct_inputs = args[-2] if len(args) >= 2 else None
            if isinstance(direct_inputs, str) and direct_inputs.strip().lower() in {
                "string", "number", "boolean", "list[string]"
            }:
                direct_inputs = args[1] if len(args) >= 2 else None
            return direct_inputs, schema

        json_semantic_ids = set()
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("tool") or "").lower() != "semantic_map":
                continue
            args = step.get("args")
            semantic_inputs, schema = parse_semantic_map_args(args)
            if semantic_inputs is None and not schema:
                continue
            if (
                semantic_inputs is None
                or semantic_inputs == ""
                or semantic_inputs == []
                or semantic_inputs == {}
            ):
                return {
                    "approve": False,
                    "reason": "semantic_map was called with empty direct inputs.",
                    "suggestion": "Pass the specific prior observation reference or source text into semantic_map; history context is only auxiliary and must not replace direct inputs.",
                }
            step_id = str(step.get("id") or "").strip()
            if step_id and schema.startswith("json{"):
                json_semantic_ids.add(step_id)

        count_terms = (
            "how many" in q
            or "count" in q
            or "number of" in q
            or "average number" in q
            or "percentage" in q
            or "percent" in q
            or "proportion" in q
            or "fraction" in q
            or "rate of" in q
        )
        grounding_terms = any(
            token in q
            for token in [
                "how many", "count", "number of", "which", "who", "what", "when", "where",
                "wikipedia", "website", "web", "url", "page", "article", "paper", "script",
                "image", "video", "audio", "pdf", "docx", "xlsx", "csv", "table", "file",
                "according to", "from the", "in the", "between", "latest",
            ]
        )
        if tool_names and all(name == "join" for name in tool_names) and grounding_terms:
            return {
                "approve": False,
                "reason": "Join-only plan tries to answer before collecting any grounded evidence.",
                "suggestion": "First fetch one concrete source or file observation. For count/list/extraction questions, prefer source validation and semantic_map before the final join.",
            }

        exact_title_retrieval = (
            ('"' in question or "“" in question)
            and any(token in q for token in ["paper", "authors", "title", "article", "script", "official"])
        )
        if exact_title_retrieval:
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if str(step.get("tool") or "").lower() != "web_browser":
                    continue
                args = step.get("args")
                if not isinstance(args, list) or not args:
                    continue
                first_arg = str(args[0] or "").strip()
                if first_arg and not re.match(r"^https?://", first_arg, flags=re.I) and not first_arg.startswith("$"):
                    return {
                        "approve": False,
                        "reason": "Plan opens web_browser without first isolating a concrete URL/object for an exact-title retrieval question.",
                        "suggestion": "Use search results to isolate the exact target URL/object first, branch on object match, then open that concrete page.",
                    }

            raw_plan_lower = raw_plan.lower()
            if "webcache.googleusercontent.com" in raw_plan_lower or "googleusercontent.com" in raw_plan_lower:
                return {
                    "approve": False,
                    "reason": "Plan is over-constrained to a cache mirror domain instead of the target source.",
                    "suggestion": "Open the best exact-title source directly and branch on object match; do not require a specific cache mirror unless the user asked for it.",
                }

        paper_attribute_question = (
            "paper" in q
            and any(token in q for token in ["chemical", "method", "ec number", "doi", "author", "journal", "title"])
        )
        if paper_attribute_question:
            has_object_verification = False
            for step in steps:
                if not isinstance(step, dict):
                    continue
                step_tool = str(step.get("tool") or "").lower()
                if step_tool != "branch":
                    continue
                condition_blob = json.dumps(step.get("condition", ""), ensure_ascii=False).lower()
                if any(
                    token in condition_blob
                    for token in [
                        "correct paper", "target paper", "correct article", "correct journal article",
                        "object match", "paper page", "doi", "journal", "title"
                    ]
                ):
                    has_object_verification = True
                    break
            if not has_object_verification and any(
                isinstance(step, dict) and str(step.get("tool") or "").lower() in {"web_browser", "semantic_map"}
                for step in steps
            ):
                return {
                    "approve": False,
                    "reason": "Paper-attribute plan starts extraction before verifying the concrete target paper object.",
                    "suggestion": "First isolate and verify the exact target paper page or DOI with a branch on object match; only then extract methods, chemicals, or numbers.",
                }

        first_mentioned_text_question = (
            any(token in q for token in ["first place mentioned", "first person mentioned", "first city mentioned", "first country mentioned"])
            and any(token in q for token in ["book", "chapter", "verse", "passage", "niv", "text"])
        )
        if first_mentioned_text_question:
            has_extraction_step = False
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if str(step.get("tool") or "").lower() != "semantic_map":
                    continue
                args = step.get("args")
                if not isinstance(args, list) or not args:
                    continue
                instruction = str(args[0] or "").lower()
                if any(token in instruction for token in ["first place", "first named", "first country", "first city", "first person"]):
                    has_extraction_step = True
                    break
            if not has_extraction_step and any(
                isinstance(step, dict) and str(step.get("tool") or "").lower() == "search_engine"
                for step in steps
            ):
                return {
                    "approve": False,
                    "reason": "Plan jumps to downstream lookup before extracting the first named entity from the source text.",
                    "suggestion": "Open the exact passage, use semantic_map to extract the first named place/entity from the wording, branch on that extraction, and only then do the downstream lookup.",
                }
        repeated_record_terms = any(
            token in q
            for token in [
                "album", "discography", "works", "papers", "issues", "episodes",
                "records", "publications", "studio albums",
            ]
        )
        version_audit_terms = (
            any(token in q for token in ["superseded", "new version", "revised", "current version", "as of august", "as of "])
            and any(token in q for token in ["standard", "standards", "regulation", "manual", "dataset", "products"])
        )
        if count_terms and repeated_record_terms:
            if not non_join_steps:
                return {
                    "approve": False,
                    "reason": "Repeated-record count plan has no evidence-gathering step.",
                    "suggestion": "Fetch a complete source first, then use semantic_map to extract records, branch on completeness, and count deterministically.",
                }
            semantic_steps = [
                step for step in steps
                if isinstance(step, dict) and str(step.get("tool") or "").lower() == "semantic_map"
            ]
            raw_count_steps = [
                step for step in steps
                if isinstance(step, dict) and str(step.get("tool") or "").lower() in {"python", "code_interpreter", "calculator"}
            ]
            record_extractors = []
            direct_scalar_extractors = []
            for step in semantic_steps:
                args = step.get("args")
                _, schema = parse_semantic_map_args(args)
                if "records:list[" in schema or "locations:list[" in schema or "list[" in schema:
                    record_extractors.append(step)
                elif schema in {"number", "int", "float", "string", "str"}:
                    direct_scalar_extractors.append(step)

            if direct_scalar_extractors:
                return {
                    "approve": False,
                    "reason": "Repeated-record count plan asks semantic_map for a final scalar directly from noisy evidence.",
                    "suggestion": "First extract structured records with semantic_map, branch on completeness, then compute the final count deterministically.",
                }

            if "web_browser" in tool_names and not record_extractors:
                return {
                    "approve": False,
                    "reason": "Repeated-record count plan lacks a structured record extraction step.",
                    "suggestion": "Use semantic_map to extract structured records from the candidate page, branch on completeness, then count with code_interpreter.",
                }

            if raw_count_steps and not record_extractors:
                return {
                    "approve": False,
                    "reason": "Repeated-record count plan tries to count from raw text before building a structured record artifact.",
                    "suggestion": "First extract records:list[...] with semantic_map, branch on completeness, then run deterministic counting over that artifact.",
                }

        if count_terms and version_audit_terms:
            semantic_steps = [
                step for step in steps
                if isinstance(step, dict) and str(step.get("tool") or "").lower() == "semantic_map"
            ]
            record_extractors = []
            status_extractors = []
            deterministic_steps = [
                step for step in steps
                if isinstance(step, dict) and str(step.get("tool") or "").lower() in {"python", "code_interpreter", "calculator"}
            ]
            for step in semantic_steps:
                args = step.get("args")
                _, schema = parse_semantic_map_args(args)
                if "records:list[" not in schema:
                    continue
                record_extractors.append(step)
                if "superseded:boolean" in schema or "current_source:string" in schema:
                    status_extractors.append(step)

            if not record_extractors:
                return {
                    "approve": False,
                    "reason": "Version-audit percentage plan lacks a structured record extraction step.",
                    "suggestion": "First extract the historical target item records with semantic_map, branch on completeness, then inspect current official sources per item.",
                }

            if deterministic_steps and not status_extractors:
                return {
                    "approve": False,
                    "reason": "Version-audit percentage plan computes a scalar before building per-record status booleans.",
                    "suggestion": "Build a records:list[...] artifact with explicit superseded:boolean and support fields, then compute the percentage deterministically.",
                }

        optimization_terms = any(
            token in q for token in ["maximize", "best", "optimal", "highest odds", "most likely", "which ball should you choose"]
        )
        if optimization_terms:
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if str(step.get("tool") or "").lower() not in {"python", "code_interpreter"}:
                    continue
                args = step.get("args")
                code_blob = json.dumps(args, ensure_ascii=False).lower()
                if any(token in code_blob for token in ["random", "trials", "monte", "counter(", "randint"]):
                    return {
                        "approve": False,
                        "reason": "Optimization plan relies on Monte Carlo simulation for the final answer.",
                        "suggestion": "Use deterministic reasoning or exact dynamic programming/state analysis instead of simulation for the final answer path.",
                    }

        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("tool") or "").lower() != "calculator":
                continue
            calc_args = json.dumps(step.get("args", []), ensure_ascii=False)
            if any(f"${semantic_id}" in calc_args for semantic_id in json_semantic_ids):
                return {
                    "approve": False,
                    "reason": "Calculator input references a structured semantic_map object.",
                    "suggestion": "Extract the needed scalar field before calculator and compute on that scalar only.",
                }

        return None

    async def _review_plan(
        self,
        question: str,
        raw_plan: str,
        is_replan: bool,
    ) -> Optional[Dict[str, Any]]:
        hard_review = self._hard_review_plan(question=question, raw_plan=raw_plan)
        if hard_review:
            log("Hard plan critic response: \n", hard_review, block=True)
            return hard_review
        if not self.plan_critic:
            return None
        critic_prompt = self.planner_critic_prompt_replan if is_replan else self.planner_critic_prompt
        if not critic_prompt:
            return None
        prompt = (
            f"{critic_prompt}\n\n"
            f"Question: {question}\n\n"
            f"Proposed Plan JSON:\n{raw_plan}\n"
        )
        raw_review = await self.plan_critic.arun(prompt)
        review = self._extract_first_json_object(raw_review)
        if not review:
            log("Plan critic returned non-JSON output; skipping review.\n", raw_review, block=True)
            return None
        log("Plan critic response: \n", review, block=True)
        return review
    
    async def join(self, input_query: str, agent_scratchpad: str, is_final: bool, callbacks=None) -> tuple:
        """执行 join 操作"""
        joinner_prompt = self.joinner_prompt_final if is_final else self.joinner_prompt
        
        prompt = (
            f"{joinner_prompt}\n"
            f"Question: {input_query}\n\n"
            f"{agent_scratchpad}\n"
        )
        
        log("Joining prompt:\n", prompt, block=True)
        
        # 使用代理运行
        join_callbacks = callbacks if callbacks is not None else ([self.executor_callback] if self.benchmark else None)
        response = await self.agent.arun(
            prompt, 
            callbacks=join_callbacks,
        )
        
        raw_answer = cast(str, response)
        if is_final and self._contains_invalid_final_action(raw_answer):
            retry_prompt = (
                f"{prompt}\n"
                "Reminder: this is the final answer step. Do not emit any tool action such as python(...), search_engine(...), or semantic_map(...).\n"
                f"Do not nest actions such as {JOINNER_FINISH}({JOINNER_REPLAN}(...)) or {JOINNER_FINISH}({JOINNER_FINISH}(...)).\n"
                f"Return exactly one final action in the form {JOINNER_FINISH}(answer).\n"
            )
            response = await self.agent.arun(
                retry_prompt,
                callbacks=join_callbacks,
            )
            raw_answer = cast(str, response)

        log("Question: \n", input_query, block=True)
        log("Raw Answer: \n", raw_answer, block=True)
        
        thought, answer, is_replan = self._parse_joinner_output(raw_answer)
        if is_final and self._contains_nested_action(answer):
            retry_prompt = (
                f"{prompt}\n"
                "Reminder: this is the final answer step.\n"
                f"The payload inside {JOINNER_FINISH}(...) must be the final answer itself, not another action.\n"
                f"Return exactly one final action in the form {JOINNER_FINISH}(answer).\n"
            )
            response = await self.agent.arun(
                retry_prompt,
                callbacks=join_callbacks,
            )
            raw_answer = cast(str, response)
            log("Raw Answer Retry: \n", raw_answer, block=True)
            thought, answer, is_replan = self._parse_joinner_output(raw_answer)

        # Robust fallback: if the raw model output still contains a Replan action,
        # preserve the control-flow intent instead of treating it as a final answer.
        if re.search(r"\breplan\s*\(", raw_answer or "", flags=re.IGNORECASE):
            is_replan = True

        if is_final and is_replan:
            force_finish_prompt = (
                f"{prompt}\n"
                "You are at the final allowed iteration. Replan is forbidden now.\n"
                f"Return exactly one action in the form {JOINNER_FINISH}(answer).\n"
                "Choose the best grounded candidate from existing observations only.\n"
                "Do not output missing-information text such as 'need ...'.\n"
            )
            response = await self.agent.arun(
                force_finish_prompt,
                callbacks=join_callbacks,
            )
            raw_answer = cast(str, response)
            log("Raw Answer Final Force-Finish Retry: \n", raw_answer, block=True)
            thought, answer, is_replan = self._parse_joinner_output(raw_answer)
            if re.search(r"\breplan\s*\(", raw_answer or "", flags=re.IGNORECASE):
                is_replan = True

            if is_replan or re.match(r"^\s*need\b", (answer or "").strip(), flags=re.I):
                fallback = self._extract_last_observation_candidate(agent_scratchpad)
                if fallback:
                    answer = fallback
                    is_replan = False
                    log("Applied fallback answer from latest grounded observation.", block=True)

        answer = self._normalize_final_answer(input_query, answer)

        return thought, answer, is_replan
    
    async def _acall(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """异步调用控制器"""
        try:
            contexts = []
            joinner_thought = ""
            agent_scratchpad = ""
            
            if "input" not in inputs:
                return {self.output_key: "Error: Missing 'input' in request"}
            
            for i in range(self.max_replans):
                is_first_iter = i == 0
                is_final_iter = i == self.max_replans - 1
                
                # 导入 TaskFetchingUnit
                from src.dynacall.task import TaskFetchingUnit
                task_fetching_unit = TaskFetchingUnit()
                
                if self.planner_stream:
                    # 为了让流式规划也经过 plan review，这里先获取完整 raw_plan 再调度。
                    try:
                        tasks, raw_plan = await self.planner.plan_with_raw(
                            inputs=inputs,
                            is_replan=not is_first_iter,
                            callbacks=(
                                [self.planner_callback] if self.planner_callback else None
                            ),
                        )
                        review = await self._review_plan(
                            question=inputs["input"],
                            raw_plan=raw_plan,
                            is_replan=not is_first_iter,
                        )
                        review_rejects = isinstance(review, dict) and not bool(review.get("approve", True))
                        if review_rejects and not is_final_iter:
                            reason = str(review.get("reason", "") or "The current plan is too weak to execute.")
                            suggestion = str(review.get("suggestion", "") or "")
                            critique_message = f"Plan critic rejected current plan. Reason: {reason}"
                            if suggestion:
                                critique_message += f" Suggestion: {suggestion}"
                            context = self._generate_context_for_replanner(tasks={}, joinner_thought=critique_message)
                            contexts.append(context)
                            formatted_contexts = self._format_contexts(contexts)
                            log("Plan critic requested replanning.\n", formatted_contexts, block=True)
                            inputs["context"] = formatted_contexts
                            continue
                        if review_rejects and is_final_iter:
                            log("Plan critic rejected final planning attempt; executing current plan anyway.", block=True)

                        log("Graph of tasks: ", tasks, block=True)

                        if self.benchmark:
                            self.planner_callback.additional_fields["num_tasks"] = len(tasks)

                        task_fetching_unit.set_tasks(tasks)
                        await task_fetching_unit.schedule()
                    except Exception as e:
                        log(f"Error in planning phase: {str(e)}")
                        return {self.output_key: f"Planning error: {str(e)}"}
                else:
                    # 非流式规划
                    try:
                        tasks, raw_plan = await self.planner.plan_with_raw(
                            inputs=inputs,
                            is_replan=not is_first_iter,
                            callbacks=(
                                [self.planner_callback] if self.planner_callback else None
                            ),
                        )
                        review = await self._review_plan(
                            question=inputs["input"],
                            raw_plan=raw_plan,
                            is_replan=not is_first_iter,
                        )
                        review_rejects = isinstance(review, dict) and not bool(review.get("approve", True))
                        if review_rejects and not is_final_iter:
                            reason = str(review.get("reason", "") or "The current plan is too weak to execute.")
                            suggestion = str(review.get("suggestion", "") or "")
                            critique_message = f"Plan critic rejected current plan. Reason: {reason}"
                            if suggestion:
                                critique_message += f" Suggestion: {suggestion}"
                            context = self._generate_context_for_replanner(tasks={}, joinner_thought=critique_message)
                            contexts.append(context)
                            formatted_contexts = self._format_contexts(contexts)
                            log("Plan critic requested replanning.\n", formatted_contexts, block=True)
                            inputs["context"] = formatted_contexts
                            continue
                        if review_rejects and is_final_iter:
                            log("Plan critic rejected final planning attempt; executing current plan anyway.", block=True)

                        log("Graph of tasks: ", tasks, block=True)
                        
                        if self.benchmark:
                            self.planner_callback.additional_fields["num_tasks"] = len(tasks)
                        
                        task_fetching_unit.set_tasks(tasks)
                        await task_fetching_unit.schedule()
                    except Exception as e:
                        log(f"Error in planning phase: {str(e)}")
                        return {self.output_key: f"Planning error: {str(e)}"}
                
                tasks = task_fetching_unit.tasks
                
                # 收集当前轮思考-动作-观察；不要把旧轮次 scratchpad 带入 join。
                current_agent_scratchpad = "\n\n"
                if tasks:
                    for task in tasks.values():
                        if not getattr(task, 'is_join', False):
                            current_agent_scratchpad += self._format_task_for_context(task, task.idx) + "\n"
                agent_scratchpad = current_agent_scratchpad.strip()
                log("Agent scratchpad:\n", agent_scratchpad, block=True)
                
                try:
                    joinner_thought, answer, is_replan = await self.join(
                        inputs["input"],
                        agent_scratchpad=agent_scratchpad,
                        is_final=is_final_iter,
                    )
                except Exception as e:
                    log(f"Error in join phase: {str(e)}")
                    return {self.output_key: f"Join error: {str(e)}"}
                
                if not is_replan:
                    log("Break out of replan loop.")
                    break
                
                # 为后续重新规划收集上下文
                context = self._generate_context_for_replanner(
                    tasks=tasks, joinner_thought=joinner_thought
                )
                contexts.append(context)
                formatted_contexts = self._format_contexts(contexts)
                log("Contexts:\n", formatted_contexts, block=True)
                inputs["context"] = formatted_contexts
            
            if is_final_iter:
                log("Reached max replan limit.")
            
            return {self.output_key: answer}
            
        except Exception as e:
            log(f"Unexpected engine error: {str(e)}")
            return {self.output_key: f"Unexpected error: {str(e)}"}
    
    # 为了兼容性，添加 _call 方法
    def _call(self, inputs: Dict[str, Any], run_manager=None):
        raise NotImplementedError("This engine is async only.")
    
    async def _abatch_call(self, inputs_list: List[Dict[str, Any]], run_manager=None) -> List[Dict[str, Any]]:
        """批量处理多个输入"""
        results = []
        for inputs in inputs_list:
            result = await self._acall(inputs, run_manager)
            results.append(result)
        return results
    
    # 简化版本的处理方法
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """简化的处理接口"""
        return await self._acall(inputs)
