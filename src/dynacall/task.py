from __future__ import annotations

import asyncio
import ast
import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Optional, Set
from src.utils.logger_utils import log

def _default_stringify_rule_for_arguments(args):
    if len(args) == 1:
        return str(args[0])
    else:
        return str(tuple(args))
def create_function_signature(task: Task) -> str:
    
    if task.args:
        
        normalized_args = []
        for arg in task.args:
            if isinstance(arg, str):
                
                normalized_arg = remove_quotes(arg)
                normalized_args.append(normalized_arg)
            else:
                normalized_args.append(arg)
        args_str = str(normalized_args)
    else:
        args_str = ""
    
    signature = f"{task.name}|{args_str}"
    return signature

def remove_quotes(arg: str) -> str:
    
    arg = arg.strip()
    
    
    if arg.startswith('"') and arg.endswith('"'):
        arg = arg[1:-1]
    
    elif arg.startswith("'") and arg.endswith("'"):
        arg = arg[1:-1]
    
    return arg.strip()

@dataclass
class Task:
    idx: int
    name: str
    tool: Callable
    args: Collection[Any]
    dependencies: Collection[int]
    stringify_rule: Optional[Callable] = None
    thought: Optional[str] = None
    observation: Optional[str] = None
    is_join: bool = False
    is_branch: bool = False
    is_replan: bool = False

    async def __call__(self) -> Any:
        log("running task")
        x = await self.tool(*self.args)
        log("done task")
        return x

    def get_though_action_observation(
        self, include_action=True, include_thought=True, include_action_idx=False
    ) -> str:
        thought_action_observation = ""
        if self.thought and include_thought:
            thought_action_observation = f"Thought: {self.thought}\n"
        if include_action:
            idx = f"{self.idx}. " if include_action_idx else ""
            if self.stringify_rule:
                # If the user has specified a custom stringify rule for the
                # function argument, use it
                thought_action_observation += f"{idx}{self.stringify_rule(self.args)}\n"
            else:
                # Otherwise, we have a default stringify rule
                thought_action_observation += (
                    f"{idx}{self.name}"
                    f"{_default_stringify_rule_for_arguments(self.args)}\n"
                )
        if self.observation is not None:
            thought_action_observation += f"Observation: {self.observation}\n"
        return thought_action_observation
    
class TaskFetchingUnit:
    
    
    def __init__(self):
        self.tasks: Dict[int, Any] = {}
        self.stream_executed_tasks: Set[str] = set()
        self.stream_executed_results: Dict[str, Any] = {}
        self.stream_task_map: Dict[int, Any] = {}
        self.stream_task_events: Dict[int, asyncio.Event] = {}
        self.cache_manager: Optional[Any] = None
        self.current_question: Optional[str] = None
        self.execution_metadata: Dict[str, Any] = {}
        self._inactive_task_ids: Set[int] = set()
        self.runtime_semantic_batch_enabled: bool = True
        self.runtime_semantic_batch_window_ms: int = 300
        self._semantic_batch_lock = asyncio.Lock()
        self._semantic_batch_waiting: Dict[int, Dict[str, Any]] = {}
    
    def set_tasks(self, tasks: Dict[int, Any]):
        
        self.tasks = tasks
        self._inactive_task_ids = set()

    def set_execution_context(self, question: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.current_question = question
        self.execution_metadata = metadata or {}
        if "runtime_semantic_batch" in self.execution_metadata:
            self.runtime_semantic_batch_enabled = bool(self.execution_metadata.get("runtime_semantic_batch"))
        if "runtime_semantic_batch_window_ms" in self.execution_metadata:
            try:
                self.runtime_semantic_batch_window_ms = max(
                    0, int(self.execution_metadata.get("runtime_semantic_batch_window_ms", 300))
                )
            except Exception:
                self.runtime_semantic_batch_window_ms = 300
    
    def set_stream_mode_data(self, executed_tasks: Set[str], executed_results: Dict[str, Any], 
                           task_map: Dict[int, Any], task_events: Dict[int, asyncio.Event],
                           cache_manager: Optional[Any] = None):  
        
        self.stream_executed_tasks = executed_tasks
        self.stream_executed_results = executed_results
        self.stream_task_map = task_map
        self.stream_task_events = task_events
        self.cache_manager = cache_manager  
    
    async def schedule(self):
        
        if not self.tasks:
            return
        
        print(f"    Executing {len(self.tasks)} tasks with dependency checking")
        
        
        task_queue = asyncio.Queue()
        for task_id, task in self.tasks.items():
            if hasattr(task, 'tool') and callable(task.tool):
                await task_queue.put((task_id, task))
        
        workers = [asyncio.create_task(self._worker(worker_id, task_queue)) 
                  for worker_id in range(min(len(self.tasks), 5))]
        
        await task_queue.join()
        
        for worker in workers:
            worker.cancel()
    
    async def _worker(self, worker_id: int, task_queue: asyncio.Queue):
        
        while True:
            try:
                task_id, task = await task_queue.get()
                if task_id in self._inactive_task_ids:
                    task_queue.task_done()
                    continue
                
                
                if await self._check_dependencies_ready(task):
                    if self._should_runtime_batch(task):
                        await self._execute_semantic_runtime_batch(task_id, task)
                    else:
                        await self._execute_task(task_id, task)
                    task_queue.task_done()
                else:
                    
                    await asyncio.sleep(0.1)
                    await task_queue.put((task_id, task))
                    task_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                task_queue.task_done()

    def _should_runtime_batch(self, task: Any) -> bool:
        if not self.runtime_semantic_batch_enabled:
            return False
        if getattr(task, "name", None) != "semantic_map":
            return False
        if getattr(task, "is_join", False) or getattr(task, "is_branch", False) or getattr(task, "is_replan", False):
            return False
        return True

    def _parse_semantic_runtime_args(self, runtime_args: Any) -> tuple[str, List[Any], str]:
        values = list(runtime_args) if isinstance(runtime_args, (list, tuple)) else [runtime_args]
        if len(values) >= 5:
            instruction = str(values[1]).strip()
            inputs = values[3]
            output_schema = str(values[4]).strip() or "string"
        elif len(values) >= 3:
            instruction = str(values[0]).strip()
            inputs = values[1]
            output_schema = str(values[2]).strip() or "string"
        else:
            raise ValueError("semantic_map requires at least 3 arguments")
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        return instruction, list(inputs), output_schema

    def _is_batch_compatible_semantic(self, task_a: Any, task_b: Any) -> bool:
        if getattr(task_b, "name", None) != "semantic_map":
            return False
        if getattr(task_b, "is_join", False) or getattr(task_b, "is_branch", False) or getattr(task_b, "is_replan", False):
            return False
        if getattr(task_a, "tool", None) != getattr(task_b, "tool", None):
            return False
        if self._contains_indexed_ref(getattr(task_a, "args", None)):
            return False
        if self._contains_indexed_ref(getattr(task_b, "args", None)):
            return False
        return True

    def _contains_indexed_ref(self, value: Any) -> bool:
        if isinstance(value, str):
            return bool(re.search(r"\$\{?\d+\}?\[\d+\]", value))
        if isinstance(value, (list, tuple)):
            return any(self._contains_indexed_ref(item) for item in value)
        if isinstance(value, dict):
            return any(self._contains_indexed_ref(item) for item in value.values())
        return False

    async def _execute_semantic_runtime_batch(self, task_id: int, task: Any):
        loop = asyncio.get_event_loop()
        my_future = loop.create_future()

        async with self._semantic_batch_lock:
            self._semantic_batch_waiting[task_id] = {"task": task, "future": my_future}

        if self.runtime_semantic_batch_window_ms > 0:
            await asyncio.sleep(self.runtime_semantic_batch_window_ms / 1000.0)

        group: List[tuple[int, Any, asyncio.Future]] = []
        async with self._semantic_batch_lock:
            if task_id not in self._semantic_batch_waiting:
                pass
            else:
                anchor = self._semantic_batch_waiting.get(task_id)
                if anchor:
                    anchor_task = anchor["task"]
                    for candidate_id in list(self._semantic_batch_waiting.keys()):
                        entry = self._semantic_batch_waiting.get(candidate_id)
                        if not entry:
                            continue
                        candidate_task = entry["task"]
                        if candidate_id in self._inactive_task_ids:
                            self._semantic_batch_waiting.pop(candidate_id, None)
                            if not entry["future"].done():
                                entry["future"].set_result(None)
                            continue
                        if not self._is_batch_compatible_semantic(anchor_task, candidate_task):
                            continue
                        if not await self._check_dependencies_ready(candidate_task):
                            continue
                        self._semantic_batch_waiting.pop(candidate_id, None)
                        group.append((candidate_id, candidate_task, entry["future"]))

        if group:
            try:
                if len(group) == 1:
                    single_id, single_task, _ = group[0]
                    await self._execute_task(single_id, single_task)
                else:
                    await self._execute_semantic_batch(group)
                for _, _, fut in group:
                    if not fut.done():
                        fut.set_result(None)
            except Exception as e:
                for _, _, fut in group:
                    if not fut.done():
                        fut.set_exception(e)
                raise

        if not my_future.done():
            my_future.set_result(None)
        await my_future

    async def _execute_semantic_batch(self, group: List[tuple[int, Any, asyncio.Future]]):
        batch_start = time.time()
        prepared: List[Dict[str, Any]] = []
        batch_items: List[Dict[str, Any]] = []
        fused_inputs: List[Any] = []
        shared_deps: List[int] = []

        for task_id, task, _ in group:
            if hasattr(task, "dependencies") and task.dependencies:
                task.args = self._replace_dependencies_in_args(task.args, task.dependencies, task.name)
            runtime_args = self._augment_semantic_map_args(task, task.args)
            instruction, inputs, output_schema = self._parse_semantic_runtime_args(runtime_args)
            input_value = inputs[0] if inputs else ""
            batch_items.append(
                {
                    "instruction": instruction,
                    "output_schema": output_schema,
                    "depends_on_result": None,
                }
            )
            fused_inputs.append(input_value)
            shared_deps.extend(list(getattr(task, "dependencies", []) or []))
            prepared.append(
                {
                    "task_id": task_id,
                    "task": task,
                    "output_schema": output_schema,
                }
            )

        batch_spec = {"mode": "batch", "output_format": "result_lines", "items": batch_items}
        fused_args = [json.dumps(batch_spec, ensure_ascii=False), fused_inputs, "batch_results"]

        first_task = prepared[0]["task"]
        try:
            if asyncio.iscoroutinefunction(first_task.tool):
                payload = await first_task.tool(*fused_args)
            else:
                loop = asyncio.get_event_loop()
                payload = await loop.run_in_executor(None, first_task.tool, *fused_args)
            payload_text = str(payload or "").strip()
            parsed_lines = self._parse_result_lines(payload_text)
            if len(parsed_lines) < len(prepared):
                raise ValueError(
                    f"runtime semantic batch parsed {len(parsed_lines)} results, expected {len(prepared)}"
                )
            for idx, item in enumerate(prepared):
                task = item["task"]
                result_text = parsed_lines[idx]
                task.observation = result_text
                if self.cache_manager:
                    self.cache_manager.set_cached_result(task, result_text)
                print(f"       Task {item['task_id']} completed in batch ({time.time() - batch_start:.3f}s)")
            print(
                f"    Runtime semantic_map batch executed: size={len(prepared)}, window={self.runtime_semantic_batch_window_ms}ms"
            )
        except Exception as e:
            print(f"    Runtime semantic_map batch fallback due to error: {e}")
            for item in prepared:
                await self._execute_task(item["task_id"], item["task"])
    
    async def _check_dependencies_ready(self, task: Any) -> bool:
        
        if not hasattr(task, 'dependencies') or not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if not await self._is_dependency_ready(dep_id):
                return False
        
        return True
    
    async def _is_dependency_ready(self, dep_id: int) -> bool:
        
        
        if dep_id in self.tasks:
            dep_task = self.tasks[dep_id]
            if hasattr(dep_task, 'observation') and dep_task.observation is not None:
                return True
            return False
        
        
        if dep_id in self.stream_task_map:
            stream_task = self.stream_task_map[dep_id]
            task_key = self._get_task_key(stream_task)
            
            
            if self.cache_manager:
                cached_result = self.cache_manager.get_cached_result(stream_task)
                if cached_result is not None:
                    return True
            
            
            if task_key in self.stream_executed_tasks and task_key not in self.stream_executed_results:
                if dep_id in self.stream_task_events:
                    try:
                        await asyncio.wait_for(self.stream_task_events[dep_id].wait(), timeout=1.0)
                        return task_key in self.stream_executed_results
                    except asyncio.TimeoutError:
                        return False
                return False
            
            
            if task_key in self.stream_executed_results:
                return True
            
            return False
        
        return False
    
    async def _execute_task(self, task_id: int, task: Any):
        
        try:
            start_time = time.time()
            if task_id in self._inactive_task_ids:
                if getattr(task, "observation", None) is None:
                    task.observation = "skipped by branch"
                return

            task_key = self._get_task_key(task)
            
            
            if self.cache_manager:
                cached_result = self.cache_manager.get_cached_result(task)
                if cached_result is not None:
                    task.observation = cached_result
                    print(f"    Using cached result for task {task_id}")
                    return
            
            
            if task_key in self.stream_executed_tasks and task_key in self.stream_executed_results:
                task.observation = self.stream_executed_results[task_key]
                print(f"    Reused stream result for task {task_id}")
                return
            
            
            if hasattr(task, 'dependencies') and task.dependencies and getattr(task, "name", None) != "branch":
                task.args = self._replace_dependencies_in_args(task.args, task.dependencies, task.name)

            runtime_args = task.args
            if getattr(task, "name", None) == "semantic_map":
                runtime_args = self._augment_semantic_map_args(task, runtime_args)
            elif getattr(task, "name", None) == "deepsearch":
                runtime_args = self._augment_deepsearch_args(task, runtime_args)
            
            
            print(f"    Executing task {task_id}: {task.name}")

            if task.name == "branch":
                decision, chosen_ids, skipped_ids = self._evaluate_branch_task(runtime_args)
                for skipped_id in skipped_ids:
                    skipped_task = self.tasks.get(skipped_id)
                    if skipped_task is None:
                        continue
                    if getattr(skipped_task, "observation", None) is None:
                        skipped_task.observation = "skipped by branch"
                    self._inactive_task_ids.add(skipped_id)
                task.observation = json.dumps(
                    {
                        "decision": bool(decision),
                        "chosen": "then" if decision else "else",
                        "activated": chosen_ids,
                        "skipped": skipped_ids,
                    },
                    ensure_ascii=False,
                )
                elapsed = time.time() - start_time
                print(f"       Task {task_id} completed in {elapsed:.3f}s")
                return

            if task.name == "replan":
                payload = runtime_args[0] if isinstance(runtime_args, (list, tuple)) and runtime_args else {}
                if isinstance(payload, dict):
                    reason = str(payload.get("reason", "Evidence is insufficient.")).strip() or "Evidence is insufficient."
                    scope = str(payload.get("scope", "local")).strip().lower() or "local"
                    task.observation = f"replan requested ({scope}): {reason}"
                else:
                    task.observation = f"replan requested: {str(payload)}"
                elapsed = time.time() - start_time
                print(f"       Task {task_id} completed in {elapsed:.3f}s")
                return
            
            
            if asyncio.iscoroutinefunction(task.tool):
                result = await task.tool(*runtime_args)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, task.tool, *runtime_args)
            
            if self.cache_manager:
                self.cache_manager.set_cached_result(task, result)
            
            task.observation = result
            elapsed = time.time() - start_time
            print(f"       Task {task_id} completed in {elapsed:.3f}s")
            
        except Exception as e:
            error_msg = f"Task execution error: {e}"
            
            task.observation = error_msg
            print(f"       Task {task_id} failed: {e}")

    def _evaluate_branch_task(self, runtime_args: Any) -> tuple[bool, List[int], List[int]]:
        predicate = ""
        then_ids: List[int] = []
        else_ids: List[int] = []

        if isinstance(runtime_args, (list, tuple)) and len(runtime_args) >= 3:
            predicate = runtime_args[0]
            then_ids = list(runtime_args[1] or [])
            else_ids = list(runtime_args[2] or [])

        decision = self._evaluate_predicate(predicate)
        return decision, (then_ids if decision else else_ids), (else_ids if decision else then_ids)

    def _json_has_empty_content(self, value: Any) -> bool:
        if isinstance(value, str):
            return value == ""
        if isinstance(value, list):
            if len(value) == 0:
                return True
            return any(self._json_has_empty_content(item) for item in value)
        if isinstance(value, dict):
            return any(self._json_has_empty_content(item) for item in value.values())
        return False

    def _contains_empty_json_slots(self, value: Any) -> bool:
        if isinstance(value, (dict, list)):
            return self._json_has_empty_content(value)
        if not isinstance(value, str):
            return False
        text = value.strip()
        if not text or text[0] not in "{[":
            return False
        parsed: Any = None
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return False
        return self._json_has_empty_content(parsed)

    def _evaluate_predicate(self, predicate: Any) -> bool:
        if isinstance(predicate, dict):
            op = str(predicate.get("op", "")).strip().lower()
            if op == "and":
                return all(self._evaluate_predicate(item) for item in (predicate.get("args") or []))
            if op == "or":
                return any(self._evaluate_predicate(item) for item in (predicate.get("args") or []))
            if op == "llm_judge":
                inputs = predicate.get("inputs") or []
                resolved = [self._resolve_observation_reference(item) for item in inputs]
                resolved_text = " ".join(str(item or "") for item in resolved).strip().lower()
                if not resolved_text:
                    return False
                if any(self._contains_empty_json_slots(item) for item in resolved):
                    return False
                negative_markers = [
                    "task result missing",
                    "skipped by branch",
                    "execution error",
                    "tool error",
                    "no result",
                    "no results",
                ]
                if any(marker in resolved_text for marker in negative_markers):
                    return False
                return True
            return False

        text = str(predicate or "").strip()
        if not text:
            return False

        if "|" in text:
            return any(self._evaluate_predicate(part.strip()) for part in text.split("|"))
        if "&" in text:
            return all(self._evaluate_predicate(part.strip()) for part in text.split("&"))

        lowered = text.lower()

        def _extract_single_ref(pattern: str) -> Optional[int]:
            match = re.match(pattern, lowered)
            if not match:
                return None
            try:
                return int(match.group(1))
            except Exception:
                return None

        dep_id = _extract_single_ref(r"^is_nonempty\(\$([0-9]+)\)$")
        if dep_id is not None:
            value = self._get_observation_by_id(dep_id)
            return bool(str(value or "").strip())

        dep_id = _extract_single_ref(r"^is_empty\(\$([0-9]+)\)$")
        if dep_id is not None:
            value = self._get_observation_by_id(dep_id)
            return not bool(str(value or "").strip())

        dep_id = _extract_single_ref(r"^contains_url\(\$([0-9]+)\)$")
        if dep_id is not None:
            value = str(self._get_observation_by_id(dep_id) or "")
            return bool(re.search(r"https?://", value))

        dep_id = _extract_single_ref(r"^contains_number\(\$([0-9]+)\)$")
        if dep_id is not None:
            value = str(self._get_observation_by_id(dep_id) or "")
            return bool(re.search(r"\b\d+(?:\.\d+)?\b", value))

        dep_id = _extract_single_ref(r"^tool_error\(\$([0-9]+)\)$")
        if dep_id is not None:
            value = str(self._get_observation_by_id(dep_id) or "").lower()
            return any(marker in value for marker in ["error", "failed", "exception", "forbidden", "404", "429", "timeout"])

        match = re.match(r"^contains_str\(\$([0-9]+),\s*(['\"])(.*)\2\)$", text, flags=re.I)
        if match:
            dep_id = int(match.group(1))
            needle = match.group(3).lower()
            value = str(self._get_observation_by_id(dep_id) or "").lower()
            return needle in value

        match = re.match(r"^matches_regex\(\$([0-9]+),\s*(['\"])(.*)\2\)$", text, flags=re.I)
        if match:
            dep_id = int(match.group(1))
            pattern = match.group(3)
            value = str(self._get_observation_by_id(dep_id) or "")
            try:
                return bool(re.search(pattern, value))
            except Exception:
                return False

        match = re.match(r"^list_length_ge\(\$([0-9]+),\s*([0-9]+)\)$", lowered)
        if match:
            dep_id = int(match.group(1))
            threshold = int(match.group(2))
            value = self._get_observation_by_id(dep_id)
            if isinstance(value, list):
                return len(value) >= threshold
            parsed = self._try_parse_json(value)
            if isinstance(parsed, list):
                return len(parsed) >= threshold
            return False

        match = re.match(r"^contains_json_field\(\$([0-9]+),\s*(['\"])(.*)\2\)$", text, flags=re.I)
        if match:
            dep_id = int(match.group(1))
            field = match.group(3)
            value = self._get_observation_by_id(dep_id)
            parsed = self._try_parse_json(value)
            if isinstance(parsed, dict):
                return field in parsed
            return False

        return bool(text)

    def _resolve_observation_reference(self, ref: Any) -> Any:
        if isinstance(ref, str):
            match = re.fullmatch(r"\$\{?([0-9]+)\}?", ref.strip())
            if match:
                return self._get_observation_by_id(int(match.group(1)))
        return ref

    def _get_observation_by_id(self, dep_id: int) -> Any:
        if dep_id in self.tasks:
            return getattr(self.tasks[dep_id], "observation", None)
        if dep_id in self.stream_task_map:
            stream_task = self.stream_task_map[dep_id]
            task_key = self._get_task_key(stream_task)
            return self.stream_executed_results.get(task_key)
        return None

    def _try_parse_json(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    
    def _replace_dependencies_in_args(self, args, dependencies: List[int], task_name: Optional[str] = None):
        
        if isinstance(args, (list, tuple)):
            return type(args)(
                self._replace_dependencies_in_args(item, dependencies, task_name)
                for item in args
            )
        elif isinstance(args, str):
            for dependency in sorted(dependencies, reverse=True):
                markers = [f"${{{dependency}}}", f"${dependency}"]
                if not any(marker in args for marker in markers):
                    continue

                if self.cache_manager:
                    dep_task = None
                    if dependency in self.tasks:
                        dep_task = self.tasks[dependency]
                    elif dependency in self.stream_task_map:
                        dep_task = self.stream_task_map[dependency]

                    if dep_task:
                        cached_result = self.cache_manager.get_cached_result(dep_task)
                        if cached_result is not None:
                            args = self._replace_dependency_references(args, dependency, cached_result, task_name)
                            continue

                if dependency in self.tasks:
                    dep_task = self.tasks[dependency]
                    if getattr(dep_task, 'observation', None) is not None:
                        args = self._replace_dependency_references(args, dependency, dep_task.observation, task_name)
                        continue

                elif dependency in self.stream_task_map:
                    stream_task = self.stream_task_map[dependency]
                    task_key = self._get_task_key(stream_task)
                    if task_key in self.stream_executed_results:
                        result = self.stream_executed_results[task_key]
                        args = self._replace_dependency_references(args, dependency, result, task_name)
                        continue
            return args
        else:
            return args

    def _replace_dependency_references(self, text: str, dep_id: int, value: Any, task_name: Optional[str]) -> str:
        pattern = re.compile(rf"\$\{{?{dep_id}\}}?(?:\[(\d+)\])?")
        dep_schema = self._infer_dependency_schema(dep_id)

        if task_name == "python":
            formatted_value = self._format_dependency_value(f"${dep_id}", value, task_name, dep_schema)

            if isinstance(value, (dict, list, bool)) or value is None:
                quoted_payload = json.dumps(value, ensure_ascii=False)
            else:
                quoted_payload = str(value)
            encoded = base64.b64encode(quoted_payload.encode("utf-8")).decode("ascii")
            quoted_formatted_value = (
                f"__import__('base64').b64decode('{encoded}').decode('utf-8', 'ignore')"
            )

            triple_quoted_markers = (
                f"r'''${{{dep_id}}}'''",
                f'r"""${{{dep_id}}}"""',
                f"R'''${{{dep_id}}}'''",
                f'R"""${{{dep_id}}}"""',
                f"'''${{{dep_id}}}'''",
                f'"""${{{dep_id}}}"""',
                f"r'''${dep_id}'''",
                f'r"""${dep_id}"""',
                f"R'''${dep_id}'''",
                f'R"""${dep_id}"""',
                f"'''${dep_id}'''",
                f'"""${dep_id}"""',
            )
            for quoted_marker in triple_quoted_markers:
                if quoted_marker in text:
                    text = text.replace(quoted_marker, quoted_formatted_value)
            quoted_markers = (
                f"r'${{{dep_id}}}'",
                f'r"${{{dep_id}}}"',
                f"R'${{{dep_id}}}'",
                f'R"${{{dep_id}}}"',
                f"'${{{dep_id}}}'",
                f'"${{{dep_id}}}"',
                f"r'${dep_id}'",
                f'r"${dep_id}"',
                f"R'${dep_id}'",
                f'R"${dep_id}"',
                f"'${dep_id}'",
                f'"${dep_id}"',
            )
            for quoted_marker in quoted_markers:
                if quoted_marker in text:
                    text = text.replace(quoted_marker, quoted_formatted_value)

        def repl(match):
            index_text = match.group(1)
            selected_value = value
            if index_text is not None:
                selected_value = self._extract_indexed_value(value, int(index_text))
            selected_schema = dep_schema
            if index_text is not None and dep_schema and dep_schema.lower().startswith("list[") and dep_schema.endswith("]"):
                selected_schema = dep_schema[5:-1].strip()
            return self._format_dependency_value(match.group(0), selected_value, task_name, selected_schema)

        return pattern.sub(repl, text)

    def _infer_dependency_schema(self, dep_id: int) -> Optional[str]:
        dep_task = None
        if dep_id in self.tasks:
            dep_task = self.tasks[dep_id]
        elif dep_id in self.stream_task_map:
            dep_task = self.stream_task_map[dep_id]
        if dep_task is None:
            return None

        if getattr(dep_task, "name", None) != "semantic_map":
            return None

        args = getattr(dep_task, "args", None)
        if isinstance(args, (list, tuple)) and len(args) >= 3:
            schema = args[2]
            if isinstance(schema, str):
                return schema
        raw_args = getattr(dep_task, "raw_args", None)
        if isinstance(raw_args, (list, tuple)) and len(raw_args) >= 3:
            schema = raw_args[2]
            if isinstance(schema, str):
                return schema
        return None

    def _extract_indexed_value(self, value: Any, index: int) -> Any:
        if isinstance(value, (list, tuple)):
            return value[index] if 0 <= index < len(value) else ""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return parsed[index] if 0 <= index < len(parsed) else ""
                except Exception:
                    pass
            result_lines = self._parse_result_lines(text)
            if result_lines:
                return result_lines[index] if 0 <= index < len(result_lines) else ""
        return ""

    def _parse_result_lines(self, text: str) -> List[str]:
        results: Dict[int, str] = {}
        current_idx: Optional[int] = None
        current_lines: List[str] = []

        def flush_current():
            if current_idx is not None:
                results[current_idx] = "\n".join(current_lines).strip()

        for line in text.splitlines():
            match = re.match(r"^\s*Results?\s*(\d+)\s*:\s*(.*)$", line, flags=re.I)
            if match:
                flush_current()
                current_idx = int(match.group(1))
                current_lines = [match.group(2).strip()]
                continue
            if current_idx is not None:
                current_lines.append(line)
        flush_current()

        if not results:
            return []
        return [results[idx] for idx in sorted(results)]

    def _format_dependency_value(
        self,
        marker: str,
        value: Any,
        task_name: Optional[str],
        source_schema: Optional[str] = None,
    ) -> str:
        def _python_json_expr(payload: Any) -> str:
            json_payload = json.dumps(payload, ensure_ascii=False)
            encoded = base64.b64encode(json_payload.encode("utf-8")).decode("ascii")
            return (
                "__import__('json').loads("
                f"__import__('base64').b64decode('{encoded}').decode('utf-8', 'ignore')"
                ")"
            )

        if task_name == "python":
            if isinstance(value, (list, dict, bool)) or value is None:
                return _python_json_expr(value)

            if isinstance(value, (int, float)):
                return repr(value)

            if isinstance(value, str):
                text = value.strip()
                lowered_schema = (source_schema or "").strip().lower()

                if lowered_schema == "string":
                    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
                    python_expr = f"__import__('base64').b64decode('{encoded}').decode('utf-8', 'ignore')"
                    return python_expr

                # Preserve scalar semantic outputs as native Python literals.
                if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
                    try:
                        parsed = json.loads(text)
                        return repr(parsed)
                    except Exception:
                        pass

                if text in {"true", "false", "null"}:
                    try:
                        parsed = json.loads(text)
                        return repr(parsed)
                    except Exception:
                        pass

                # Preserve structured semantic_map outputs as Python literals.
                if (text.startswith("[") and text.endswith("]")) or (
                    text.startswith("{") and text.endswith("}")
                ):
                    try:
                        parsed = json.loads(text)
                        return _python_json_expr(parsed)
                    except Exception:
                        try:
                            parsed = ast.literal_eval(text)
                            return _python_json_expr(parsed)
                        except Exception:
                            pass

                encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
                python_expr = f"__import__('base64').b64decode('{encoded}').decode('utf-8', 'ignore')"
                return python_expr

            return repr(value)
        value_text = str(value)
        return value_text

    def _replace_dependency_placeholder(self, args: str, arg_mask: str, value: Any, task_name: Optional[str]) -> str:
        value_text = str(value)

        if task_name == "python":
            encoded = base64.b64encode(value_text.encode("utf-8")).decode("ascii")
            python_expr = f"__import__('base64').b64decode('{encoded}').decode('utf-8', 'ignore')"
            for quoted_mask in (f"'{arg_mask}'", f'"{arg_mask}"'):
                if quoted_mask in args:
                    return args.replace(quoted_mask, python_expr)
            return args.replace(arg_mask, python_expr)

        return args.replace(arg_mask, value_text)

    def _augment_semantic_map_args(self, task: Any, runtime_args: Any):
        if not isinstance(runtime_args, (list, tuple)):
            return runtime_args
        if len(runtime_args) < 3 or len(runtime_args) >= 6:
            return runtime_args

        direct_dependency_ids = list(getattr(task, "dependencies", []) or [])
        direct_dependency_set = set(direct_dependency_ids)
        provenance = []
        direct_observation_refs = {}
        for dep_id in direct_dependency_ids:
            dep_task = None
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
            elif dep_id in self.stream_task_map:
                dep_task = self.stream_task_map[dep_id]
            if dep_task is None:
                continue

            action_repr = dep_task.get_though_action_observation(
                include_thought=False,
                include_action=True,
                include_action_idx=False,
            ).splitlines()[0]
            provenance.append(
                {
                    "task_id": dep_id,
                    "tool": getattr(dep_task, "name", "unknown"),
                    "action": action_repr,
                    "observation": getattr(dep_task, "observation", None),
                }
            )
            direct_observation_refs[dep_id] = len(provenance) - 1

        history_observations = []
        current_plan = []
        for task_id, plan_task in sorted((self.tasks or {}).items(), key=lambda item: item[0]):
            action_repr = plan_task.get_though_action_observation(
                include_thought=False,
                include_action=True,
                include_action_idx=False,
            ).splitlines()[0]
            current_plan.append(
                {
                    "task_id": task_id,
                    "tool": getattr(plan_task, "name", "unknown"),
                    "action": action_repr,
                }
            )
            observation = getattr(plan_task, "observation", None)
            if observation is not None:
                item = {
                    "task_id": task_id,
                    "tool": getattr(plan_task, "name", "unknown"),
                    "action": action_repr,
                    "observation": observation,
                }
                if task_id in direct_dependency_set:
                    item["observation_ref"] = f"inputs_with_provenance[{direct_observation_refs.get(task_id, 0)}].observation"
                    item["observation"] = "[see direct input observation]"
                history_observations.append(item)

        is_new_style = len(runtime_args) >= 5
        direct_inputs = runtime_args[3] if is_new_style else runtime_args[1]
        if direct_inputs in (None, "", [], {}):
            rescued_inputs = []
            for item in reversed(history_observations):
                tool_name = str(item.get("tool", "") or "").lower()
                observation_text = str(item.get("observation", "") or "").strip()
                lowered = observation_text.lower()
                if not observation_text:
                    continue
                if tool_name in {"branch", "replan", "join"}:
                    continue
                if any(
                    marker in lowered
                    for marker in [
                        "skipped by branch",
                        "task result missing",
                        "no answer generated",
                        "semantic_map expects",
                        "execution error",
                        "tool error",
                    ]
                ):
                    continue
                rescued_inputs.append(observation_text)
                if len(rescued_inputs) >= 2:
                    break
            if rescued_inputs:
                direct_inputs = list(reversed(rescued_inputs))

        local_question = ""
        try:
            local_question = str(runtime_args[1] if is_new_style else runtime_args[0]).strip()
        except Exception:
            local_question = ""

        plan_context = ""
        try:
            if is_new_style:
                plan_context = str(runtime_args[2]).strip()
        except Exception:
            plan_context = ""
        if not plan_context:
            plan_context = local_question

        context = {
            "global_question": self.current_question or "",
            "local_question": local_question,
            "plan_context": plan_context,
            "history_observations": history_observations,
            "current_plan": current_plan,
            "inputs_with_provenance": provenance,
        }
        if self.execution_metadata:
            context.update(self.execution_metadata)
        context_blob = json.dumps(context, ensure_ascii=False)

        if is_new_style:
            augmented_args = list(runtime_args)
            augmented_args[3] = direct_inputs
            if len(augmented_args) == 5:
                augmented_args.append(context_blob)
            else:
                augmented_args[-1] = context_blob
            return augmented_args

        instruction = str(runtime_args[0]).strip()
        output_schema = str(runtime_args[2]).strip() if len(runtime_args) > 2 else "string"
        return [
            self.current_question or "",
            instruction,
            plan_context,
            direct_inputs,
            output_schema,
            context_blob,
        ]

    def _augment_deepsearch_args(self, task: Any, runtime_args: Any):
        if not isinstance(runtime_args, (list, tuple)):
            return runtime_args
        if len(runtime_args) != 1:
            return runtime_args

        context = {
            "global_question": self.current_question or "",
        }
        if self.execution_metadata:
            context.update(self.execution_metadata)
        return list(runtime_args) + [json.dumps(context, ensure_ascii=False)]
    
    def _get_task_key(self, task: Any) -> str:
        task_key = create_function_signature(task)
        question_namespace = self.execution_metadata.get("question_id") or self.current_question
        if question_namespace:
            return f"{question_namespace}::{task_key}"
        return task_key
