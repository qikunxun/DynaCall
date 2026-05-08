from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

from src.dynacall.task import Task


@dataclass
class FusionOpportunity:
    question: str
    original_plan: Dict[int, Task]
    optimized_plan: Dict[int, Task]
    savings: int
    fused_edges: List[Tuple[int, int, str]]


class PhysicalFusionOptimizer:
    """Physical-plan optimizer that fuses safe producer-consumer pairs."""

    PRODUCER_KIND = {
        "search_engine": "retrieve",
        "arxiv_search": "retrieve",
        "web_browser": "browse",
        "file_reader": "read",
        "pdf_viewer": "read",
        "spreadsheet_reader": "read",
        "powerpoint_viewer": "read",
        "text_reader": "read",
        "archive_explorer": "read",
        "ocr": "read",
        "extract_text_from_image": "read",
        "speech_to_text": "read",
        "analyze_image": "read",
        "image_recognition": "read",
    }

    CONSUMER_KIND = {
        "web_browser": "browse",
        "python": "compute",
        "calculator": "compute",
        "execute_code_multilang": "compute",
    }

    ALLOWED_PATTERNS = {
        ("retrieve", "browse"): "retrieve_then_browse",
        ("browse", "compute"): "browse_then_compute",
        ("read", "compute"): "read_then_compute",
    }

    def __init__(self, max_fusions_per_plan: int = 8):
        self.max_fusions_per_plan = max_fusions_per_plan

    async def optimize_tool_chains(self, batch_plans: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        print("🔧 Applying physical operator fusion...")

        optimized_plans: Dict[str, Dict[int, Task]] = {}
        opportunities: List[FusionOpportunity] = []

        for qid, plan_data in batch_plans.items():
            plan = plan_data["plan"]
            question = plan_data["question"]

            optimized_plan, fused_edges = self._optimize_plan(plan)
            optimized_plans[qid] = optimized_plan

            if len(optimized_plan) < len(plan):
                opportunities.append(
                    FusionOpportunity(
                        question=question,
                        original_plan=plan,
                        optimized_plan=optimized_plan,
                        savings=len(plan) - len(optimized_plan),
                        fused_edges=fused_edges,
                    )
                )
                print(
                    f"✅ Physical fusion for {qid}: {len(plan)} -> {len(optimized_plan)} tasks; "
                    f"fused {len(fused_edges)} edge(s)"
                )
            else:
                print(f"ℹ️  No physical fusion opportunities for {qid}")

        return {
            "optimized_plans": optimized_plans,
            "fusion_opportunities": opportunities,
            "stats": self._calculate_stats(opportunities, batch_plans, optimized_plans),
        }

    def _optimize_plan(self, original_plan: Dict[int, Task]) -> Tuple[Dict[int, Task], List[Tuple[int, int, str]]]:
        plan = dict(original_plan)
        fused_edges: List[Tuple[int, int, str]] = []

        for _ in range(self.max_fusions_per_plan):
            dependents = self._build_dependents(plan)
            fused = False

            for parent_id in sorted(plan.keys()):
                if parent_id not in plan:
                    continue
                parent = plan[parent_id]
                if getattr(parent, "is_join", False) or not callable(getattr(parent, "tool", None)):
                    continue

                child_ids = dependents.get(parent_id, [])
                if len(child_ids) != 1:
                    continue

                child_id = child_ids[0]
                if child_id not in plan:
                    continue
                child = plan[child_id]
                if getattr(child, "is_join", False) or not callable(getattr(child, "tool", None)):
                    continue

                pattern = self._infer_pattern(parent, child)
                if pattern is None:
                    continue
                if not self._is_safe_to_fuse(plan, dependents, parent, child):
                    continue

                print(f"🔗 Fusing edge {parent.idx}({parent.name}) -> {child.idx}({child.name}) as {pattern}")
                plan[child_id] = self._build_fused_task(parent, child, pattern)
                del plan[parent_id]
                fused_edges.append((parent.idx, child.idx, pattern))
                fused = True
                break

            if not fused:
                break

        return plan, fused_edges

    def _build_dependents(self, plan: Dict[int, Task]) -> Dict[int, List[int]]:
        dependents: Dict[int, List[int]] = {task_id: [] for task_id in plan}
        for task_id, task in plan.items():
            for dep in getattr(task, "dependencies", []) or []:
                if dep in dependents:
                    dependents[dep].append(task_id)
        return dependents

    def _infer_pattern(self, parent: Task, child: Task) -> Optional[str]:
        parent_kind = self.PRODUCER_KIND.get(parent.name)
        child_kind = self.CONSUMER_KIND.get(child.name)
        if not parent_kind or not child_kind:
            return None
        return self.ALLOWED_PATTERNS.get((parent_kind, child_kind))

    def _is_safe_to_fuse(
        self,
        plan: Dict[int, Task],
        dependents: Dict[int, List[int]],
        parent: Task,
        child: Task,
    ) -> bool:
        if parent.idx not in (getattr(child, "dependencies", []) or []):
            return False
        if len(dependents.get(parent.idx, [])) != 1:
            return False
        if not self._args_reference_dependency(child.args, parent.idx):
            return False
        if getattr(parent, "is_join", False) or getattr(child, "is_join", False):
            return False
        return True

    def _args_reference_dependency(self, value: Any, dep_id: int) -> bool:
        markers = {f"${dep_id}", f"${{{dep_id}}}"}
        if isinstance(value, str):
            return any(marker in value for marker in markers)
        if isinstance(value, (list, tuple)):
            return any(self._args_reference_dependency(item, dep_id) for item in value)
        if isinstance(value, dict):
            return any(self._args_reference_dependency(item, dep_id) for item in value.values())
        return False

    def _build_fused_task(self, parent: Task, child: Task, pattern: str) -> Task:
        fused_spec = json.dumps(
            {
                "pattern": pattern,
                "producer_idx": parent.idx,
                "producer_name": parent.name,
                "consumer_idx": child.idx,
                "consumer_name": child.name,
            },
            sort_keys=True,
        )

        fused_deps = sorted(
            {
                dep
                for dep in list(getattr(parent, "dependencies", []) or [])
                + list(getattr(child, "dependencies", []) or [])
                if dep != parent.idx
            }
        )

        async def fused_tool(spec: str, parent_args: Any, child_args: Any) -> Any:
            del spec
            parent_result = await self._invoke_tool(parent.tool, parent_args)
            child_runtime_args = self._replace_dependency_value(child_args, parent.idx, parent_result, child.name)
            return await self._invoke_tool(child.tool, child_runtime_args)

        def stringify_rule(args: Collection[Any]) -> str:
            return f"fused_op([{parent.name} -> {child.name}])"

        fused_task = Task(
            idx=child.idx,
            name="fused_op",
            tool=fused_tool,
            args=[fused_spec, parent.args, child.args],
            dependencies=fused_deps,
            stringify_rule=stringify_rule,
            thought=child.thought or parent.thought,
            is_join=False,
        )
        return fused_task

    async def _invoke_tool(self, tool: Any, args: Any) -> Any:
        call_args = tuple(args) if isinstance(args, (list, tuple)) else (args,)
        if asyncio.iscoroutinefunction(tool):
            return await tool(*call_args)
        result = tool(*call_args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _replace_dependency_value(self, args: Any, dep_id: int, value: Any, task_name: str) -> Any:
        if isinstance(args, (list, tuple)):
            return type(args)(self._replace_dependency_value(item, dep_id, value, task_name) for item in args)
        if isinstance(args, dict):
            return {k: self._replace_dependency_value(v, dep_id, value, task_name) for k, v in args.items()}
        if not isinstance(args, str):
            return args

        markers = [f"${{{dep_id}}}", f"${dep_id}"]
        updated = args
        for marker in markers:
            if marker in updated:
                updated = self._replace_marker(updated, marker, value, task_name)
        return updated

    def _replace_marker(self, text: str, marker: str, value: Any, task_name: str) -> str:
        value_text = str(value)
        if task_name == "python":
            encoded = base64.b64encode(value_text.encode("utf-8")).decode("ascii")
            python_expr = f"__import__('base64').b64decode('{encoded}').decode('utf-8', 'ignore')"
            for quoted_marker in (f"'{marker}'", f'"{marker}"'):
                if quoted_marker in text:
                    return text.replace(quoted_marker, python_expr)
            return text.replace(marker, python_expr)
        return text.replace(marker, value_text)

    def _calculate_stats(
        self,
        opportunities: List[FusionOpportunity],
        original_plans: Dict[str, Dict[str, Any]],
        optimized_plans: Dict[str, Dict[int, Task]],
    ) -> Dict[str, Any]:
        total_original = sum(len(data["plan"]) for data in original_plans.values())
        total_optimized = sum(len(plan) for plan in optimized_plans.values())
        total_savings = total_original - total_optimized
        return {
            "original_tasks": total_original,
            "optimized_tasks": total_optimized,
            "savings": total_savings,
            "opportunities": len(opportunities),
            "optimization_rate": (total_savings / total_original) if total_original else 0,
        }
