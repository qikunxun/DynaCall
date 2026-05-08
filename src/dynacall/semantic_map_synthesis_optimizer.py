from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.dynacall.task import Task


@dataclass
class SemanticMapSynthesisOpportunity:
    question: str
    original_plan: Dict[int, Task]
    optimized_plan: Dict[int, Task]
    savings: int
    fused_groups: List[List[int]]


@dataclass
class GuardedFusionInfo:
    group: List[int]
    branches: List[int]
    replans: List[int]


class SemanticMapSynthesisOptimizer:
    """Batch multiple semantic_map nodes into one multi-output semantic_map node."""

    def __init__(self, min_group_size: int = 2, max_dependency_depth_gap: int = 1):
        self.min_group_size = min_group_size
        self.max_dependency_depth_gap = max_dependency_depth_gap

    async def optimize_tool_chains(self, batch_plans: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        print("🔧 Applying semantic_map synthesis optimization...")

        optimized_plans: Dict[str, Dict[int, Task]] = {}
        opportunities: List[SemanticMapSynthesisOpportunity] = []

        for qid, plan_data in batch_plans.items():
            original_plan = plan_data["plan"]
            question = plan_data["question"]

            optimized_plan, fused_groups = self._optimize_plan(original_plan)
            optimized_plans[qid] = optimized_plan

            if len(optimized_plan) < len(original_plan):
                opportunities.append(
                    SemanticMapSynthesisOpportunity(
                        question=question,
                        original_plan=original_plan,
                        optimized_plan=optimized_plan,
                        savings=len(original_plan) - len(optimized_plan),
                        fused_groups=fused_groups,
                    )
                )
                print(
                    f"✅ semantic_map synthesis for {qid}: "
                    f"{len(original_plan)} -> {len(optimized_plan)} tasks; fused {len(fused_groups)} group(s)"
                )
            else:
                print(f"ℹ️  No semantic_map synthesis opportunities for {qid}")

        return {
            "optimized_plans": optimized_plans,
            "synthesis_opportunities": opportunities,
            "stats": self._calculate_stats(opportunities, batch_plans, optimized_plans),
        }

    def _optimize_plan(self, original_plan: Dict[int, Task]) -> Tuple[Dict[int, Task], List[List[int]]]:
        if not original_plan:
            return original_plan, []

        plan = dict(original_plan)
        groups = self._find_fusion_groups(plan)
        if not groups:
            return original_plan, []

        fused_group_by_task: Dict[int, List[int]] = {}
        for group in groups:
            for task_id in group:
                fused_group_by_task[task_id] = group

        guarded_group_info: Dict[Tuple[int, ...], GuardedFusionInfo] = {}
        guarded_branch_to_group: Dict[int, Tuple[int, ...]] = {}
        guarded_replan_to_group: Dict[int, Tuple[int, ...]] = {}
        for group in groups:
            group_key = tuple(group)
            info = self._get_guarded_fusion_info(group, plan)
            if info is None:
                continue
            guarded_group_info[group_key] = info
            for branch_id in info.branches:
                guarded_branch_to_group[branch_id] = group_key
            for replan_id in info.replans:
                guarded_replan_to_group[replan_id] = group_key

        old_to_new: Dict[int, int] = {}
        indexed_mapping: Dict[int, Tuple[int, int]] = {}
        new_plan: Dict[int, Task] = {}
        next_idx = 1
        created_groups: set[Tuple[int, ...]] = set()
        combined_branch_new_ids: set[int] = set()

        group_placement_member = {
            tuple(group): self._select_group_placement_member(group, plan) for group in groups
        }

        for old_idx in sorted(plan.keys()):
            task = plan[old_idx]
            guarded_group_key = guarded_branch_to_group.get(old_idx)
            if guarded_group_key is not None:
                info = guarded_group_info[guarded_group_key]
                if old_idx != min(info.branches):
                    continue

                branch_idx = next_idx
                fused_idx = next_idx + 1
                replan_idx = next_idx + 2 if info.replans else None

                combined_branch = self._build_combined_branch_task(info, plan, branch_idx, fused_idx, replan_idx)
                new_plan[branch_idx] = combined_branch
                combined_branch_new_ids.add(branch_idx)
                for branch_id in info.branches:
                    old_to_new[branch_id] = branch_idx

                fused_task = self._build_fused_task(info.group, plan, fused_idx)
                new_plan[fused_idx] = fused_task
                for offset, semantic_task_id in enumerate(info.group):
                    old_to_new[semantic_task_id] = fused_idx
                    indexed_mapping[semantic_task_id] = (fused_idx, offset)

                if replan_idx is not None:
                    combined_replan = self._build_combined_replan_task(info, plan, replan_idx, branch_idx)
                    new_plan[replan_idx] = combined_replan
                    for replan_id in info.replans:
                        old_to_new[replan_id] = replan_idx
                    next_idx += 3
                else:
                    next_idx += 2
                continue

            if tuple(fused_group_by_task.get(old_idx, [])) in guarded_group_info:
                continue
            if old_idx in guarded_replan_to_group:
                continue

            group = fused_group_by_task.get(old_idx)
            if group is None:
                cloned = self._clone_task(task, next_idx)
                new_plan[next_idx] = cloned
                old_to_new[old_idx] = next_idx
                next_idx += 1
                continue

            group_key = tuple(group)
            if old_idx != group_placement_member[group_key]:
                continue

            fused_task = self._build_fused_task(group, plan, next_idx)
            new_plan[next_idx] = fused_task
            for offset, semantic_task_id in enumerate(group):
                old_to_new[semantic_task_id] = next_idx
                indexed_mapping[semantic_task_id] = (next_idx, offset)
            created_groups.add(group_key)
            next_idx += 1

        for new_idx, task in list(new_plan.items()):
            if new_idx in combined_branch_new_ids and isinstance(task.args, (list, tuple)) and task.args:
                values = list(task.args)
                values[0] = self._rewrite_args(values[0], old_to_new, indexed_mapping)
                task.args = values
            else:
                task.args = self._rewrite_args(task.args, old_to_new, indexed_mapping)
            if task.name == "Calculate":
                indexed_refs = self._collect_indexed_reference_bases(task.args)
                refs = []
                for dep in getattr(task, "dependencies", []) or []:
                    mapped = old_to_new.get(dep, dep)
                    if mapped in indexed_refs:
                        continue
                    ref = f"${mapped}"
                    if ref not in refs:
                        refs.append(ref)
                if isinstance(task.args, (list, tuple)):
                    values = list(task.args)
                else:
                    values = [task.args]
                if len(values) == 1 and refs:
                    values.append(refs)
                elif len(values) >= 2 and isinstance(values[1], list):
                    merged = []
                    for ref in list(values[1]) + refs:
                        if isinstance(ref, int):
                            ref = f"${ref}"
                        elif isinstance(ref, str) and ref.isdigit():
                            ref = f"${ref}"
                        if ref not in merged:
                            merged.append(ref)
                    values[1] = merged
                task.args = values
            task.dependencies = self._rewrite_dependencies(task.dependencies, old_to_new)
            new_plan[new_idx] = task

        if not self._is_topologically_valid(new_plan):
            print("⚠️  semantic_map synthesis skipped: optimized plan would violate dependency order")
            return original_plan, []

        return new_plan, groups

    def _find_fusion_groups(self, plan: Dict[int, Task]) -> List[List[int]]:
        dependents: Dict[int, List[int]] = {task_id: [] for task_id in plan}
        for task_id, task in plan.items():
            for dep in getattr(task, "dependencies", []) or []:
                if dep in dependents:
                    dependents[dep].append(task_id)
        dependency_depth = self._compute_dependency_depth(plan)

        groups: List[List[int]] = []
        seen: set[Tuple[int, ...]] = set()

        # Case A: directly connected chains such as semantic_map -> semantic_map.
        for task_id in sorted(plan.keys()):
            if task_id in {item for group in groups for item in group}:
                continue
            if not self._is_simple_semantic_map(plan[task_id]):
                continue
            chain = [task_id]
            current = task_id
            while True:
                semantic_children = [
                    child_id
                    for child_id in dependents.get(current, [])
                    if self._is_chain_child(plan.get(child_id), current)
                ]
                if len(semantic_children) != 1:
                    break
                child_id = semantic_children[0]
                if child_id in chain:
                    break
                chain.append(child_id)
                current = child_id

            if len(chain) >= self.min_group_size:
                group = tuple(chain)
                seen.add(group)
                groups.append(chain)

        already_grouped = {task_id for group in groups for task_id in group}

        # Case B: sibling semantic_maps that feed the same downstream node.
        for consumer_id in sorted(plan.keys()):
            consumer = plan[consumer_id]
            semantic_dependencies = []
            for dep_id in getattr(consumer, "dependencies", []) or []:
                if dep_id in already_grouped:
                    continue
                dep_task = plan.get(dep_id)
                if dep_task is None or dep_task.name != "semantic_map" or dep_task.is_join:
                    continue
                if not self._is_simple_semantic_map(dep_task):
                    continue
                semantic_dependencies.append(dep_id)

            if len(semantic_dependencies) < self.min_group_size:
                continue

            for depth_cluster in self._cluster_by_dependency_depth(semantic_dependencies, dependency_depth):
                if len(depth_cluster) < self.min_group_size:
                    continue
                group = tuple(sorted(depth_cluster))
                if group in seen:
                    continue
                seen.add(group)
                groups.append(list(group))

        already_grouped = {task_id for group in groups for task_id in group}

        # Case C: independent semantic_maps that are all ready at the same earlier point,
        # even when each feeds a different downstream node such as separate Calculate calls.
        # Example:
        #   search1; search2; semantic_map($1); Calculate($3); semantic_map($2); Calculate($5)
        # becomes:
        #   search1; search2; semantic_map([$1, $2]); Calculate($3[0]); Calculate($3[1])
        candidates = [
            task_id
            for task_id in sorted(plan.keys())
            if task_id not in already_grouped and self._is_simple_semantic_map(plan[task_id])
        ]
        consumed: set[int] = set()
        for anchor in candidates:
            if anchor in consumed:
                continue
            group: List[int] = []
            for candidate in candidates:
                if candidate in consumed or candidate < anchor:
                    continue
                if not self._can_move_semantic_map_to_anchor(plan, dependents, candidate, anchor):
                    continue
                if not self._depths_are_close(anchor, candidate, dependency_depth):
                    continue
                group.append(candidate)

            if len(group) < self.min_group_size:
                continue

            group_key = tuple(group)
            if group_key in seen:
                continue
            seen.add(group_key)
            groups.append(group)
            consumed.update(group)

        return groups

    def _compute_dependency_depth(self, plan: Dict[int, Task]) -> Dict[int, int]:
        depth: Dict[int, int] = {}
        for task_id in sorted(plan.keys()):
            deps = [dep for dep in list(getattr(plan[task_id], "dependencies", []) or []) if dep in plan]
            if not deps:
                depth[task_id] = 0
            else:
                depth[task_id] = 1 + max(depth.get(dep, 0) for dep in deps)
        return depth

    def _depths_are_close(self, base_id: int, candidate_id: int, depth: Dict[int, int]) -> bool:
        return abs(depth.get(base_id, 0) - depth.get(candidate_id, 0)) <= self.max_dependency_depth_gap

    def _cluster_by_dependency_depth(self, task_ids: List[int], depth: Dict[int, int]) -> List[List[int]]:
        if not task_ids:
            return []
        sorted_ids = sorted(task_ids, key=lambda tid: (depth.get(tid, 0), tid))
        clusters: List[List[int]] = []
        current = [sorted_ids[0]]
        current_min = depth.get(sorted_ids[0], 0)
        current_max = current_min

        for task_id in sorted_ids[1:]:
            d = depth.get(task_id, 0)
            next_min = min(current_min, d)
            next_max = max(current_max, d)
            if next_max - next_min <= self.max_dependency_depth_gap:
                current.append(task_id)
                current_min, current_max = next_min, next_max
            else:
                clusters.append(current)
                current = [task_id]
                current_min = d
                current_max = d
        clusters.append(current)
        return clusters

    def _select_group_placement_member(self, group: List[int], plan: Dict[int, Task]) -> int:
        group_set = set(group)

        # Chain fusion should stay at the last member because later items may depend
        # on earlier fused results.
        for task_id in group:
            deps = set(getattr(plan[task_id], "dependencies", []) or [])
            if deps & group_set:
                return max(group)

        # Same-consumer fusion should also stay at the last member. This preserves
        # upstream guards such as branch nodes before each semantic_map while still
        # putting the fused result before the shared downstream consumer.
        consumer_sets = []
        for task_id in group:
            consumers = {
                candidate_id
                for candidate_id, candidate in plan.items()
                if task_id in set(getattr(candidate, "dependencies", []) or [])
            }
            consumer_sets.append(consumers)
        if consumer_sets and set.intersection(*consumer_sets):
            return max(group)

        # Independent ready-at-same-stage maps can be hoisted to the first member so
        # interleaved downstream tasks consume indexed fused outputs.
        return min(group)

    def _get_guarded_fusion_info(self, group: List[int], plan: Dict[int, Task]) -> Optional[GuardedFusionInfo]:
        branches: List[int] = []
        replans: List[int] = []

        for semantic_id in group:
            semantic_task = plan.get(semantic_id)
            if semantic_task is None:
                return None
            controlling_branches = []
            for dep_id in list(getattr(semantic_task, "dependencies", []) or []):
                dep_task = plan.get(dep_id)
                if dep_task is None or getattr(dep_task, "name", None) != "branch":
                    continue
                condition, then_ids, else_ids = self._parse_branch_args(dep_task)
                if semantic_id in then_ids:
                    controlling_branches.append(dep_id)

            if len(controlling_branches) != 1:
                return None

            branch_id = controlling_branches[0]
            branch_task = plan[branch_id]
            _, then_ids, else_ids = self._parse_branch_args(branch_task)
            if then_ids != [semantic_id]:
                return None

            for else_id in else_ids:
                else_task = plan.get(else_id)
                if else_task is None or getattr(else_task, "name", None) != "replan":
                    return None
                replans.append(else_id)
            branches.append(branch_id)

        if len(set(branches)) != len(group):
            return None

        return GuardedFusionInfo(group=list(group), branches=branches, replans=replans)

    def _parse_branch_args(self, task: Task) -> Tuple[Any, List[int], List[int]]:
        args = list(task.args) if isinstance(task.args, (list, tuple)) else [task.args]
        condition = args[0] if len(args) >= 1 else ""
        then_ids = list(args[1] or []) if len(args) >= 2 else []
        else_ids = list(args[2] or []) if len(args) >= 3 else []
        return condition, then_ids, else_ids

    def _build_combined_branch_task(
        self,
        info: GuardedFusionInfo,
        plan: Dict[int, Task],
        new_idx: int,
        fused_idx: int,
        replan_idx: Optional[int],
    ) -> Task:
        conditions = []
        dependencies: List[int] = []
        thoughts = []
        for branch_id in info.branches:
            branch_task = plan[branch_id]
            condition, _, _ = self._parse_branch_args(branch_task)
            condition_text = str(condition).strip()
            if condition_text:
                conditions.append(condition_text)
            dependencies.extend(list(getattr(branch_task, "dependencies", []) or []))
            if branch_task.thought:
                thoughts.append(branch_task.thought)

        combined_condition = " & ".join(conditions)
        else_ids = [replan_idx] if replan_idx is not None else []
        return Task(
            idx=new_idx,
            name="branch",
            tool=plan[info.branches[0]].tool,
            args=[combined_condition, [fused_idx], else_ids],
            dependencies=sorted(set(dependencies)),
            stringify_rule=plan[info.branches[0]].stringify_rule,
            thought=thoughts[0] if thoughts else None,
            is_branch=True,
        )

    def _build_combined_replan_task(
        self,
        info: GuardedFusionInfo,
        plan: Dict[int, Task],
        new_idx: int,
        branch_idx: int,
    ) -> Task:
        source = plan[info.replans[0]]
        reasons = []
        scopes = []
        for replan_id in info.replans:
            replan_task = plan[replan_id]
            args = list(replan_task.args) if isinstance(replan_task.args, (list, tuple)) else [replan_task.args]
            payload = args[0] if args else {}
            if isinstance(payload, dict):
                reason = str(payload.get("reason", "")).strip()
                scope = str(payload.get("scope", "")).strip()
                if reason:
                    reasons.append(reason)
                if scope:
                    scopes.append(scope)
            elif payload:
                reasons.append(str(payload).strip())

        payload = {
            "reason": "One or more fused branch guards failed: " + "; ".join(reasons),
            "scope": scopes[0] if scopes else "local",
        }
        return Task(
            idx=new_idx,
            name="replan",
            tool=source.tool,
            args=[payload],
            dependencies=[branch_idx],
            stringify_rule=source.stringify_rule,
            thought=source.thought,
            is_replan=True,
        )

    def _is_chain_child(self, task: Optional[Task], parent_id: int) -> bool:
        if task is None or not self._is_simple_semantic_map(task):
            return False
        dependencies = list(getattr(task, "dependencies", []) or [])
        if parent_id not in dependencies:
            return False
        _, inputs, _ = self._parse_semantic_args(task)
        return any(self._references_task(item, parent_id) for item in inputs)

    def _is_simple_semantic_map(self, task: Task) -> bool:
        if task.name != "semantic_map":
            return False
        try:
            _, inputs, _ = self._parse_semantic_args(task)
        except Exception:
            return False
        if self._has_indexed_reference(getattr(task, "args", None)):
            return False
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
            return False
        return True

    def _semantic_schema(self, task: Task) -> str:
        _, _, output_schema = self._parse_semantic_args(task)
        return output_schema

    def _parse_semantic_args(self, task: Task) -> Tuple[str, List[Any], str]:
        args = list(task.args) if isinstance(task.args, (list, tuple)) else [task.args]
        if len(args) >= 5:
            instruction = str(args[1]).strip()
            inputs = args[3]
            output_schema = str(args[4]).strip()
        elif len(args) >= 3:
            instruction = str(args[0]).strip()
            inputs = args[1]
            output_schema = str(args[2]).strip()
        else:
            raise ValueError("semantic_map requires at least 3 args")
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        return instruction, list(inputs), output_schema or "string"

    def _references_task(self, value: Any, task_id: int) -> bool:
        if isinstance(value, str):
            return bool(re.search(rf"\$\{{?{task_id}\}}?(?:\[\d+\])?", value))
        if isinstance(value, (list, tuple)):
            return any(self._references_task(item, task_id) for item in value)
        if isinstance(value, dict):
            return any(self._references_task(item, task_id) for item in value.values())
        return False

    def _has_indexed_reference(self, value: Any) -> bool:
        if isinstance(value, str):
            return bool(re.search(r"\$\{?\d+\}?\[\d+\]", value))
        if isinstance(value, (list, tuple)):
            return any(self._has_indexed_reference(item) for item in value)
        if isinstance(value, dict):
            return any(self._has_indexed_reference(item) for item in value.values())
        return False

    def _can_move_semantic_map_to_anchor(
        self,
        plan: Dict[int, Task],
        dependents: Dict[int, List[int]],
        task_id: int,
        anchor: int,
    ) -> bool:
        task = plan[task_id]
        referenced_ids = set(getattr(task, "dependencies", []) or [])
        referenced_ids.update(self._extract_reference_ids(getattr(task, "args", None)))

        # The fused node is emitted at the earliest group member. Every input used by
        # every fused item must already be available there.
        if any(dep_id >= anchor for dep_id in referenced_ids if dep_id in plan):
            return False

        # A consumer cannot appear before the fused node. This protects non-contiguous
        # groups where a downstream Calculate may sit between two semantic_map nodes.
        if any(consumer_id <= anchor for consumer_id in dependents.get(task_id, [])):
            return False

        return True

    def _extract_reference_ids(self, value: Any) -> set[int]:
        refs: set[int] = set()
        if isinstance(value, str):
            for match in re.finditer(r"\$\{?(\d+)\}?(?:\[\d+\])?", value):
                refs.add(int(match.group(1)))
        elif isinstance(value, (list, tuple)):
            for item in value:
                refs.update(self._extract_reference_ids(item))
        elif isinstance(value, dict):
            for item in value.values():
                refs.update(self._extract_reference_ids(item))
        return refs

    def _build_fused_task(self, group: List[int], plan: Dict[int, Task], new_idx: int) -> Task:
        members = [plan[task_id] for task_id in group]
        first = members[0]

        batch_spec = {
            "mode": "batch",
            "output_format": "result_lines",
            "items": [],
        }
        fused_inputs: List[Any] = []
        fused_deps: List[int] = []
        group_position = {task_id: offset for offset, task_id in enumerate(group)}

        for task_id, member in zip(group, members):
            instruction, inputs, output_schema = self._parse_semantic_args(member)
            input_value = inputs[0]
            depends_on_result = None
            for dep_id in getattr(member, "dependencies", []) or []:
                if dep_id in group_position:
                    depends_on_result = group_position[dep_id] + 1
                    break
            batch_spec["items"].append(
                {
                    "instruction": instruction,
                    "output_schema": output_schema,
                    "depends_on_result": depends_on_result,
                }
            )
            if depends_on_result is None:
                fused_inputs.append(input_value)
            else:
                fused_inputs.append(f"Use Result {depends_on_result} as the input for this item.")
            fused_deps.extend(
                dep_id for dep_id in list(getattr(member, "dependencies", []) or [])
                if dep_id not in group_position
            )

        fused_output_schema = "batch_results"
        thought = next((member.thought for member in members if member.thought), None)

        def stringify_rule(args):
            item_count = len(batch_spec["items"])
            input_count = len(args[1]) if len(args) > 1 and isinstance(args[1], list) else item_count
            return f"semantic_map(batch_items={item_count}, inputs={input_count}, schema='batch_results')"

        return Task(
            idx=new_idx,
            name="semantic_map",
            tool=first.tool,
            args=[json.dumps(batch_spec, ensure_ascii=False), fused_inputs, fused_output_schema],
            dependencies=sorted(set(fused_deps)),
            stringify_rule=stringify_rule,
            thought=thought,
            is_join=False,
        )

    def _clone_task(self, task: Task, new_idx: int) -> Task:
        return Task(
            idx=new_idx,
            name=task.name,
            tool=task.tool,
            args=task.args,
            dependencies=list(getattr(task, "dependencies", []) or []),
            stringify_rule=task.stringify_rule,
            thought=task.thought,
            observation=task.observation,
            is_join=task.is_join,
            is_branch=getattr(task, "is_branch", False),
            is_replan=getattr(task, "is_replan", False),
        )

    def _rewrite_dependencies(self, dependencies: Any, old_to_new: Dict[int, int]) -> List[int]:
        new_dependencies = []
        for dep in list(dependencies or []):
            mapped = old_to_new.get(dep, dep)
            if mapped not in new_dependencies:
                new_dependencies.append(mapped)
        return new_dependencies

    def _rewrite_args(
        self,
        args: Any,
        old_to_new: Dict[int, int],
        indexed_mapping: Dict[int, Tuple[int, int]],
    ) -> Any:
        if isinstance(args, (list, tuple)):
            return type(args)(self._rewrite_args(item, old_to_new, indexed_mapping) for item in args)
        if isinstance(args, dict):
            return {k: self._rewrite_args(v, old_to_new, indexed_mapping) for k, v in args.items()}
        if isinstance(args, int):
            return old_to_new.get(args, args)
        if not isinstance(args, str):
            return args

        import re

        pattern = re.compile(r"\$\{?(\d+)\}?(?:\[(\d+)\])?")

        def repl(match):
            old_id = int(match.group(1))
            existing_index = match.group(2)
            if old_id in indexed_mapping:
                new_id, offset = indexed_mapping[old_id]
                if existing_index is not None:
                    return match.group(0)
                return f"${new_id}[{offset}]"
            if old_id in old_to_new:
                new_id = old_to_new[old_id]
                if existing_index is not None:
                    return f"${new_id}[{existing_index}]"
                return f"${new_id}"
            return match.group(0)

        return pattern.sub(repl, args)

    def _collect_indexed_reference_bases(self, value: Any) -> set[int]:
        refs: set[int] = set()
        if isinstance(value, str):
            for match in re.finditer(r"\$\{?(\d+)\}?\[\d+\]", value):
                refs.add(int(match.group(1)))
        elif isinstance(value, (list, tuple)):
            for item in value:
                refs.update(self._collect_indexed_reference_bases(item))
        elif isinstance(value, dict):
            for item in value.values():
                refs.update(self._collect_indexed_reference_bases(item))
        return refs

    def _is_topologically_valid(self, plan: Dict[int, Task]) -> bool:
        task_ids = set(plan.keys())
        for task_id, task in plan.items():
            for dep_id in list(getattr(task, "dependencies", []) or []):
                if dep_id in task_ids and dep_id >= task_id:
                    return False
        return True

    def _calculate_stats(
        self,
        opportunities: List[SemanticMapSynthesisOpportunity],
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
