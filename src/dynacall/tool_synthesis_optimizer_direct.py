import re
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from src.dynacall.task import Task

@dataclass
class TreeNode:
    
    task_id: int
    task: Task
    children: List['TreeNode']
    parent: Optional['TreeNode'] = None
    
    def __repr__(self):
        return f"TreeNode({self.task_id}:{self.task.name})"

@dataclass
class SynthesisOpportunity:
    
    question: str
    original_plan: Dict[int, Task]
    optimized_plan: Dict[int, Task]
    savings: int
    computation_chains: List[List[int]]

class DirectToolSynthesisOptimizer:
    
    
    def __init__(self, max_merge_count: int = 5):

        self.max_merge_count = max_merge_count
    
    async def optimize_tool_chains(self, batch_plans: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        
        print("🔧 Applying call-tree based tool synthesis optimization...")
        
        all_opportunities = []
        optimized_plans = {}
        
        for qid, plan_data in batch_plans.items():
            original_plan = plan_data['plan']
            question = plan_data['question']
            
            print(f"\n📋 Processing: {question[:50]}...")
            print(f"   Original tasks: {len(original_plan)}")
            
            
            print("📝 ORIGINAL PLAN:")
            self._print_plan(original_plan)
            
            
            optimized_plan, optimization_info = await self._optimize_with_call_tree(original_plan)
            
            if optimized_plan and len(optimized_plan) < len(original_plan):
                savings = len(original_plan) - len(optimized_plan)
                opportunity = SynthesisOpportunity(
                    question=question,
                    original_plan=original_plan,
                    optimized_plan=optimized_plan,
                    savings=savings,
                    computation_chains=optimization_info.get('dependency_chains', [])
                )
                all_opportunities.append(opportunity)
                optimized_plans[qid] = optimized_plan
                
                print(f"✅ Call-tree optimized: {len(original_plan)} -> {len(optimized_plan)} tasks")
                print(f"   Tasks saved: {savings}")
                
                
                print("📝 OPTIMIZED PLAN:")
                self._print_plan(optimized_plan)
            else:
                optimized_plans[qid] = original_plan
                print("ℹ️  No optimization achieved")
        
        self._print_final_stats(all_opportunities, batch_plans, optimized_plans)
        
        return {
            'optimized_plans': optimized_plans,
            'synthesis_opportunities': all_opportunities,
            'stats': self._calculate_stats(all_opportunities, batch_plans, optimized_plans)
        }
    
    def _print_plan(self, plan: Dict[int, Task]):
        
        for task_id, task in sorted(plan.items()):
            if task.is_join:
                print(f"  {task_id}. {task.name}()")
            else:
                args_str = self._format_args_for_display(task.args)
                print(f"  {task_id}. {task.name}({args_str})")
    
    def _format_args_for_display(self, args) -> str:
        
        if not args:
            return ""
        
        if isinstance(args, (list, tuple)):
            formatted_args = []
            for arg in args:
                if isinstance(arg, str):
                    
                    formatted_args.append(f"'{arg}'")
                elif isinstance(arg, (list, tuple)):
                    
                    inner_args = [f"'{item}'" if isinstance(item, str) else str(item) for item in arg]
                    formatted_args.append(f"[{', '.join(inner_args)}]")
                else:
                    formatted_args.append(str(arg))
            return ", ".join(formatted_args)
        else:
            return f"'{str(args)}'" if isinstance(args, str) else str(args)
    
    async def _optimize_with_call_tree(self, original_plan: Dict[int, Task]) -> Tuple[Dict[int, Task], Dict[str, Any]]:
        
        if not original_plan:
            return original_plan, {}
        
        
        dependency_chains = self._analyze_dependency_chains(original_plan)
        print(f"🔗 Found {len(dependency_chains)} dependency chain(s)")
        
        
        call_tree = self._build_call_tree(original_plan, dependency_chains)
        if not call_tree:
            return original_plan, {'dependency_chains': dependency_chains}
        
        print("🌳 Original call tree:")
        self._print_tree(call_tree)
        
        
        merged_tree = self._merge_node_recursive(call_tree, original_plan)
        print("🌳 Merged call tree:")
        self._print_tree(merged_tree)
        
        
        optimized_plan = self._generate_plan_from_tree(merged_tree, original_plan)

        return optimized_plan, {
            'dependency_chains': dependency_chains,
            'original_tree': call_tree,
            'merged_tree': merged_tree
        }
    
    def _analyze_dependency_chains(self, plan: Dict[int, Task]) -> List[List[int]]:
        
        dependency_graph = {}
        
        
        for task_id, task in plan.items():
            dependency_graph[task_id] = {
                'task': task,
                'dependencies': task.dependencies if hasattr(task, 'dependencies') else [],
                'dependents': []
            }
        
        
        for task_id, data in dependency_graph.items():
            for dep_id in data['dependencies']:
                if dep_id in dependency_graph:
                    dependency_graph[dep_id]['dependents'].append(task_id)
        
        
        print("🔍 DEPENDENCY ANALYSIS:")
        for task_id, data in dependency_graph.items():
            task_name = plan[task_id].name + ("(join)" if plan[task_id].is_join else "")
            print(f"  Task {task_id}({task_name}): depends_on={data['dependencies']}, depended_by={data['dependents']}")
        
        
        chains = []
        visited = set()
        
        
        leaf_nodes = [task_id for task_id, data in dependency_graph.items() 
                     if not data['dependencies']]
        
        print(f"  Leaf nodes: {leaf_nodes}")
        
        
        for leaf in leaf_nodes:
            if leaf not in visited:
                paths = self._find_complete_paths(leaf, dependency_graph, visited)
                chains.extend(paths)
        
        print(f"  Found {len(chains)} dependency chains: {chains}")
        return chains
    
    def _find_complete_paths(self, start_task: int, graph: Dict[int, Any], visited: Set[int]) -> List[List[int]]:
        
        paths = []
        
        def dfs(current: int, path: List[int]):
            path.append(current)
            visited.add(current)
            
            
            if not graph[current]['dependents']:
                paths.append(path.copy())
            else:
                
                for next_task in graph[current]['dependents']:
                    if next_task not in path:  
                        dfs(next_task, path)
            
            path.pop()
        
        dfs(start_task, [])
        return paths
    
    def _build_call_tree(self, plan: Dict[int, Task], dependency_chains: List[List[int]]) -> Optional[TreeNode]:
        
        if not dependency_chains:
            return None
        
        
        root_candidates = defaultdict(int)
        for chain in dependency_chains:
            if chain:
                root_candidates[chain[-1]] += 1
        
        if not root_candidates:
            return None
        
        root_task_id = max(root_candidates.items(), key=lambda x: x[1])[0]
        root_task = plan[root_task_id]
        
        
        root_node = TreeNode(task_id=root_task_id, task=root_task, children=[])
        node_map = {root_task_id: root_node}
        
        
        for chain in dependency_chains:
            if root_task_id in chain:
                
                root_index = chain.index(root_task_id)
                current_parent = root_node
                
                
                for i in range(root_index - 1, -1, -1):
                    child_task_id = chain[i]
                    if child_task_id not in node_map:
                        child_task = plan[child_task_id]
                        child_node = TreeNode(
                            task_id=child_task_id, 
                            task=child_task, 
                            children=[],
                            parent=current_parent
                        )
                        node_map[child_task_id] = child_node
                        current_parent.children.append(child_node)
                        current_parent = child_node
                    else:
                        current_parent = node_map[child_task_id]
        
        return root_node

    def _merge_node_recursive(self, node: TreeNode, original_plan: Dict[int, Task], 
                             merge_count: Dict[int, int] = None) -> TreeNode:
        if merge_count is None:
            merge_count = {}
        
        
        current_merge_count = merge_count.get(node.task_id, 0)
        if current_merge_count >= self.max_merge_count:
            print(f"⏹️  Node {node.task_id}({node.task.name}) reached max merge count ({self.max_merge_count}), skip merging")
            
            new_children = []
            for child in node.children:
                merged_child = self._merge_node_recursive(child, original_plan, merge_count)
                if merged_child is not child:
                    merged_child.parent = node
                    new_children.append(merged_child)
                else:
                    new_children.append(child)
            node.children = new_children
            return node
        
        
        new_children = []
        for child in node.children:
            
            merged_child = self._merge_node_recursive(child, original_plan, merge_count)
            
            
            if merged_child is not child:
                
                merged_child.parent = node
                new_children.append(merged_child)
            else:
                
                new_children.append(child)
        
        
        node.children = new_children
        

        C = [c for c in node.children if self._can_merge_tasks(c.task, node.task)]

        # 只要有相同类型的孩子节点就可以合并，不需要所有孩子都相同
        if C and self._is_llm_based(node.task):
            # 找出与父节点相同工具类型的孩子节点
            same_tool_children = [c for c in C if c.task.name == node.task.name]

            if same_tool_children:

                print(f"🔄 Merging {node.task_id}({node.task.name}) with {len(same_tool_children)} children of same type (merge count: {current_merge_count + 1}/{self.max_merge_count})")


                grand_children = []
                for child in same_tool_children:
                    grand_children.extend(child.children)


                merged_task = self._coalesce_functions(node.task, [c.task for c in same_tool_children])


                merged_node = TreeNode(
                    task_id=node.task_id,
                    task=merged_task,
                    children=[],
                    parent=node.parent
                )


                for grand_child in grand_children:
                    grand_child.parent = merged_node
                merged_node.children = grand_children


                for child in node.children:
                    if child not in same_tool_children:
                        child.parent = merged_node
                        merged_node.children.append(child)


                merge_count[node.task_id] = current_merge_count + 1



                return self._merge_node_recursive(merged_node, original_plan, merge_count)


        return node
    
    def _is_llm_based(self, task: Task) -> bool:



        return self._task_has_dollar_reference(task)

    def _can_merge_tasks(self, child_task: Task, parent_task: Task) -> bool:
        
        
        if child_task.name != parent_task.name:
            return False
        
        child_has_dollar = self._task_has_dollar_reference(child_task)
        parent_has_dollar = self._task_has_dollar_reference(parent_task)
        
        
        return child_has_dollar and parent_has_dollar

    def _task_has_dollar_reference(self, task: Task) -> bool:
        
        
        if task.args and len(task.args) > 0:
            description = str(task.args[0]) if isinstance(task.args[0], str) else str(task.args[0])
            if '$' in description:
                return True
        
        
        if task.args and len(task.args) > 1:
            dependencies = task.args[1]
            if isinstance(dependencies, (list, tuple)):
                for dep in dependencies:
                    if isinstance(dep, str) and '$' in dep:
                        return True
        
        
        if hasattr(task, 'dependencies') and task.dependencies:
            for dep in task.dependencies:
                if isinstance(dep, str) and '$' in dep:
                    return True
        
        return False
    
    def _coalesce_functions(self, parent_task: Task, child_tasks: List[Task]) -> Task:        
        i_p = parent_task.idx
        f = parent_task.name
        t_p = self._extract_instruction(parent_task)  
        d_p = self._extract_dependencies(parent_task)  
        
        
        t_merged = t_p
        
        
        
        dollar_refs = re.findall(r'\$(\d+)', t_p)


        # 创建child_tasks的ID到任务的映射
        child_task_map = {child.idx: child for child in child_tasks}

        placeholder_mapping = {}
        param_chain = []


        for ref in dollar_refs:
            placeholder = f"${ref}"
            ref_id = int(ref)


            # 只有当引用的任务在被合并的子任务中时，才进行替换
            if ref_id in child_task_map:
                child_task = child_task_map[ref_id]
                t_k = self._extract_instruction(child_task)
                placeholder_mapping[placeholder] = t_k


                param_chain.append({
                    'placeholder': placeholder,
                    'child_instruction': t_k,
                    'child_deps': self._extract_dependencies(child_task),
                    'child_params': self._extract_parameters(child_task)
                })
        
        
        
        
        
        all_base_deps = []
        for child in child_tasks:
            child_deps = self._extract_dependencies(child)
            
            for dep in child_deps:
                if isinstance(dep, int):
                    if dep not in all_base_deps:
                        all_base_deps.append(dep)
                elif isinstance(dep, str) and dep.startswith('$'):
                    try:
                        base_dep = int(dep[1:])
                        if base_dep not in all_base_deps:
                            all_base_deps.append(base_dep)
                    except ValueError:
                        pass
        
        
        all_deps = sorted(all_base_deps)
        
        
        
        sorted_placeholders = sorted(
            placeholder_mapping.keys(),
            key=lambda x: t_merged.rfind(x),
            reverse=True
        )
        
        for placeholder in sorted_placeholders:
            if placeholder in t_merged:
                t_merged = t_merged.replace(placeholder, placeholder_mapping[placeholder])
        
        
        
        remaining_refs = re.findall(r'\$(\d+)', t_merged)
        if remaining_refs:
            
            for ref in remaining_refs:
                try:
                    ref_num = int(ref)
                    if ref_num not in all_deps:
                        all_deps.append(ref_num)
                except ValueError:
                    pass
        
        
        
        v_merged = []
        
        
        
        for match in re.finditer(r'\$(\d+)', t_merged):
            ref = match.group(1)
            try:
                ref_num = int(ref)
                if ref_num not in all_deps:
                    all_deps.append(ref_num)
            except ValueError:
                pass
        
        
        merged_args = [t_merged]  
        if all_deps:  
            merged_args.append(all_deps)
        
        merged_task = Task(
            idx=i_p,
            name=f,
            tool=parent_task.tool,
            args=merged_args,
            dependencies=all_deps,
            is_join=parent_task.is_join
        )
        
        
        if hasattr(parent_task, 'observation'):
            merged_task.observation = parent_task.observation
        if hasattr(parent_task, 'thought'):
            merged_task.thought = parent_task.thought
        if hasattr(parent_task, 'action'):
            merged_task.action = parent_task.action
        
        print(f"   Original instruction: {t_p}")
        print(f"   Merged instruction: {t_merged}")
        print(f"   All dependencies: {all_deps}")
        
        return merged_task
    
    def _extract_instruction(self, task: Task) -> str:
        
        if not task.args or len(task.args) == 0:
            return ""
        
        
        if isinstance(task.args[0], str):
            return task.args[0]
        elif isinstance(task.args[0], (list, tuple)) and task.args[0]:
            
            first_arg = task.args[0]
            if isinstance(first_arg, (list, tuple)) and first_arg:
                return str(first_arg[0])
            else:
                return str(first_arg)
        else:
            return str(task.args[0])
    
    def _extract_parameters(self, task: Task) -> List[str]:
        
        params = []
        
        if not task.args or len(task.args) <= 1:
            return params
        
        
        
        for i in range(2, len(task.args)):
            arg = task.args[i]
            if isinstance(arg, str):
                params.append(arg)
            elif isinstance(arg, (list, tuple)):
                
                for item in arg:
                    if isinstance(item, str):
                        params.append(item)
        
        return params
    
    def _extract_dependencies(self, task: Task) -> List[int]:
        
        if hasattr(task, 'dependencies') and task.dependencies:
            return task.dependencies[:]  
        
        
        if task.args and len(task.args) > 1:
            deps = task.args[1]
            if isinstance(deps, (list, tuple)):
                
                result = []
                for dep in deps:
                    if isinstance(dep, int):
                        result.append(dep)
                    elif isinstance(dep, str) and dep.startswith('$'):
                        try:
                            result.append(int(dep[1:]))
                        except ValueError:
                            pass
                return result
        
        return []
    
    def _generate_plan_from_tree(self, root: TreeNode, original_plan: Dict[int, Task]) -> Dict[int, Task]:
        
        plan = {}
        task_id_mapping = {}
        next_id = 1
        
        
        all_nodes = self._collect_nodes_in_order(root)
        
        
        for node in all_nodes:
            new_id = next_id
            task_id_mapping[node.task_id] = new_id
            next_id += 1
        
        print(f"🔍 Task ID mapping: {task_id_mapping}")
        
        
        for node in all_nodes:
            new_id = task_id_mapping[node.task_id]
            new_task = self._copy_task(node.task, new_id)
            
            
            if hasattr(new_task, 'dependencies'):
                updated_deps = []
                for dep in new_task.dependencies:
                    if dep in task_id_mapping:
                        updated_deps.append(task_id_mapping[dep])
                    else:
                        
                        continue
                new_task.dependencies = updated_deps
            
            
            if new_task.is_join:
                new_task.args = ()
            elif new_task.args is not None:
                updated_args = []
                for arg in new_task.args:
                    if isinstance(arg, str):
                        updated_arg = re.sub(
                            r'\$(\d+)',
                            lambda m: f"${task_id_mapping.get(int(m.group(1)), m.group(1))}",
                            arg
                        )
                        updated_args.append(updated_arg)
                    elif isinstance(arg, (list, tuple)):
                        updated_list = []
                        for item in arg:
                            if isinstance(item, str):
                                updated_item = re.sub(
                                    r'\$(\d+)',
                                    lambda m: f"${task_id_mapping.get(int(m.group(1)), m.group(1))}",
                                    item
                                )
                                updated_list.append(updated_item)
                            elif isinstance(item, int):
                                if item in task_id_mapping:
                                    updated_list.append(f"${task_id_mapping[item]}")
                                else:
                                    updated_list.append(f"${item}")
                            else:
                                updated_list.append(item)
                        updated_args.append(updated_list)
                    else:
                        updated_args.append(arg)
                new_task.args = updated_args
            
            plan[new_id] = new_task
        
        return plan
    
    def _collect_nodes_in_order(self, root: TreeNode) -> List[TreeNode]:
        
        nodes = []
        
        def collect_postorder(node: TreeNode):
            
            for child in node.children:
                collect_postorder(child)
            
            nodes.append(node)
        
        collect_postorder(root)
        return nodes
    
    def _copy_task(self, task: Task, new_id: int) -> Task:
        
        new_task = Task(
            idx=new_id,
            name=task.name,
            tool=task.tool,
            args=task.args[:] if task.args else None,
            dependencies=task.dependencies[:] if hasattr(task, 'dependencies') and task.dependencies else [],
            is_join=task.is_join
        )
        
        if hasattr(task, 'observation'):
            new_task.observation = task.observation
        if hasattr(task, 'thought'):
            new_task.thought = task.thought
        if hasattr(task, 'action'):
            new_task.action = task.action
        
        return new_task
    
    def _print_tree(self, node: TreeNode, level: int = 0):
        
        indent = "  " * level
        print(f"{indent}{node.task_id}({node.task.name})")
        for child in node.children:
            self._print_tree(child, level + 1)
    
    def _print_final_stats(self, opportunities: List[SynthesisOpportunity], 
                          original_plans: Dict, optimized_plans: Dict):
        
        total_original = sum(len(data['plan']) for data in original_plans.values())
        total_optimized = sum(len(plan) for plan in optimized_plans.values())
        total_savings = sum(opp.savings for opp in opportunities)
        
        print(f"\n🎯 CALL-TREE OPTIMIZATION SUMMARY:")
        print(f"   Questions processed: {len(original_plans)}")
        print(f"   Original tasks: {total_original}")
        print(f"   Optimized tasks: {total_optimized}")
        print(f"   Tasks saved: {total_savings}")
        print(f"   Optimization rate: {total_savings/total_original:.1%}" if total_original > 0 else "N/A")
        print(f"   Opportunities found: {len(opportunities)}")
    
    def _calculate_stats(self, opportunities: List[SynthesisOpportunity], 
                        original_plans: Dict, optimized_plans: Dict) -> Dict[str, Any]:
        
        total_original = sum(len(data['plan']) for data in original_plans.values())
        total_optimized = sum(len(plan) for plan in optimized_plans.values())
        total_savings = sum(opp.savings for opp in opportunities)
        
        return {
            'original_tasks': total_original,
            'optimized_tasks': total_optimized,
            'savings': total_savings,
            'opportunities': len(opportunities),
            'optimization_rate': total_savings / total_original if total_original > 0 else 0
        }
