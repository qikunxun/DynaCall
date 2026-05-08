import asyncio
import time
import traceback
import json
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
from src.dynacall.semantic_map_synthesis_optimizer import SemanticMapSynthesisOptimizer
from src.dynacall.task import TaskFetchingUnit, create_function_signature

@dataclass
class ResourceBudget:
    max_questions: int
    max_concurrent: int

@dataclass
class MapTask:
    question_id: str
    question: str
    plan: Optional[Dict[int, Any]] = None
    optimized_plan: Optional[Dict[int, Any]] = None
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # 每个问题独立的统计
    shared_task_hits: int = 0          # 从其他问题复用的任务数
    cached_task_hits: int = 0          # 从缓存复用的任务数
    early_executed_tasks: int = 0      # 自己提前执行的任务数（early_execution）
    executed_tasks_count: int = 0      # map阶段自己执行的任务数
    leaf_task_num: int = 0
    
    # 记录该问题产生的所有 task_key，用于清理
    owned_task_keys: Set[str] = field(default_factory=set)
    
    # 记录该问题在 early_execution 中执行的任务
    early_executed_task_keys: Set[str] = field(default_factory=set)
    # Stage wall-clock timing (seconds)
    plan_start_time: Optional[float] = None
    plan_end_time: Optional[float] = None
    map_start_time: Optional[float] = None
    map_end_time: Optional[float] = None
    reduce_start_time: Optional[float] = None
    reduce_end_time: Optional[float] = None
    join_duration: float = 0.0
    planner_callback: Any = None
    executor_callback: Any = None

class SharedTaskManager:
    """全局共享任务管理器 - 所有问题共享，支持按问题清理"""
    def __init__(self):
        self.task_registry: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_events: Dict[str, asyncio.Event] = {}
        self.task_errors: Dict[str, Exception] = {}
        self.task_owners: Dict[str, str] = {}  # task_key -> question_id 记录哪个问题首次创建
        
        # 按问题索引的任务列表，用于批量清理
        self.question_tasks: Dict[str, Set[str]] = {}  # question_id -> set of task_keys
        
    def register_task(self, task_key: str, owner_id: str = None) -> bool:
        """
        注册任务，返回是否是新任务
        如果是新任务，返回 True；如果已存在，返回 False
        """
        if task_key in self.task_registry:
            return False
        
        self.task_events[task_key] = asyncio.Event()
        self.task_owners[task_key] = owner_id
        
        # 记录该问题拥有的任务
        if owner_id not in self.question_tasks:
            self.question_tasks[owner_id] = set()
        self.question_tasks[owner_id].add(task_key)
        
        return True
    
    def get_task_result(self, task_key: str) -> Optional[Any]:
        return self.task_results.get(task_key)
    
    def get_task_error(self, task_key: str) -> Optional[Exception]:
        return self.task_errors.get(task_key)
    
    def is_task_completed(self, task_key: str) -> bool:
        return task_key in self.task_results or task_key in self.task_errors
    
    def is_task_running(self, task_key: str) -> bool:
        return task_key in self.task_registry and not self.task_registry[task_key].done()
    
    async def wait_for_task(self, task_key: str, timeout: Optional[float] = None) -> bool:
        if task_key not in self.task_events:
            return False
        
        try:
            await asyncio.wait_for(self.task_events[task_key].wait(), timeout=timeout)
            return task_key in self.task_results
        except asyncio.TimeoutError:
            return False
    
    def set_task_result(self, task_key: str, result: Any):
        self.task_results[task_key] = result
        if task_key in self.task_events:
            self.task_events[task_key].set()
    
    def set_task_error(self, task_key: str, error: Exception):
        self.task_errors[task_key] = error
        if task_key in self.task_events:
            self.task_events[task_key].set()
    
    def create_task(self, task_key: str, coro, owner_id: str) -> asyncio.Task:
        if task_key in self.task_registry:
            return self.task_registry[task_key]
        
        task = asyncio.create_task(coro)
        self.task_registry[task_key] = task
        self.task_owners[task_key] = owner_id
        
        # 记录该问题拥有的任务
        if owner_id not in self.question_tasks:
            self.question_tasks[owner_id] = set()
        self.question_tasks[owner_id].add(task_key)
        
        return task
    
    def get_task_owner(self, task_key: str) -> Optional[str]:
        """获取任务的创建者"""
        return self.task_owners.get(task_key)
    
    def cleanup_question_tasks(self, question_id: str):
        """
        清理指定问题产生的所有任务记录
        只有当没有其他问题依赖这些任务时，才真正删除
        """
        if question_id not in self.question_tasks:
            return
        
        task_keys = self.question_tasks[question_id].copy()
        removed_count = 0
        
        for task_key in task_keys:
            # 检查是否有其他问题还在依赖这个任务
            other_dependents = False
            for owner, tasks in self.question_tasks.items():
                if owner != question_id and task_key in tasks:
                    other_dependents = True
                    break
            
            if not other_dependents:
                # 没有其他依赖，可以安全删除
                self._remove_task(task_key)
                removed_count += 1
            else:
                # 还有其他问题依赖，只从当前问题的记录中移除
                print(f"    Keeping shared task {task_key} (used by other questions)")
        
        # 删除该问题的任务集合
        del self.question_tasks[question_id]
        
        if removed_count > 0:
            print(f"    Cleaned up {removed_count} tasks owned by question {question_id}")
    
    def _remove_task(self, task_key: str):
        """内部方法：删除任务的所有记录"""
        if task_key in self.task_registry:
            del self.task_registry[task_key]
        
        if task_key in self.task_results:
            del self.task_results[task_key]
        if task_key in self.task_errors:
            del self.task_errors[task_key]
        if task_key in self.task_events:
            del self.task_events[task_key]
        if task_key in self.task_owners:
            del self.task_owners[task_key]
        
        # 从所有问题的任务集合中删除
        for question_id in list(self.question_tasks.keys()):
            if task_key in self.question_tasks[question_id]:
                self.question_tasks[question_id].discard(task_key)
    
    def get_shared_task_stats(self) -> Dict[str, Any]:
        """获取共享任务统计信息"""
        total_tasks = len(self.task_results) + len(self.task_errors)
        running_tasks = sum(1 for t in self.task_registry.values() if not t.done())
        
        return {
            'total_tasks_created': len(self.task_owners),
            'completed_tasks': len(self.task_results),
            'failed_tasks': len(self.task_errors),
            'running_tasks': running_tasks,
            'cached_results': len(self.task_results),
            'active_questions': len(self.question_tasks)
        }

class CacheManager:
    def __init__(self, cache_file: Optional[str] = None, max_cache_size: int = 1000):
        self.cache_file = cache_file
        self.cache: Dict[str, Any] = {}
        self.enabled = cache_file is not None
        self.max_cache_size = max_cache_size
        
        if self.enabled:
            self._load_cache()
            print(f" Cache ENABLED: {self.cache_file} ({len(self.cache)} entries)")
        else:
            print(" Cache DISABLED")
    
    def _load_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"📂 Loaded cache from {self.cache_file}: {len(self.cache)} entries")
            except Exception as e:
                print(f" Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        if not self.enabled or not self.cache_file:
            return
            
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f" Failed to save cache: {e}")
    
    def get_cached_result(self, task: Any) -> Optional[Any]:
        if not self.enabled:
            return None
            
        cache_key = create_function_signature(task)
        if cache_key in self.cache:
            return self.cache[cache_key]
        return None
    
    def set_cached_result(self, task: Any, result: Any):
        if not self.enabled:
            return
            
        cache_key = create_function_signature(task)
        if 'context=' in cache_key or '$' in cache_key: 
            return
        
        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        asyncio.create_task(self._async_save_cache())
    
    async def _async_save_cache(self):
        await asyncio.sleep(0.1)
        self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.enabled:
            return {'enabled': False}
        return {
            'enabled': True,
            'cache_entries': len(self.cache),
            'max_cache_size': self.max_cache_size
        }

class Scheduler:
    def __init__(self, agent, budget: ResourceBudget, cache_file: Optional[str] = None, 
                 enable_early_execution: bool = False, result_callback=None, 
                 enable_function_coalescing: bool = False,
                 max_shared_task_age: int = 100,
                 enable_aggressive_cleanup: bool = True):
        self.agent = agent
        self.budget = budget
        self.enable_early_execution = enable_early_execution
        self.result_callback = result_callback
        self.enable_function_coalescing = enable_function_coalescing
        self.enable_aggressive_cleanup = enable_aggressive_cleanup
        
        self.cache_file = cache_file
        self.synthesis_optimizer = SemanticMapSynthesisOptimizer()
        
        # 任务存储
        self.map_tasks: Dict[str, MapTask] = {}
        self.completed_questions: Dict[str, Dict] = {}
        
        # 全局共享管理器 - 跨问题共享执行结果
        self.shared_task_manager = SharedTaskManager()
        
        # 缓存管理器
        self.cache_manager = CacheManager(cache_file)
        
        # Stream execution 相关（仅用于 early_execution）
        self.stream_executed_results: Dict[str, Any] = {}
        self.stream_executed_tasks: Set[str] = set()
        self.stream_pending_tasks: Dict[str, asyncio.Task] = {}
        self.stream_task_events: Dict[str, asyncio.Event] = {}
        
        # 队列和池
        self.pending_questions = deque()
        self.processing_pool: Set[str] = set()
        self.pending_planner_tasks = deque()
        self.running_planner_tasks: Set[str] = set()
        self.ready_map_tasks = deque()
        self.running_map_tasks: Set[str] = set()
        self.ready_reduce_tasks = deque()
        self.running_reduce_tasks: Set[str] = set()
        
        # 全局统计
        self.stats = {
            'total_questions': 0,
            'completed_questions': 0,
            'peak_shared_tasks': 0
        }
        
        self.semaphore = asyncio.Semaphore(budget.max_concurrent)

    def _ensure_question_callbacks(self, map_task: MapTask):
        if not getattr(self.agent, "benchmark", False):
            return
        if map_task.planner_callback is None and getattr(self.agent, "planner_callback", None) is not None:
            cb_cls = type(self.agent.planner_callback)
            map_task.planner_callback = cb_cls(stream=getattr(self.agent.planner_callback, "stream", False))
        if map_task.executor_callback is None and getattr(self.agent, "executor_callback", None) is not None:
            cb_cls = type(self.agent.executor_callback)
            map_task.executor_callback = cb_cls(stream=getattr(self.agent.executor_callback, "stream", False))

    def _get_question_llm_stats(self, map_task: MapTask) -> Dict[str, Any]:
        planner_stats = map_task.planner_callback.get_stats() if map_task.planner_callback else {}
        executor_stats = map_task.executor_callback.get_stats() if map_task.executor_callback else {}
        total: Dict[str, Any] = {}
        for key in set(planner_stats) | set(executor_stats):
            planner_value = planner_stats.get(key, 0)
            executor_value = executor_stats.get(key, 0)
            if isinstance(planner_value, list) or isinstance(executor_value, list):
                total[key] = (
                    (planner_value if isinstance(planner_value, list) else [])
                    + (executor_value if isinstance(executor_value, list) else [])
                )
            elif isinstance(planner_value, (int, float)) and isinstance(executor_value, (int, float)):
                total[key] = planner_value + executor_value
        return {"planner": planner_stats, "executor": executor_stats, "total": total}

    def _get_task_key(self, task: Any) -> str:
        return create_function_signature(task)

    async def process_questions(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f" Starting processing for {len(questions)} questions")
        if self.enable_early_execution:
            print(" Early execution mode: ENABLED")
        print(" Shared task optimization: ENABLED")
        
        cache_stats = self.cache_manager.get_stats()
        if cache_stats.get('enabled', False):
            print(f" Cache optimization: ENABLED ({cache_stats['cache_entries']} entries)")
        else:
            print(" Cache optimization: DISABLED")
        
        self.stats['total_questions'] = len(questions)
        
        for q_data in questions:
            map_task = MapTask(
                question_id=q_data['id'], 
                question=q_data['question']
            )
            self.map_tasks[q_data['id']] = map_task
            self.pending_questions.append(q_data['id'])
        
        await self._scheduling_loop()
        
        # 打印汇总统计
        total_shared = sum(t.shared_task_hits for t in self.map_tasks.values())
        total_cached = sum(t.cached_task_hits for t in self.map_tasks.values())
        total_executed = sum(t.executed_tasks_count for t in self.map_tasks.values())
        total_early = sum(t.early_executed_tasks for t in self.map_tasks.values())
        
        print(f" Processing completed")
        print(f" Shared task stats: {total_shared} tasks reused from other questions")
        print(f" Cache hits: {total_cached} tasks")
        print(f" Early executed: {total_early} tasks (self, during planning)")
        print(f" Map executed: {total_executed} tasks")
        
        shared_stats = self.shared_task_manager.get_shared_task_stats()
        print(f" Shared task manager: {shared_stats['completed_tasks']} completed, "
              f"{shared_stats['running_tasks']} running, "
              f"{shared_stats['active_questions']} active questions")
        
        cache_stats = self.cache_manager.get_stats()
        if cache_stats.get('enabled', False):
            print(f" Cache final: {cache_stats['cache_entries']} entries")
        
        return self.completed_questions

    async def _scheduling_loop(self):
        while (self.pending_questions or self.processing_pool or 
               self.pending_planner_tasks or self.ready_map_tasks or self.ready_reduce_tasks or
               self.running_planner_tasks or self.running_map_tasks or self.running_reduce_tasks):
            
            await self._fill_processing_pool()
            await self._process_planner_stage()
            await self._process_execution_stage()
            await self._process_join_stage()
            
            shared_stats = self.shared_task_manager.get_shared_task_stats()
            peak = shared_stats['completed_tasks'] + shared_stats['running_tasks']
            self.stats['peak_shared_tasks'] = max(self.stats['peak_shared_tasks'], peak)
            
            await asyncio.sleep(0.001)

    async def _fill_processing_pool(self):
        while (len(self.processing_pool) < self.budget.max_questions and 
               self.pending_questions):
            qid = self.pending_questions.popleft()
            self.processing_pool.add(qid)
            self.pending_planner_tasks.append(qid)
            self.map_tasks[qid].start_time = time.time()
            print(f" Added question {qid} to processing pool")

    async def _process_planner_stage(self):
        available_slots = self.budget.max_concurrent - len(self.running_planner_tasks)
        while available_slots > 0 and self.pending_planner_tasks:
            qid = self.pending_planner_tasks.popleft()
            async with self.semaphore:
                asyncio.create_task(self._execute_planner_task(qid))
                self.running_planner_tasks.add(qid)
                available_slots -= 1

    async def _process_execution_stage(self):
        available_slots = self.budget.max_concurrent - (len(self.running_map_tasks) + len(self.running_reduce_tasks))
        while available_slots > 0 and self.ready_map_tasks:
            qid = self.ready_map_tasks.popleft()
            async with self.semaphore:
                asyncio.create_task(self._execute_map_task(qid))
                self.running_map_tasks.add(qid)
                available_slots -= 1

    async def _process_join_stage(self):
        available_slots = self.budget.max_concurrent - (len(self.running_map_tasks) + len(self.running_reduce_tasks))
        while available_slots > 0 and self.ready_reduce_tasks:
            qid = self.ready_reduce_tasks.popleft()
            async with self.semaphore:
                asyncio.create_task(self._execute_reduce_task(qid))
                self.running_reduce_tasks.add(qid)
                available_slots -= 1

    async def _execute_planner_task(self, question_id: str):
        map_task = self.map_tasks[question_id]
        
        try:
            self._ensure_question_callbacks(map_task)
            map_task.plan_start_time = time.time()
            print(f" Planning question {question_id}: {map_task.question[:50]}...")
            
            inputs = {"input": map_task.question}
            
            if self.enable_early_execution and hasattr(self.agent.planner, 'aplan'):
                plan, leaf_num = await self._execute_planner_with_stream(question_id, inputs)
                map_task.leaf_task_num = leaf_num
            else:
                callbacks = [map_task.planner_callback] if map_task.planner_callback else None
                plan = await self.agent.planner.plan(inputs=inputs, is_replan=False, callbacks=callbacks)

            map_task.plan = plan
            
            if self.enable_function_coalescing:
                optimized_plan = await self._apply_tool_synthesis(question_id, map_task.plan)
                map_task.optimized_plan = optimized_plan
            else:
                map_task.optimized_plan = plan
            self.ready_map_tasks.append(question_id)
            map_task.plan_end_time = time.time()
            print(f" Planned question {question_id}, ready for map")
                
        except Exception as e:
            print(f" Planner failed for {question_id}: {str(e)}")
            traceback.print_exc()
            await self._record_failed_question(question_id, f"Planner Error: {str(e)}")
            self.processing_pool.discard(question_id)
        
        finally:
            self.running_planner_tasks.discard(question_id)

    async def _execute_planner_with_stream(self, question_id: str, inputs: Dict[str, Any], max_retry_num=3) -> Tuple[Dict[int, Any], int]:
        """
        带 early_execution 的规划
        注意：这里执行的任务都是当前问题自己的任务，不算 shared_task_hits
        """
        print(f" Stream planning for question {question_id}")
        
        map_task = self.map_tasks[question_id]
        self._ensure_question_callbacks(map_task)
        count = 0
        
        for i in range(max_retry_num):
            if count > 0: 
                break
                
            collected_tasks = {}
            task_queue = asyncio.Queue()
            
            if count == 0 and i == 0:
                planning_task = asyncio.create_task(
                    self.agent.planner.aplan(
                        inputs=inputs,
                        task_queue=task_queue,
                        is_replan=False,
                        callbacks=[map_task.planner_callback] if map_task.planner_callback else None,
                    )
                )
            else:
                print('Trying replanning!')
                planning_task = asyncio.create_task(
                    self.agent.planner.aplan(
                        inputs={'input': inputs['input'], 'context': 
            '(Please ensure that there is at least one task (e.g., search) in the plan that does not rely on the output of other tasks)'},
                        task_queue=task_queue,
                        is_replan=True,
                        callbacks=[map_task.planner_callback] if map_task.planner_callback else None,
                    )
                )
                
            try:
                while True:
                    try:
                        task_data = await asyncio.wait_for(task_queue.get(), timeout=0.1)
                        if task_data is None:
                            break
                        
                        task_id = len(collected_tasks) + 1
                        if hasattr(task_data, 'idx'):
                            task_data.idx = task_id
                        collected_tasks[task_id] = task_data
                        
                        # 检查是否是叶子节点
                        if self._is_leaf_task(task_data):
                            count += 1
                            task_key = self._get_task_key(task_data)
                            
                            # 1. 先检查缓存
                            cached_result = self.cache_manager.get_cached_result(task_data)
                            if cached_result is not None:
                                task_data.observation = cached_result
                                self.stream_executed_results[task_key] = cached_result
                                self.stream_executed_tasks.add(task_key)
                                map_task.cached_task_hits += 1  # 缓存命中
                                print(f"    Using cached result for: {task_key}")
                                continue
                            
                            # 2. 检查其他问题执行过的任务 - 这才是真正的 shared_task_hits!
                            if self.shared_task_manager.is_task_completed(task_key):
                                owner = self.shared_task_manager.get_task_owner(task_key)
                                if owner and owner != question_id:  # 必须是其他问题执行的
                                    result = self.shared_task_manager.get_task_result(task_key)
                                    if result is not None:
                                        task_data.observation = result
                                        self.stream_executed_results[task_key] = result
                                        self.stream_executed_tasks.add(task_key)
                                        map_task.shared_task_hits += 1  # 从其他问题复用
                                        print(f"    Reusing shared task from question {owner}: {task_key}")
                                        continue
                            
                            elif self.shared_task_manager.is_task_running(task_key):
                                owner = self.shared_task_manager.get_task_owner(task_key)
                                if owner and owner != question_id:  # 必须是其他问题执行的
                                    # 正在运行的任务，标记为共享命中
                                    map_task.shared_task_hits += 1
                                    print(f"    Will reuse running shared task from question {owner}: {task_key}")
                                    # 不立即获取结果，等后续 wait
                            
                            # 3. 需要执行新任务 - 这是当前问题自己的 early_execution
                            # 注意：这里不是 shared_task_hits，是当前问题自己的执行
                            is_new = self.shared_task_manager.register_task(task_key, question_id)
                            if is_new:
                                # 记录该问题拥有的任务
                                map_task.owned_task_keys.add(task_key)
                                map_task.early_executed_task_keys.add(task_key)  # 记录是提前执行的
                                map_task.early_executed_tasks += 1  # 统计提前执行的任务数
                                
                                event = asyncio.Event()
                                self.stream_task_events[task_key] = event
                                
                                task_coro = self._execute_leaf_task_async(task_data, task_key, event, question_id)
                                self.shared_task_manager.create_task(task_key, task_coro, question_id)
                                self.stream_pending_tasks[task_key] = self.shared_task_manager.task_registry[task_key]
                                self.stream_executed_tasks.add(task_key)
                                print(f"    Early executing own task: {task_key}")
                            else:
                                # 这种情况不应该发生，因为前面已经检查过 is_task_completed 和 is_task_running
                                print(f"    Warning: Task {task_key} already registered but not completed/running?")
                        
                        task_queue.task_done()
                        
                    except asyncio.TimeoutError:
                        if planning_task.done():
                            break
                        continue
                        
            except Exception as e:
                print(f"  Stream planning interrupted: {e}")
                planning_task.cancel()
                raise
            
            if not planning_task.done():
                try:
                    await asyncio.wait_for(planning_task, timeout=1.0)
                except asyncio.TimeoutError:
                    planning_task.cancel()
        
        print(f" Stream planning completed: {len(collected_tasks)} tasks, early executed: {map_task.early_executed_tasks}")
        return collected_tasks, count

    async def _execute_leaf_task_async(self, task: Any, task_key: str, event: asyncio.Event, owner_id: str):
        """执行叶子任务"""
        try:
            if hasattr(task, 'tool') and callable(task.tool):
                print(f"    Async executing leaf task {task_key} (owner: {owner_id})")
                
                cached_result = self.cache_manager.get_cached_result(task)
                if cached_result is not None:
                    result = cached_result
                    print(f"       Late cache hit for {task_key}")
                else:
                    if asyncio.iscoroutinefunction(task.tool):
                        result = await task.tool(*task.args)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, task.tool, *task.args)
                    
                    self.cache_manager.set_cached_result(task, result)
                
                self.shared_task_manager.set_task_result(task_key, result)
                self.stream_executed_results[task_key] = result
                task.observation = result
                print(f"       Leaf task {task_key} executed: {str(result)[:100]}...")
                
        except Exception as e:
            error_msg = f"Leaf task execution error: {e}"
            self.shared_task_manager.set_task_error(task_key, e)
            self.stream_executed_results[task_key] = error_msg
            task.observation = error_msg
            print(f"       Leaf task {task_key} failed: {e}")
        
        finally:
            event.set()
            if task_key in self.stream_pending_tasks:
                del self.stream_pending_tasks[task_key]
            if task_key in self.stream_task_events:
                del self.stream_task_events[task_key]

    async def _execute_map_task(self, question_id: str):
        """执行 map 阶段 - 主要执行逻辑"""
        map_task = self.map_tasks[question_id]
        
        try:
            map_task.map_start_time = time.time()
            print(f"  Mapping question {question_id}")
            
            plan = map_task.optimized_plan or map_task.plan
            if not plan:
                raise ValueError("No plan available for execution")
            
            # 重置当前问题的 map 阶段统计
            map_task.shared_task_hits = 0  # 注意：early_execution 已经加过 shared_task_hits，这里不要重置！
            map_task.cached_task_hits = 0  # 重置缓存命中（map阶段自己的缓存命中）
            map_task.executed_tasks_count = 0  # map阶段执行的任务数
            
            tf_unit = TaskFetchingUnit()
            runtime_batch_enabled = str(os.getenv("RUNTIME_SEMANTIC_BATCH", "1")).strip().lower() not in {"0", "false", "off", "no"}
            runtime_batch_window_ms = int(os.getenv("RUNTIME_SEMANTIC_BATCH_WINDOW_MS", "150"))
            tf_unit.set_execution_context(
                question=map_task.question,
                metadata={
                    "question_id": question_id,
                    "runtime_semantic_batch": runtime_batch_enabled,
                    "runtime_semantic_batch_window_ms": runtime_batch_window_ms,
                },
            )
            
            # 设置 stream mode 数据（如果启用了 early_execution）
            if self.enable_early_execution:
                stream_task_map = {}
                for task_id, task in plan.items():
                    stream_task_map[task_id] = task
                
                tf_unit.set_stream_mode_data(
                    executed_tasks=self.stream_executed_tasks,
                    executed_results=self.stream_executed_results,
                    task_map=stream_task_map,
                    task_events=self.stream_task_events,
                    shared_task_manager=self.shared_task_manager,
                    cache_manager=self.cache_manager
                )
            
            # 遍历所有任务，决定如何执行
            tasks_to_execute = {}
            
            for task_id, task in plan.items():
                if (hasattr(task, 'tool') and callable(task.tool) and not getattr(task, 'is_join', False)):
                    task_key = self._get_task_key(task)
                    
                    # 检查是否已经在 early_execution 中执行过了
                    if self.enable_early_execution and task_key in map_task.early_executed_task_keys:
                        # 任务已经在规划阶段执行过了，直接使用结果
                        if task_key in self.stream_executed_results:
                            task.observation = self.stream_executed_results[task_key]
                            print(f"    Using early executed result for task {task_id}")
                            continue
                    
                    # 1. 检查缓存
                    cached_result = self.cache_manager.get_cached_result(task)
                    if cached_result is not None:
                        task.observation = cached_result
                        map_task.cached_task_hits += 1
                        print(f"    Using cached result for task {task_id}")
                        continue
                    
                    # 2. 检查共享任务管理器 - 只有其他问题执行的任务才算 shared_task_hits
                    if self.shared_task_manager.is_task_completed(task_key):
                        owner = self.shared_task_manager.get_task_owner(task_key)
                        if owner and owner != question_id:  # 必须是其他问题执行的
                            result = self.shared_task_manager.get_task_result(task_key)
                            if result is not None:
                                task.observation = result
                                map_task.shared_task_hits += 1  # 从其他问题复用
                                print(f"    Reusing shared task from question {owner}: {task_id}")
                                continue
                            else:
                                error = self.shared_task_manager.get_task_error(task_key)
                                if error:
                                    task.observation = f"Shared task error: {error}"
                                    print(f"     Shared task had error for {task_id}: {error}")
                                    continue
                        else:
                            # 如果是自己执行过的任务，直接复用但不计数
                            result = self.shared_task_manager.get_task_result(task_key)
                            if result is not None:
                                task.observation = result
                                print(f"    Reusing own previously executed task: {task_id}")
                                continue
                    
                    elif self.shared_task_manager.is_task_running(task_key):
                        owner = self.shared_task_manager.get_task_owner(task_key)
                        if owner and owner != question_id:  # 必须是其他问题执行的
                            # 等待正在运行的任务
                            print(f"    Waiting for running shared task from question {owner}: {task_id}")
                            success = await self.shared_task_manager.wait_for_task(task_key, timeout=30.0)
                            if success:
                                result = self.shared_task_manager.get_task_result(task_key)
                                task.observation = result
                                map_task.shared_task_hits += 1  # 从其他问题复用
                                print(f"     Got result from shared task {task_id}")
                                continue
                            else:
                                task.observation = "Timeout waiting for shared task"
                                print(f"     Timeout waiting for shared task {task_id}")
                                continue
                        else:
                            # 等待自己的任务
                            print(f"    Waiting for own running task: {task_id}")
                            success = await self.shared_task_manager.wait_for_task(task_key, timeout=30.0)
                            if success:
                                result = self.shared_task_manager.get_task_result(task_key)
                                task.observation = result
                                print(f"     Got result from own task {task_id}")
                                continue
                    
                    # 3. 需要执行的新任务
                    is_new = self.shared_task_manager.register_task(task_key, question_id)
                    if is_new:
                        tasks_to_execute[task_id] = task
                        map_task.owned_task_keys.add(task_key)
                    else:
                        # 理论上不应该走到这里
                        print(f"     Task {task_id} should have been handled above")
                        tasks_to_execute[task_id] = task
                        map_task.owned_task_keys.add(task_key)
            
            # 执行需要执行的任务
            if tasks_to_execute:
                print(f"    Executing {len(tasks_to_execute)} new tasks with dependency-aware scheduling")
                map_task.executed_tasks_count += len(tasks_to_execute)
                tf_unit.set_tasks(tasks_to_execute)
                await tf_unit.schedule()
                
                # 将执行结果存入共享管理器
                for task_id, task in tasks_to_execute.items():
                    if hasattr(task, 'observation') and task.observation is not None:
                        task_key = self._get_task_key(task)
                        self.shared_task_manager.set_task_result(task_key, task.observation)
                        self.cache_manager.set_cached_result(task, task.observation)
            else:
                print(f"    All tasks already cached, shared, or early executed")
            
            self.ready_reduce_tasks.append(question_id)
            map_task.map_end_time = time.time()
            print(f" Mapped question {question_id}, ready for reduce "
                  f"(shared from others: {map_task.shared_task_hits}, "
                  f"cached: {map_task.cached_task_hits}, "
                  f"early: {map_task.early_executed_tasks}, "
                  f"map executed: {map_task.executed_tasks_count})")
            
        except Exception as e:
            print(f" Map failed for {question_id}: {str(e)}")
            traceback.print_exc()
            await self._record_failed_question(question_id, f"Map Error: {str(e)}")
            self.processing_pool.discard(question_id)
        
        finally:
            self.running_map_tasks.discard(question_id)

    async def _execute_reduce_task(self, question_id: str):
        """reduce 阶段 - 生成最终答案并清理资源"""
        map_task = self.map_tasks[question_id]
        
        try:
            map_task.reduce_start_time = time.time()
            print(f" Reducing question {question_id}")
            plan = map_task.optimized_plan or map_task.plan
            
            if not plan:
                raise ValueError("No plan available for execution")
            
            # 确保所有任务都有结果
            await self._ensure_all_tasks_have_results(plan, question_id)
            
            # 生成答案
            print(f"    Generating final answer for {question_id}")
            answer = await self._call_join_function(question_id, plan, map_task.question)
            total_time = time.time() - map_task.start_time
            map_task.end_time = time.time()
            map_task.reduce_end_time = map_task.end_time

            plan_time = (
                (map_task.plan_end_time - map_task.plan_start_time)
                if map_task.plan_start_time is not None and map_task.plan_end_time is not None
                else 0.0
            )
            map_time = (
                (map_task.map_end_time - map_task.map_start_time)
                if map_task.map_start_time is not None and map_task.map_end_time is not None
                else 0.0
            )
            reduce_time = (
                (map_task.reduce_end_time - map_task.reduce_start_time)
                if map_task.reduce_start_time is not None and map_task.reduce_end_time is not None
                else 0.0
            )
            join_time = float(getattr(map_task, "join_duration", 0.0) or 0.0)
            non_join_reduce_time = max(0.0, reduce_time - join_time)
            other_time = max(0.0, total_time - (plan_time + map_time + reduce_time))
            
            # 收集统计数据
            original_tasks = len(map_task.plan) if map_task.plan else 0
            optimized_tasks = len(plan)
            executed_tasks = map_task.executed_tasks_count
            shared_hits = map_task.shared_task_hits  # 只统计从其他问题复用的任务
            cached_hits = map_task.cached_task_hits
            early_tasks = map_task.early_executed_tasks
            leaf_tasks = getattr(map_task, 'leaf_task_num', 0)
            llm_stats = self._get_question_llm_stats(map_task)
            
            result_data = {
                "id": question_id,
                "question": map_task.question,
                "answer": answer,
                "time": total_time,
                "status": "success",
                "start_time": map_task.start_time,
                "end_time": map_task.end_time,
                "original_tasks": original_tasks,
                "optimized_tasks": optimized_tasks,
                "executed_tasks": executed_tasks,
                "early_executed_tasks": early_tasks,
                "shared_task_hits": shared_hits,
                "cached_tasks": cached_hits,
                "leaf_tasks": leaf_tasks,
                "owned_tasks": len(map_task.owned_task_keys),
                "plan_time": plan_time,
                "map_time": map_time,
                "reduce_time": reduce_time,
                "join_time": join_time,
                "non_join_reduce_time": non_join_reduce_time,
                "other_time": other_time,
                "stats": llm_stats,
            }
            
            self.completed_questions[question_id] = result_data
            self.stats['completed_questions'] += 1
            
            if self.result_callback:
                self.result_callback(question_id, result_data)
            
            print(f" Reduced question {question_id} in {total_time:.2f}s "
                  f"(executed: {executed_tasks}, early: {early_tasks}, "
                  f"shared from others: {shared_hits}, cached: {cached_hits})")
            print(
                f"    Stage timing for {question_id}: "
                f"plan={plan_time:.3f}s, map={map_time:.3f}s, "
                f"reduce={reduce_time:.3f}s (join={join_time:.3f}s, non_join={non_join_reduce_time:.3f}s), "
                f"other={other_time:.3f}s, total={total_time:.3f}s"
            )
            
            # 清理资源
            if self.enable_aggressive_cleanup:
                print(f"    Cleaning up resources for question {question_id}")
                self._cleanup_question_resources(question_id)
            
        except Exception as e:
            print(f" Reduce failed for {question_id}: {str(e)}")
            traceback.print_exc()
            await self._record_failed_question(question_id, f"Reduce Error: {str(e)}")
        
        finally:
            self.running_reduce_tasks.discard(question_id)
            self.processing_pool.discard(question_id)

    def _cleanup_question_resources(self, question_id: str):
        """清理问题产生的所有资源"""
        # 1. 从共享任务管理器中清理该问题拥有的任务
        self.shared_task_manager.cleanup_question_tasks(question_id)
        
        # 2. 清理 stream 执行相关资源
        if self.enable_early_execution:
            task_keys_to_remove = []
            for task_key in self.stream_executed_tasks:
                owner = self.shared_task_manager.get_task_owner(task_key)
                if owner == question_id:
                    task_keys_to_remove.append(task_key)
            
            for task_key in task_keys_to_remove:
                if task_key in self.stream_executed_results:
                    del self.stream_executed_results[task_key]
                if task_key in self.stream_executed_tasks:
                    self.stream_executed_tasks.discard(task_key)
            
            if task_keys_to_remove:
                print(f"    Cleaned up {len(task_keys_to_remove)} stream tasks")
        
        # 3. 清理 map_task 中的任务集合
        if question_id in self.map_tasks:
            self.map_tasks[question_id].owned_task_keys.clear()
            self.map_tasks[question_id].early_executed_task_keys.clear()

    async def _wait_for_stream_tasks(self, plan: Dict[int, Any]):
        """等待 stream 任务完成（仅 early_execution）"""
        if not self.enable_early_execution:
            return
            
        tasks_to_wait = []
        for task_id, task in plan.items():
            if hasattr(task, 'tool') and callable(task.tool) and not getattr(task, 'is_join', False):
                task_key = self._get_task_key(task)
                if task_key in self.stream_task_events:
                    tasks_to_wait.append(self.stream_task_events[task_key].wait())
                elif self.shared_task_manager.is_task_running(task_key):
                    tasks_to_wait.append(self.shared_task_manager.wait_for_task(task_key))
        
        if tasks_to_wait:
            print(f"    Waiting for {len(tasks_to_wait)} stream/shared tasks to complete...")
            await asyncio.gather(*tasks_to_wait)
            print(f"    All stream/shared tasks completed")

    async def _ensure_all_tasks_have_results(self, plan: Dict[int, Any], question_id: str):
        """确保所有任务都有结果"""
        map_task = self.map_tasks[question_id]
        
        await self._wait_for_stream_tasks(plan)
        
        missing_results = []
        for task_id, task in plan.items():
            if hasattr(task, 'tool') and callable(task.tool) and not getattr(task, 'is_join', False):
                task_key = self._get_task_key(task)
                
                # 1. 检查缓存
                cached_result = self.cache_manager.get_cached_result(task)
                if cached_result is not None:
                    task.observation = cached_result
                    continue
                
                # 2. 检查共享任务
                if self.shared_task_manager.is_task_completed(task_key):
                    result = self.shared_task_manager.get_task_result(task_key)
                    if result is not None:
                        task.observation = result
                        continue
                
                # 3. 检查 stream 结果
                if self.enable_early_execution:
                    if task_key in self.stream_executed_results:
                        task.observation = self.stream_executed_results[task_key]
                        continue
                
                # 4. 仍然缺失
                if not hasattr(task, 'observation') or task.observation is None:
                    missing_results.append(f"{task_id}:{getattr(task, 'name', 'unknown')}")
        
        if missing_results:
            print(f"     Missing results for {len(missing_results)} tasks: {missing_results}")
            for task_id, task in plan.items():
                if hasattr(task, 'tool') and callable(task.tool) and not getattr(task, 'is_join', False):
                    if not hasattr(task, 'observation') or task.observation is None:
                        task.observation = f"Task {getattr(task, 'name', 'unknown')} result missing"

    def _is_leaf_task(self, task: Any) -> bool:
        """判断是否为叶子任务"""
        return (hasattr(task, 'tool') and callable(task.tool) and 
                not getattr(task, 'is_join', False) and 
                not getattr(task, 'dependencies', []))

    async def _apply_tool_synthesis(self, question_id: str, plan: Dict[int, Any]) -> Dict[int, Any]:
        """应用工具链优化"""
        try:
            batch_plan_data = {
                question_id: {
                    'plan': plan,
                    'question': self.map_tasks[question_id].question
                }
            }
            
            synthesis_result = await self.synthesis_optimizer.optimize_tool_chains(batch_plan_data)
            optimized_plans = synthesis_result.get('optimized_plans', {})
            optimized_plan = optimized_plans.get(question_id, plan)
            
            if optimized_plan and len(optimized_plan) < len(plan):
                savings = len(plan) - len(optimized_plan)
                print(f" Tool synthesis: {len(plan)} → {len(optimized_plan)} tasks (saved {savings})")
            
            return optimized_plan
            
        except Exception as e:
            print(f"  Tool synthesis failed for {question_id}: {e}")
            return plan

    async def _call_join_function(self, qid: str, tasks: Dict[str, Any], question: str) -> str:
        """调用 join 函数生成答案"""
        try:
            agent_scratchpad = self._build_agent_scratchpad(tasks)
            if hasattr(self.agent, 'join'):
                map_task = self.map_tasks.get(qid)
                if map_task is not None:
                    self._ensure_question_callbacks(map_task)
                callbacks = (
                    [map_task.executor_callback]
                    if map_task is not None and map_task.executor_callback is not None
                    else None
                )
                join_start = time.time()
                thought, answer, is_replan = await self.agent.join(
                    input_query=question,
                    agent_scratchpad=agent_scratchpad,
                    is_final=True,
                    callbacks=callbacks,
                )
                join_elapsed = time.time() - join_start
                if map_task is not None:
                    map_task.join_duration = float(join_elapsed)
                print(f"    Join duration for {qid}: {join_elapsed:.3f}s")
                return answer if answer else "No answer generated"
            else:
                return "Join function not available"
        except Exception as e:
            return f"Join error: {str(e)}"

    def _build_agent_scratchpad(self, tasks: Dict[int, Any]) -> str:
        """构建 agent scratchpad"""
        scratchpad = "\n\n"
        if tasks:
            sorted_tasks = sorted(tasks.values(), key=lambda t: getattr(t, 'idx', 0))
            for task in sorted_tasks:
                if not getattr(task, 'is_join', False):
                    task_str = task.get_though_action_observation(
                        include_action=True, include_thought=True, 
                    )
                    scratchpad += task_str
        return scratchpad.strip()

    async def _record_failed_question(self, question_id: str, error_msg: str):
        """记录失败的问题并清理资源"""
        map_task = self.map_tasks[question_id]
        total_time = time.time() - map_task.start_time
        
        result_data = {
            "id": question_id,
            "question": map_task.question,
            "answer": error_msg,
            "time": total_time,
            "status": "error",
            "start_time": map_task.start_time,
            "end_time": time.time(),
            "cached_tasks": getattr(map_task, 'cached_task_hits', 0),
            "shared_task_hits": getattr(map_task, 'shared_task_hits', 0),
            "early_executed_tasks": getattr(map_task, 'early_executed_tasks', 0),
            "executed_tasks": getattr(map_task, 'executed_tasks_count', 0),
            "owned_tasks": len(getattr(map_task, 'owned_task_keys', set())),
            "stats": self._get_question_llm_stats(map_task),
        }
        
        self.completed_questions[question_id] = result_data
        self.stats['completed_questions'] += 1
        
        if self.result_callback:
            self.result_callback(question_id, result_data)
        
        if self.enable_aggressive_cleanup:
            print(f"    Cleaning up resources for failed question {question_id}")
            self._cleanup_question_resources(question_id)
            
        self.processing_pool.discard(question_id)
