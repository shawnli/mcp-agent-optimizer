"""
并行执行器 - 异步执行工具调用

基于LLMCompiler的并行执行思想
"""
import asyncio
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from ..core.tool import Tool, ToolCall, TaskStatus
from .planner import ExecutionPlan, TaskNode


class ParallelExecutor:
    """
    并行执行器
    
    功能:
    1. 异步并行执行无依赖的工具调用
    2. 管理依赖关系,确保执行顺序正确
    3. 错误处理和重试
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Args:
            max_workers: 最大并行工作线程数
            timeout: 单个工具调用的超时时间(秒)
            max_retries: 失败重试次数
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_plan(
        self,
        plan: ExecutionPlan,
        tools: Dict[str, Tool],
        planner
    ) -> Dict[str, Any]:
        """
        执行完整的执行计划
        
        Args:
            plan: ExecutionPlan对象
            tools: 可用工具字典 {tool_id: Tool}
            planner: ExecutionPlanner对象
        
        Returns:
            执行结果字典 {task_id: result}
        """
        # 获取并行批次
        batches = planner.get_parallel_batches(plan)
        
        # 存储结果
        results = {}
        
        # 逐批次执行
        for i, batch in enumerate(batches):
            print(f"Executing Batch {i + 1}/{len(batches)} ({len(batch)} tasks)")
            
            batch_results = self._execute_batch(batch, tools, results)
            results.update(batch_results)
        
        return results
    
    def _execute_batch(
        self,
        batch: List[TaskNode],
        tools: Dict[str, Tool],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        并行执行一个批次的任务
        
        Args:
            batch: 任务节点列表
            tools: 可用工具
            previous_results: 之前批次的结果(用于解析依赖)
        
        Returns:
            本批次的结果
        """
        futures = {}
        
        # 提交所有任务到线程池
        for task in batch:
            # 解析参数中的依赖引用
            resolved_params = self._resolve_parameters(
                task.parameters,
                previous_results
            )
            
            # 获取工具
            tool = tools.get(task.tool_id)
            if tool is None:
                print(f"Warning: Tool {task.tool_id} not found")
                continue
            
            # 提交执行
            future = self.executor.submit(
                self._execute_tool_with_retry,
                task.task_id,
                tool,
                resolved_params
            )
            futures[future] = task.task_id
        
        # 收集结果
        results = {}
        
        for future in as_completed(futures, timeout=self.timeout * len(batch)):
            task_id = futures[future]
            
            try:
                result = future.result(timeout=self.timeout)
                results[task_id] = result
                print(f"  ✓ {task_id} completed")
            except Exception as e:
                print(f"  ✗ {task_id} failed: {str(e)}")
                results[task_id] = {"error": str(e), "status": "failed"}
        
        return results
    
    def _execute_tool_with_retry(
        self,
        task_id: str,
        tool: Tool,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        执行工具,支持重试
        
        Args:
            task_id: 任务ID
            tool: Tool对象
            parameters: 参数
        
        Returns:
            执行结果
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # 执行工具
                result = tool.execute(**parameters)
                
                latency = (time.time() - start_time) * 1000  # ms
                
                return {
                    "status": "success",
                    "result": result,
                    "latency_ms": latency,
                    "attempts": attempt + 1
                }
            
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    # 指数退避
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        
        # 所有重试都失败
        return {
            "status": "failed",
            "error": str(last_error),
            "attempts": self.max_retries
        }
    
    def _resolve_parameters(
        self,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        解析参数中的依赖引用
        
        例如: {"text": "$t1.result"} -> {"text": <t1的实际结果>}
        
        Args:
            parameters: 原始参数
            previous_results: 之前的结果
        
        Returns:
            解析后的参数
        """
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                # 解析引用: $task_id.field
                ref = value[1:]  # 去掉$
                
                if "." in ref:
                    task_id, field = ref.split(".", 1)
                else:
                    task_id, field = ref, "result"
                
                # 查找引用的结果
                if task_id in previous_results:
                    task_result = previous_results[task_id]
                    
                    if isinstance(task_result, dict) and field in task_result:
                        resolved[key] = task_result[field]
                    else:
                        resolved[key] = task_result
                else:
                    # 引用未找到,保持原值
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved
    
    async def execute_plan_async(
        self,
        plan: ExecutionPlan,
        tools: Dict[str, Tool],
        planner
    ) -> Dict[str, Any]:
        """
        异步版本的执行计划
        
        使用asyncio而非ThreadPoolExecutor
        """
        batches = planner.get_parallel_batches(plan)
        results = {}
        
        for i, batch in enumerate(batches):
            print(f"Executing Batch {i + 1}/{len(batches)} (async)")
            
            # 创建异步任务
            tasks = []
            for task_node in batch:
                resolved_params = self._resolve_parameters(
                    task_node.parameters,
                    results
                )
                
                tool = tools.get(task_node.tool_id)
                if tool:
                    coro = self._execute_tool_async(
                        task_node.task_id,
                        tool,
                        resolved_params
                    )
                    tasks.append(coro)
            
            # 并行执行
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集结果
            for task_node, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results[task_node.task_id] = {
                        "status": "failed",
                        "error": str(result)
                    }
                else:
                    results[task_node.task_id] = result
        
        return results
    
    async def _execute_tool_async(
        self,
        task_id: str,
        tool: Tool,
        parameters: Dict[str, Any]
    ) -> Any:
        """异步执行工具"""
        try:
            start_time = time.time()
            
            # 在线程池中执行同步工具
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                tool.execute,
                **parameters
            )
            
            latency = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "result": result,
                "latency_ms": latency
            }
        
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            "max_workers": self.max_workers,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
