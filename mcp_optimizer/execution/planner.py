"""
执行规划器 - LLMCompiler风格的DAG规划
基于论文: An LLM Compiler for Parallel Function Calling (ICML 2024)
"""
from typing import List, Dict, Any, Set, Tuple
import networkx as nx
from pydantic import BaseModel
from ..core.tool import Tool, ToolCall


class TaskNode(BaseModel):
    """任务节点"""
    task_id: str
    tool_id: str
    parameters: Dict[str, Any]
    dependencies: List[str] = []  # 依赖的task_id列表
    
    class Config:
        arbitrary_types_allowed = True


class ExecutionPlan(BaseModel):
    """执行计划"""
    tasks: List[TaskNode]
    dag: Any = None  # networkx.DiGraph,用于拓扑排序
    
    class Config:
        arbitrary_types_allowed = True


class ExecutionPlanner:
    """
    执行规划器
    
    功能:
    1. 解析LLM生成的工具调用序列
    2. 分析依赖关系,构建DAG
    3. 识别可并行执行的任务
    """
    
    def __init__(self):
        pass
    
    def create_plan(
        self,
        tool_calls: List[Dict[str, Any]],
        available_tools: Dict[str, Tool]
    ) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            tool_calls: LLM生成的工具调用列表,格式:
                [
                    {
                        "task_id": "t1",
                        "tool_id": "search_api",
                        "parameters": {"query": "AI news"},
                        "depends_on": []
                    },
                    {
                        "task_id": "t2",
                        "tool_id": "summarize",
                        "parameters": {"text": "$t1.result"},
                        "depends_on": ["t1"]
                    }
                ]
            available_tools: 可用工具字典
        
        Returns:
            ExecutionPlan对象
        """
        tasks = []
        dag = nx.DiGraph()
        
        # 构建任务节点
        for call in tool_calls:
            task_id = call["task_id"]
            tool_id = call["tool_id"]
            parameters = call["parameters"]
            depends_on = call.get("depends_on", [])
            
            # 验证工具存在
            if tool_id not in available_tools:
                raise ValueError(f"Tool {tool_id} not found in available tools")
            
            # 创建任务节点
            task = TaskNode(
                task_id=task_id,
                tool_id=tool_id,
                parameters=parameters,
                dependencies=depends_on
            )
            tasks.append(task)
            
            # 添加到DAG
            dag.add_node(task_id, task=task)
            for dep in depends_on:
                dag.add_edge(dep, task_id)
        
        # 检测循环依赖
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Circular dependency detected in execution plan")
        
        return ExecutionPlan(tasks=tasks, dag=dag)
    
    def get_parallel_batches(self, plan: ExecutionPlan) -> List[List[TaskNode]]:
        """
        获取可并行执行的任务批次
        
        使用拓扑排序,将任务分组为多个批次,
        同一批次内的任务可以并行执行
        
        Returns:
            List of batches, 每个batch是可并行执行的任务列表
        """
        dag = plan.dag
        batches = []
        
        # 计算每个节点的入度
        in_degree = {node: dag.in_degree(node) for node in dag.nodes()}
        
        # 当前可执行的任务(入度为0)
        ready = [node for node, degree in in_degree.items() if degree == 0]
        
        while ready:
            # 当前批次
            current_batch = []
            for task_id in ready:
                task = dag.nodes[task_id]["task"]
                current_batch.append(task)
            
            batches.append(current_batch)
            
            # 更新入度,找出下一批次
            next_ready = []
            for task_id in ready:
                for successor in dag.successors(task_id):
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        next_ready.append(successor)
            
            ready = next_ready
        
        return batches
    
    def estimate_speedup(self, plan: ExecutionPlan) -> float:
        """
        估算并行执行相比串行执行的加速比
        
        Returns:
            加速比 (串行时间 / 并行时间)
        """
        batches = self.get_parallel_batches(plan)
        
        # 串行执行时间 = 所有任务数量
        serial_time = len(plan.tasks)
        
        # 并行执行时间 = 批次数量(假设每个任务耗时相同)
        parallel_time = len(batches)
        
        if parallel_time == 0:
            return 1.0
        
        return serial_time / parallel_time
    
    def visualize_dag(self, plan: ExecutionPlan) -> str:
        """
        生成DAG的文本可视化
        
        Returns:
            DAG的文本表示
        """
        batches = self.get_parallel_batches(plan)
        
        output = ["Execution Plan (DAG):", "=" * 50]
        
        for i, batch in enumerate(batches):
            output.append(f"\nBatch {i + 1} (Parallel):")
            for task in batch:
                deps = f" <- {task.dependencies}" if task.dependencies else ""
                output.append(f"  - {task.task_id}: {task.tool_id}{deps}")
        
        output.append(f"\nEstimated Speedup: {self.estimate_speedup(plan):.2f}x")
        
        return "\n".join(output)
