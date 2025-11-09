"""
MCP Agent Optimizer - 大规模MCP服务优化框架

实现了报告中的关键优化策略:
- 分层语义路由
- 强化学习工具选择(带进步奖励)
- 并行执行规划
- 智能缓存
"""

__version__ = "0.1.0"

from .core.tool import Tool, MCPService, ToolCall, ToolParameter
from .routing.hierarchical_router import HierarchicalRouter
from .execution.planner import ExecutionPlanner, ExecutionPlan
from .execution.executor import ParallelExecutor
from .rl.reward import ProgressReward, ProcessRewardModel, RewardShaper
from .rl.grpo import GRPO
from .engineering.cache import PromptCache, ToolCallCache, IntelligentCache

__all__ = [
    "Tool",
    "MCPService",
    "ToolCall",
    "ToolParameter",
    "HierarchicalRouter",
    "ExecutionPlanner",
    "ExecutionPlan",
    "ParallelExecutor",
    "ProgressReward",
    "ProcessRewardModel",
    "RewardShaper",
    "GRPO",
    "PromptCache",
    "ToolCallCache",
    "IntelligentCache",
]
