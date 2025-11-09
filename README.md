# MCP Agent Optimizer

大规模MCP服务集成的优化框架,实现了最新的算法和工程实践,用于构建高效、智能的AI Agent系统。

## 🌟 核心特性

### 算法优化

1. **分层语义路由 (Hierarchical Semantic Routing)**
   - 基于Tool-to-Agent Retrieval论文(2025)
   - 统一向量空间中的工具和服务检索
   - BM25 + Dense Vector混合检索
   - 在大规模工具场景下提升19.4%的准确率

2. **强化学习工具选择 (RL-based Tool Selection)**
   - **进步奖励 (Progress Reward)**: 对比相邻步骤的改进,解决稀疏奖励问题
   - **进程奖励模型 (Process Reward Model)**: 为每一步提供细粒度反馈
   - **GRPO算法**: Group Relative Policy Optimization,稳定高效的策略优化
   - 支持从专家演示学习(InversePRM)

3. **并行执行规划 (Parallel Execution Planning)**
   - 基于LLMCompiler论文(ICML 2024)
   - 自动构建工具调用依赖图(DAG)
   - 识别并并行执行无依赖任务
   - 最高可实现3.7倍加速

### 工程优化

1. **智能缓存系统**
   - 提示缓存:减少重复处理静态上下文
   - 工具调用缓存:避免重复执行相同参数的调用
   - LLM驱动的缓存决策:让模型自主判断是否使用缓存

2. **上下文工程**
   - 即时(Just-in-Time)上下文加载
   - 动态工具描述管理
   - 避免上下文腐化

3. **分布式架构支持**
   - 编排Agent + 工作Agent模式
   - 高内聚、低耦合
   - 易于扩展和维护

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/shawnli/mcp-agent-optimizer.git
cd mcp-agent-optimizer

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 🚀 快速开始

### 1. 分层语义路由

```python
from mcp_optimizer import HierarchicalRouter, MCPService, Tool

# 创建MCP服务和工具
email_service = MCPService(
    id="email_service",
    name="Email Service",
    description="Send and manage emails",
    category="communication"
)

search_tool = Tool(
    id="search_emails",
    name="Search Emails",
    description="Search emails by keywords",
    parent_service_id="email_service"
)

email_service.add_tool(search_tool)

# 创建路由器
router = HierarchicalRouter(
    services=[email_service],
    bm25_weight=0.3,
    dense_weight=0.7
)

# 路由查询到最相关的服务
query = "find emails about project updates"
top_services = router.route(query, top_k=3)

print(f"Top services: {[s.name for s in top_services]}")
```

### 2. 进步奖励计算

```python
from mcp_optimizer import ProgressReward

# 创建进步奖励计算器
progress_reward = ProgressReward()

# 定义状态
prev_state = {
    "completed_subtasks": 2,
    "total_subtasks": 5
}

curr_state = {
    "completed_subtasks": 3,
    "total_subtasks": 5
}

goal = "Complete all subtasks"

# 计算进步奖励
reward = progress_reward.compute_progress_reward(prev_state, curr_state, goal)
print(f"Progress reward: {reward}")  # 正值表示有进步
```

### 3. 并行执行规划

```python
from mcp_optimizer import ExecutionPlanner, ParallelExecutor

# 创建执行计划
planner = ExecutionPlanner()

tool_calls = [
    {
        "task_id": "t1",
        "tool_id": "search_api",
        "parameters": {"query": "AI news"},
        "depends_on": []
    },
    {
        "task_id": "t2",
        "tool_id": "search_api",
        "parameters": {"query": "ML papers"},
        "depends_on": []
    },
    {
        "task_id": "t3",
        "tool_id": "summarize",
        "parameters": {"text": "$t1.result"},
        "depends_on": ["t1"]
    }
]

plan = planner.create_plan(tool_calls, available_tools)

# 可视化DAG
print(planner.visualize_dag(plan))

# 并行执行
executor = ParallelExecutor(max_workers=10)
results = executor.execute_plan(plan, tools, planner)

print(f"Estimated speedup: {planner.estimate_speedup(plan):.2f}x")
```

### 4. 智能缓存

```python
from mcp_optimizer import IntelligentCache, ToolCallCache

# 创建缓存
tool_cache = ToolCallCache(max_size=5000, default_ttl=3600)
intelligent_cache = IntelligentCache(tool_cache)

# 执行工具调用(自动使用缓存)
result, from_cache = intelligent_cache.execute_with_cache(
    tool_id="search_api",
    parameters={"query": "AI news"},
    execute_fn=lambda **kwargs: search_api.execute(**kwargs),
    context="User wants latest AI news"
)

print(f"Result from cache: {from_cache}")
print(f"Cache stats: {tool_cache.get_stats()}")
```

## 📚 完整示例

查看`examples/`目录获取更多示例:

- `basic_routing.py`: 分层路由的完整示例
- `rl_tool_selection.py`: 强化学习工具选择
- `parallel_execution.py`: 并行执行优化

## 🧪 运行示例

```bash
# 分层路由示例
python examples/basic_routing.py

# RL工具选择示例
python examples/rl_tool_selection.py

# 并行执行示例
python examples/parallel_execution.py
```

## 📖 核心概念

### 进步奖励 (Progress Reward)

这是用户提出的创新想法,对应强化学习中的核心概念:

**理论基础**:
- **优势函数**: A(s,a) = Q(s,a) - V(s)
- **时序差分**: TD_error = r_t + γ*V(s_{t+1}) - V(s_t)
- **进程奖励**: 评估每一步的质量,而非只看最终结果

**实践价值**:
- ✅ 解决稀疏奖励问题
- ✅ 精确的信用分配
- ✅ 每一步都有即时反馈
- ✅ 更快的学习速度

**实现方式**:
```python
# 方法1: 状态价值对比
V_prev = value_model(state_t-1)
V_curr = value_model(state_t)
progress_reward = V_curr - V_prev

# 方法2: 目标距离对比
distance_prev = distance_to_goal(state_t-1, goal)
distance_curr = distance_to_goal(state_t, goal)
progress_reward = distance_prev - distance_curr

# 方法3: LLM评判
progress_score = llm_judge(state_t-1, state_t, goal)
progress_reward = normalize(progress_score)
```

### 分层语义路由

将扁平的工具列表重构为分层的知识图谱:

```
查询 → 检索Top-N实体(工具+服务) → 聚合到父服务 → 选择Top-K服务
```

**优势**:
- 搜索空间从O(N)降到O(log N)
- 避免上下文过载
- 更高的选择准确性

### 并行执行规划

"先规划,后执行"的范式:

```
LLM生成计划 → 构建DAG → 拓扑排序 → 批次并行执行
```

**优势**:
- 大幅降低延迟(最高3.7倍)
- 减少LLM调用次数(最高6.7倍)
- 全局视野,识别并行机会

## 🔬 技术细节

### 支持的算法

- **路由**: BM25, Dense Vector, Hybrid Retrieval
- **RL**: PPO, GRPO, Monte Carlo, GAE
- **奖励**: Outcome, Process, Progress, Contrastive, Shaped
- **缓存**: LRU, LFU, TTL, LLM-driven

### 性能指标

基于相关论文的实验结果:

| 指标 | 传统方法 | 优化后 | 提升 |
|:---|:---|:---|:---|
| 工具选择准确率 | 基线 | +19.4% | Tool-to-Agent Retrieval |
| 执行延迟 | 基线 | -73% (3.7x) | LLMCompiler |
| LLM调用成本 | 基线 | -85% (6.7x) | LLMCompiler |
| 样本效率 | 基线 | 显著提升 | AgentPRM |

## 📄 参考文献

本项目基于以下前沿研究:

1. **LLMCompiler** (ICML 2024): An LLM Compiler for Parallel Function Calling
2. **Tool-to-Agent Retrieval** (2025): Bridging Tools and Agents for Scalable LLM Multi-Agent Systems
3. **AgentPRM** (2025): Process Reward Models for LLM Agents
4. **LLM-dCache** (HiPC 2024): Improving tool-augmented LLMs with GPT-driven localized data caching
5. **VisTA** (2025): A Reinforcement Learning Framework for Visual Tool Selection

完整参考文献见[研究报告](docs/mcp_optimization_report.md)。

## 🤝 贡献

欢迎贡献!请查看[CONTRIBUTING.md](CONTRIBUTING.md)了解详情。

## 📝 许可证

MIT License

## 🙏 致谢

感谢所有相关论文的作者,以及提出"进步奖励"创新想法的用户。

## 📧 联系

- Issues: https://github.com/shawnli/mcp-agent-optimizer/issues
- Email: shawnli@example.com

---

**注意**: 这是一个研究原型,部分模块(如神经网络训练)需要根据实际需求进一步完善。
