"""
完整示例:整合所有优化策略

演示如何将分层路由、进步奖励、并行执行和智能缓存整合到一个Agent系统中
"""
from mcp_optimizer import (
    MCPService, Tool, ToolParameter,
    HierarchicalRouter,
    ExecutionPlanner, ParallelExecutor,
    ProgressReward, ProcessRewardModel, RewardShaper,
    ToolCallCache, IntelligentCache
)


def create_mock_services():
    """创建模拟的MCP服务和工具"""
    
    # 1. Email服务
    email_service = MCPService(
        id="email_service",
        name="Email Service",
        description="Send, receive and manage emails",
        category="communication"
    )
    
    email_service.add_tool(Tool(
        id="send_email",
        name="Send Email",
        description="Send an email to recipients",
        parameters=[
            ToolParameter(name="to", type="string", description="Recipient email"),
            ToolParameter(name="subject", type="string", description="Email subject"),
            ToolParameter(name="body", type="string", description="Email body")
        ],
        parent_service_id="email_service",
        execute_fn=lambda **kwargs: {"status": "sent", "message_id": "12345"}
    ))
    
    email_service.add_tool(Tool(
        id="search_emails",
        name="Search Emails",
        description="Search emails by keywords",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query")
        ],
        parent_service_id="email_service",
        execute_fn=lambda query: {"results": [f"Email about {query}"]}
    ))
    
    # 2. 数据分析服务
    data_service = MCPService(
        id="data_service",
        name="Data Analysis Service",
        description="Analyze and visualize data",
        category="data_analysis"
    )
    
    data_service.add_tool(Tool(
        id="load_data",
        name="Load Data",
        description="Load data from a source",
        parameters=[
            ToolParameter(name="source", type="string", description="Data source")
        ],
        parent_service_id="data_service",
        execute_fn=lambda source: {"data": [1, 2, 3, 4, 5], "rows": 5}
    ))
    
    data_service.add_tool(Tool(
        id="analyze_data",
        name="Analyze Data",
        description="Perform statistical analysis on data",
        parameters=[
            ToolParameter(name="data", type="array", description="Input data")
        ],
        parent_service_id="data_service",
        execute_fn=lambda data: {"mean": 3.0, "std": 1.41}
    ))
    
    # 3. 搜索服务
    search_service = MCPService(
        id="search_service",
        name="Web Search Service",
        description="Search the web for information",
        category="search"
    )
    
    search_service.add_tool(Tool(
        id="web_search",
        name="Web Search",
        description="Search the web",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query")
        ],
        parent_service_id="search_service",
        execute_fn=lambda query: {"results": [f"Result for {query}"]}
    ))
    
    return [email_service, data_service, search_service]


def demo_hierarchical_routing():
    """演示分层语义路由"""
    print("=" * 60)
    print("1. 分层语义路由演示")
    print("=" * 60)
    
    services = create_mock_services()
    router = HierarchicalRouter(services=services)
    
    queries = [
        "send an email to john@example.com",
        "analyze sales data from last month",
        "search for AI research papers"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        top_services = router.route(query, top_k=2)
        print(f"推荐服务:")
        for i, service in enumerate(top_services, 1):
            print(f"  {i}. {service.name} ({service.category})")


def demo_progress_reward():
    """演示进步奖励计算"""
    print("\n" + "=" * 60)
    print("2. 进步奖励演示")
    print("=" * 60)
    
    progress_reward = ProgressReward()
    
    # 模拟一个任务的多个步骤
    states = [
        {"completed_subtasks": 0, "total_subtasks": 5},
        {"completed_subtasks": 1, "total_subtasks": 5},
        {"completed_subtasks": 2, "total_subtasks": 5},
        {"completed_subtasks": 2, "total_subtasks": 5},  # 没有进步
        {"completed_subtasks": 4, "total_subtasks": 5},  # 大进步
    ]
    
    goal = "Complete all subtasks"
    
    print(f"\n目标: {goal}")
    print(f"初始状态: {states[0]}")
    
    for i in range(1, len(states)):
        reward = progress_reward.compute_progress_reward(
            states[i-1], states[i], goal
        )
        print(f"\n步骤 {i}:")
        print(f"  状态: {states[i]}")
        print(f"  进步奖励: {reward:.3f}")
        
        if reward > 0:
            print(f"  ✓ 有进步!")
        elif reward < 0:
            print(f"  ✗ 退步了!")
        else:
            print(f"  → 无变化")


def demo_parallel_execution():
    """演示并行执行规划"""
    print("\n" + "=" * 60)
    print("3. 并行执行规划演示")
    print("=" * 60)
    
    services = create_mock_services()
    tools = {}
    for service in services:
        for tool in service.tools:
            tools[tool.id] = tool
    
    # 定义一个复杂的工具调用序列
    tool_calls = [
        {
            "task_id": "t1",
            "tool_id": "web_search",
            "parameters": {"query": "AI news"},
            "depends_on": []
        },
        {
            "task_id": "t2",
            "tool_id": "web_search",
            "parameters": {"query": "ML papers"},
            "depends_on": []
        },
        {
            "task_id": "t3",
            "tool_id": "load_data",
            "parameters": {"source": "database"},
            "depends_on": []
        },
        {
            "task_id": "t4",
            "tool_id": "analyze_data",
            "parameters": {"data": "$t3.data"},
            "depends_on": ["t3"]
        },
        {
            "task_id": "t5",
            "tool_id": "send_email",
            "parameters": {
                "to": "user@example.com",
                "subject": "Analysis Results",
                "body": "$t4.result"
            },
            "depends_on": ["t1", "t2", "t4"]
        }
    ]
    
    # 创建执行计划
    planner = ExecutionPlanner()
    plan = planner.create_plan(tool_calls, tools)
    
    # 可视化DAG
    print("\n" + planner.visualize_dag(plan))
    
    # 执行计划
    print("\n开始执行...")
    executor = ParallelExecutor(max_workers=5)
    results = executor.execute_plan(plan, tools, planner)
    
    print(f"\n执行完成!")
    print(f"总任务数: {len(results)}")
    print(f"成功: {sum(1 for r in results.values() if r.get('status') == 'success')}")


def demo_intelligent_cache():
    """演示智能缓存"""
    print("\n" + "=" * 60)
    print("4. 智能缓存演示")
    print("=" * 60)
    
    services = create_mock_services()
    tools = {}
    for service in services:
        for tool in service.tools:
            tools[tool.id] = tool
    
    # 创建缓存
    tool_cache = ToolCallCache(max_size=100, default_ttl=3600)
    intelligent_cache = IntelligentCache(tool_cache)
    
    # 执行相同的搜索多次
    query = "AI news"
    
    print(f"\n执行搜索: {query}")
    
    for i in range(3):
        result, from_cache = intelligent_cache.execute_with_cache(
            tool_id="web_search",
            parameters={"query": query},
            execute_fn=tools["web_search"].execute,
            context="User wants latest news"
        )
        
        print(f"\n第 {i+1} 次调用:")
        print(f"  结果: {result}")
        print(f"  来自缓存: {from_cache}")
    
    # 显示缓存统计
    stats = tool_cache.get_stats()
    print(f"\n缓存统计:")
    print(f"  缓存大小: {stats['size']}")
    print(f"  命中次数: {stats['hits']}")
    print(f"  未命中次数: {stats['misses']}")
    print(f"  命中率: {stats['hit_rate']:.2%}")


def demo_reward_shaping():
    """演示奖励塑形"""
    print("\n" + "=" * 60)
    print("5. 奖励塑形演示")
    print("=" * 60)
    
    shaper = RewardShaper(
        outcome_weight=1.0,
        progress_weight=0.5,
        efficiency_weight=0.1,
        safety_weight=2.0
    )
    
    # 模拟一个任务序列
    scenarios = [
        {
            "name": "正常进步",
            "prev_state": {"completed_subtasks": 1, "total_subtasks": 5},
            "curr_state": {"completed_subtasks": 2, "total_subtasks": 5},
            "is_terminal": False,
            "outcome_reward": 0.0
        },
        {
            "name": "任务完成",
            "prev_state": {"completed_subtasks": 4, "total_subtasks": 5},
            "curr_state": {"completed_subtasks": 5, "total_subtasks": 5},
            "is_terminal": True,
            "outcome_reward": 10.0
        },
        {
            "name": "违反安全约束",
            "prev_state": {"completed_subtasks": 2, "total_subtasks": 5},
            "curr_state": {
                "completed_subtasks": 3,
                "total_subtasks": 5,
                "safety_violation": True
            },
            "is_terminal": False,
            "outcome_reward": 0.0
        }
    ]
    
    goal = "Complete all subtasks safely"
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        
        rewards = shaper.compute_shaped_reward(
            prev_state=scenario["prev_state"],
            curr_state=scenario["curr_state"],
            action={},
            goal=goal,
            is_terminal=scenario["is_terminal"],
            outcome_reward=scenario["outcome_reward"]
        )
        
        print(f"  奖励组成:")
        for key, value in rewards.items():
            if key != "total":
                print(f"    {key}: {value:.3f}")
        print(f"  总奖励: {rewards['total']:.3f}")


def main():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("MCP Agent Optimizer - 完整演示")
    print("=" * 60)
    
    demo_hierarchical_routing()
    demo_progress_reward()
    demo_parallel_execution()
    demo_intelligent_cache()
    demo_reward_shaping()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
