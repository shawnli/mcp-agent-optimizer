"""
工具(Tool)和MCP服务的核心定义
"""
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Optional[Any] = None


class Tool(BaseModel):
    """工具定义"""
    id: str = Field(..., description="工具唯一标识符")
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具功能描述")
    parameters: List[ToolParameter] = Field(default_factory=list)
    parent_service_id: str = Field(..., description="所属MCP服务ID")
    version: str = Field(default="1.0.0", description="工具版本")
    embedding: Optional[List[float]] = Field(default=None, description="工具描述的向量嵌入")
    
    # 实际执行函数(可选,用于模拟)
    execute_fn: Optional[Callable] = Field(default=None, exclude=True)
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具"""
        if self.execute_fn:
            return self.execute_fn(**kwargs)
        return {"status": "success", "result": f"Tool {self.name} executed with {kwargs}"}
    
    class Config:
        arbitrary_types_allowed = True


class MCPService(BaseModel):
    """MCP服务定义"""
    id: str = Field(..., description="服务唯一标识符")
    name: str = Field(..., description="服务名称")
    description: str = Field(..., description="服务功能描述")
    tools: List[Tool] = Field(default_factory=list)
    version: str = Field(default="1.0.0", description="服务版本")
    embedding: Optional[List[float]] = Field(default=None, description="服务描述的向量嵌入")
    
    # 服务级别的元数据
    category: Optional[str] = Field(default=None, description="服务分类,如'data_analysis', 'email', 'code'")
    tags: List[str] = Field(default_factory=list, description="服务标签")
    
    def add_tool(self, tool: Tool):
        """添加工具到服务"""
        tool.parent_service_id = self.id
        self.tools.append(tool)
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """根据ID获取工具"""
        for tool in self.tools:
            if tool.id == tool_id:
                return tool
        return None


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ToolCall(BaseModel):
    """工具调用记录"""
    tool_id: str
    service_id: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    status: TaskStatus = TaskStatus.PENDING
    latency_ms: Optional[float] = None
    error: Optional[str] = None
