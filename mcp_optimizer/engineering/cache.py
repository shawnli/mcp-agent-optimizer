"""
智能缓存系统

实现:
1. 提示缓存(Prompt Caching)
2. 工具调用缓存(Tool Call Caching)
3. LLM-dCache风格的智能缓存决策
"""
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    hit_count: int = 0
    ttl: Optional[int] = None  # Time-to-live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def update_hit(self):
        """更新命中次数"""
        self.hit_count += 1


class PromptCache:
    """
    提示缓存
    
    缓存静态的提示部分(如系统提示、工具描述)
    减少重复处理的token数量
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
    
    def _generate_key(self, prompt: str) -> str:
        """生成缓存键"""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[Any]:
        """获取缓存的提示"""
        key = self._generate_key(prompt)
        
        if key in self.cache:
            entry = self.cache[key]
            
            if not entry.is_expired():
                entry.update_hit()
                return entry.value
            else:
                # 过期,删除
                del self.cache[key]
        
        return None
    
    def set(
        self,
        prompt: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """设置缓存"""
        key = self._generate_key(prompt)
        
        # 如果缓存满了,使用LRU策略删除
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = CacheEntry(
            key=key,
            value=value,
            ttl=ttl
        )
    
    def _evict_lru(self):
        """驱逐最少使用的条目"""
        if not self.cache:
            return
        
        # 找到hit_count最小的条目
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].hit_count
        )
        del self.cache[lru_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_hits = sum(entry.hit_count for entry in self.cache.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "avg_hits": total_hits / len(self.cache) if self.cache else 0
        }


class ToolCallCache:
    """
    工具调用缓存
    
    缓存(tool_id, parameters) -> result的映射
    避免重复执行相同参数的工具调用
    """
    
    def __init__(
        self,
        max_size: int = 5000,
        default_ttl: int = 3600  # 1小时
    ):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # 统计
        self.hits = 0
        self.misses = 0
    
    def _generate_key(
        self,
        tool_id: str,
        parameters: Dict[str, Any]
    ) -> str:
        """生成缓存键"""
        # 将参数排序后序列化,确保一致性
        sorted_params = json.dumps(parameters, sort_keys=True)
        key_str = f"{tool_id}:{sorted_params}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        tool_id: str,
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """获取缓存的工具调用结果"""
        key = self._generate_key(tool_id, parameters)
        
        if key in self.cache:
            entry = self.cache[key]
            
            if not entry.is_expired():
                entry.update_hit()
                self.hits += 1
                return entry.value
            else:
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        result: Any,
        ttl: Optional[int] = None
    ):
        """设置缓存"""
        key = self._generate_key(tool_id, parameters)
        
        if len(self.cache) >= self.max_size:
            self._evict_lfu()  # 使用LFU策略
        
        self.cache[key] = CacheEntry(
            key=key,
            value=result,
            ttl=ttl or self.default_ttl,
            metadata={"tool_id": tool_id}
        )
    
    def _evict_lfu(self):
        """驱逐最不常用的条目(LFU)"""
        if not self.cache:
            return
        
        lfu_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].hit_count
        )
        del self.cache[lfu_key]
    
    def invalidate_tool(self, tool_id: str):
        """使某个工具的所有缓存失效"""
        to_delete = [
            key for key, entry in self.cache.items()
            if entry.metadata.get("tool_id") == tool_id
        ]
        
        for key in to_delete:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class IntelligentCache:
    """
    智能缓存决策器
    
    基于LLM-dCache的思想:
    将缓存操作本身作为工具,让LLM决定是否使用缓存
    """
    
    def __init__(
        self,
        tool_cache: ToolCallCache,
        llm_judge=None  # LLM判断器(可选)
    ):
        self.tool_cache = tool_cache
        self.llm_judge = llm_judge
        
        # 启发式规则
        self.heuristics = {
            "always_cache": ["search", "api_call", "database_query"],
            "never_cache": ["random", "timestamp", "current_time"],
            "conditional": ["compute", "analyze"]
        }
    
    def should_use_cache(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        context: str = ""
    ) -> bool:
        """
        决定是否应该使用缓存
        
        Args:
            tool_id: 工具ID
            parameters: 参数
            context: 上下文(用于LLM判断)
        
        Returns:
            True表示应该使用缓存
        """
        # 1. 启发式规则
        if tool_id in self.heuristics["always_cache"]:
            return True
        
        if tool_id in self.heuristics["never_cache"]:
            return False
        
        # 2. 检查是否有缓存
        cached_result = self.tool_cache.get(tool_id, parameters)
        if cached_result is None:
            return False  # 没有缓存,无法使用
        
        # 3. LLM判断(可选)
        if self.llm_judge and tool_id in self.heuristics["conditional"]:
            return self._llm_decision(tool_id, parameters, context, cached_result)
        
        # 4. 默认:使用缓存
        return True
    
    def _llm_decision(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        context: str,
        cached_result: Any
    ) -> bool:
        """
        使用LLM判断是否应该使用缓存
        
        这是LLM-dCache的核心思想
        """
        if self.llm_judge is None:
            return True
        
        prompt = f"""
        工具: {tool_id}
        参数: {json.dumps(parameters, indent=2)}
        上下文: {context}
        
        缓存中有一个之前的结果:
        {json.dumps(cached_result, indent=2)}
        
        问题:这个缓存的结果在当前上下文中是否仍然有效和有用?
        回答(yes/no):
        """
        
        # 调用LLM
        response = self.llm_judge(prompt)
        
        return "yes" in response.lower()
    
    def execute_with_cache(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        execute_fn,
        context: str = "",
        force_refresh: bool = False
    ) -> Tuple[Any, bool]:
        """
        执行工具调用,智能使用缓存
        
        Args:
            tool_id: 工具ID
            parameters: 参数
            execute_fn: 实际执行函数
            context: 上下文
            force_refresh: 强制刷新缓存
        
        Returns:
            (result, from_cache)
        """
        # 检查是否应该使用缓存
        if not force_refresh and self.should_use_cache(tool_id, parameters, context):
            cached = self.tool_cache.get(tool_id, parameters)
            if cached is not None:
                return cached, True
        
        # 执行工具
        result = execute_fn(**parameters)
        
        # 存入缓存
        self.tool_cache.set(tool_id, parameters, result)
        
        return result, False
