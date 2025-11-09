"""
分层语义路由器 - Tool-to-Agent Retrieval实现
基于论文: Tool-to-Agent Retrieval (arXiv 2025)
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
from ..core.tool import Tool, MCPService


class HierarchicalRouter:
    """
    分层语义路由器
    
    实现统一向量空间中的工具和Agent检索,支持:
    1. 工具级检索 + Agent聚合
    2. BM25 + Dense Vector混合检索
    3. 元数据关系遍历
    """
    
    def __init__(
        self,
        services: List[MCPService],
        embedding_model=None,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7
    ):
        """
        初始化路由器
        
        Args:
            services: MCP服务列表
            embedding_model: 向量嵌入模型(如sentence-transformers)
            bm25_weight: BM25检索权重
            dense_weight: 稠密向量检索权重
        """
        self.services = {s.id: s for s in services}
        self.embedding_model = embedding_model
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # 构建统一目录
        self.tool_catalog = {}  # tool_id -> Tool
        self.service_catalog = {}  # service_id -> MCPService
        self.tool_to_service = {}  # tool_id -> service_id
        
        self._build_catalog()
    
    def _build_catalog(self):
        """构建统一的工具-服务目录"""
        for service in self.services.values():
            self.service_catalog[service.id] = service
            
            for tool in service.tools:
                self.tool_catalog[tool.id] = tool
                self.tool_to_service[tool.id] = service.id
    
    def route(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 3,
        top_n: int = 20
    ) -> List[MCPService]:
        """
        执行分层路由
        
        Args:
            query: 用户查询
            query_embedding: 查询的向量嵌入(可选)
            top_k: 返回的top-K个服务
            top_n: 初步检索的top-N个实体(N >> K)
        
        Returns:
            排序后的top-K个MCP服务
        """
        # Step 1: 检索top-N个实体(工具+服务)
        top_entities = self._retrieve_top_n(query, query_embedding, top_n)
        
        # Step 2: 聚合到父服务并计分
        service_scores = self._aggregate_to_services(top_entities)
        
        # Step 3: 选择top-K个服务
        sorted_services = sorted(
            service_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [self.services[service_id] for service_id, _ in sorted_services]
    
    def _retrieve_top_n(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        top_n: int
    ) -> List[Tuple[str, float, str]]:
        """
        检索top-N个实体(工具或服务)
        
        Returns:
            List of (entity_id, score, entity_type)
        """
        results = []
        
        # BM25检索(简化实现,实际应使用rank_bm25库)
        bm25_scores = self._bm25_search(query)
        
        # 稠密向量检索
        if query_embedding is not None:
            dense_scores = self._dense_search(query_embedding)
        else:
            dense_scores = {}
        
        # 混合评分
        all_entities = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        for entity_id in all_entities:
            bm25_score = bm25_scores.get(entity_id, 0.0)
            dense_score = dense_scores.get(entity_id, 0.0)
            
            # 归一化后加权组合
            combined_score = (
                self.bm25_weight * bm25_score +
                self.dense_weight * dense_score
            )
            
            # 判断实体类型
            entity_type = "tool" if entity_id in self.tool_catalog else "service"
            results.append((entity_id, combined_score, entity_type))
        
        # 排序并返回top-N
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def _bm25_search(self, query: str) -> Dict[str, float]:
        """
        BM25关键词检索(简化实现)
        
        实际应使用rank_bm25库或Elasticsearch
        """
        scores = {}
        query_terms = set(query.lower().split())
        
        # 检索工具
        for tool_id, tool in self.tool_catalog.items():
            doc_terms = set((tool.name + " " + tool.description).lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                scores[tool_id] = overlap / len(query_terms)
        
        # 检索服务
        for service_id, service in self.service_catalog.items():
            doc_terms = set((service.name + " " + service.description).lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                scores[service_id] = overlap / len(query_terms)
        
        return scores
    
    def _dense_search(self, query_embedding: List[float]) -> Dict[str, float]:
        """
        稠密向量检索(余弦相似度)
        """
        scores = {}
        query_vec = np.array(query_embedding)
        
        # 检索工具
        for tool_id, tool in self.tool_catalog.items():
            if tool.embedding:
                tool_vec = np.array(tool.embedding)
                similarity = self._cosine_similarity(query_vec, tool_vec)
                scores[tool_id] = similarity
        
        # 检索服务
        for service_id, service in self.service_catalog.items():
            if service.embedding:
                service_vec = np.array(service.embedding)
                similarity = self._cosine_similarity(query_vec, service_vec)
                scores[service_id] = similarity
        
        return scores
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _aggregate_to_services(
        self,
        entities: List[Tuple[str, float, str]]
    ) -> Dict[str, float]:
        """
        将实体聚合到父服务
        
        Args:
            entities: List of (entity_id, score, entity_type)
        
        Returns:
            service_id -> aggregated_score
        """
        service_scores = defaultdict(float)
        service_counts = defaultdict(int)
        
        for entity_id, score, entity_type in entities:
            if entity_type == "tool":
                # 通过元数据找到父服务
                service_id = self.tool_to_service.get(entity_id)
                if service_id:
                    service_scores[service_id] += score
                    service_counts[service_id] += 1
            else:
                # 直接是服务
                service_scores[entity_id] += score
                service_counts[entity_id] += 1
        
        # 平均分数(或使用最大值、加权和等策略)
        return {
            service_id: total_score / service_counts[service_id]
            for service_id, total_score in service_scores.items()
        }
