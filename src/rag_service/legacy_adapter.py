"""LegacyGraphRAGAdapter：包装现有 GraphRAGService 为统一 RAGService 接口。

核心设计：
1. 内部持有 get_graphrag_service() 单例，所有方法直接委托。
2. 使用 inspect.signature 过滤参数：当 GraphRAGService 底层方法未声明
   某个参数（如 similarity_search 缺失 session_scope/include_global）时，
   自动跳过，避免 TypeError；同时保证前向兼容（以后底层加了参数自动生效）。
3. 暴露 settings 和 driver 属性代理，供 routes.py 的 health / diagnose
   endpoint 直接访问底层配置和 Neo4j driver。
"""

import inspect
from typing import Any, Dict, List, Optional

from graphrag_module.service import get_graphrag_service, GraphRAGService
from graphrag_module.config import GraphRAGSettings

from .interface import RAGService


class LegacyGraphRAGAdapter(RAGService):
    """适配器：将现有 GraphRAGService 包装为统一 RAGService 接口。

    Phase 1 的默认后端实现。内部通过 get_graphrag_service() 持有
    GraphRAGService 单例实例，所有方法直接委托，签名完全对齐。
    """

    def __init__(self, settings: Optional[GraphRAGSettings] = None):
        # GraphRAGService 单例由 get_graphrag_service() 管理（含 initialize）
        self._service: GraphRAGService = get_graphrag_service()

    # ---------- 属性代理（供 routes.py health / diagnose endpoint 直接访问） ----------

    @property
    def settings(self) -> GraphRAGSettings:
        """GraphRAGSettings 配置对象。"""
        return self._service.settings

    @property
    def driver(self):
        """Neo4j driver 实例，供 diagnose endpoint 直接操作。"""
        return self._service.driver

    # ---------- 通用参数过滤委托 ----------

    def _delegate(self, method_name: str, **kwargs: Any) -> Any:
        """按底层方法签名过滤参数后调用。

        解决 GraphRAGService.similarity_search 未声明 session_scope / include_global
        但 routes.py 和 rag_search 内部都传递这两个参数的问题。
        """
        target = getattr(self._service, method_name)
        sig = inspect.signature(target)
        accepted: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in sig.parameters:
                accepted[key] = value
        return target(**accepted)

    # ---------- 生命周期 ----------

    def initialize(self) -> None:
        self._service.initialize()

    def close(self) -> None:
        self._service.close()

    # ---------- 数据库 / 索引管理 ----------

    def setup_local_database(
        self,
        create_vector_index: bool = True,
        force_recreate_index: bool = False,
        ontology_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._delegate(
            "setup_local_database",
            create_vector_index=create_vector_index,
            force_recreate_index=force_recreate_index,
            ontology_file_path=ontology_file_path,
        )

    def ensure_graph_schema(self) -> Dict[str, Any]:
        return self._delegate("ensure_graph_schema")

    def ensure_vector_index(self, force_recreate: bool = False) -> Dict[str, Any]:
        return self._delegate("ensure_vector_index", force_recreate=force_recreate)

    # ---------- 数据写入 ----------

    def upsert_paper_chunks(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._delegate("upsert_paper_chunks", rows=rows)

    def sync_from_mysql_knowledge_base(
        self,
        paper_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        auto_extract_entities: bool = True,
    ) -> Dict[str, Any]:
        return self._delegate(
            "sync_from_mysql_knowledge_base",
            paper_ids=paper_ids,
            limit=limit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            auto_extract_entities=auto_extract_entities,
        )

    # ---------- 临时 OCR 会话 ----------

    def upsert_temp_ocr_content(
        self,
        title: str,
        content: str,
        session_scope: Optional[str] = None,
        ttl_hours: int = 24,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        auto_extract_entities: bool = True,
    ) -> Dict[str, Any]:
        return self._delegate(
            "upsert_temp_ocr_content",
            title=title,
            content=content,
            session_scope=session_scope,
            ttl_hours=ttl_hours,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            auto_extract_entities=auto_extract_entities,
        )

    def cleanup_temp_scope(self, session_scope: str) -> Dict[str, Any]:
        return self._delegate("cleanup_temp_scope", session_scope=session_scope)

    def cleanup_expired_temp_data(self) -> Dict[str, Any]:
        return self._delegate("cleanup_expired_temp_data")

    # ---------- 检索 ----------

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
        strict_paper_filter: bool = False,
        session_scope: Optional[str] = None,
        include_global: bool = False,
    ) -> Dict[str, Any]:
        return self._delegate(
            "similarity_search",
            query_text=query_text,
            top_k=top_k,
            paper_ids=paper_ids,
            strict_paper_filter=strict_paper_filter,
            session_scope=session_scope,
            include_global=include_global,
        )

    def rag_search(
        self,
        query_text: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
        session_scope: Optional[str] = None,
        include_global: bool = False,
    ) -> Dict[str, Any]:
        return self._delegate(
            "rag_search",
            query_text=query_text,
            top_k=top_k,
            paper_ids=paper_ids,
            session_scope=session_scope,
            include_global=include_global,
        )

    # ---------- 摘要 / 图谱上下文 ----------

    def get_graph_refined_context(
        self,
        paper_id: int,
        top_entities: int = 25,
        snippets_per_entity: int = 5,
        neighbor_limit: int = 5,
        section_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._delegate(
            "get_graph_refined_context",
            paper_id=paper_id,
            top_entities=top_entities,
            snippets_per_entity=snippets_per_entity,
            neighbor_limit=neighbor_limit,
            section_filter=section_filter,
        )

    def generate_paper_summary(
        self,
        paper_id: int,
        top_entities: int = 10,
        snippets_per_entity: int = 2,
        neighbor_limit: int = 5,
        recursive_group_size: int = 4,
        section_aware: bool = False,
    ) -> Dict[str, Any]:
        return self._delegate(
            "generate_paper_summary",
            paper_id=paper_id,
            top_entities=top_entities,
            snippets_per_entity=snippets_per_entity,
            neighbor_limit=neighbor_limit,
            recursive_group_size=recursive_group_size,
            section_aware=section_aware,
        )

    def generate_query_concept_summary(
        self,
        query_text: str,
        paper_ids: List[int],
        anchor_entity_limit: int = 10,
        concept_max_hops: int = 2,
        direct_snippets_per_paper: int = 6,
        expanded_snippets_per_paper: int = 6,
    ) -> Dict[str, Any]:
        return self._delegate(
            "generate_query_concept_summary",
            query_text=query_text,
            paper_ids=paper_ids,
            anchor_entity_limit=anchor_entity_limit,
            concept_max_hops=concept_max_hops,
            direct_snippets_per_paper=direct_snippets_per_paper,
            expanded_snippets_per_paper=expanded_snippets_per_paper,
        )

    # ---------- 相关论文推荐 ----------

    def related_papers_by_id(
        self,
        paper_id: int,
        top_k: int = 10,
        per_chunk_k: int = 8,
        source_chunk_limit: int = 8,
        evidence_limit: int = 3,
        concept_weight: float = 0.65,
        vector_weight: float = 0.35,
        min_shared_concepts: int = 1,
    ) -> Dict[str, Any]:
        return self._delegate(
            "related_papers_by_id",
            paper_id=paper_id,
            top_k=top_k,
            per_chunk_k=per_chunk_k,
            source_chunk_limit=source_chunk_limit,
            evidence_limit=evidence_limit,
            concept_weight=concept_weight,
            vector_weight=vector_weight,
            min_shared_concepts=min_shared_concepts,
        )

    # ---------- 调试扩展 ----------

    def debug_info(self) -> Dict[str, Any]:
        return {
            **super().debug_info(),
            "backend": "legacy_graphrag",
            "driver_exists": self._service.driver is not None,
        }
