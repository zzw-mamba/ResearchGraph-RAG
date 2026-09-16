"""RAG 服务抽象接口。

定义所有 RAG 后端实现必须满足的统一契约。
LegacyGraphRAGAdapter 和未来的 LightRAGAdapter 都必须实现该接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RAGService(ABC):
    """RAG 服务统一抽象基类。

    返回类型统一为 Dict[str, Any]，与现有 GraphRAGService 保持一致。
    """

    # ---------------- 生命周期 ----------------

    @abstractmethod
    def initialize(self) -> None:
        """初始化后端依赖（Neo4j 连接、Embedding 客户端、检索器等）。"""

    @abstractmethod
    def close(self) -> None:
        """释放后端资源。"""

    # ---------------- 属性 ----------------

    @property
    @abstractmethod
    def settings(self) -> Any:
        """后端配置对象（GraphRAGSettings 或 LightRAG 对应配置）。"""

    # ---------------- 数据库 / 索引管理 ----------------

    @abstractmethod
    def setup_local_database(
        self,
        create_vector_index: bool = True,
        force_recreate_index: bool = False,
        ontology_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """准备本地图数据库环境并返回当前统计信息。"""

    @abstractmethod
    def ensure_graph_schema(self) -> Dict[str, Any]:
        """创建图谱约束与索引（幂等）。"""

    @abstractmethod
    def ensure_vector_index(self, force_recreate: bool = False) -> Dict[str, Any]:
        """确保向量索引存在，可按需强制重建。"""

    # ---------------- 数据写入 ----------------

    @abstractmethod
    def upsert_paper_chunks(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """写入论文切片、实体关系与结构化关联边。"""

    @abstractmethod
    def sync_from_mysql_knowledge_base(
        self,
        paper_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        auto_extract_entities: bool = True,
    ) -> Dict[str, Any]:
        """从 MySQL 知识库同步论文数据到后端，支持自动实体提取。"""

    # ---------------- 临时 OCR 会话 ----------------

    @abstractmethod
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
        """将 OCR 内容写入临时图谱作用域，仅用于当前会话检索。"""

    @abstractmethod
    def cleanup_temp_scope(self, session_scope: str) -> Dict[str, Any]:
        """删除指定临时会话作用域下的 OCR 图数据。"""

    @abstractmethod
    def cleanup_expired_temp_data(self) -> Dict[str, Any]:
        """清理已过期临时 OCR 图数据。"""

    # ---------------- 检索 ----------------

    @abstractmethod
    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
        strict_paper_filter: bool = False,
        session_scope: Optional[str] = None,
        include_global: bool = False,
    ) -> Dict[str, Any]:
        """基于向量索引执行相似度检索。"""

    @abstractmethod
    def rag_search(
        self,
        query_text: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
        session_scope: Optional[str] = None,
        include_global: bool = False,
    ) -> Dict[str, Any]:
        """检索相关 chunks 后调用 LLM 生成回答。"""

    # ---------------- 摘要 / 图谱上下文 ----------------

    @abstractmethod
    def get_graph_refined_context(
        self,
        paper_id: int,
        top_entities: int = 25,
        snippets_per_entity: int = 5,
        neighbor_limit: int = 5,
        section_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """从后端提取关键实体及其上下文。"""

    @abstractmethod
    def generate_paper_summary(
        self,
        paper_id: int,
        top_entities: int = 10,
        snippets_per_entity: int = 2,
        neighbor_limit: int = 5,
        recursive_group_size: int = 4,
        section_aware: bool = False,
    ) -> Dict[str, Any]:
        """生成论文摘要，支持章节感知的递归聚合。"""

    @abstractmethod
    def generate_query_concept_summary(
        self,
        query_text: str,
        paper_ids: List[int],
        anchor_entity_limit: int = 10,
        concept_max_hops: int = 2,
        direct_snippets_per_paper: int = 6,
        expanded_snippets_per_paper: int = 6,
    ) -> Dict[str, Any]:
        """基于 query + concept 本体扩展的跨论文摘要。"""

    # ---------------- 相关论文推荐 ----------------

    @abstractmethod
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
        """基于 Concept 对齐 + 向量证据返回相关文章。"""

    # ---------------- 调试信息（可选） ----------------

    def debug_info(self) -> Dict[str, Any]:
        """返回后端调试信息，可选实现。"""
        return {"backend": getattr(self.settings, "__class__", type(self.settings)).__name__}
