"""rag_service：RAG 后端统一接口层。

Phase 1 完成：抽象接口 RAGService + LegacyGraphRAGAdapter 包装现有 GraphRAGService。
Phase 2 完成：LightRAGAdapter 完整实现，支持 RAG_BACKEND 后端切换。

公开 API：
- RAGService          : 抽象基类
- LegacyGraphRAGAdapter : Phase 1 默认后端（Neo4j + GraphRAGService）
- LightRAGAdapter       : Phase 2 新后端（LightRAG + LocalEmbeddings）
- get_rag_service()   : 后端路由单例工厂
- RAG_BACKEND          : 当前后端标识
- GRAPHRAG_IMPORT_ERROR: 从 graphrag_module 转导出（routes.py 等依赖）
- DEFAULT_ONTOLOGY_FILE: 从 graphrag_module 转导出（routes.py SetupLocalDBRequest 默认值依赖）
"""

from typing import Optional

from .config import RAG_BACKEND
from .interface import RAGService
from .legacy_adapter import LegacyGraphRAGAdapter
from .lightrag_adapter import LightRAGAdapter  # Phase 2 完整实现（替换 stub）

# ---- 从 graphrag_module 转导出，供 routes.py / template.py 改导入后继续使用 ----
from graphrag_module.service import GRAPHRAG_IMPORT_ERROR, DEFAULT_ONTOLOGY_FILE  # noqa: F401


# ---- 全局单例工厂（与 get_graphrag_service() 相同的模块级单例模式） ----

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """获取全局单例 RAG 服务实例，按 RAG_BACKEND 环境变量路由。

    - "legacy"  -> LegacyGraphRAGAdapter（包装现有 GraphRAGService + Neo4j）
    - "lightrag" -> LightRAGAdapter（LightRAG + LocalEmbeddings，无需 Neo4j）
    """
    global _rag_service
    if _rag_service is None:
        if RAG_BACKEND == "legacy":
            _rag_service = LegacyGraphRAGAdapter()
        elif RAG_BACKEND == "lightrag":
            _rag_service = LightRAGAdapter()
        else:
            raise RuntimeError(f"Unknown RAG_BACKEND: {RAG_BACKEND}")
    return _rag_service


__all__ = [
    "DEFAULT_ONTOLOGY_FILE",
    "GRAPHRAG_IMPORT_ERROR",
    "LegacyGraphRAGAdapter",
    "LightRAGAdapter",
    "RAGService",
    "RAG_BACKEND",
    "get_rag_service",
]
