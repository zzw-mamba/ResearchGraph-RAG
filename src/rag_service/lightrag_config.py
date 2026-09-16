"""LightRAG 专属配置。

从 .env 读取，独立于 GraphRAGSettings（后者面向 Neo4j）。
字段名对齐 LightRAG dataclass 的 constructor 参数，LightRAGAdapter 会
把这些值映射到 LightRAG(working_dir=..., kv_storage=..., ...)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass
class LightRAGSettings:
    # --- 存储后端 ---
    graph_storage: str = field(default_factory=lambda: _env_str("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"))
    vector_storage: str = field(default_factory=lambda: _env_str("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"))
    kv_storage: str = field(default_factory=lambda: _env_str("LIGHTRAG_KV_STORAGE", "JsonKVStorage"))
    doc_status_storage: str = field(default_factory=lambda: _env_str("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"))

    # --- 路径 / 隔离 ---
    working_dir: str = field(default_factory=lambda: _env_str("LIGHTRAG_WORKING_DIR", "./rag_storage"))
    workspace: str = field(default_factory=lambda: _env_str("LIGHTRAG_WORKSPACE", ""))

    # --- 查询参数 ---
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 40))
    chunk_top_k: int = field(default_factory=lambda: _env_int("CHUNK_TOP_K", 20))
    max_entity_tokens: int = field(default_factory=lambda: _env_int("MAX_ENTITY_TOKENS", 6000))
    max_relation_tokens: int = field(default_factory=lambda: _env_int("MAX_RELATION_TOKENS", 8000))
    max_total_tokens: int = field(default_factory=lambda: _env_int("MAX_TOTAL_TOKENS", 30000))
    enable_rerank: bool = field(default_factory=lambda: os.getenv("RERANK_BINDING", "null").lower() != "null")

    # --- 并发 ---
    max_parallel_insert: int = field(default_factory=lambda: _env_int("MAX_PARALLEL_INSERT", 3))
    max_async_llm: int = field(default_factory=lambda: _env_int("MAX_ASYNC_LLM", 4))
    max_async_embedding: int = field(default_factory=lambda: _env_int("EMBEDDING_FUNC_MAX_ASYNC", 8))

    # --- Rerank 可选（LightRAGAdapter 当前不启用 rerank，保留 env 读取即可）---
    rerank_binding: str = field(default_factory=lambda: _env_str("RERANK_BINDING", "null"))
    rerank_model: str = field(default_factory=lambda: _env_str("RERANK_MODEL", ""))
    rerank_host: str = field(default_factory=lambda: _env_str("RERANK_BINDING_HOST", ""))
    rerank_api_key: str = field(default_factory=lambda: _env_str("RERANK_BINDING_API_KEY", ""))

    # --- Embedding 维度（Qwen3-Embedding-8B = 4096）---
    embedding_dim: int = field(default_factory=lambda: _env_int("GRAPHRAG_EMBEDDING_DIMENSIONS", 4096))

    def to_lightrag_kwargs(self) -> Dict[str, Any]:
        """把字段名对齐成 LightRAG dataclass constructor 参数名。"""
        return {
            "working_dir": self.working_dir,
            # BUG-4 修复：空字符串是 LightRAG 合法 default workspace 值，不要转成 None
            "workspace": self.workspace,
            "graph_storage": self.graph_storage,
            "vector_storage": self.vector_storage,
            "kv_storage": self.kv_storage,
            "doc_status_storage": self.doc_status_storage,
            "top_k": self.top_k,
            "chunk_top_k": self.chunk_top_k,
            "max_entity_tokens": self.max_entity_tokens,
            "max_relation_tokens": self.max_relation_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_parallel_insert": self.max_parallel_insert,
            "llm_model_max_async": self.max_async_llm,
            "embedding_func_max_async": self.max_async_embedding,
        }

    def debug_info(self) -> str:
        return (
            f"[LightRAGSettings]\n"
            f"  working_dir={self.working_dir}\n"
            f"  workspace={self.workspace or '(default)'}\n"
            f"  storages: graph={self.graph_storage} vector={self.vector_storage} kv={self.kv_storage} doc_status={self.doc_status_storage}\n"
            f"  top_k={self.top_k}  chunk_top_k={self.chunk_top_k}\n"
            f"  max_tokens(entity/relation/total)={self.max_entity_tokens}/{self.max_relation_tokens}/{self.max_total_tokens}\n"
            f"  max_parallel_insert={self.max_parallel_insert}  max_async(llm/embed)={self.max_async_llm}/{self.max_async_embedding}\n"
            f"  embedding_dim={self.embedding_dim}  enable_rerank={self.enable_rerank}\n"
        )


# 单例缓存
_lightrag_settings_instance: LightRAGSettings | None = None


def get_lightrag_settings() -> LightRAGSettings:
    global _lightrag_settings_instance
    if _lightrag_settings_instance is None:
        _lightrag_settings_instance = LightRAGSettings()
    return _lightrag_settings_instance
