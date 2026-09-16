"""LightRAG EmbeddingFunc 适配器。

将项目现有的同步 LocalEmbeddings（HTTP + SQLite 缓存 + 批量）包装成
LightRAG 要求的 async EmbeddingFunc，签名：async (list[str]) -> np.ndarray。

依赖：
- LocalEmbeddings.embed_documents(texts: List[str]) -> List[List[float]]  （同步）
- LightRAG 的 wrap_embedding_func_with_attrs 装饰器
"""

from __future__ import annotations

import asyncio
from typing import List

import numpy as np

try:
    from lightrag.utils import wrap_embedding_func_with_attrs
except Exception as _e:  # pragma: no cover - 运行时由 get_rag_service 触发
    wrap_embedding_func_with_attrs = None
    _LIGHTRAG_IMPORT_ERROR = _e
else:
    _LIGHTRAG_IMPORT_ERROR = None


def make_local_embedding_func(local_embeddings, embedding_dim: int = 4096):
    """工厂：返回可直接传给 LightRAG(embedding_func=...) 的 EmbeddingFunc 实例。

    Args:
        local_embeddings: graphrag_module.service.LocalEmbeddings 实例（已初始化，带 SQLite 缓存）。
        embedding_dim: 向量维度，默认 4096（Qwen3-Embedding-8B）。

    Returns:
        lightrag.utils.EmbeddingFunc —— 带 embedding_dim / max_token_size / model_name 属性的可调用对象。
        调用签名：async __call__(texts: list[str]) -> np.ndarray  (shape=[N, dim])
    """
    if _LIGHTRAG_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"lightrag-hku 未安装，无法创建 EmbeddingFunc。原错误: {_LIGHTRAG_IMPORT_ERROR}"
        )

    model_name = getattr(local_embeddings, "model", "") or ""

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        model_name=model_name or "local-embedding",
    )
    async def _embed(texts: List[str]) -> np.ndarray:
        """async embed：分批 + sync→async 桥接 + 返回 np.ndarray。"""
        if not texts:
            return np.empty((0, embedding_dim), dtype=np.float32)

        # 每批 20 条。LocalEmbeddings.embed_documents 内部已实现缓存过滤 + 批量 HTTP。
        batch_size = 20
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = await asyncio.to_thread(local_embeddings.embed_documents, batch)
            all_vectors.extend(vectors)

        arr = np.array(all_vectors, dtype=np.float32)
        # 防御：如果 dim 不匹配，打印警告但不 hard-fail
        if arr.size and arr.shape[1] != embedding_dim:
            import logging
            logging.getLogger(__name__).warning(
                "Embedding dim mismatch: got %d, expected %d. LocalEmbeddings.model=%s",
                arr.shape[1],
                embedding_dim,
                model_name,
            )
        return arr

    return _embed
