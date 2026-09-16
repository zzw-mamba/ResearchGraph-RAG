"""RAG 后端选择配置。

通过环境变量 RAG_BACKEND 选择要启用的 RAG 实现。
Phase 1 只有 "legacy"（现有 GraphRAGService），"lightrag" 为 stub。
"""

import os

RAG_BACKEND = os.getenv("RAG_BACKEND", "legacy").strip().lower()
assert RAG_BACKEND in {"legacy", "lightrag"}, (
    f"RAG_BACKEND 必须是 'legacy' 或 'lightrag'，当前值: {RAG_BACKEND}"
)
