"""LightRAGAdapter：RAGService 接口的 LightRAG 实现。

设计要点：
1. LightRAG 核心是 async，本 Adapter 内部使用"守护线程 + 持久化事件循环"模式桥接 sync→async。
2. CSO 本体对齐已移除 —— 原委托给 GraphRAGService 的 4 个方法（generate_paper_summary、
   related_papers_by_id、get_graph_refined_context、generate_query_concept_summary）
   均降级返回 {"status": "unavailable"} 占位字典，待后续 LightRAG 原生能力接入。
3. Embedding 复用 LocalEmbeddings（带 SQLite 缓存），与 Legacy 后端共享同一套
   向量服务 (Qwen3-Embedding-8B, dim=4096, localhost:9091)。
4. 临时 OCR 内容通过 LightRAG 的 workspace 隔离：workspace="temp_{hash(session_scope)}"。
5. 所有公共方法用 exception 隔离 —— LightRAG 内部异常包装成 RuntimeError，
   不让实现细节穿透到 FastAPI handler。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---- 现有自研实现（LightRAG 后端复用 LocalEmbeddings）----
from graphrag_module.service import LocalEmbeddings
from graphrag_module.config import GraphRAGSettings, get_graphrag_settings

from .interface import RAGService
from .lightrag_config import LightRAGSettings, get_lightrag_settings


# ---------------------------------------------------------------------------
# 轻量 proxy：让 routes.py 的 health/diagnose 在 LightRAG 后端也能正常访问
# settings.vector_index_name / settings.neo4j_uri 等字段 —— 这些字段在
# LightRAG 后端下是"兼容桩"，diagnose 中通过 driver is None 来区分即可。
# ---------------------------------------------------------------------------


class _LightRAGSettingsProxy:
    """属性代理：把 LightRAGSettings 暴露为 health endpoint 期望的形态。"""

    def __init__(self, lr_settings: LightRAGSettings, gr_settings: GraphRAGSettings):
        self._lr = lr_settings
        self._gr = gr_settings

    # LightRAG 专属
    @property
    def graph_storage(self) -> str:
        return self._lr.graph_storage

    @property
    def vector_storage(self) -> str:
        return self._lr.vector_storage

    @property
    def kv_storage(self) -> str:
        return self._lr.kv_storage

    @property
    def doc_status_storage(self) -> str:
        return self._lr.doc_status_storage

    @property
    def working_dir(self) -> str:
        return self._lr.working_dir

    @property
    def workspace(self) -> str:
        return self._lr.workspace

    # GraphRAG 兼容桩（routes.py / template.py 会访问）
    @property
    def vector_index_name(self) -> str:
        return f"lightrag_{self._lr.vector_storage.lower()}"

    @property
    def vector_node_label(self) -> str:
        return "Chunk"

    @property
    def embedding_model(self) -> str:
        return self._gr.embedding_model

    @property
    def llm_model(self) -> str:
        return self._gr.llm_model

    @property
    def neo4j_uri(self) -> str:
        return "(LightRAG backend, no Neo4j)"

    @property
    def neo4j_user(self) -> str:
        return ""

    # LightRAGSettings 本身也能完整访问
    @property
    def lightrag(self) -> LightRAGSettings:
        return self._lr

    @property
    def graphrag(self) -> GraphRAGSettings:
        return self._gr

    def debug_info(self) -> str:
        return (
            "[LightRAGAdapter settings proxy]\n"
            + self._lr.debug_info()
            + "\n"
            + f"  embedding_model={self.embedding_model}  llm_model={self.llm_model}"
        )


# ---------------------------------------------------------------------------
# 核心适配类
# ---------------------------------------------------------------------------


class LightRAGAdapter(RAGService):
    """LightRAG 后端的 RAGService 实现。"""

    def __init__(
        self,
        graphrag_settings: Optional[GraphRAGSettings] = None,
        lightrag_settings: Optional[LightRAGSettings] = None,
    ):
        self._gr_settings = graphrag_settings or get_graphrag_settings()
        self._lr_settings = lightrag_settings or get_lightrag_settings()
        self._settings_proxy = _LightRAGSettingsProxy(self._lr_settings, self._gr_settings)

        # LocalEmbeddings（带 SQLite 缓存）—— 与 Legacy 共享同一套向量服务
        self.local_embedder = LocalEmbeddings(
            base_url=self._gr_settings.local_embedding_base_url,
            api_path=self._gr_settings.local_embedding_api_path,
            model=self._gr_settings.embedding_model,
            timeout=self._gr_settings.local_embedding_timeout,
        )

        # LightRAG 运行时状态
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._rag = None  # LightRAG 实例（import 延迟到 initialize，避免 stub 模式下 LightRAG 未安装）
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # 属性（RAGService ABC 要求）
    # ------------------------------------------------------------------

    @property
    def settings(self) -> _LightRAGSettingsProxy:
        """返回同时含 LightRAG + 兼容 GraphRAG 字段的 proxy。"""
        return self._settings_proxy

    @property
    def driver(self):
        """LightRAG 不使用 Neo4j，返回 None。routes.py diagnose 通过此值做分支。"""
        return None

    # ------------------------------------------------------------------
    # 线程 + 事件循环桥接
    # ------------------------------------------------------------------

    def _start_event_loop_thread(self) -> None:
        """启动持久化事件循环线程（daemon），供所有 LightRAG async 操作复用。"""
        if self._thread is not None and self._thread.is_alive():
            return
        ready = threading.Event()
        loop_holder: Dict[str, Optional[asyncio.AbstractEventLoop]] = {"loop": None}

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder["loop"] = loop
            self._loop = loop
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True, name="LightRAG-EventLoop")
        self._thread.start()

        # 等待 loop 就绪，最多 3s
        if not ready.wait(timeout=3.0):
            raise RuntimeError("LightRAG 事件循环线程启动超时")

    def _submit(self, coro_factory, timeout: float = 120.0):
        """把协程提交到持久化事件循环并等待结果。"""
        if self._loop is None:
            raise RuntimeError("LightRAG 事件循环未启动")
        future = asyncio.run_coroutine_threadsafe(coro_factory(), self._loop)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            # 让 traceback 更清晰地定位是 LightRAG 内部哪一层抛的
            tb_str = traceback.format_exc()
            raise RuntimeError(f"LightRAG async 调用失败: {e}\n{tb_str}") from e

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("LightRAGAdapter 未调用 initialize()，请先启动服务")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self._start_event_loop_thread()

        async def _async_init():
            from lightrag import LightRAG
            from rag_service.embedding_wrapper import make_local_embedding_func
            from rag_service.llm_wrapper import make_llm_func

            embedding_func = make_local_embedding_func(
                self.local_embedder, embedding_dim=self._lr_settings.embedding_dim
            )
            llm_func = make_llm_func(self._gr_settings.llm_model)

            rag_kwargs = self._lr_settings.to_lightrag_kwargs()
            rag_kwargs.update(
                {
                    "llm_model_func": llm_func,
                    "embedding_func": embedding_func,
                }
            )
            self._rag = LightRAG(**rag_kwargs)
            await self._rag.initialize_storages()
            self._initialized = True
            print(
                f"[LightRAGAdapter] ✓ 初始化完成: workspace={self._lr_settings.workspace or '(default)'}  "
                f"working_dir={self._lr_settings.working_dir}"
            )

        try:
            self._submit(_async_init, timeout=120)
        except Exception as e:
            # 清理已启动的 loop，避免悬挂
            self._initialized = False
            raise RuntimeError(f"LightRAG 初始化失败: {e}") from e

    def close(self) -> None:
        # 先标记未初始化，防止后续并发调用
        self._initialized = False

        # 1. 关闭 LightRAG 存储
        if self._rag is not None:
            try:

                async def _async_finalize():
                    await self._rag.finalize_storages()

                self._submit(_async_finalize, timeout=30)
            except Exception as e:
                print(f"[LightRAGAdapter] finalize_storages 警告: {e}")

        # 2. 停止事件循环线程
        if self._loop is not None and self._thread is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                print(
                    "[LightRAGAdapter] ⚠ 事件循环线程在 5s 内未退出（daemon 模式，进程退出时会被回收）"
                )

        self._rag = None
        self._loop = None
        self._thread = None

    # ------------------------------------------------------------------
    # 数据库 / 索引管理 （LightRAG 内置，这些方法主要是占位 + 返回元信息）
    # ------------------------------------------------------------------

    def setup_local_database(
        self,
        create_vector_index: bool = True,
        force_recreate_index: bool = False,
        ontology_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """LightRAG 存储由 initialize() 自动创建；此方法只返回当前存储元信息。"""
        self._ensure_initialized()
        return {
            "status": "ok",
            "backend": "lightrag",
            "graph_storage": self._lr_settings.graph_storage,
            "vector_storage": self._lr_settings.vector_storage,
            "kv_storage": self._lr_settings.kv_storage,
            "doc_status_storage": self._lr_settings.doc_status_storage,
            "working_dir": self._lr_settings.working_dir,
            "workspace": self._lr_settings.workspace or "",
        }

    def ensure_graph_schema(self) -> Dict[str, Any]:
        """LightRAG (NetworkX) 无需手动建约束。"""
        self._ensure_initialized()
        return {"status": "ok", "message": "LightRAG 存储无显式 schema 约束"}

    def ensure_vector_index(self, force_recreate: bool = False) -> Dict[str, Any]:
        """LightRAG 向量存储 (NanoVectorDB / Qdrant 等) 在 upsert 时自动建。"""
        self._ensure_initialized()
        return {"status": "ok", "force_recreate": force_recreate}

    # ------------------------------------------------------------------
    # 数据写入
    # ------------------------------------------------------------------

    def upsert_paper_chunks(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将前端已切好的 chunks 注入 LightRAG。

        rows 格式（与 LegacyGraphRAGAdapter 一致）：
        [{"paper_id": 123, "title": "...", "chunk_text": "...", "chunk_index": 0}, ...]

        Phase 2 简化：将同 paper 的 chunks 拼接成完整文档，用 ainsert 路径（带
        file_paths）写入，让 LightRAG 自行切片 + 实体抽取。file_path 设为
        "paper_{pid}.pdf"，方便 similarity_search 按 paper_id 精确过滤。
        自研 _hierarchical_semantic_chunking 将在后续 Phase 3 回归。
        """
        self._ensure_initialized()
        if not rows:
            return {"inserted_docs": 0, "inserted_chunks": 0}

        # 按 paper_id 聚合
        by_paper: Dict[int, List[Dict[str, Any]]] = {}
        titles: Dict[int, str] = {}
        for r in rows:
            pid = int(r.get("paper_id") or 0)
            by_paper.setdefault(pid, []).append(r)
            if r.get("title") and pid not in titles:
                titles[pid] = str(r["title"])

        async def _insert_all() -> Dict[str, Any]:
            """串行插入所有 paper，返回统计 dict（消除 nonlocal）。"""
            inserted_docs = 0
            inserted_chunks = 0
            errors: List[str] = []
            for pid, chunks in by_paper.items():
                sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
                chunk_texts = [c["chunk_text"] for c in sorted_chunks if c.get("chunk_text")]
                if not chunk_texts:
                    continue
                full_text = "\n\n".join(chunk_texts)
                doc_id = f"paper-{pid}"
                file_path = f"paper_{pid}.pdf"
                try:
                    await self._rag.ainsert(
                        [full_text],
                        ids=[doc_id],
                        file_paths=[file_path],
                    )
                    inserted_docs += 1
                    inserted_chunks += len(chunk_texts)  # ainsert 后 chunk 数量由 LightRAG 决定，这里用近似
                except Exception as e:
                    errors.append(f"paper_id={pid}: {e}")
            return {
                "inserted_docs": inserted_docs,
                "inserted_chunks": inserted_chunks,
                "errors": errors,
            }

        return self._submit(_insert_all, timeout=300)

    def sync_from_mysql_knowledge_base(
        self,
        paper_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        auto_extract_entities: bool = True,
    ) -> Dict[str, Any]:
        """从 MySQL 知识库同步到 LightRAG。

        Phase 2 简化：直接用 ainsert 路径（带 file_paths），让 LightRAG 自行切片
        + 实体抽取。file_path 设为 "paper_{pid}.pdf"，确保 similarity_search 能按
        paper_id 精确过滤。自研 _hierarchical_semantic_chunking 将在 Phase 3 回归。
        auto_extract_entities 由 LightRAG 内部流水线自动完成。
        """
        self._ensure_initialized()
        from database import SessionLocal
        from models import KnowledgeBase

        db = SessionLocal()
        try:
            query = db.query(KnowledgeBase)
            if paper_ids:
                query = query.filter(KnowledgeBase.id.in_(paper_ids))
            if limit:
                query = query.limit(limit)
            papers = list(query.all())
        finally:
            db.close()

        if not papers:
            return {"synced": 0, "inserted_chunks": 0, "inserted_docs": 0}

        inserted_docs = 0
        errors: List[str] = []

        async def _insert_one(
            paper_id: int, title: str, content: str, doc_id: str, file_path: str
        ) -> Optional[int]:
            """单条插入：成功返回 paper_id（用作计数标记），失败返回 None 并追加 errors。"""
            try:
                await self._rag.ainsert(
                    [content],
                    ids=[doc_id],
                    file_paths=[file_path],
                )
                return paper_id
            except Exception as e:
                errors.append(f"paper_id={paper_id} ({title}): {e}")
                return None

        async def _sync_all():
            tasks: List[Any] = []
            for p in papers:
                content = (p.abstract or "") + "\n\n" + (p.content or "")
                if not content.strip():
                    continue
                pid = p.id
                # BUG-5 修复：p.id 可能为 None（防御检查）
                if pid is None:
                    errors.append(f"paper_id=None ({p.title}): 跳过")
                    continue
                doc_id = f"paper-{pid}"
                file_path = f"paper_{pid}.pdf"
                title = p.title or f"paper_{pid}"
                tasks.append(_insert_one(pid, title, content, doc_id, file_path))

            # 并发跑；return_exceptions=True 确保单个异常不会中断整体
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        results = self._submit(_sync_all, timeout=600)

        # BUG-5 修复：用 gather + 结果收集模式，消除 nonlocal 竞态
        for r in results:
            if isinstance(r, Exception):
                errors.append(f"gather 捕获异常: {r}")
            elif r is not None:
                inserted_docs += 1

        return {
            "synced": len(papers),
            "inserted_docs": inserted_docs,
            "inserted_chunks": inserted_docs,  # LightRAG 内部切片数未知，这里用 docs 数做占位
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # 临时 OCR （用 workspace 隔离）
    # ------------------------------------------------------------------

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
        self._ensure_initialized()
        if not session_scope:
            # 退化：把 temp 内容直接写入主 workspace
            if not content.strip():
                return {"inserted_chunks": 0, "note": "empty content"}
            doc_id = f"temp-{hashlib.md5((title + content).encode()).hexdigest()[:12]}"
            file_path = f"temp_{doc_id}.txt"

            async def _do():
                await self._rag.ainsert(
                    [content],
                    ids=[doc_id],
                    file_paths=[file_path],
                )

            self._submit(_do, timeout=120)
            return {"inserted_chunks": 0, "doc_id": doc_id}

        # 用独立 workspace 隔离临时数据
        ws_hash = hashlib.md5(session_scope.encode("utf-8")).hexdigest()[:12]
        temp_workspace = f"temp_{ws_hash}"
        expires_at_iso = (
            datetime.now(timezone.utc).timestamp() + ttl_hours * 3600
        )  # 存入 KV 便于过期清理

        # 创建一个临时 LightRAG 实例指向同一 working_dir + 隔离 workspace
        async def _do_insert():
            from lightrag import LightRAG

            temp_settings = self._lr_settings
            temp_rag = LightRAG(
                working_dir=temp_settings.working_dir,
                workspace=temp_workspace,
                graph_storage=temp_settings.graph_storage,
                vector_storage=temp_settings.vector_storage,
                kv_storage=temp_settings.kv_storage,
                doc_status_storage=temp_settings.doc_status_storage,
                llm_model_func=self._rag.llm_model_func,
                embedding_func=self._rag.embedding_func,
                top_k=temp_settings.top_k,
                chunk_top_k=temp_settings.chunk_top_k,
                max_entity_tokens=temp_settings.max_entity_tokens,
                max_relation_tokens=temp_settings.max_relation_tokens,
                max_total_tokens=temp_settings.max_total_tokens,
                max_parallel_insert=temp_settings.max_parallel_insert,
            )
            await temp_rag.initialize_storages()

            doc_id = f"temp-{temp_workspace}-{hashlib.md5(content.encode()).hexdigest()[:8]}"
            file_path = f"temp_{doc_id}.txt"
            await temp_rag.ainsert(
                [content],
                ids=[doc_id],
                file_paths=[file_path],
            )
            await temp_rag.finalize_storages()

            # 把过期时间存到主 workspace 的 full_docs metadata
            # LightRAG ainsert_custom_chunks 不写自定义 metadata，退化方案：
            # 存到文件 .lightrag_temp_ttl.json
            import json
            ttl_path = os.path.join(temp_settings.working_dir, ".temp_ttl.json")
            record = {
                "session_scope": session_scope,
                "workspace": temp_workspace,
                "title": title,
                "expires_at": expires_at_iso,
                "doc_id": doc_id,
            }
            existing = {}
            if os.path.exists(ttl_path):
                try:
                    with open(ttl_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            existing[session_scope] = record
            with open(ttl_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            return {"inserted_chunks": 0, "doc_id": doc_id, "workspace": temp_workspace}

        return self._submit(_do_insert, timeout=120)

    def cleanup_temp_scope(self, session_scope: str) -> Dict[str, Any]:
        """删除指定 workspace 的临时数据（通过文件系统 + LightRAG drop）。"""
        self._ensure_initialized()

        import json
        import shutil

        ttl_path = os.path.join(self._lr_settings.working_dir, ".temp_ttl.json")
        target_workspace = None
        if os.path.exists(ttl_path):
            try:
                with open(ttl_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                record = registry.pop(session_scope, None)
                if record:
                    target_workspace = record.get("workspace")
                with open(ttl_path, "w", encoding="utf-8") as f:
                    json.dump(registry, f, ensure_ascii=False, indent=2)
            except Exception as e:
                return {"status": "error", "message": f"读取 TTL registry 失败: {e}"}

        # LightRAG 的 workspace 数据在 working_dir/workspace_{ws_name}/ 下（具体取决于 storage impl）。
        # NetworkXStorage / JsonKVStorage / JsonDocStatusStorage 把数据写到 working_dir 下的
        # 以 workspace 命名的子目录中；NanoVectorDBStorage 同理。
        deleted = False
        if target_workspace:
            # 尝试删除以 workspace 名开头的所有子目录
            for entry in os.listdir(self._lr_settings.working_dir):
                full = os.path.join(self._lr_settings.working_dir, entry)
                if os.path.isdir(full) and entry.startswith(target_workspace):
                    try:
                        shutil.rmtree(full)
                        deleted = True
                    except Exception:
                        pass
            # 也尝试完整 workspace 名的目录
            full_path = os.path.join(self._lr_settings.working_dir, target_workspace)
            if os.path.isdir(full_path):
                try:
                    shutil.rmtree(full_path)
                    deleted = True
                except Exception:
                    pass

        return {"status": "ok", "workspace": target_workspace, "deleted": deleted}

    def cleanup_expired_temp_data(self) -> Dict[str, Any]:
        """扫描 TTL registry，删除过期 session。"""
        import json
        import shutil

        ttl_path = os.path.join(self._lr_settings.working_dir, ".temp_ttl.json")
        if not os.path.exists(ttl_path):
            return {"status": "ok", "expired_count": 0}

        try:
            with open(ttl_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"读取 TTL registry 失败: {e}"}

        now = datetime.now(timezone.utc).timestamp()
        expired_scopes = [
            scope for scope, rec in registry.items() if float(rec.get("expires_at", 0)) < now
        ]

        for scope in expired_scopes:
            self.cleanup_temp_scope(scope)

        return {"status": "ok", "expired_count": len(expired_scopes), "scopes": expired_scopes}

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def _parse_paper_ids_from_file_path(self, file_path: str) -> Optional[int]:
        """从 LightRAG chunk 的 file_path 中解析 paper_id。

        写入路径约定：
          - 正式论文：ainsert(..., file_paths=["paper_{pid}.pdf"]) -> 返回 int
          - 临时 OCR：ainsert(..., file_paths=["temp_xxx.txt"]) -> 返回 None
          - 未知来源 ("unknown_source") -> 返回 None
        """
        if not file_path:
            return None
        # 兼容几种可能的命名：
        #   "paper_123.pdf" / "paper-123.pdf" / "paper_123"
        m = re.match(r"^paper[_\-](\d+)", file_path)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
        strict_paper_filter: bool = False,
        session_scope: Optional[str] = None,
        include_global: bool = False,
    ) -> Dict[str, Any]:
        """LightRAG naive 向量检索 + mix 图检索，返回与 Legacy 一致的 Dict 格式。"""
        self._ensure_initialized()
        from lightrag import QueryParam

        # 1. LightRAG 的 mix/global 模式内部会自行做关键词扩展，
        #    这里直接用原始 query 即可，保留 expanded_query 变量名保持下游兼容
        expanded_query = query_text

        # 2. 并行跑 naive 向量模式 + mix 图模式，取并集
        async def _do_search():
            naive_param = QueryParam(
                mode="naive",
                top_k=max(top_k * 3, 20),
                chunk_top_k=max(top_k * 3, 20),
            )
            mix_param = QueryParam(
                mode="mix",
                top_k=max(top_k * 2, 15),
                chunk_top_k=max(top_k * 2, 15),
            )
            naive_data = await self._rag.aquery_data(expanded_query, param=naive_param)
            mix_data = await self._rag.aquery_data(expanded_query, param=mix_param)
            return naive_data, mix_data

        naive_data, mix_data = self._submit(_do_search, timeout=60)

        # 3. 合并 chunks 并转成 Legacy 格式
        combined_chunks: List[Dict[str, Any]] = []
        seen = set()

        for payload in (naive_data, mix_data):
            if not isinstance(payload, dict):
                continue
            data_section = payload.get("data") or {}
            for chunk in data_section.get("chunks", []) or []:
                content = chunk.get("content", "")
                if not content or content in seen:
                    continue
                seen.add(content)

                # 从 aquery_data 返回的 chunk 里拿 file_path，解析 paper_id
                chunk_file_path = chunk.get("file_path", "")
                parsed_paper_id = self._parse_paper_ids_from_file_path(chunk_file_path)

                # 尝试用 entities + relationships 提供图上下文
                graph_snippets: List[str] = []
                if payload is mix_data:
                    for e in data_section.get("entities", [])[:5]:
                        en = e.get("entity_name", "")
                        ed = e.get("description", "")
                        if en and ed:
                            graph_snippets.append(f"{en}: {ed}")

                combined_chunks.append(
                    {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "text": content,
                        "score": 0.0,  # LightRAG 不返回原始向量距离
                        "chunk_index": None,
                        "paper_id": parsed_paper_id,
                        "title": None,
                        "year": None,
                        "session_scope": None,
                        "is_temp": parsed_paper_id is None,
                        "file_path": chunk_file_path,
                        "graph_context": graph_snippets[:5] if graph_snippets else None,
                    }
                )

        # 4. 按 paper_ids 过滤（Phase 2 已启用，因为 ainsert 写入时带了 file_path）
        target_ids = set(paper_ids) if paper_ids else None
        if target_ids:
            filtered = []
            for c in combined_chunks:
                cid = c.get("paper_id")
                if cid is not None and cid in target_ids:
                    filtered.append(c)
            combined_chunks = filtered
            if not combined_chunks:
                print(
                    f"[LightRAGAdapter] similarity_search: paper_ids={paper_ids} 过滤后无结果，"
                    f"strict_paper_filter={strict_paper_filter}"
                )

        # 5. truncate
        results = combined_chunks[:top_k]

        return {
            "query_text": query_text,
            "expanded_query": expanded_query,
            "top_k": top_k,
            "results": results,
            "search_mode": "lightrag_mix_naive_union",
            "count": len(results),
        }

    def rag_search(
        self,
        query_text: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
        session_scope: Optional[str] = None,
        include_global: bool = False,
    ) -> Dict[str, Any]:
        """用 LightRAG mix 模式的 aquery_llm 生成 LLM 答案。"""
        self._ensure_initialized()
        from lightrag import QueryParam

        # LightRAG 的 mix/global 模式内部会自行做关键词扩展，直接用原始 query
        expanded_query = query_text

        async def _do():
            param = QueryParam(
                mode="mix",
                top_k=top_k,
                chunk_top_k=top_k,
            )
            result = await self._rag.aquery_llm(expanded_query, param=param)
            return result

        result = self._submit(_do, timeout=120)

        # LightRAG aquery_llm 返回 dict: {"llm_response": {"content": str, ...}, "data": {...}, ...}
        llm_response = (result.get("llm_response") if isinstance(result, dict) else None) or {}
        answer = llm_response.get("content", "") or ""

        data_section = {}
        if isinstance(result, dict):
            # LightRAG aquery_llm 直接返回 {"data": {...}, "llm_response": {...}, ...} — 无 raw_data 包装层
            data_section = result.get("data") or {}

        chunks = data_section.get("chunks", []) or []
        normalized_chunks = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("content", ""),
                "file_path": c.get("file_path", ""),
            }
            for c in chunks
        ]

        return {
            "query_text": query_text,
            "expanded_query": expanded_query,
            "answer": answer.strip(),
            "results": normalized_chunks,
            "top_k": top_k,
        }

    # ------------------------------------------------------------------
    # 摘要 / 图谱上下文 —— CSO 本体对齐已移除，以下方法均降级返回 unavailable
    # ------------------------------------------------------------------

    def get_graph_refined_context(
        self,
        paper_id: int,
        top_entities: int = 25,
        snippets_per_entity: int = 5,
        neighbor_limit: int = 5,
        section_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """CSO 本体对齐已移除，LightRAG 不支持 Concept 级图精炼。"""
        return {
            "paper_id": paper_id,
            "backend": "lightrag",
            "status": "unavailable",
            "reason": "CSO 本体对齐已移除，LightRAG 不支持 Concept 级图精炼",
            "graph_context": [],
        }

    def generate_paper_summary(
        self,
        paper_id: int,
        top_entities: int = 10,
        snippets_per_entity: int = 2,
        neighbor_limit: int = 5,
        recursive_group_size: int = 4,
        section_aware: bool = False,
    ) -> Dict[str, Any]:
        """CSO 本体对齐已移除，章节感知摘要暂不可用。"""
        return {
            "paper_id": paper_id,
            "backend": "lightrag",
            "status": "unavailable",
            "reason": "CSO 本体对齐已移除，章节感知摘要暂不可用。LightRAG 原生摘要功能待后续接入。",
            "summary": "",
        }

    def generate_query_concept_summary(
        self,
        query_text: str,
        paper_ids: List[int],
        anchor_entity_limit: int = 10,
        concept_max_hops: int = 2,
        direct_snippets_per_paper: int = 6,
        expanded_snippets_per_paper: int = 6,
    ) -> Dict[str, Any]:
        """CSO 本体对齐已移除，Concept 级跨论文摘要暂不可用。"""
        return {
            "query_text": query_text,
            "backend": "lightrag",
            "status": "unavailable",
            "reason": "CSO 本体对齐已移除，Concept 级跨论文摘要暂不可用",
            "summary": "",
        }

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
        """CSO 本体对齐已移除，Concept 对齐的相关论文推荐暂不可用。"""
        return {
            "paper_id": paper_id,
            "backend": "lightrag",
            "status": "unavailable",
            "reason": "CSO 本体对齐已移除，Concept 对齐的相关论文推荐暂不可用",
            "related_papers": [],
        }

    # ------------------------------------------------------------------
    # 调试信息
    # ------------------------------------------------------------------

    def debug_info(self) -> Dict[str, Any]:
        base = super().debug_info()
        base.update(
            {
                "backend": "lightrag",
                "initialized": self._initialized,
                "working_dir": self._lr_settings.working_dir,
                "workspace": self._lr_settings.workspace or "(default)",
                "embedding_model": self._gr_settings.embedding_model,
                "llm_model": self._gr_settings.llm_model,
                "thread_alive": self._thread.is_alive() if self._thread else False,
            }
        )
        return base
