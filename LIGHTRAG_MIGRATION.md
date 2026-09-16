# LightRAG 迁移记录

> Phase 1 + Phase 2 实施完成（2026-09-16）
> 目标：将 LightRAG 接入现有 RAG 体系，逐步替代自研 GraphRAG 模块

---

## 一、改了什么

### 1.1 新建文件（8 个，全部在 `src/rag_service/`）

| 文件 | 行数 | 职责 |
|------|------|------|
| `config.py` | 12 | `RAG_BACKEND` 环境变量，路由 legacy / lightrag |
| `interface.py` | 175 | `RAGService` ABC 抽象接口，17 个 abstractmethod |
| `legacy_adapter.py` | 267 | `LegacyGraphRAGAdapter`，包装现有 `GraphRAGService`，参数签名对齐 |
| `lightrag_config.py` | 111 | `LightRAGSettings` dataclass，从 `.env` 读取 LightRAG 专属配置 |
| `embedding_wrapper.py` | 75 | `make_local_embedding_func()`，将 `LocalEmbeddings` 包装成 LightRAG `EmbeddingFunc` |
| `llm_wrapper.py` | 58 | `make_llm_func()`，将 `ask_messages` 包装成 LightRAG async llm_func |
| `lightrag_adapter.py` | 920 | `LightRAGAdapter` 完整实现，守护线程事件循环桥接 sync→async |
| `__init__.py` | 57 | 导出层 + `get_rag_service()` 单例工厂 + 转导出 `GRAPHRAG_IMPORT_ERROR` / `DEFAULT_ONTOLOGY_FILE` |

### 1.2 修改的文件（3 个）

| 文件 | 改动 |
|------|------|
| `src/graphrag_module/routes.py` | 第 25 行 import 源从 `graphrag_module.service` 改为 `rag_service`；第 131 行 `get_graphrag_service()` → `get_rag_service()`；新增 `is_lightrag_backend` 检测分支处理 LightRAG 后端 diagnose |
| `src/routers/template.py` | 第 44 行 import 源从 `graphrag_module.service` 改为 `rag_service`；第 174 行 `get_graphrag_service()` → `get_rag_service()` |
| `requirements.txt` | 末尾追加 `lightrag-hku>=1.0.0` |

### 1.3 未修改的文件（有意保持原样）

- `src/graphrag_module/service.py` — `GraphRAGService` 完整保留，由 `LegacyGraphRAGAdapter` 持有（LightRAG 后端已移除 CSO 本体对齐依赖）
- `src/graphrag_module/config.py` — `GraphRAGSettings` 未动
- `src/graphrag_module/lifecycle.py` — 保持原样，LightRAGAdapter 初始化在 main.py lifespan 里自动触发
- `src/main.py` — 未改导入和路由注册（它们间接通过 `rag_service` 走）

---

## 二、架构变更

### 之前

```
routes.py → get_graphrag_service() → GraphRAGService (单例)
                                    ├─ LocalEmbeddings (SQLite cache)
                                    ├─ neo4j_driver (bolt://localhost:7687)
                                    └─ VectorRetriever
```

### 现在

```
routes.py / template.py → get_rag_service()  ← 路由切换
    │
    ├─ RAG_BACKEND=legacy
    │   └─ LegacyGraphRAGAdapter → GraphRAGService (同之前)
    │
    └─ RAG_BACKEND=lightrag
        ├─ LightRAG (NetworkXStorage + NanoVectorDBStorage + JsonKVStorage)
        │   ├─ ainsert / aquery_data / aquery_llm
        │   ├─ aquery (mix/naive/local/global/hybrid/bypass)
        │   └─ initialize_storages / finalize_storages
        ├─ LocalEmbeddings (复用，SQLite cache + 批量)
        └─ ask_messages (复用，asyncio.to_thread 桥接)
```

### 关键设计决策

1. **Adapter/Service Layer 模式**：`RAGService` ABC 定义统一接口，`LegacyGraphRAGAdapter` 和 `LightRAGAdapter` 分别实现，上层 routes.py 零感知
2. **CSO 本体对齐已移除**：原委托给 `GraphRAGService` 的 4 个方法（`generate_paper_summary`、`related_papers_by_id`、`get_graph_refined_context`、`generate_query_concept_summary`）在 LightRAG 后端均降级返回 `{"status": "unavailable"}` 占位字典，后续按需要补回原生实现；**LightRAGAdapter 不再依赖 Neo4j**（LegacyGraphRAGAdapter 仍依赖）
3. **守护线程事件循环**：LightRAG 全 async，但 FastAPI 路由是 sync；用 daemon 线程跑 `run_forever()`，所有 `async X` 通过 `asyncio.run_coroutine_threadsafe` 提交
4. **import 延迟**：`lightrag_adapter.py` 顶部无任何 `from lightrag`，所有 LightRAG import 都在方法内部；Legacy 模式不装 lightrag-hku 也能正常运行
5. **文件写入用 ainsert + file_paths**：`paper_{pid}.pdf` 格式让 LightRAG chunk 保留 paper_id 元数据，`similarity_search(paper_ids=[...])` 精确过滤

---

## 三、Bug 修复记录

| Bug | 严重度 | 位置 | 根因 | 修复 |
|-----|--------|------|------|------|
| BUG-1 rag_search chunks 永远空 | Critical | `lightrag_adapter.py` L802-L805 | `aquery_llm` 返回 `{"llm_response", "data"}`，代码却读 `result.get("raw_data").get("data")` | 改为 `result.get("data")` |
| BUG-2 llm_wrapper 顶部硬 import utils.model | High | `llm_wrapper.py` L20 | Legacy 模式下 import rag_service 包会触发 utils.model 失败 | 移到 `make_llm_func` 工厂函数内部 |
| BUG-3 diagnose LightRAG 后端崩溃 | High | `routes.py` L154-L212 | `LightRAGAdapter.driver=None` → `.session()` AttributeError | 加 `is_lightrag_backend` 检测分支 |
| BUG-4 workspace 空字符串被错转 None | Medium | `lightrag_config.py` L75 | `self.workspace or None` 把空字符串转 None | 改为 `self.workspace` |
| BUG-5 nonlocal 并发竞态 | Medium | `lightrag_adapter.py` `upsert_paper_chunks` + `sync_from_mysql` | `nonlocal` 在 async closure 里修改外层变量，语义不清且 `inserted_chunks` 从未增加 | 用 `gather return_exceptions=True` + 返回值收集；增加 `p.id is None` 防御 |
| BUG-6 max_tokens 参数名 fallback 缺失 | Medium | `llm_wrapper.py` L45 | LightRAG 不同版本可能传 `max_output_token` | 改为 `kwargs.get("max_tokens") or kwargs.get("max_output_token", 4096)` |

---

## 四、启动命令

### Legacy 模式（默认）

```bash
cd /Users/mambaweizzw/Desktop/实验室/ResearchGraph-RAG
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### LightRAG 模式

```bash
cd /Users/mambaweizzw/Desktop/实验室/ResearchGraph-RAG

# 1. 安装依赖
pip install -r requirements.txt
pip install -e ./LightRAG      # 或 pip install lightrag-hku

# 2. 创建 .env（从 example 复制后追加 LightRAG 段）
[ -f .env ] || cp .env.example .env
cat >> .env << 'EOF'

# ===== LightRAG Backend =====
RAG_BACKEND=lightrag
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage
LIGHTRAG_WORKING_DIR=./rag_storage
TOP_K=40
CHUNK_TOP_K=20
MAX_ENTITY_TOKENS=6000
MAX_RELATION_TOKENS=8000
MAX_TOTAL_TOKENS=30000
MAX_PARALLEL_INSERT=3
MAX_ASYNC_LLM=4
EMBEDDING_FUNC_MAX_ASYNC=8
EOF

# 3. 启动
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 验证端点

```bash
# LightRAG 健康检查
curl http://localhost:8000/api/graphrag/health
curl http://localhost:8000/api/graphrag/diagnose  # backend=lightrag

# 写入 + 检索
curl -X POST http://localhost:8000/api/graphrag/sync-from-mysql -H "Content-Type: application/json" -d '{"limit": 5}'
curl -X POST http://localhost:8000/api/graphrag/similarity-search -H "Content-Type: application/json" -d '{"query_text": "graph neural network", "top_k": 5}'
curl -X POST http://localhost:8000/api/graphrag/search -H "Content-Type: application/json" -d '{"query_text": "graph neural network", "top_k": 3}'
```

### 回退 Legacy

```bash
RAG_BACKEND=legacy uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 五、能力对比

| 能力 | Legacy (自研 GraphRAG) | LightRAG 模式 | 匹配度 |
|------|----------------------|---------------|--------|
| 文档导入 | MySQL KB → 自研切片 | MySQL KB → LightRAG ainsert | 高 |
| 文档切分 | 自研 5 层（章节→标题→段落→语义跳变→语言自适应） | LightRAG 内置 fixed-token（Phase 3 回归自研切片） | 中 |
| Embedding | LocalEmbeddings（SQLite cache + 批量） | 复用 LocalEmbeddings + EmbeddingFunc 包装 | 高 |
| 向量检索 | Neo4j Vector Index CALL | LightRAG mix + naive 双路 | 高 |
| 知识图谱 | 自研 Cypher Paper/Chunk/Entity/Concept | LightRAG 自管理实体/关系图 | 中 |
| 混合检索 | 向量+关键词+放宽 paper_ids 三层回退 | LightRAG mix 模式（含图+向量） | 高 |
| **Rerank** | ❌ 未实现 | ✅ LightRAG 内置（需配 Cohere/Jina/vLLM） | **LightRAG 反超** |
| Query 实体扩展 | _expand_query_with_entities() | LightRAG mix/global 模式内部扩展 | 中 |
| 章节感知摘要 | ✅ 自研递归 LLM | ❌ 已移除（CSO 本体对齐） | **Legacy 独有** |
| 相关论文推荐 | ✅ Concept 对齐 + 向量加权 | ❌ 已移除（CSO 本体对齐） | **Legacy 独有** |
| 临时 OCR (session_scope+TTL) | ✅ 负数 paper_id + session_scope | LightRAG workspace 隔离 + 手动 TTL | 中 |
| 数据持久化 | Neo4j | NetworkX + NanoVectorDB + JsonKV（可切 PG） | 可配置 |
| 存储后端可选 | 仅 Neo4j | 12+ 后端可通过 .env 切换 | **LightRAG 反超** |
| **Neo4j 依赖** | ✅ 启动必需 | ❌ 完全不需要 | **LightRAG 反超** |

---

## 六、已知局限

1. **自研切片未回归**：LightRAG Adapter 用 `ainsert` 让 LightRAG 自行切片，丢失了章节感知 + 语言自适应 + 章节丢弃（references/acknowledgment）。Phase 3 回归 `ainsert_custom_chunks` + 自研切片。
2. **Paper_ids 过滤依赖 ainsert file_path**：如果文档写入时没设 `file_paths=["paper_{pid}.pdf"]`，`similarity_search(paper_ids=[...])` 无法精确过滤。当前 sync_from_mysql 和 upsert_paper_chunks 都正确设置了。
3. **Embedding 维度必须统一**：`.env` 里 `GRAPHRAG_EMBEDDING_DIMENSIONS=4096`，LightRAG `embedding_dim` 从这个值读取。如果实际 Embedding 服务返回不同维度，会在 adapter 初始化时打 warning。
4. **4 个 CSO 方法降级**：`get_graph_refined_context`、`generate_paper_summary`、`generate_query_concept_summary`、`related_papers_by_id` 在 LightRAG 后端返回 `{"status": "unavailable"}` 占位字典，前端需判断此状态并展示"暂不可用"。

---

## 七、后续 Phase 路线图

- **Phase 3**：回归自研切片（`ainsert_custom_chunks` + `_hierarchical_semantic_chunking`），Rerank 补配
- **Phase 4**：评估 LightRAG 与 Legacy 的检索准确率 / Recall / 响应时间
- **Phase 5**：根据评估结果决定是否清理 Legacy 代码
- **Phase 6**：数据迁移脚本（从 MySQL KB 全量重建 LightRAG 知识库）

---

*文档生成时间：2026-09-16 | 实施范围：LightRAG 接入 Phase 1 + Phase 2*
