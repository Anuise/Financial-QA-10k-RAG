# 階段六:部署與介面 (Deployment & Interface)

## 1. 系統定位與核心價值 (Context & Value)

### 1.1 階段定位

部署與介面是 RAG 系統的最終交付階段,負責將前五個階段建構的核心能力封裝為可用的服務,並透過友善的使用者介面讓終端使用者能夠輕鬆存取財務問答功能。這個階段決定了系統的可用性與使用者體驗。

### 1.2 核心痛點

將 RAG 系統從開發環境遷移至生產環境面臨以下挑戰:

**挑戰一:服務化**

- **問題**:如何將 Jupyter Notebook 中的原型轉換為穩定的 API 服務?
- **需求**:高可用性、錯誤處理、請求驗證

**挑戰二:使用者體驗**

- **問題**:如何讓非技術使用者也能輕鬆使用系統?
- **需求**:直觀的介面、即時回饋、來源追溯

**挑戰三:效能與成本**

- **問題**:如何在有限資源下支撐多使用者並發?
- **需求**:快取機制、負載平衡、成本控制

**解決方案**:

1. **FastAPI Backend**:提供 RESTful API,支援非同步處理
2. **Streamlit Frontend**:快速建構互動式 Web 介面
3. **監控與日誌**:即時追蹤系統健康狀態與使用情況

### 1.3 預期效果

完成本階段後,系統將具備:

1. **生產級 API**:穩定的 `/search` 與 `/chat` Endpoints
2. **友善介面**:支援對話歷史、來源跳轉、Confidence 顯示
3. **可觀測性**:完整的日誌、指標與追蹤
4. **可擴展性**:支援水平擴展與負載平衡

---

> **架構銜接說明**:
> 了解部署的挑戰後,下一層將說明如何透過「Backend 服務→Frontend 介面→監控系統」的三層架構,實現完整的生產環境部署。

---

## 2. 工作流程與架構 (Workflow & Architecture)

### 2.1 整體架構

部署架構採用前後端分離設計:

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Streamlit Web Interface                  │   │
│  │  - Chat UI                                       │   │
│  │  - Source Citation Display                       │   │
│  │  - Confidence Score Visualization                │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         ↓ HTTP/REST
┌─────────────────────────────────────────────────────────┐
│                    Backend Layer                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │         FastAPI Application                      │   │
│  │  - /search: 單次查詢                             │   │
│  │  - /chat: 對話式問答                             │   │
│  │  - /health: 健康檢查                             │   │
│  └──────────────────────────────────────────────────┘   │
│                         ↓                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │         RAG Pipeline                             │   │
│  │  Retrieval → Reranking → Generation              │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  ChromaDB   │  │ BM25 Index  │  │   Cache     │     │
│  │  (Vectors)  │  │  (Sparse)   │  │   (Redis)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  Monitoring Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Logging   │  │   Metrics   │  │   Tracing   │     │
│  │  (File/DB)  │  │ (Prometheus)│  │  (Jaeger)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Backend: FastAPI 服務

**為何選擇 FastAPI?**

| 特性         | FastAPI       | Flask     | Django    |
| ------------ | ------------- | --------- | --------- |
| **效能**     | 高 (非同步)   | 中 (同步) | 中 (同步) |
| **自動文檔** | ✅ (Swagger)  | ❌        | ❌        |
| **型別檢查** | ✅ (Pydantic) | ❌        | 部分      |
| **學習曲線** | 低            | 低        | 高        |
| **適用場景** | API 服務      | 小型應用  | 全端應用  |

**API 設計**:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="Financial QA RAG API",
    description="10-K Financial Reports Question Answering System",
    version="1.0.0"
)

# 請求模型
class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    alpha: float = 0.5
    include_sources: bool = True

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    max_history: int = 5

# 回應模型
class Source(BaseModel):
    source_id: int
    chunk_id: str
    document: str
    section: str
    excerpt: str
    rerank_score: float

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]
    confidence: dict
    metadata: dict

# Endpoints
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    單次查詢 Endpoint

    Args:
        request: 查詢請求

    Returns:
        查詢結果
    """
    try:
        # 執行 RAG Pipeline
        result = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha
        )

        return SearchResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    對話式問答 Endpoint

    支援多輪對話,自動維護上下文
    """
    try:
        # 載入對話歷史
        history = conversation_manager.get_history(request.conversation_id)

        # 執行查詢
        result = rag_pipeline.query_with_history(
            query=request.query,
            history=history[-request.max_history:]
        )

        # 儲存對話
        conversation_manager.add_turn(
            conversation_id=request.conversation_id,
            query=request.query,
            answer=result['answer']
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "chromadb": chromadb_client.heartbeat(),
        "bm25_loaded": bm25_index is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2.3 Frontend: Streamlit 介面

**Streamlit 優勢**:

- 純 Python 開發,無需 HTML/CSS/JS
- 內建元件豐富 (Chat, Sidebar, Metrics)
- 自動處理狀態管理

**介面設計**:

```python
import streamlit as st
import requests
from typing import List, Dict

# 頁面配置
st.set_page_config(
    page_title="Financial QA System",
    page_icon="💰",
    layout="wide"
)

# 側邊欄配置
with st.sidebar:
    st.title("⚙️ 設定")

    # 檢索參數
    top_k = st.slider("Top-K", min_value=5, max_value=50, value=20)
    alpha = st.slider("Alpha (Dense 權重)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # 模型選擇
    llm_model = st.selectbox("LLM 模型", ["gpt-4", "gpt-3.5-turbo", "claude-3"])

    # 清除歷史
    if st.button("清除對話歷史"):
        st.session_state.messages = []
        st.rerun()

# 主標題
st.title("💰 Financial QA System")
st.caption("基於 10-K 財報的智慧問答系統")

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示對話歷史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 顯示來源引用
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 查看來源"):
                for source in message["sources"]:
                    st.markdown(f"""
                    **[Source {source['source_id']}]** {source['document']} - {source['section']}

                    > {source['excerpt']}

                    *Rerank Score: {source['rerank_score']:.3f}*
                    """)

            # 顯示 Confidence
            if "confidence" in message:
                conf = message["confidence"]
                st.metric(
                    "Confidence",
                    f"{conf['overall_confidence']:.2%}",
                    delta=conf['confidence_level']
                )

# 使用者輸入
if prompt := st.chat_input("請輸入您的問題..."):
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 呼叫 API
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = requests.post(
                "http://localhost:8000/search",
                json={
                    "query": prompt,
                    "top_k": top_k,
                    "alpha": alpha
                }
            )

            if response.status_code == 200:
                result = response.json()

                # 顯示答案
                st.markdown(result["answer"])

                # 儲存至歷史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "confidence": result["confidence"]
                })

                st.rerun()
            else:
                st.error(f"API 錯誤: {response.status_code}")
```

### 2.4 對話管理

**對話歷史儲存**:

```python
from typing import List, Dict, Optional
import json
from pathlib import Path

class ConversationManager:
    """對話管理器"""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(exist_ok=True)

    def get_history(self, conversation_id: Optional[str]) -> List[Dict]:
        """取得對話歷史"""
        if not conversation_id:
            return []

        history_file = self.storage_dir / f"{conversation_id}.json"
        if not history_file.exists():
            return []

        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def add_turn(
        self,
        conversation_id: str,
        query: str,
        answer: str
    ):
        """新增對話輪次"""
        history = self.get_history(conversation_id)

        history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer
        })

        # 儲存
        history_file = self.storage_dir / f"{conversation_id}.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
```

---

> **細節銜接說明**:
> 確立了「FastAPI Backend + Streamlit Frontend」的架構後,以下將深入說明部署配置、監控系統與效能優化策略。

---

## 3. 技術規格與實作細節 (Detailed Specification)

### 3.1 部署配置

#### 3.1.1 Docker 容器化

**Dockerfile (Backend)**:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式
COPY . .

# 下載模型 (可選,也可掛載 Volume)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-m3')"

# 暴露端口
EXPOSE 8000

# 啟動服務
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:

```yaml
version: "3.8"

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMADB_HOST=chromadb
    depends_on:
      - chromadb
      - redis

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chromadb_data:/chroma/chroma

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  chromadb_data:
  redis_data:
```

#### 3.1.2 環境變數管理

**.env 範例**:

```bash
# LLM API
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
CHROMADB_HOST=localhost
CHROMADB_PORT=8001

# Cache
REDIS_HOST=localhost
REDIS_PORT=6379

# Application
LOG_LEVEL=INFO
MAX_WORKERS=4
CACHE_TTL=3600
```

**配置載入**:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """應用程式設定"""

    # LLM
    openai_api_key: str
    anthropic_api_key: str = None

    # Database
    chromadb_host: str = "localhost"
    chromadb_port: int = 8001

    # Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl: int = 3600

    # Application
    log_level: str = "INFO"
    max_workers: int = 4

    class Config:
        env_file = ".env"

settings = Settings()
```

### 3.2 快取策略

#### 3.2.1 Redis 快取

```python
import redis
import json
import hashlib

class QueryCache:
    """查詢快取"""

    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl

    def get_cache_key(self, query: str, params: dict) -> str:
        """生成快取鍵"""
        key_str = f"{query}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, params: dict) -> Optional[dict]:
        """取得快取"""
        key = self.get_cache_key(query, params)
        cached = self.redis.get(key)

        if cached:
            return json.loads(cached)
        return None

    def set(self, query: str, params: dict, result: dict):
        """設定快取"""
        key = self.get_cache_key(query, params)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(result, ensure_ascii=False)
        )
```

#### 3.2.2 快取整合至 API

```python
@app.post("/search")
async def search_with_cache(request: SearchRequest):
    """帶快取的查詢"""

    # 1. 檢查快取
    cache_params = {
        "top_k": request.top_k,
        "alpha": request.alpha
    }

    cached_result = query_cache.get(request.query, cache_params)
    if cached_result:
        logger.info(f"Cache hit for query: {request.query}")
        return SearchResponse(**cached_result)

    # 2. 執行查詢
    result = rag_pipeline.query(
        query=request.query,
        top_k=request.top_k,
        alpha=request.alpha
    )

    # 3. 儲存快取
    query_cache.set(request.query, cache_params, result)

    return SearchResponse(**result)
```

### 3.3 監控與日誌

#### 3.3.1 結構化日誌

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """結構化日誌記錄器"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 檔案處理器
        handler = logging.FileHandler('app.log')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_query(
        self,
        query: str,
        result: dict,
        latency_ms: float,
        cache_hit: bool
    ):
        """記錄查詢日誌"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "query",
            "query": query,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "confidence": result['confidence']['overall_confidence'],
            "num_sources": len(result['sources']),
            "tokens_used": result['metadata']['tokens_used']['total']
        }

        self.logger.info(json.dumps(log_entry, ensure_ascii=False))

    def log_error(self, error: Exception, context: dict):
        """記錄錯誤日誌"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }

        self.logger.error(json.dumps(log_entry, ensure_ascii=False))
```

#### 3.3.2 Prometheus 指標

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# 定義指標
query_counter = Counter('rag_queries_total', 'Total number of queries')
query_latency = Histogram('rag_query_latency_seconds', 'Query latency in seconds')
cache_hit_counter = Counter('rag_cache_hits_total', 'Total cache hits')
confidence_gauge = Gauge('rag_confidence_score', 'Current confidence score')

@app.post("/search")
async def search_with_metrics(request: SearchRequest):
    """帶指標的查詢"""

    start_time = time.time()

    try:
        # 執行查詢
        result = rag_pipeline.query(request.query)

        # 記錄指標
        query_counter.inc()
        query_latency.observe(time.time() - start_time)
        confidence_gauge.set(result['confidence']['overall_confidence'])

        return SearchResponse(**result)

    except Exception as e:
        logger.log_error(e, {"query": request.query})
        raise
```

#### 3.3.3 監控儀表板

**Grafana Dashboard 配置**:

```json
{
  "dashboard": {
    "title": "RAG System Monitoring",
    "panels": [
      {
        "title": "Query Rate",
        "targets": [
          {
            "expr": "rate(rag_queries_total[5m])"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rag_query_latency_seconds)"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(rag_cache_hits_total[5m]) / rate(rag_queries_total[5m])"
          }
        ]
      },
      {
        "title": "Average Confidence",
        "targets": [
          {
            "expr": "avg_over_time(rag_confidence_score[5m])"
          }
        ]
      }
    ]
  }
}
```

### 3.4 效能優化

#### 3.4.1 非同步處理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRAGPipeline:
    """非同步 RAG Pipeline"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def query_async(self, query: str) -> dict:
        """非同步查詢"""

        # 1. 並行執行檢索
        dense_task = asyncio.create_task(self._dense_search_async(query))
        sparse_task = asyncio.create_task(self._sparse_search_async(query))

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # 2. 融合結果
        fused_results = self._fuse_results(dense_results, sparse_results)

        # 3. Reranking (在執行緒池中執行,避免阻塞)
        loop = asyncio.get_event_loop()
        reranked = await loop.run_in_executor(
            self.executor,
            self._rerank,
            query,
            fused_results
        )

        # 4. LLM 生成
        answer = await self._generate_async(query, reranked)

        return answer

    async def _dense_search_async(self, query: str):
        """非同步向量檢索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.dense_retriever.search,
            query
        )
```

#### 3.4.2 連線池

```python
from chromadb import HttpClient

class ChromaDBPool:
    """ChromaDB 連線池"""

    def __init__(self, host: str, port: int, pool_size: int = 10):
        self.clients = [
            HttpClient(host=host, port=port)
            for _ in range(pool_size)
        ]
        self.current_idx = 0

    def get_client(self) -> HttpClient:
        """取得客戶端 (Round-robin)"""
        client = self.clients[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.clients)
        return client
```

### 3.5 安全性

#### 3.5.1 Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/search")
@limiter.limit("10/minute")  # 每分鐘最多 10 次請求
async def search_with_rate_limit(request: Request, search_request: SearchRequest):
    """帶速率限制的查詢"""
    return await search(search_request)
```

#### 3.5.2 輸入驗證

```python
from pydantic import BaseModel, validator

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    alpha: float = 0.5

    @validator('query')
    def validate_query(cls, v):
        """驗證查詢"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        if len(v) > 500:
            raise ValueError("Query too long (max 500 characters)")
        return v.strip()

    @validator('top_k')
    def validate_top_k(cls, v):
        """驗證 Top-K"""
        if v < 1 or v > 100:
            raise ValueError("top_k must be between 1 and 100")
        return v

    @validator('alpha')
    def validate_alpha(cls, v):
        """驗證 Alpha"""
        if v < 0 or v > 1:
            raise ValueError("alpha must be between 0 and 1")
        return v
```

### 3.6 部署檢查清單

#### 3.6.1 上線前檢查

```markdown
## 部署檢查清單

### 功能測試

- [ ] API Endpoints 正常運作
- [ ] Frontend 介面正常顯示
- [ ] 來源引用正確連結
- [ ] Confidence Score 計算正確

### 效能測試

- [ ] 單次查詢延遲 < 3s
- [ ] 並發 10 使用者無錯誤
- [ ] 快取命中率 > 30%

### 安全性

- [ ] API Key 已設定環境變數
- [ ] Rate Limiting 已啟用
- [ ] 輸入驗證已實作

### 監控

- [ ] 日誌正常寫入
- [ ] Prometheus 指標可存取
- [ ] Grafana 儀表板正常顯示

### 文檔

- [ ] API 文檔已生成 (Swagger)
- [ ] 使用者手冊已撰寫
- [ ] 故障排除指南已準備
```

---

## 4. 專案總結

完成階段六後,整個 Financial-QA-10k-RAG 系統已具備完整的生產能力:

### 4.1 系統能力總覽

| 階段       | 核心能力     | 關鍵產出                       |
| ---------- | ------------ | ------------------------------ |
| **階段一** | 資料工程     | 高品質 Chunks (8420 個)        |
| **階段二** | 雙索引構建   | BGE-M3 Embeddings + BM25 Index |
| **階段三** | 混合檢索     | RRF/Weighted Sum 融合演算法    |
| **階段四** | 重排序與生成 | Cross-Encoder + GPT-4 生成     |
| **階段五** | 系統評估     | RAGAS Score 0.78 (目標 > 0.75) |
| **階段六** | 部署與介面   | FastAPI + Streamlit 生產服務   |

### 4.2 技術亮點

1. **混合檢索架構**:結合 Dense 與 Sparse 索引,兼顧語意理解與精確匹配
2. **Kaggle-Local 分流**:GPU 密集任務在 Kaggle 執行,本地僅需 CPU
3. **可觀測性**:完整的日誌、指標與追蹤系統
4. **使用者友善**:Streamlit 介面支援來源追溯與 Confidence 顯示

### 4.3 未來優化方向

1. **模型微調**:針對財報領域 Fine-tune BGE-M3 與 Reranker
2. **多模態支援**:處理財報中的圖表與視覺化內容
3. **即時更新**:支援增量索引更新,無需重建整個資料庫
4. **多語言支援**:擴展至中文、日文等其他語言的財報

---

**恭喜!您已完成 Financial-QA-10k-RAG 系統的完整技術規劃。**
