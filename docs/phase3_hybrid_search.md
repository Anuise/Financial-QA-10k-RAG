# 階段三:混合檢索邏輯 (Hybrid Search)

## 1. 系統定位與核心價值 (Context & Value)

### 1.1 階段定位

混合檢索是 RAG 系統的核心檢索引擎,負責整合階段二建立的 Dense Index (向量) 與 Sparse Index (BM25),透過智慧融合演算法,同時發揮語意理解與精確匹配的優勢,為使用者查詢找出最相關的文本片段。

### 1.2 核心痛點

單一檢索策略在財務問答場景中存在明顯的偏差:

**場景一:語意查詢**

- **問題**: "Which companies showed strong revenue growth in 2023?"
- **Dense Only**: ✅ 能理解 "strong growth" 的語意,找到相關段落
- **Sparse Only**: ❌ 可能錯過使用 "increased sales" 等同義表達的段落

**場景二:精確查詢**

- **問題**: "What is Apple's EBITDA for Q4 2023?"
- **Dense Only**: ❌ 可能返回其他公司或其他季度的 EBITDA
- **Sparse Only**: ✅ 精確匹配 "Apple", "EBITDA", "Q4 2023"

**混合檢索的價值**:透過融合兩種分數,系統能夠:

1. 在語意查詢時不遺漏同義表達
2. 在精確查詢時優先返回關鍵詞完全匹配的結果
3. 透過 Alpha 參數動態調整兩者權重

### 1.3 預期效果

完成本階段後,系統將具備:

1. **自適應檢索**:根據查詢類型自動調整 Dense/Sparse 權重
2. **高召回率**:Top-20 結果中包含 95%+ 的相關文檔
3. **高精確度**:Top-5 結果中至少 3 個高度相關
4. **低延遲**:單次查詢耗時 < 200ms

---

> **架構銜接說明**:
> 了解混合檢索的必要性後,下一層將說明如何透過「並行檢索→分數融合→結果排序」的流程,實現高效的混合檢索引擎。

---

## 2. 工作流程與架構 (Workflow & Architecture)

### 2.1 整體流程

混合檢索遵循以下處理流程:

```
使用者查詢 (Query)
    ↓
[2.2] 查詢預處理 (Query Processing)
    ↓
┌─────────────────────────────────┬─────────────────────────────────┐
│   Dense Retrieval               │   Sparse Retrieval              │
│   (Vector Search)               │   (BM25 Search)                 │
├─────────────────────────────────┼─────────────────────────────────┤
│ 1. Query → BGE-M3 Embedding     │ 1. Query → Tokenization         │
│ 2. ChromaDB.query(embedding)    │ 2. BM25.get_scores(tokens)      │
│ 3. 取得 Top-K (K=50)            │ 3. 取得 Top-K (K=50)            │
│ 4. 輸出: [(id, score_dense)]    │ 4. 輸出: [(id, score_sparse)]   │
└─────────────────────────────────┴─────────────────────────────────┘
    ↓                                   ↓
    └───────────────┬───────────────────┘
                    ↓
        [2.3] 分數融合 (Score Fusion)
            - Reciprocal Rank Fusion (RRF)
            - 或 Weighted Sum
                    ↓
        [2.4] 結果排序與過濾
            - 取 Top-N (N=20)
            - 去重與品質過濾
                    ↓
        返回最終結果 (Ranked Chunks)
```

### 2.2 查詢預處理 (Query Processing)

**目標**:標準化使用者查詢,提升檢索品質。

**處理步驟**:

1. **拼寫修正**:修正常見錯誤 (如 "reveue" → "revenue")
2. **查詢擴展**:新增同義詞 (如 "profit" → "profit OR net income")
3. **實體識別**:提取公司名稱、日期、數值等關鍵實體
4. **查詢分類**:判斷查詢類型 (事實型/分析型/比較型)

```python
from typing import Dict, List
import re

class QueryProcessor:
    """查詢預處理器"""

    # 財經同義詞字典
    SYNONYMS = {
        'profit': ['net income', 'earnings'],
        'revenue': ['sales', 'turnover'],
        'debt': ['liabilities', 'borrowings']
    }

    # 查詢類型模式
    QUERY_PATTERNS = {
        'factual': re.compile(r'\b(what|how much|how many)\b', re.I),
        'analytical': re.compile(r'\b(why|analyze|explain)\b', re.I),
        'comparative': re.compile(r'\b(compare|versus|vs|difference)\b', re.I)
    }

    def process(self, query: str) -> Dict:
        """
        處理查詢

        Returns:
            {
                'original': 原始查詢,
                'processed': 處理後查詢,
                'entities': 提取的實體,
                'query_type': 查詢類型,
                'suggested_alpha': 建議的 Alpha 值
            }
        """
        # 1. 實體識別
        entities = self._extract_entities(query)

        # 2. 查詢分類
        query_type = self._classify_query(query)

        # 3. 查詢擴展
        expanded_query = self._expand_query(query)

        # 4. 建議 Alpha 值
        suggested_alpha = self._suggest_alpha(query_type, entities)

        return {
            'original': query,
            'processed': expanded_query,
            'entities': entities,
            'query_type': query_type,
            'suggested_alpha': suggested_alpha
        }

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """提取實體"""
        entities = {
            'companies': [],
            'dates': [],
            'metrics': []
        }

        # 公司代碼 (1-5 個大寫字母)
        entities['companies'] = re.findall(r'\b[A-Z]{1,5}\b', query)

        # 日期 (年份、季度)
        entities['dates'] = re.findall(r'\b(20\d{2}|Q[1-4])\b', query)

        # 財務指標
        metrics_pattern = r'\b(revenue|profit|EBITDA|EPS|debt|assets)\b'
        entities['metrics'] = re.findall(metrics_pattern, query, re.I)

        return entities

    def _classify_query(self, query: str) -> str:
        """分類查詢類型"""
        for qtype, pattern in self.QUERY_PATTERNS.items():
            if pattern.search(query):
                return qtype
        return 'factual'  # 預設為事實型

    def _expand_query(self, query: str) -> str:
        """查詢擴展"""
        expanded = query
        for term, synonyms in self.SYNONYMS.items():
            if term in query.lower():
                # 新增同義詞
                expanded += ' ' + ' '.join(synonyms)
        return expanded

    def _suggest_alpha(self, query_type: str, entities: Dict) -> float:
        """建議 Alpha 值 (Dense 權重)"""
        # 若包含精確實體 (公司代碼、日期),降低 Dense 權重
        if entities['companies'] or entities['dates']:
            return 0.3  # 更依賴 Sparse (精確匹配)

        # 分析型查詢更依賴語意理解
        if query_type == 'analytical':
            return 0.7  # 更依賴 Dense (語意)

        # 預設平衡
        return 0.5
```

### 2.3 並行檢索

**Dense Retrieval** (向量檢索):

```python
def dense_search(query: str, collection, top_k: int = 50) -> List[Tuple[str, float]]:
    """
    向量檢索

    Returns:
        [(chunk_id, score), ...]
    """
    # 1. Query Embedding
    query_embedding = embed_query(query)  # 使用 BGE-M3

    # 2. ChromaDB 查詢
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["distances"]
    )

    # 3. 轉換為 (id, score) 格式
    # ChromaDB 返回距離,需轉換為相似度分數
    chunk_ids = results['ids'][0]
    distances = results['distances'][0]
    scores = [1 / (1 + dist) for dist in distances]  # 距離 → 相似度

    return list(zip(chunk_ids, scores))
```

**Sparse Retrieval** (BM25 檢索):

```python
def sparse_search(query: str, bm25_index, tokenizer, top_k: int = 50) -> List[Tuple[str, float]]:
    """
    BM25 檢索

    Returns:
        [(chunk_id, score), ...]
    """
    # 1. Tokenization
    query_tokens = tokenizer.tokenize(query)

    # 2. BM25 計分
    scores = bm25_index.get_scores(query_tokens)

    # 3. 取 Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]

    # 4. 取得對應的 Chunk IDs
    chunk_ids = [bm25_index.chunk_ids[i] for i in top_indices]

    return list(zip(chunk_ids, top_scores))
```

### 2.4 分數融合策略

**策略一:Reciprocal Rank Fusion (RRF)**

**優勢**:不需要正規化分數,對分數尺度不敏感。

**公式**:

$$
\text{RRF\_Score}(d) = \sum_{r \in \{r_{\text{dense}}, r_{\text{sparse}}\}} \frac{1}{k + \text{rank}_r(d)}
$$

其中 $k$ 是平滑參數 (預設 60)。

**實作**:

```python
def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    RRF 融合

    Args:
        dense_results: Dense 檢索結果 [(id, score), ...]
        sparse_results: Sparse 檢索結果 [(id, score), ...]
        k: 平滑參數

    Returns:
        融合後的結果 [(id, rrf_score), ...]
    """
    rrf_scores = {}

    # 計算 Dense 的 RRF 分數
    for rank, (chunk_id, _) in enumerate(dense_results, 1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)

    # 計算 Sparse 的 RRF 分數
    for rank, (chunk_id, _) in enumerate(sparse_results, 1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)

    # 排序
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results
```

**策略二:Weighted Sum**

**優勢**:可透過 Alpha 參數精細控制 Dense/Sparse 權重。

**公式**:

$$
\text{Score}(d) = \alpha \cdot \text{Score}_{\text{dense}}(d) + (1 - \alpha) \cdot \text{Score}_{\text{sparse}}(d)
$$

**實作**:

```python
def weighted_sum_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    alpha: float = 0.5
) -> List[Tuple[str, float]]:
    """
    加權求和融合

    Args:
        dense_results: Dense 檢索結果
        sparse_results: Sparse 檢索結果
        alpha: Dense 權重 (0-1)

    Returns:
        融合後的結果
    """
    # 1. 正規化分數至 [0, 1]
    dense_dict = normalize_scores(dict(dense_results))
    sparse_dict = normalize_scores(dict(sparse_results))

    # 2. 加權求和
    all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())
    fused_scores = {}

    for chunk_id in all_ids:
        dense_score = dense_dict.get(chunk_id, 0)
        sparse_score = sparse_dict.get(chunk_id, 0)
        fused_scores[chunk_id] = alpha * dense_score + (1 - alpha) * sparse_score

    # 3. 排序
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-Max 正規化"""
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

    if max_score == min_score:
        return {k: 1.0 for k in scores}

    return {
        k: (v - min_score) / (max_score - min_score)
        for k, v in scores.items()
    }
```

---

> **細節銜接說明**:
> 確立了「並行檢索→分數融合」的工作流程後,以下將深入說明 Alpha 參數調優、查詢效能優化與錯誤處理機制。

---

## 3. 技術規格與實作細節 (Detailed Specification)

### 3.1 Alpha 參數調優

#### 3.1.1 參數影響分析

Alpha 值決定了 Dense 與 Sparse 的權重比例:

| Alpha | Dense 權重 | Sparse 權重 | 適用場景                            |
| ----- | ---------- | ----------- | ----------------------------------- |
| 0.0   | 0%         | 100%        | 純關鍵字匹配 (如搜尋公司代碼)       |
| 0.3   | 30%        | 70%         | 精確查詢為主 (如 "AAPL Q4 revenue") |
| 0.5   | 50%        | 50%         | 平衡型查詢 (預設值)                 |
| 0.7   | 70%        | 30%         | 語意查詢為主 (如 "growth trends")   |
| 1.0   | 100%       | 0%          | 純語意匹配 (如 "risk factors")      |

#### 3.1.2 自動調優策略

**基於查詢特徵的動態 Alpha**:

```python
class AlphaOptimizer:
    """Alpha 參數自動調優器"""

    def __init__(self):
        # 基於歷史查詢的最佳 Alpha 值
        self.query_type_alphas = {
            'factual': 0.4,      # 事實型查詢偏向精確匹配
            'analytical': 0.7,   # 分析型查詢偏向語意理解
            'comparative': 0.6   # 比較型查詢平衡兩者
        }

    def optimize(self, query: str, entities: Dict) -> float:
        """
        根據查詢特徵優化 Alpha

        Args:
            query: 使用者查詢
            entities: 提取的實體

        Returns:
            最佳 Alpha 值
        """
        # 1. 基礎 Alpha (根據查詢類型)
        query_type = self._classify_query(query)
        alpha = self.query_type_alphas.get(query_type, 0.5)

        # 2. 根據實體調整
        # 若包含公司代碼或精確日期,降低 Alpha (更依賴 Sparse)
        if entities.get('companies') or entities.get('dates'):
            alpha -= 0.2

        # 3. 根據查詢長度調整
        # 長查詢更適合語意理解
        if len(query.split()) > 10:
            alpha += 0.1

        # 4. 限制範圍
        alpha = max(0.1, min(0.9, alpha))

        return alpha
```

#### 3.1.3 離線評估調優

**使用驗證集進行網格搜尋**:

```python
from sklearn.model_selection import ParameterGrid
import numpy as np

def grid_search_alpha(
    val_queries: List[str],
    val_relevance: List[List[str]],
    search_func: callable
) -> float:
    """
    網格搜尋最佳 Alpha

    Args:
        val_queries: 驗證集查詢
        val_relevance: 每個查詢的相關文檔 ID 列表
        search_func: 檢索函數 (接受 query, alpha,返回結果)

    Returns:
        最佳 Alpha 值
    """
    alphas = np.arange(0.0, 1.1, 0.1)
    best_alpha = 0.5
    best_score = 0

    for alpha in alphas:
        # 計算 MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []

        for query, relevant_ids in zip(val_queries, val_relevance):
            results = search_func(query, alpha)
            result_ids = [r[0] for r in results]

            # 找到第一個相關文檔的排名
            for rank, doc_id in enumerate(result_ids, 1):
                if doc_id in relevant_ids:
                    reciprocal_ranks.append(1 / rank)
                    break
            else:
                reciprocal_ranks.append(0)

        mrr = np.mean(reciprocal_ranks)

        if mrr > best_score:
            best_score = mrr
            best_alpha = alpha

    print(f"Best Alpha: {best_alpha:.1f}, MRR: {best_score:.4f}")
    return best_alpha
```

### 3.2 查詢效能優化

#### 3.2.1 並行檢索

**使用多執行緒加速**:

```python
from concurrent.futures import ThreadPoolExecutor
import time

def hybrid_search_parallel(
    query: str,
    chromadb_collection,
    bm25_index,
    alpha: float = 0.5,
    top_k: int = 50
) -> List[Tuple[str, float]]:
    """
    並行執行 Dense 與 Sparse 檢索

    Returns:
        融合後的結果
    """
    start_time = time.time()

    # 並行執行兩種檢索
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(dense_search, query, chromadb_collection, top_k)
        sparse_future = executor.submit(sparse_search, query, bm25_index, tokenizer, top_k)

        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # 融合結果
    fused_results = weighted_sum_fusion(dense_results, sparse_results, alpha)

    elapsed = time.time() - start_time
    print(f"⚡ Search completed in {elapsed*1000:.1f}ms")

    return fused_results[:20]  # 返回 Top-20
```

#### 3.2.2 查詢快取

**快取熱門查詢結果**:

```python
from functools import lru_cache
import hashlib

class SearchCache:
    """檢索結果快取"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def get_cache_key(self, query: str, alpha: float, top_k: int) -> str:
        """生成快取鍵"""
        key_str = f"{query}_{alpha}_{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, alpha: float, top_k: int):
        """取得快取結果"""
        key = self.get_cache_key(query, alpha, top_k)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, query: str, alpha: float, top_k: int, results):
        """儲存快取結果"""
        key = self.get_cache_key(query, alpha, top_k)

        # 若快取已滿,移除最少使用的項目
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]

        self.cache[key] = results
        self.access_count[key] = 0
```

#### 3.2.3 Top-K 優化

**動態調整 Top-K**:

```python
def adaptive_top_k(query: str, base_k: int = 50) -> int:
    """
    根據查詢複雜度動態調整 Top-K

    Args:
        query: 使用者查詢
        base_k: 基礎 K 值

    Returns:
        調整後的 K 值
    """
    # 簡單查詢 (< 5 個詞) 減少 K
    if len(query.split()) < 5:
        return base_k // 2

    # 複雜查詢 (> 15 個詞) 增加 K
    if len(query.split()) > 15:
        return base_k * 2

    return base_k
```

### 3.3 結果後處理

#### 3.3.1 去重

**移除高度相似的 Chunks**:

```python
def deduplicate_results(
    results: List[Tuple[str, float]],
    chunks: Dict[str, str],
    similarity_threshold: float = 0.9
) -> List[Tuple[str, float]]:
    """
    去除重複結果

    Args:
        results: 檢索結果 [(chunk_id, score), ...]
        chunks: Chunk ID → 文本的映射
        similarity_threshold: 相似度閾值

    Returns:
        去重後的結果
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(results) <= 1:
        return results

    # 取得文本
    texts = [chunks[chunk_id] for chunk_id, _ in results]

    # 計算 TF-IDF 相似度
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(tfidf_matrix)

    # 去重
    keep_indices = []
    for i in range(len(results)):
        # 檢查是否與已保留的結果過於相似
        is_duplicate = False
        for j in keep_indices:
            if similarities[i, j] > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep_indices.append(i)

    return [results[i] for i in keep_indices]
```

#### 3.3.2 品質過濾

**過濾低品質結果**:

```python
def filter_low_quality(
    results: List[Tuple[str, float]],
    chunks: Dict[str, Dict],
    min_score: float = 0.1,
    min_length: int = 50
) -> List[Tuple[str, float]]:
    """
    過濾低品質結果

    Args:
        results: 檢索結果
        chunks: Chunk ID → Metadata 的映射
        min_score: 最低分數閾值
        min_length: 最短長度 (詞數)

    Returns:
        過濾後的結果
    """
    filtered = []

    for chunk_id, score in results:
        # 檢查分數
        if score < min_score:
            continue

        # 檢查長度
        chunk_data = chunks[chunk_id]
        if chunk_data['metadata']['word_count'] < min_length:
            continue

        filtered.append((chunk_id, score))

    return filtered
```

### 3.4 錯誤處理

#### 3.4.1 空結果處理

```python
def handle_empty_results(
    query: str,
    results: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """處理空結果情況"""

    if not results:
        print(f"⚠️ No results found for query: '{query}'")

        # 嘗試放寬查詢條件
        # 1. 移除停用詞
        simplified_query = remove_stopwords(query)

        # 2. 降低 Top-K 閾值
        # 3. 使用查詢擴展

        return []

    return results
```

#### 3.4.2 檢索超時處理

```python
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds: int):
    """設定執行時間限制"""
    def signal_handler(signum, frame):
        raise TimeoutException("Search timeout")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def safe_search(query: str, timeout: int = 5):
    """帶超時保護的檢索"""
    try:
        with time_limit(timeout):
            return hybrid_search_parallel(query)
    except TimeoutException:
        print(f"⚠️ Search timeout after {timeout}s")
        return []
```

### 3.5 評估指標

#### 3.5.1 檢索品質指標

**Recall@K**:

```python
def recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    計算 Recall@K

    Args:
        retrieved: 檢索到的文檔 ID 列表
        relevant: 相關文檔 ID 列表
        k: 截斷位置

    Returns:
        Recall@K 分數
    """
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    return len(retrieved_k & relevant_set) / len(relevant_set)
```

**MRR (Mean Reciprocal Rank)**:

```python
def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]]
) -> float:
    """計算 MRR"""
    reciprocal_ranks = []

    for retrieved, relevant in zip(all_retrieved, all_relevant):
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)

    return np.mean(reciprocal_ranks)
```

#### 3.5.2 效能指標

**查詢延遲分布**:

```python
import time
import matplotlib.pyplot as plt

def benchmark_search(queries: List[str], num_runs: int = 100):
    """效能基準測試"""
    latencies = []

    for _ in range(num_runs):
        query = np.random.choice(queries)

        start = time.time()
        hybrid_search_parallel(query)
        latency = (time.time() - start) * 1000  # ms

        latencies.append(latency)

    # 統計
    print(f"Mean Latency: {np.mean(latencies):.1f}ms")
    print(f"P50 Latency: {np.percentile(latencies, 50):.1f}ms")
    print(f"P95 Latency: {np.percentile(latencies, 95):.1f}ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.1f}ms")

    # 視覺化
    plt.hist(latencies, bins=50)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Search Latency Distribution')
    plt.savefig('latency_distribution.png')
```

---

## 4. 與下一階段的銜接

完成混合檢索後,系統已能返回 Top-20 相關文檔。**階段四:重排序與生成**將:

1. 使用 Cross-Encoder 對 Top-20 結果進行精細重排序
2. 組裝 Context Window,準備 LLM 輸入
3. 透過 Prompt Engineering 生成高品質答案

> **關鍵依賴**:
>
> - 階段四將接收本階段產出的 Top-20 Chunks
> - `metadata.has_table` 等欄位將影響 Reranking 權重
> - 檢索分數將作為 Confidence Score 的參考依據
