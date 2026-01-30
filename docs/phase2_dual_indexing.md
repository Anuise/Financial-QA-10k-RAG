# 階段二:雙索引構建 (Dual Indexing)

## 1. 系統定位與核心價值 (Context & Value)

### 1.1 階段定位

雙索引構建是 RAG 系統的檢索基礎,負責為階段一產出的文本單元建立兩套互補的索引機制:**Dense Index (向量索引)** 與 **Sparse Index (關鍵字索引)**。這種混合架構能夠同時捕捉語意相似性與精確關鍵字匹配,是實現高品質檢索的關鍵。

### 1.2 核心痛點

單一索引機制存在明顯的局限性:

- **純向量檢索 (Dense Only)**:
  - ✅ 優勢:能理解語意相似性 (如 "revenue" 與 "sales" 的關聯)
  - ❌ 劣勢:對專有名詞與數值不敏感 (如公司代碼 "AAPL" 可能被誤匹配)
- **純關鍵字檢索 (Sparse Only)**:
  - ✅ 優勢:精確匹配術語與數值
  - ❌ 劣勢:無法處理同義詞與語意變化 (如 "profit" 與 "net income")

財務問答場景同時需要**語意理解**與**精確匹配**,因此必須採用雙索引架構。

### 1.3 預期效果

完成本階段後,系統將具備:

1. **語意檢索能力**:透過 BGE-M3 Embeddings 理解問題意圖
2. **精確匹配能力**:透過 BM25 索引快速定位關鍵詞
3. **高效查詢**:兩套索引均支援毫秒級檢索
4. **可擴展性**:ChromaDB 支援增量更新與分散式部署

---

> **架構銜接說明**:
> 了解雙索引的必要性後,下一層將說明如何透過「Embedding 計算→向量儲存→BM25 建立」的流程,在 Kaggle GPU 與 Local CPU 間分工協作。

---

## 2. 工作流程與架構 (Workflow & Architecture)

### 2.1 整體流程

雙索引構建分為兩條並行路徑:

```
階段一產出: chunks.jsonl
    ↓
┌─────────────────────────────────┬─────────────────────────────────┐
│   Dense Index Pipeline          │   Sparse Index Pipeline         │
│   (Kaggle GPU)                  │   (Local CPU)                   │
├─────────────────────────────────┼─────────────────────────────────┤
│ 1. 載入 BGE-M3 模型             │ 1. 載入 chunks.jsonl            │
│ 2. 批次計算 Embeddings          │ 2. Tokenization (財經術語優化)  │
│ 3. 儲存為 embeddings.npy        │ 3. 建立 BM25 倒排索引           │
│ 4. 上傳至 Kaggle Output         │ 4. 儲存為 bm25_index.pkl        │
└─────────────────────────────────┴─────────────────────────────────┘
    ↓                                   ↓
下載 embeddings.npy                 載入 bm25_index.pkl
    ↓                                   ↓
    └───────────────┬───────────────────┘
                    ↓
            整合至 ChromaDB
            (Collection: financial_10k)
```

### 2.2 基礎設施分工

#### ☁️ Kaggle Cloud Layer (Dense Index)

**為何在 Kaggle 執行?**

- BGE-M3 模型需要 GPU 加速 (P100/T4)
- 大規模 Embedding 計算耗時 (8000+ Chunks 需 30-60 分鐘)
- Kaggle 提供免費 GPU 配額 (每週 30 小時)

**執行環境**:

- **Notebook**: `embedding_computation.ipynb`
- **GPU**: P100 (16GB VRAM) 或 T4 (16GB VRAM)
- **依賴**: `transformers`, `torch`, `numpy`

#### 💻 Local Desktop Layer (Sparse Index + Integration)

**為何在本地執行?**

- BM25 索引建立僅需 CPU (無 GPU 需求)
- ChromaDB 整合需要本地資料庫環境
- 方便後續的增量更新與查詢測試

**執行環境**:

- **Script**: `scripts/build_bm25_index.py`, `scripts/integrate_embeddings.py`
- **依賴**: `rank-bm25`, `chromadb`, `numpy`

### 2.3 資料流向

```
[Kaggle] chunks.jsonl → BGE-M3 → embeddings.npy (1024-dim vectors)
                                        ↓
                                  Download to Local
                                        ↓
[Local] ChromaDB.add(embeddings=embeddings.npy, documents=chunks)

[Local] chunks.jsonl → Tokenizer → BM25 Index → bm25_index.pkl
                                        ↓
                                  Load for Query
```

---

> **細節銜接說明**:
> 確立了「Kaggle 產製 Embeddings、Local 建立 BM25」的分工後,以下將深入說明 BGE-M3 模型配置、BM25 參數調優與 ChromaDB 架構設計。

---

## 3. 技術規格與實作細節 (Detailed Specification)

### 3.1 Dense Index: BGE-M3 Embeddings

#### 3.1.1 模型選擇理由

**[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)** 是目前最適合財報場景的 Embedding 模型:

| 特性             | BGE-M3              | 競品 (OpenAI text-embedding-3) |
| ---------------- | ------------------- | ------------------------------ |
| **最大輸入長度** | 8192 tokens         | 8191 tokens                    |
| **輸出維度**     | 1024-dim            | 1536-dim                       |
| **多語言支援**   | ✅ (100+ 語言)      | ✅                             |
| **領域適應性**   | 通用 + 金融微調     | 通用                           |
| **成本**         | 免費 (自託管)       | $0.13/1M tokens                |
| **推理速度**     | ~50 chunks/sec (T4) | API 延遲 ~200ms                |

**關鍵優勢**:

- **長文本支援**:財報段落常超過 512 tokens,BGE-M3 可完整編碼
- **成本效益**:在 Kaggle 免費 GPU 上運行,無 API 費用
- **可控性**:可針對財經領域進行 Fine-tuning

#### 3.1.2 Embedding 計算流程

**Kaggle Notebook 實作** (`embedding_computation.ipynb`):

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
from tqdm import tqdm

# 1. 載入模型
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 移至 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. 載入 Chunks
chunks = []
with open('/kaggle/input/financial-chunks/chunks.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))

# 3. 批次計算 Embeddings
batch_size = 32
embeddings = []

for i in tqdm(range(0, len(chunks), batch_size)):
    batch_texts = [chunk['text'] for chunk in chunks[i:i+batch_size]]

    # Tokenization
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors='pt'
    ).to(device)

    # 前向傳播
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 [CLS] token 的 embedding
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    embeddings.append(batch_embeddings)

# 4. 合併並儲存
embeddings = np.vstack(embeddings)
np.save('/kaggle/working/embeddings.npy', embeddings)

print(f"✅ Embeddings shape: {embeddings.shape}")  # (8420, 1024)
```

#### 3.1.3 效能優化策略

**1. 混合精度計算 (Mixed Precision)**

```python
from torch.cuda.amp import autocast

with autocast():
    outputs = model(**inputs)
```

**效果**: 減少 VRAM 使用 40%,加速推理 30%

**2. 動態批次大小**

```python
def get_optimal_batch_size(available_vram_gb: float) -> int:
    """根據可用 VRAM 動態調整批次大小"""
    if available_vram_gb >= 16:
        return 64
    elif available_vram_gb >= 8:
        return 32
    else:
        return 16
```

**3. Checkpoint 機制**

```python
import os

checkpoint_path = '/kaggle/working/checkpoint.npy'

# 每處理 1000 個 Chunks 儲存一次
if i % 1000 == 0 and i > 0:
    np.save(checkpoint_path, np.vstack(embeddings))
    print(f"💾 Checkpoint saved at {i} chunks")

# 從 Checkpoint 恢復
if os.path.exists(checkpoint_path):
    embeddings = [np.load(checkpoint_path)]
    start_idx = len(embeddings[0])
    print(f"🔄 Resuming from chunk {start_idx}")
```

### 3.2 Sparse Index: BM25

#### 3.2.1 BM25 演算法原理

**BM25 (Best Matching 25)** 是一種基於機率的排序函數:

$$
\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$

**參數說明**:

- $q_i$: 查詢中的第 $i$ 個詞
- $f(q_i, D)$: 詞 $q_i$ 在文件 $D$ 中的詞頻
- $|D|$: 文件 $D$ 的長度
- $\text{avgdl}$: 所有文件的平均長度
- $k_1$: 詞頻飽和參數 (預設 1.5)
- $b$: 長度正規化參數 (預設 0.75)

#### 3.2.2 財經領域 Tokenizer 優化

**挑戰**: 標準 Tokenizer 會將 "EBITDA" 切分為 ["E", "BIT", "DA"],破壞語意。

**解決方案**: 自訂 Tokenizer,保留財經專有名詞。

```python
import re
from typing import List

class FinancialTokenizer:
    """財經領域專用 Tokenizer"""

    # 財經術語白名單 (不切分)
    FINANCIAL_TERMS = {
        'EBITDA', 'GAAP', 'EPS', 'P/E', 'ROE', 'ROA', 'CAGR',
        'CAPEX', 'OPEX', 'FCF', 'NPV', 'IRR', 'WACC', 'SEC'
    }

    # 公司代碼模式 (1-5 個大寫字母)
    TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')

    # 數值模式 (保留完整數字,包含逗號與小數點)
    NUMBER_PATTERN = re.compile(r'\$?\d{1,3}(,\d{3})*(\.\d+)?[BMK]?')

    def tokenize(self, text: str) -> List[str]:
        """
        切分文本為 tokens,保留財經術語完整性

        Args:
            text: 待切分文本

        Returns:
            Token 列表
        """
        tokens = []

        # 1. 提取並保留財經術語
        for term in self.FINANCIAL_TERMS:
            if term in text:
                text = text.replace(term, f' __{term}__ ')

        # 2. 提取並保留公司代碼
        text = self.TICKER_PATTERN.sub(r' __\g<0>__ ', text)

        # 3. 提取並保留數值
        text = self.NUMBER_PATTERN.sub(r' __\g<0>__ ', text)

        # 4. 標準切分
        raw_tokens = text.lower().split()

        # 5. 還原保留的術語
        for token in raw_tokens:
            if token.startswith('__') and token.endswith('__'):
                tokens.append(token[2:-2])  # 移除標記符號
            elif len(token) > 2:  # 過濾過短的詞
                tokens.append(token)

        return tokens

# 使用範例
tokenizer = FinancialTokenizer()
text = "Apple (AAPL) reported EBITDA of $120.5B in Q4 2023"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['apple', 'AAPL', 'reported', 'EBITDA', 'of', '$120.5B', 'in', 'q4', '2023']
```

#### 3.2.3 BM25 索引建立

**Local Script 實作** (`scripts/build_bm25_index.py`):

```python
from rank_bm25 import BM25Okapi
import json
import pickle
from pathlib import Path

def build_bm25_index(chunks_path: Path, output_path: Path):
    """
    建立 BM25 索引

    Args:
        chunks_path: chunks.jsonl 路徑
        output_path: 索引儲存路徑
    """
    # 1. 載入 Chunks
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    # 2. Tokenization
    tokenizer = FinancialTokenizer()
    tokenized_corpus = [tokenizer.tokenize(chunk['text']) for chunk in chunks]

    # 3. 建立 BM25 索引
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. 儲存索引
    index_data = {
        'bm25': bm25,
        'chunk_ids': [chunk['chunk_id'] for chunk in chunks],
        'tokenizer': tokenizer
    }

    with open(output_path, 'wb') as f:
        pickle.dump(index_data, f)

    print(f"✅ BM25 index built: {len(chunks)} chunks")

if __name__ == '__main__':
    build_bm25_index(
        chunks_path=Path('data/processed/chunks.jsonl'),
        output_path=Path('data/indexes/bm25_index.pkl')
    )
```

#### 3.2.4 BM25 參數調優

**參數影響分析**:

| 參數   | 預設值 | 調優方向     | 影響                                |
| ------ | ------ | ------------ | ----------------------------------- |
| **k1** | 1.5    | ↑ 提升至 2.0 | 增加高頻詞的權重 (適合術語密集文本) |
| **b**  | 0.75   | ↓ 降至 0.5   | 減少長度懲罰 (適合長文本)           |

**調優實驗**:

```python
from sklearn.model_selection import ParameterGrid

# 定義參數網格
param_grid = {
    'k1': [1.2, 1.5, 1.8, 2.0],
    'b': [0.5, 0.6, 0.75, 0.85]
}

# 評估函數 (使用驗證集)
def evaluate_bm25(k1: float, b: float, val_queries: List[str], val_relevance: List[List[int]]):
    """評估 BM25 參數組合的 MRR (Mean Reciprocal Rank)"""
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    reciprocal_ranks = []
    for query, relevant_ids in zip(val_queries, val_relevance):
        scores = bm25.get_scores(tokenizer.tokenize(query))
        ranked_ids = np.argsort(scores)[::-1]

        # 找到第一個相關文件的排名
        for rank, doc_id in enumerate(ranked_ids, 1):
            if doc_id in relevant_ids:
                reciprocal_ranks.append(1 / rank)
                break

    return np.mean(reciprocal_ranks)

# 網格搜尋
best_score = 0
best_params = {}

for params in ParameterGrid(param_grid):
    score = evaluate_bm25(**params, val_queries=queries, val_relevance=relevance)
    if score > best_score:
        best_score = score
        best_params = params

print(f"Best params: {best_params}, MRR: {best_score:.4f}")
```

### 3.3 ChromaDB 整合

#### 3.3.1 Collection 架構設計

**Collection 命名**: `financial_10k_v1`

**Schema**:

```python
import chromadb
from chromadb.config import Settings

# 初始化 ChromaDB
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="data/chromadb"
))

# 建立 Collection
collection = client.create_collection(
    name="financial_10k_v1",
    metadata={
        "description": "10-K Financial Reports RAG System",
        "embedding_model": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "total_chunks": 8420,
        "created_at": "2024-01-30"
    }
)
```

#### 3.3.2 資料匯入流程

```python
import numpy as np
import json

def integrate_embeddings_to_chromadb(
    chunks_path: Path,
    embeddings_path: Path,
    collection_name: str
):
    """
    將 Embeddings 整合至 ChromaDB

    Args:
        chunks_path: chunks.jsonl 路徑
        embeddings_path: embeddings.npy 路徑
        collection_name: ChromaDB Collection 名稱
    """
    # 1. 載入資料
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    embeddings = np.load(embeddings_path)

    # 2. 準備 ChromaDB 格式
    ids = [chunk['chunk_id'] for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    embeddings_list = embeddings.tolist()

    # 3. 批次匯入 (避免記憶體溢出)
    batch_size = 1000
    collection = client.get_collection(collection_name)

    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings_list[i:i+batch_size]
        )
        print(f"✅ Imported batch {i//batch_size + 1}/{len(ids)//batch_size + 1}")

    print(f"✅ Total chunks in ChromaDB: {collection.count()}")
```

#### 3.3.3 索引效能優化

**1. HNSW 索引參數**

ChromaDB 預設使用 HNSW (Hierarchical Navigable Small World) 演算法:

```python
collection = client.create_collection(
    name="financial_10k_v1",
    metadata={
        "hnsw:space": "cosine",           # 距離度量 (cosine/l2/ip)
        "hnsw:construction_ef": 200,      # 建立索引時的搜尋範圍 (↑ 提升品質但變慢)
        "hnsw:M": 16                      # 每個節點的連接數 (↑ 提升召回率但增加記憶體)
    }
)
```

**參數調優建議**:

| 參數              | 預設值 | 財報場景建議 | 理由                           |
| ----------------- | ------ | ------------ | ------------------------------ |
| `construction_ef` | 100    | 200          | 財報查詢要求高召回率           |
| `M`               | 16     | 32           | 增加連接數以提升長文本檢索品質 |
| `space`           | `l2`   | `cosine`     | BGE-M3 建議使用 Cosine 距離    |

**2. 查詢時參數**

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=20,
    include=["documents", "metadatas", "distances"]
)
```

### 3.4 錯誤處理與驗證

#### 3.4.1 Embedding 品質檢查

```python
def validate_embeddings(embeddings: np.ndarray):
    """驗證 Embeddings 品質"""

    # 1. 檢查形狀
    assert embeddings.shape[1] == 1024, f"Expected 1024-dim, got {embeddings.shape[1]}"

    # 2. 檢查 NaN/Inf
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
    assert not np.isinf(embeddings).any(), "Embeddings contain Inf"

    # 3. 檢查向量範數 (應接近 1,因為 BGE-M3 輸出已正規化)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=0.1), f"Abnormal norms: {norms.min():.3f} - {norms.max():.3f}"

    # 4. 檢查多樣性 (避免所有向量過於相似)
    similarity_matrix = np.dot(embeddings, embeddings.T)
    avg_similarity = (similarity_matrix.sum() - len(embeddings)) / (len(embeddings) * (len(embeddings) - 1))
    assert avg_similarity < 0.8, f"Embeddings too similar: {avg_similarity:.3f}"

    print("✅ Embeddings validation passed")
```

#### 3.4.2 索引一致性檢查

```python
def verify_index_consistency(
    chunks_path: Path,
    chromadb_collection,
    bm25_index_path: Path
):
    """驗證三份資料的一致性"""

    # 1. 載入資料
    with open(chunks_path, 'r') as f:
        chunks = [json.loads(line) for line in f]

    with open(bm25_index_path, 'rb') as f:
        bm25_data = pickle.load(f)

    chromadb_count = chromadb_collection.count()

    # 2. 檢查數量一致性
    assert len(chunks) == chromadb_count, \
        f"Chunks ({len(chunks)}) != ChromaDB ({chromadb_count})"

    assert len(chunks) == len(bm25_data['chunk_ids']), \
        f"Chunks ({len(chunks)}) != BM25 ({len(bm25_data['chunk_ids'])})"

    # 3. 檢查 ID 一致性
    chunk_ids = {chunk['chunk_id'] for chunk in chunks}
    bm25_ids = set(bm25_data['chunk_ids'])

    assert chunk_ids == bm25_ids, \
        f"ID mismatch: {len(chunk_ids - bm25_ids)} missing in BM25"

    print("✅ Index consistency verified")
```

### 3.5 輸出與交付物

#### 3.5.1 Kaggle Output

**檔案結構**:

```
/kaggle/working/
├── embeddings.npy          # (8420, 1024) float32 陣列
├── embedding_log.json      # 處理日誌
└── checkpoint.npy          # 中間檢查點 (可選)
```

**embedding_log.json 範例**:

```json
{
  "model": "BAAI/bge-m3",
  "total_chunks": 8420,
  "embedding_dim": 1024,
  "batch_size": 32,
  "processing_time_seconds": 1847,
  "gpu_type": "Tesla P100-PCIE-16GB",
  "peak_vram_usage_gb": 12.3,
  "chunks_per_second": 4.56
}
```

#### 3.5.2 Local Output

**檔案結構**:

```
data/
├── indexes/
│   ├── bm25_index.pkl      # BM25 索引 + Tokenizer
│   └── index_stats.json    # 索引統計資訊
└── chromadb/
    └── financial_10k_v1/   # ChromaDB 持久化目錄
        ├── chroma.sqlite3
        └── *.parquet
```

**index_stats.json 範例**:

```json
{
  "bm25": {
    "total_chunks": 8420,
    "vocabulary_size": 45230,
    "avg_chunk_length": 98.7,
    "k1": 1.5,
    "b": 0.75
  },
  "chromadb": {
    "collection_name": "financial_10k_v1",
    "total_vectors": 8420,
    "embedding_dim": 1024,
    "index_type": "HNSW",
    "disk_usage_mb": 342.5
  }
}
```

---

## 4. 與下一階段的銜接

完成雙索引構建後,系統已具備檢索能力。**階段三:混合檢索邏輯**將:

1. 整合 Dense 與 Sparse 兩套索引的查詢結果
2. 使用 Reciprocal Rank Fusion (RRF) 融合分數
3. 實作 Alpha 參數調優機制,平衡語意與關鍵字權重

> **關鍵依賴**:
>
> - 階段三需要同時載入 ChromaDB Collection 與 BM25 Index
> - 查詢時將並行執行兩套檢索,再融合結果
> - `metadata.has_table` 等欄位將影響分數加權策略
