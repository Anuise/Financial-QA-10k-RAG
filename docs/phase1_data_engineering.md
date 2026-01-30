# 階段一:資料工程 (Data Engineering)

## 1. 系統定位與核心價值 (Context & Value)

### 1.1 階段定位

資料工程是整個 RAG 系統的基石,負責將原始的 10-K 財務報表轉換為結構化、可檢索的文本單元 (Chunks)。這個階段的品質直接決定了後續檢索與生成的上限——即使擁有最先進的 Embedding 模型與 LLM,若資料切分不當,系統仍無法提供精確的答案。

### 1.2 核心痛點

- **QA 資料集特性**：資料以 Question-Answer Pairs 形式存在，需解析 CSV 格式而非原始文本。
- **Context 重複性**：同一段 Context 可能對應多個問題，需進行去重 (Deduplication) 以避免索引冗餘。
- **資料品質**：Context 欄位可能包含不完整句子或特殊字元，需進行品質過濾。
- **欄位映射**：需正確將 CSV 欄位映射至 RAG 系統所需的 Metadata (如 ticker, filing year)。

### 1.3 預期效果

完成本階段後,系統將具備:

1. **高品質文本單元**:每個 Chunk 語意完整,不會截斷關鍵資訊
2. **結構保留**:表格與章節標題的完整性得到維護
3. **可追溯性**:每個 Chunk 都能回溯到原始文件的位置
4. **高效處理**:支援批次處理大量 10-K 文件

---

> **架構銜接說明**:
> 了解資料工程的價值後,下一層將說明如何透過「清洗→切分→驗證」的三階段流程,將原始財報轉換為可用的文本單元。

---

## 2. 工作流程與架構 (Workflow & Architecture)

### 2.1 整體流程

資料工程遵循以下處理流程:

```
Financial-QA-10k.csv
    ↓
[階段 2.2] CSV 解析與欄位驗證
    ↓
[階段 2.3] Context 去重與清洗
    ↓
[階段 2.4] 品質驗證
    ↓
Processed Chunks (.jsonl)
```

### 2.2 資料清洗 (Data Cleaning)

**目標**:移除格式噪音,標準化文本格式。

**處理規則**:

1. **HTML 標籤移除**:去除 `<div>`, `<p>`, `<table>` 等標籤
2. **Unicode 正規化**:修正亂碼字元 (如 `\u00a0` → 空格)
3. **空白字元處理**:
   - 移除多餘的空行 (連續超過 2 行)
   - 統一縮排格式 (Tab → 4 Spaces)
4. **特殊符號保留**:保留財務符號 ($, %, ±) 與專有名詞 (GAAP, EBITDA)

### 2.3 Context 處理策略 (Context Processing)

**目標**:從 QA 資料集中提取 Context 並轉換為標準化的文本單元，同時保留問答對作為評估資料。

#### 策略：Context-as-Chunk

由於資料集已經提供了從財報中擷取出的 `context` 欄位，我們將直接以此作為基礎文本單元 (Chunk)，而不進行額外的複雜切分。

**處理邏輯**:

1. **去重 (Deduplication)**:
   - 檢查 `context` 內容是否重複。
   - 若多個 Question 對應同一個 Context，僅保留一份 Context 作為索引，並將相關 Questions 記錄在 Metadata 中。

2. **長度檢查**:
   - 統計 Context 的 Token 數。
   - 若 Context 超過 Embedding 模型限制 (如 512 tokens)，則進行次級切分 (以句子為單位)，但通常 QA 資料集的 Context 較短，此情況較少見。

3. **Metadata 綁定**:
   - 將 `ticker` (股票代碼) 與 `filing` (年份) 綁定至 Chunk。
   - 保留原始 `question` 與 `answer`，用於後續 Retrieval Evaluation。

### 2.4 品質驗證 (Quality Validation)

**驗證指標**:

1. **Chunk 數量統計**:確保每份文件產生合理數量的 Chunks (預期 50-200 個)
2. **長度分布檢查**:繪製 Chunk 長度直方圖,識別異常值
3. **重複檢測**:使用 MinHash 演算法偵測高度相似的 Chunks
4. **關鍵詞覆蓋**:確認財務術語 (如 Revenue, Assets) 在 Chunks 中的分布

---

> **細節銜接說明**:
> 確立了「清洗→切分→驗證」的工作流程後,以下將深入說明具體的技術規格、參數調優策略與錯誤處理機制。

---

## 3. 技術規格與實作細節 (Detailed Specification)

### 3.1 Chunking 演算法詳解

#### 3.1.1 Recursive Character Splitter 實作

**核心邏輯**:

```python
def recursive_split(text: str, chunk_size: int, overlap: int, separators: List[str]) -> List[str]:
    """
    遞迴式文本切分演算法

    Args:
        text: 待切分文本
        chunk_size: 目標 Chunk 大小 (tokens)
        overlap: 重疊區域大小 (tokens)
        separators: 分隔符優先順序列表

    Returns:
        切分後的 Chunk 列表
    """
    chunks = []
    current_chunk = ""

    # 以第一個分隔符切分
    segments = text.split(separators[0])

    for segment in segments:
        # 若加入此段落後超過 chunk_size
        if len(current_chunk) + len(segment) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                # 保留 overlap 部分
                current_chunk = current_chunk[-overlap:] + segment
            else:
                # 若單一段落超過 chunk_size,使用下一個分隔符遞迴切分
                if len(separators) > 1:
                    chunks.extend(recursive_split(segment, chunk_size, overlap, separators[1:]))
                else:
                    chunks.append(segment[:chunk_size])
        else:
            current_chunk += separators[0] + segment

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
```

**參數調優建議**:

| 參數       | 預設值                 | 調優方向      | 影響                            |
| ---------- | ---------------------- | ------------- | ------------------------------- |
| Chunk Size | 512 tokens             | ↑ 提升至 1024 | 更完整的語意,但檢索精度下降     |
| Overlap    | 50 tokens              | ↑ 提升至 100  | 減少邊界資訊遺失,但增加儲存成本 |
| Separators | `["\n\n", "\n", ". "]` | 新增 `: `     | 更細緻的切分,適合列表型內容     |

#### 3.1.2 Structure-Aware Splitting 實作

**表格偵測正則表達式**:

```python
import re

# 偵測 Markdown 風格表格
TABLE_PATTERN = re.compile(r'(\|[^\n]+\|\n)+')

# 偵測 Tab 分隔表格
TAB_TABLE_PATTERN = re.compile(r'([^\n]+\t[^\n]+\n){3,}')

def detect_tables(text: str) -> List[Tuple[int, int]]:
    """
    偵測文本中的表格位置

    Returns:
        表格的 (起始位置, 結束位置) 列表
    """
    tables = []

    # Markdown 表格
    for match in TABLE_PATTERN.finditer(text):
        tables.append((match.start(), match.end()))

    # Tab 表格
    for match in TAB_TABLE_PATTERN.finditer(text):
        tables.append((match.start(), match.end()))

    return tables
```

**標題綁定邏輯**:

```python
def bind_heading_to_table(text: str, table_start: int) -> int:
    """
    找到表格上方最近的標題,返回標題起始位置

    Args:
        text: 完整文本
        table_start: 表格起始位置

    Returns:
        標題起始位置 (若無標題則返回 table_start)
    """
    # 向上搜尋最近的 H1-H3 標題
    heading_pattern = re.compile(r'^#{1,3}\s+.+$', re.MULTILINE)

    # 取得表格之前的文本
    before_table = text[:table_start]

    # 找到最後一個標題
    headings = list(heading_pattern.finditer(before_table))
    if headings:
        last_heading = headings[-1]
        # 若標題距離表格不超過 2 行,則綁定
        lines_between = before_table[last_heading.end():table_start].count('\n')
        if lines_between <= 2:
            return last_heading.start()

    return table_start
```

### 3.2 資料清洗規則

#### 3.2.1 HTML 標籤處理

**使用工具**: BeautifulSoup4

```python
from bs4 import BeautifulSoup

def clean_html(text: str) -> str:
    """移除 HTML 標籤,保留純文本"""
    soup = BeautifulSoup(text, 'html.parser')

    # 移除 script 與 style 標籤
    for tag in soup(['script', 'style']):
        tag.decompose()

    # 取得純文本
    clean_text = soup.get_text()

    # 移除多餘空白
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)

    return clean_text.strip()
```

#### 3.2.2 Unicode 正規化

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    """標準化 Unicode 字元"""
    # NFKC 正規化 (相容性分解 + 標準組合)
    text = unicodedata.normalize('NFKC', text)

    # 替換常見亂碼
    replacements = {
        '\u00a0': ' ',      # Non-breaking space
        '\u2019': "'",      # Right single quotation mark
        '\u201c': '"',      # Left double quotation mark
        '\u201d': '"',      # Right double quotation mark
        '\u2013': '-',      # En dash
        '\u2014': '--',     # Em dash
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text
```

### 3.3 品質驗證機制

#### 3.3.1 Chunk 長度分布分析

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_chunk_distribution(chunks: List[str], output_path: str):
    """
    分析 Chunk 長度分布,生成視覺化報告

    Args:
        chunks: Chunk 列表
        output_path: 圖表儲存路徑
    """
    lengths = [len(chunk.split()) for chunk in chunks]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
    plt.axvline(np.median(lengths), color='green', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
    plt.xlabel('Chunk Length (words)')
    plt.ylabel('Frequency')
    plt.title('Chunk Length Distribution')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

    # 輸出統計資訊
    print(f"Total Chunks: {len(chunks)}")
    print(f"Mean Length: {np.mean(lengths):.2f} words")
    print(f"Std Dev: {np.std(lengths):.2f} words")
    print(f"Min Length: {np.min(lengths)} words")
    print(f"Max Length: {np.max(lengths)} words")
```

#### 3.3.2 重複檢測 (MinHash)

```python
from datasketch import MinHash, MinHashLSH

def detect_duplicates(chunks: List[str], threshold: float = 0.8) -> List[Tuple[int, int]]:
    """
    使用 MinHash LSH 偵測高度相似的 Chunks

    Args:
        chunks: Chunk 列表
        threshold: 相似度閾值 (0-1)

    Returns:
        重複 Chunk 的索引對列表
    """
    # 建立 LSH 索引
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # 為每個 Chunk 建立 MinHash
    minhashes = []
    for i, chunk in enumerate(chunks):
        m = MinHash(num_perm=128)
        for word in chunk.split():
            m.update(word.encode('utf-8'))
        lsh.insert(f"chunk_{i}", m)
        minhashes.append(m)

    # 找出重複對
    duplicates = []
    for i, m in enumerate(minhashes):
        results = lsh.query(m)
        for result in results:
            j = int(result.split('_')[1])
            if i < j:  # 避免重複記錄
                duplicates.append((i, j))

    return duplicates
```

### 3.4 錯誤處理策略

#### 3.4.1 異常 Chunk 處理

```python
class ChunkValidator:
    """Chunk 驗證器"""

    def __init__(self, min_length: int = 50, max_length: int = 2048):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, chunk: str, chunk_id: str) -> Tuple[bool, Optional[str]]:
        """
        驗證 Chunk 是否符合品質標準

        Returns:
            (是否通過, 錯誤訊息)
        """
        # 檢查長度
        length = len(chunk.split())
        if length < self.min_length:
            return False, f"Chunk {chunk_id} too short: {length} words"
        if length > self.max_length:
            return False, f"Chunk {chunk_id} too long: {length} words"

        # 檢查是否為空白或重複字元
        if len(set(chunk.replace(' ', ''))) < 10:
            return False, f"Chunk {chunk_id} contains mostly repeated characters"

        # 檢查是否包含可讀文字
        alpha_ratio = sum(c.isalpha() for c in chunk) / len(chunk)
        if alpha_ratio < 0.5:
            return False, f"Chunk {chunk_id} has low alphabetic ratio: {alpha_ratio:.2%}"

        return True, None
```

#### 3.4.2 處理日誌記錄

```python
import logging
from pathlib import Path

def setup_processing_logger(log_dir: Path) -> logging.Logger:
    """設定資料處理日誌"""
    logger = logging.getLogger('data_engineering')
    logger.setLevel(logging.INFO)

    # 檔案處理器
    fh = logging.FileHandler(log_dir / 'processing.log')
    fh.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger
```

### 3.5 效能優化

#### 3.5.1 批次處理

```python
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

def batch_process_files(
    file_paths: List[Path],
    process_func: Callable,
    num_workers: int = 4
) -> List[Any]:
    """
    多進程批次處理檔案

    Args:
        file_paths: 檔案路徑列表
        process_func: 處理函數 (接受 Path,返回結果)
        num_workers: 工作進程數

    Returns:
        處理結果列表
    """
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_func, file_paths))
    return results
```

#### 3.5.2 記憶體管理

```python
def process_large_file(file_path: Path, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """
    以串流方式處理大型檔案,避免記憶體溢出

    Args:
        file_path: 檔案路徑
        chunk_size: 每次讀取的位元組數

    Yields:
        文本區塊
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data
```

### 3.6 輸出格式

#### 3.6.1 JSONL 結構

每個處理後的 Chunk 將以 JSONL 格式儲存:

```json
{
  "chunk_id": "NVDA_2023_10K_chunk_001",
  "document_id": "NVDA_2023_10K",
  "text": "Since our original focus on PC graphics, we have expanded into various markets.",
  "metadata": {
    "ticker": "NVDA",
    "filing": "2023_10K",
    "source_question": "What area did NVIDIA initially focus on before expanding into other markets?",
    "source_answer": "NVIDIA initially focused on PC graphics.",
    "word_count": 14,
    "token_count": 18
  }
}
```

#### 3.6.2 處理報告

生成 `processing_report.json` 記錄處理統計:

```json
{
  "total_documents": 150,
  "total_chunks": 8420,
  "processing_time_seconds": 342.5,
  "statistics": {
    "avg_chunks_per_doc": 56.13,
    "avg_chunk_length_words": 98.7,
    "chunks_with_tables": 1240,
    "duplicates_removed": 23
  },
  "errors": [
    {
      "document_id": "TSLA_10K_2023",
      "error": "Failed to parse table on line 4520",
      "severity": "warning"
    }
  ]
}
```

---

## 4. 與下一階段的銜接

完成資料工程後,產出的 `chunks.jsonl` 將作為**階段二:雙索引構建**的輸入。下一階段將:

1. 使用 BGE-M3 模型為每個 Chunk 計算 Embedding (Dense Index)
2. 建立 BM25 倒排索引 (Sparse Index)
3. 將兩套索引儲存至 ChromaDB 與本地檔案

> **關鍵依賴**:
>
> - 階段二依賴本階段產出的 `chunks.jsonl`
> - Chunk 的 `metadata.chunk_id` 將作為索引的主鍵
> - `metadata.has_table` 標記將影響 Reranking 階段的權重調整
