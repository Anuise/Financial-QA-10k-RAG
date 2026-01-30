# Financial-QA-10k-RAG：財務報表智慧問答系統

## 1. 專案背景與核心價值 (Project Context & Value)

### 1.1 系統定位

本專案旨在解決財務領域中「長文本理解」與「精確數值檢索」的雙重挑戰。透過建構一個專為 10-K 財務報表設計的**檢索增強生成 (RAG)** 系統，我們能夠讓使用者透過自然語言提問，快速獲取精確的財務數據與趨勢分析，大幅降低閱讀與查找財報的時間成本。

### 1.2 核心痛點解決

傳統關鍵字搜尋在處理財報時常面臨詞彙不匹配或語意模糊的問題；而通用 LLM 直接回答則容易產生幻覺 (Hallucination)。本系統透過以下方式解決：

- **混合檢索 (Hybrid Search)**：結合關鍵字精確度與向量語意理解。
- **領域專用優化**：針對財報結構設計特定的 Chunking 與 Reranking 策略。

### 1.3 資料來源

專案採用 **[Financial Q&A 10-K Dataset](https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k)**，包含 **7,000 筆** 從 S&P 500 公司年度報告 (10-K) 中擷取的問答對 (Question-Answer Pairs)，並附帶原始上下文 (Context)。

- **資料格式**：CSV
- **核心欄位**：`question`, `answer`, `context`, `ticker`, `filing`
- **用途**：RAG 檢索語料庫 (`context`) 與 評估基準 (`question`/`answer`)。

---

> **架構銜接說明**：
> 了解系統目標後，下一層將說明如何透過「雙層基礎設施」與「六階段流水線」來實現上述價值。本專案特別針對算力需求進行了架構分流。

---

## 2. 系統架構與工作流 (Workflow & Architecture)

### 2.1 基礎設施分流策略 (Infrastructure Strategy)

為了在效能與成本間取得平衡，本專案採用 **Kaggle (Cloud GPU)** 與 **Local (CPU/Lite GPU)** 的混合部署模式：

#### ☁️ Kaggle Cloud Layer (GPU Tier)

**核心職責**：處理高算力消耗的重型任務。

- **依賴性**：必須使用 GPU (P100/T4)。
- **主要任務**：
  1. **Embedding 計算**：使用 BAAI/bge-m3 進行大規模文本向量化。
  2. **模型微調 (Fine-tuning)**：若有需求，在此層進行 Cross-Encoder 或 LLM 的微調。
  3. **資料前處理**：執行複雜的語意切分 (Semantic Chunking)。

#### 💻 Local Desktop Layer (CPU Tier)

**核心職責**：處理即時檢索、資料庫管理與應用層邏輯。

- **依賴性**：僅需 CPU 或輕量級顯卡。
- **主要任務**：
  1. **向量資料庫 (ChromaDB)**：載入由 Kaggle 預算好的 Embeddings。
  2. **混合檢索運算**：執行 Vector + BM25 的加權計算。
  3. **應用服務 (API/UI)**：運行 FastAPI 與 Streamlit 前端介面。

### 2.2 核心數據流程 (Data Pipeline)

資料流向遵循以下路徑：
`Financial-QA-10k.csv` → `Extract Context` → `Kaggle GPU (Embedding)` → `Export Vectors` → `Local DB Import` → `RAG Inference`

---

> **細節銜接說明**：
> 確立了「Kaggle 產製、Local 服務」的架構後，以下將深入拆解實作 RAG 系統的六個具體技術階段，包含參數配置與演算法選擇。

---

## 3. 技術規格詳解 (Detailed Specification)

本專案依據開發時序分為六大階段，每階段均有明確的技術指標與學習目標。

### 階段一：資料工程 (Data Engineering)

針對財報的半結構化特性（表格、標題、大段落）進行處理。

- **清洗 rules**：解析 CSV 格式，處理 CSV 跳脫字元，去除重複的 Context。
- **Chunking 策略**：
  - **Context-as-Chunk**：直接使用資料集提供的 `context` 欄位作為基礎文本單元。
  - **Metadata Mapping**：將 `question` 與 `answer` 保留為 metadata，用於後續檢索評估。

### 階段二：雙索引構建 (Dual Indexing)

建立兩套獨立的索引機制以支援混合檢索。

- **Dense Index (向量)**
  - **Model**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
  - **特性**：支援多語言、8192 token 長度，適合財報長文。
  - **Storage**: ChromaDB (Collections 分類管理)。
- **Sparse Index (關鍵字)**
  - **Algorithm**: Rank-BM25
  - **Tokenizer**: 針對財經術語優化的斷詞器 (保留 GAAP, EBITDA 等專有名詞)。

### 階段三：混合檢索邏輯 (Hybrid Search)

整合兩套索引的優勢。

- **融合算法**：Reciprocal Rank Fusion (RRF) 或 Weighted Sum。
- **Alpha 參數調優**：
  - `Score = Alpha * Vector_Score + (1 - Alpha) * BM25_Score`
  - 預設 Alpha：0.5 (可根據 Evaluation 結果動態調整)。

### 階段四：重排序與生成 (Reranking & Generation)

- **Reranker**：採用 Cross-Encoder (如 BGE-Reranker-v2-m3) 對 Top-K 初篩結果進行精細排序。
- **Prompt Engineering**：
  - Context Window 控制：確保不超過 LLM 最大輸入限制。
  - 提示詞模板：包含 "Role", "Constraints" (如：僅回答財報內容), "Source Citation"。

### 階段五：系統評估 (Evaluation)

使用 **RAGAS** 框架進行量化測試。

- **主要指標**：
  - **Context Recall**：是否找全了所有相關段落？
  - **Faithfulness**：回答是否忠於原文，無幻覺？
  - **Answer Correctness**：與標準答案 (Ground Truth) 的相似度。

### 階段六：部署與介面 (Deployment)

- **Backend**: FastAPI (提供 `/search`, `/chat` Endpoints)。
- **Frontend**: Streamlit (提供對話視窗、引用來源跳轉、Confidence Score 顯示)。
